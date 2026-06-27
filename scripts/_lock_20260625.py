"""THROWAWAY lock driver (2026-06-25 CUR v IVC + ECU v GER). No pipeline changes.
Fetches fresh AMERICAN odds, prices all 20 rows with confirmed router keys, prints lock sheet."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib.odds_api import fetch_event_odds
from odds_lib import slate
from odds_lib.field_model import FieldMeanEstimator
from odds_lib.optimizer import optimize
from odds_lib.edge import compute_edge_table, classify, K_PRIOR, edge_submit
from odds_lib.measurement import LOG_PATH, build_edge_frame
from odds_lib.player_prop_pricing import price_player_prop
import json

SPORT = "soccer_fifa_world_cup"
MKTS = ("h2h,totals,btts,h2h_h1,spreads,totals_h1,team_totals,"
        "player_shots_on_target,alternate_totals_corners,alternate_team_totals_corners,"
        "alternate_totals_cards,alternate_team_totals,alternate_totals_h2,corners_1x2,alternate_totals")

FIXTURES = {
    "Curacao vs Ivory Coast": ("a5a0b544a984ab16e564264f3e859b43", [
        ("Q1", "Will Curacao be caught offside 2+ times?",          "team_offsides_over",   "Curaçao",     "1.5", None),
        ("Q2", "Will Ivory Coast commit more fouls than Curacao?",  "team_more_fouls",      "Ivory Coast", "",    None),
        ("Q3", "Will Ivory Coast score the first goal of 2H?",      "team_first_goal_2h",   "Ivory Coast", "",    None),
        ("Q4", "Will 2H have more goals than 1H?",                  "second_half_more_goals","",          "",    None),
        ("Q5", "Will Curacao have more SOT than IVC in 2H?",        "team_more_sot_2h",     "Curaçao",     "",    None),
        ("Q6", "Will Curacao receive more cards than IVC?",         "team_more_cards",      "Curaçao",     "",    None),
        ("Q7", "Will Curacao score at least 1 goal?",              "team_score_any",       "Curaçao",     "0.5", None),
        ("Q8", "Will Ivory Coast have 5+ corners?",                "team_corners_over",    "Ivory Coast", "4.5", None),
        ("Q9", "Will Leandro Bacuna have 1+ SOT?",                 "player_sot_over",      "",            "0.5", "Leandro Bacuna"),
        ("Q10","Will Ibrahim Sangare have 1+ SOT?",               "player_sot_over",      "",            "0.5", "Ibrahim Sangaré"),
    ]),
    "Ecuador vs Germany": ("0ec28b84ec399bd4dffbe8d1bc72b3c4", [
        ("Q1", "At HT both teams 1+ SOT each?",                    "both_teams_sot_1h",    "",            "0.5", None),
        ("Q2", "Will Ecuador commit more fouls than Germany?",     "team_more_fouls",      "Ecuador",     "",    None),
        ("Q3", "Will Ecuador have more corners than GER in 2H?",   "team_more_corners_2h", "Ecuador",     "",    None),
        ("Q4", "Will Germany score more goals than ECU in 2H?",    "team_more_goals_2h",   "Germany",     "",    None),
        ("Q5", "Will Germany have more SOT than ECU in 2H?",       "team_more_sot_2h",     "Germany",     "",    None),
        ("Q6", "Will a penalty be awarded OR a red shown?",        "penalty_or_red",       "",            "",    None),
        ("Q7", "Will Ecuador be caught offside 2+ times?",         "team_offsides_over",   "Ecuador",     "1.5", None),
        ("Q8", "BTTS AND 3+ total goals?",                         "compound_btts_over_2_5","",           "2.5", None),
        ("Q9", "Will Ecuador score at least 1 goal?",             "team_score_any",       "Ecuador",     "0.5", None),
        ("Q10","Will the match have 2 or fewer goals?",           "match_total_under",    "",            "2.5", None),
    ]),
}

field = FieldMeanEstimator()
edge_table = (compute_edge_table(build_edge_frame(pd.read_csv(LOG_PATH, dtype=str)))
              if LOG_PATH.exists() else pd.DataFrame())
bacuna_raw = None

import glob, time
def fresh_or_fetch(eid):
    """Reuse a cache written in the last 10 min (this run's fresh fetch); else fetch."""
    fs = glob.glob(f"data/raw/soccer_fifa_world_cup__event-{eid}__*.json")
    if fs:
        newest = max(fs, key=lambda p: Path(p).stat().st_mtime)
        if time.time() - Path(newest).stat().st_mtime < 600:
            print(f"[reuse fresh cache] {Path(newest).name}")
            return newest
    return fetch_event_odds(SPORT, eid, markets=MKTS, regions="us,uk,eu")

print("="*100)
for match, (eid, qs) in FIXTURES.items():
    cache = fresh_or_fetch(eid)
    game = json.loads(Path(cache).read_text()); game = game if isinstance(game, dict) else game[0]
    home, away = game["home_team"], game["away_team"]
    c = slate.build_consensus([game], market_keys=("h2h","totals","btts","h2h_h1"))
    model = slate.build_model(c, home, away, game_json=game)
    h1_src = model[4] if len(model) > 4 else "?"
    print(f"\n### {match}   home={home} | away={away}   (H1 src={h1_src})")
    print(f"{'Q':4}{'qt':24}{'target':14}{'line':5}{'tier':22}{'k':5}{'c_hat':7}{'raw':8}{'SUBMIT':8} src")
    for qn, _txt, qt, tgt, line, player in qs:
        row = {"question_type": qt, "target_team": tgt, "line": line, "target_player": player}
        if player:  # prop: direct pricer (lineup-gate handled in notes), treat confirmed-starter for lock
            pr = price_player_prop(qt, player, float(line), game)
            if not pr.mapped:
                tier, raw = "PENDING", None
            else:
                tier = slate._prop_tier(pr); raw = pr.market_prob_vig_adjusted
            mkt_raw = pr.market_prob_raw if pr.mapped else None
            books = pr.books_used if pr.mapped else ""
            if player == "Leandro Bacuna" and raw is not None:
                bacuna_raw = raw
        else:
            tier, raw, _ = slate.resolve_row(row, c, game, model, lineup=None)
            mkt_raw = None; books = ""
        fe = field.estimate(qt)
        sub = optimize(tier=tier, question_type=qt, p_hat=raw, shadow=fe.q_hat, table=edge_table,
                       lower_bound=(tier == "PROP_proxy_floor"))
        raws = f"{raw:.4f}" if raw is not None else "  -  "
        extra = f"  [vig_raw={mkt_raw:.4f} books={books}]" if player and mkt_raw is not None else ""
        print(f"{qn:4}{qt:24}{(tgt or '-'):14}{(line or '-'):5}{tier:22}{sub.k:<5.2f}{fe.q_hat:<7.3f}{raws:8}{sub.q:<8.4f}{sub.source_class}{extra}")

# explicit override proof
print("\n--- Bacuna override proof: edge_submit(raw, c_hat, k=1.0) == raw ---")
if bacuna_raw is not None:
    c_hat = field.estimate("player_sot_over").q_hat
    k040 = K_PRIOR[("PROP", "thin")]
    print(f"raw={bacuna_raw:.4f}  c_hat={c_hat:.4f}")
    print(f"pipeline (k={k040}):   edge_submit(raw,c_hat,{k040}) = {edge_submit(bacuna_raw, c_hat, k040):.4f}")
    print(f"OVERRIDE (k=1.0):      edge_submit(raw,c_hat,1.0)  = {edge_submit(bacuna_raw, c_hat, 1.0):.4f}  == raw? {abs(edge_submit(bacuna_raw, c_hat, 1.0)-bacuna_raw)<1e-9}")
else:
    print("Bacuna prop NOT mapped/quoted -> cannot price (report PENDING)")
