"""Authoritative coverage resolver: market + engine(goals) + SOT rate layer +
props, for every question row. Prints per-row status and honest counts by
tier so we never hand-count again.

Tiers:
  MARKET        - de-vigged market p (highest confidence)
  PROP          - one-sided prop, vig-adjusted (liquidity-flagged)
  ENGINE_GOALS  - market-calibrated goals simulation
  RATE_SOT      - SOT sublayer (market lambdas, UNCALIBRATED constants)
  PENDING       - corners/cards/fouls/offsides/etc. (no model yet) or shadow

    python scripts/coverage_summary.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.odds_api import json_to_markets
from odds_lib.odds import process_match
from odds_lib import match_engine as E
from odds_lib import rate_layer as R
from odds_lib.player_prop_pricing import price_player_prop, PROP_EQUIVALENCE

GOALS = {"team_score_any", "team_score_1h", "team_score_2h",
         "second_half_more_goals", "team_more_goals_2h",
         "compound_first_goal_score_2h", "compound_btts_over_2_5"}
MARKET_DIRECT = {"team_win", "match_total_over", "match_total_under", "halftime_draw"}
SOT = {"team_sot_over", "team_sot_2h_over", "team_more_sot_2h", "total_sot_2h_over"}
PROPS = set(PROP_EQUIVALENCE)


def ev_cache(eid):
    fs = sorted(glob.glob(f"data/raw/soccer_fifa_world_cup__event-{eid}__*.json"))
    return fs[-1] if fs else None


def build():
    bulk = sorted(glob.glob("data/raw/soccer_fifa_world_cup__h2h-totals__*.json"))[-1]
    return {g["id"]: g for g in json.loads(Path(bulk).read_text())}


def consensus(eid, bulk_by_id):
    frames = [json_to_markets([bulk_by_id[eid]], "x", market_keys=("h2h", "totals"))]
    c = ev_cache(eid)
    if c:
        frames.append(json_to_markets(json.loads(Path(c).read_text()), "x",
                                      market_keys=("btts", "h2h_h1")))
    _, cc, _ = process_match(pd.concat(frames, ignore_index=True), on_incomplete="drop")
    return cc


def mp(c, mkey, oc, line=None):
    s = c[c["market_key"] == mkey]
    if line is not None:
        s = s[s["line"] == float(line)]
    s = s[s["outcome"].str.lower() == oc.lower()]
    return float(s["market_prob"].iloc[0]) if not s.empty else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/submission_sheets/2026-06-23_questions.csv")
    args = ap.parse_args()

    q = pd.read_csv(args.questions, dtype=str).fillna("")
    bulk_by_id = build()
    cons = {eid: consensus(eid, bulk_by_id) for eid in q["event_id"].unique()}
    games = {}

    def model(eid, match):
        if eid not in games:
            home, away = match.split(" vs ")
            c = cons[eid]
            m = E.calibrate(home, away, mp(c, "h2h", home), mp(c, "totals", "Over", 2.5))
            games[eid] = (m, E.simulate(m, n=120_000), home, away)
        return games[eid]

    rows = []
    for _, r in q.iterrows():
        qt = r["question_type"].strip().lower()
        eid, c, t = r["event_id"], cons[r["event_id"]], r["target_team"].strip()
        tier, p, conf, detail = "PENDING", None, "", "rate_layer/shadow"

        if qt in PROPS:
            g = json.loads(Path(ev_cache(eid)).read_text())
            line = float(r["line"]) if r["line"] else None
            pr = price_player_prop(qt, r["target_player"] or None, line, g)
            if pr.mapped:
                tier, p, conf, detail = "PROP", pr.market_prob_vig_adjusted, pr.liquidity_flag, pr.source_tag
            else:
                detail = pr.status
        elif qt in MARKET_DIRECT:
            p = (mp(c, "h2h", t) if qt == "team_win" else
                 mp(c, "totals", "Over", 2.5) if qt == "match_total_over" else
                 mp(c, "totals", "Under", 2.5) if qt == "match_total_under" else
                 mp(c, "h2h_h1", "Draw"))
            if p is not None:
                tier, conf, detail = "MARKET", "market", "de-vigged"
        elif qt in GOALS:
            m, sim, home, away = model(eid, r["match"])
            fn = {"team_score_any": lambda: E.p_team_score_any(sim, t),
                  "team_score_1h": lambda: E.p_team_score_1h(sim, t),
                  "team_score_2h": lambda: E.p_team_score_2h(sim, t),
                  "second_half_more_goals": lambda: E.p_second_half_more_goals(sim),
                  "team_more_goals_2h": lambda: E.p_team_more_goals_2h(sim, t),
                  "compound_btts_over_2_5": lambda: E.p_compound_btts_over_2_5(sim),
                  "compound_first_goal_score_2h": lambda: E.p_compound_first_goal_score_2h(sim, home, away)}[qt]
            tier, p, conf, detail = "ENGINE_GOALS", round(fn(), 3), "calibrated", "sim"
        elif qt in SOT:
            m, sim, home, away = model(eid, r["match"])
            other = away if t.lower() == home.lower() else home
            lam = {home.lower(): m.lam_home, away.lower(): m.lam_away}
            line = float(r["line"]) if r["line"] else 0.5
            if qt == "team_sot_over":
                rr = R.price_team_sot_over(lam[t.lower()], line)
            elif qt == "team_sot_2h_over":
                rr = R.price_team_sot_2h_over(lam[t.lower()], line, m.h1_share)
            elif qt == "team_more_sot_2h":
                rr = R.price_team_more_sot_2h(lam[t.lower()], lam[other.lower()], m.h1_share)
            else:  # total_sot_2h_over
                rr = R.price_total_sot_2h_over(m.lam_home, m.lam_away, line, m.h1_share)
            tier, p, conf, detail = "RATE_SOT", round(rr.p, 3), rr.confidence, rr.detail

        rows.append({"match": r["match"][:20], "q": r["question_number"], "type": qt,
                     "tier": tier, "p": p if p is not None else "", "conf": conf})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", 60)
    print(res.to_string(index=False))
    print("\n--- COUNTS BY TIER ---")
    print(res["tier"].value_counts().to_string())
    ready = res[res["tier"] != "PENDING"]
    print(f"\nrows with a model/market p: {len(ready)} / {len(res)}")
    print("  market-grade (MARKET/ENGINE_GOALS/PROP-ok): "
          f"{len(res[(res.tier.isin(['MARKET','ENGINE_GOALS'])) | ((res.tier=='PROP') & (res.conf=='ok'))])}")
    print(f"  lower-confidence (RATE_SOT uncalibrated + thin props): "
          f"{len(res[(res.tier=='RATE_SOT') | ((res.tier=='PROP') & (res.conf=='thin'))])}")


if __name__ == "__main__":
    main()
