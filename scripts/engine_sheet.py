"""Goals-based engine: calibrate to market, price the goals-based rows.

For each match: pull de-vigged 1X2 + total from cache, calibrate scoring
rates, simulate, then (1) print a market cross-check proving calibration
reproduces 1X2/total/BTTS/HT, and (2) emit engine p for the goals-based
question rows in the question CSV.

Only goals-based question types are priced here. Shots/corners/cards rows
are reported as still-pending (need the rate layer).

    python scripts/engine_sheet.py \\
        --questions data/submission_sheets/2026-06-23_questions.csv
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

GOALS_BASED = {
    "team_score_any", "team_score_1h", "team_score_2h",
    "second_half_more_goals", "team_more_goals_2h",
    "compound_first_goal_score_2h", "compound_btts_over_2_5",
}


def latest_event_cache(event_id: str) -> Path | None:
    fs = sorted(glob.glob(f"data/raw/soccer_fifa_world_cup__event-{event_id}__*.json"))
    return Path(fs[-1]) if fs else None


def latest_bulk() -> dict:
    f = sorted(glob.glob("data/raw/soccer_fifa_world_cup__h2h-totals__*.json"))[-1]
    return {g["id"]: g for g in json.loads(Path(f).read_text())}


def consensus_for(event_id: str, bulk_by_id: dict) -> pd.DataFrame:
    frames = [json_to_markets([bulk_by_id[event_id]], "eng", market_keys=("h2h", "totals"))]
    cache = latest_event_cache(event_id)
    if cache is not None:
        frames.append(json_to_markets(
            json.loads(cache.read_text()), "eng",
            market_keys=("btts", "h2h_h1")))
    mk = pd.concat(frames, ignore_index=True)
    _, c, _ = process_match(mk, on_incomplete="drop")
    return c


def market_inputs(c: pd.DataFrame, home: str, away: str):
    h2h = c[c["market_key"] == "h2h"]
    p_home = float(h2h[h2h["outcome"].str.lower() == home.lower()]["market_prob"].iloc[0])
    tot = c[(c["market_key"] == "totals") & (c["line"] == 2.5)]
    over = tot[tot["outcome"].str.lower() == "over"]
    p_over = float(over["market_prob"].iloc[0])
    # optional market checks
    def _opt(mkey, oc, line=None):
        sl = c[c["market_key"] == mkey]
        if line is not None:
            sl = sl[sl["line"] == line]
        sl = sl[sl["outcome"].str.lower() == oc.lower()]
        return float(sl["market_prob"].iloc[0]) if not sl.empty else None
    return p_home, p_over, {
        "btts_yes": _opt("btts", "Yes"),
        "ht_draw": _opt("h2h_h1", "Draw"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/submission_sheets/2026-06-23_questions.csv")
    ap.add_argument("--n", type=int, default=200_000)
    args = ap.parse_args()

    q = pd.read_csv(args.questions, dtype=str).fillna("")
    bulk_by_id = latest_bulk()

    models: dict[str, tuple] = {}  # event_id -> (model, sim, home, away)
    print("=" * 84)
    print("CALIBRATION CROSS-CHECK (engine must reproduce the market it was fit to)")
    print("=" * 84)
    print(f"{'match':24s} {'p_home mkt/fit':>16s} {'over2.5 mkt/fit':>16s} "
          f"{'btts mkt/sim':>14s} {'HTdraw mkt/sim':>15s}")
    for eid in q["event_id"].unique():
        rows = q[q["event_id"] == eid]
        match = rows["match"].iloc[0]
        home, away = match.split(" vs ")
        c = consensus_for(eid, bulk_by_id)
        p_home, p_over, checks = market_inputs(c, home, away)
        model = E.calibrate(home, away, p_home, p_over)
        sim = E.simulate(model, n=args.n)
        models[eid] = (model, sim, home, away)
        btts_m = checks["btts_yes"]; ht_m = checks["ht_draw"]
        print(f"{match:24s} "
              f"{model.market_p_home:6.3f}/{model.fit_p_home:<6.3f}   "
              f"{model.market_p_over:6.3f}/{model.fit_p_over:<6.3f}   "
              f"{(btts_m if btts_m else float('nan')):5.3f}/{E.p_btts(sim):<5.3f}  "
              f"{(ht_m if ht_m else float('nan')):5.3f}/{E.p_halftime_draw(sim):<5.3f}")
        print(f"    -> lam_{home}={model.lam_home:.2f}  lam_{away}={model.lam_away:.2f}")

    # ---- price goals-based rows ----
    print("\n" + "=" * 84)
    print("ENGINE PROBABILITIES — goals-based rows")
    print("=" * 84)
    out = []
    for _, r in q.iterrows():
        qt = r["question_type"].strip().lower()
        if qt not in GOALS_BASED:
            continue
        model, sim, home, away = models[r["event_id"]]
        team = r["target_team"].strip()
        p = None
        if qt == "team_score_any":
            p = E.p_team_score_any(sim, team)
        elif qt == "team_score_1h":
            p = E.p_team_score_1h(sim, team)
        elif qt == "team_score_2h":
            p = E.p_team_score_2h(sim, team)
        elif qt == "second_half_more_goals":
            p = E.p_second_half_more_goals(sim)
        elif qt == "team_more_goals_2h":
            p = E.p_team_more_goals_2h(sim, team)
        elif qt == "compound_btts_over_2_5":
            p = E.p_compound_btts_over_2_5(sim)
        elif qt == "compound_first_goal_score_2h":
            # "Jordan score first AND Algeria score in 2H"
            p = E.p_compound_first_goal_score_2h(sim, first_team=home, score_team=away)
        out.append({"match": r["match"], "q": r["question_number"],
                    "type": qt, "team": team, "engine_p": round(p, 3),
                    "before": r["before_coverage"]})
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    print(f"\ngoals-based rows now priced by engine: {len(res)}")


if __name__ == "__main__":
    main()
