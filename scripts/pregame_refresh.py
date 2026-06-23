"""Pregame refresh for ONE match: refetch its odds at/near lock, recompute
only its 10 questions, and diff against the overnight baseline.

Isolated by design: touches only the named match, writes a per-match sheet,
never modifies the baseline or other (possibly already-locked) matches.
Shadow rows don't move with odds, so only market/engine/prop/SOT rows change.

    python scripts/pregame_refresh.py --match "Argentina vs Austria"
    python scripts/pregame_refresh.py --event-id be6c63f4... [--no-fetch]

action column:
    KEEP       |delta| < 2%
    REVIEW     2% <= |delta| < 5%
    UPDATE     |delta| >= 5%
    MANUAL_QA  coverage lost (market/prop disappeared or mapping changed)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.odds_api import fetch_event_odds
from odds_lib import slate
from odds_lib.field_model import FieldMeanEstimator
from odds_lib.optimizer import optimize
from odds_lib.edge import compute_edge_table
from odds_lib.lineups import load_lineup
from odds_lib.player_prop_pricing import is_lower_bound_prop
from odds_lib.measurement import LOG_PATH, build_edge_frame

SPORT = "soccer_fifa_world_cup"
REFRESH_MARKETS = ("h2h,totals,btts,h2h_h1,spreads,totals_h1,team_totals,"
                   "player_goal_scorer_anytime,player_shots_on_target")
REVIEW_THR, UPDATE_THR = 0.02, 0.05


def latest_event_cache(eid):
    fs = sorted(glob.glob(f"data/raw/soccer_fifa_world_cup__event-{eid}__*.json"))
    return Path(fs[-1]) if fs else None


def decide(old_tier, new_tier, delta, p_hat):
    lost = (str(new_tier) == "PENDING" and str(old_tier) not in ("PENDING", "nan", ""))
    if lost or (str(new_tier).startswith("PROP") and p_hat is None):
        return "MANUAL_QA", f"coverage lost (was {old_tier}, now {new_tier})"
    if abs(delta) >= UPDATE_THR:
        reason = f"moved {delta:+.1%}"
    elif abs(delta) >= REVIEW_THR:
        reason = f"moved {delta:+.1%}"
    else:
        reason = "stable" if new_tier != "PENDING" else "shadow (odds-independent)"
    if str(old_tier) != str(new_tier) and not lost:
        reason = f"tier {old_tier}->{new_tier}; " + reason
    action = ("UPDATE" if abs(delta) >= UPDATE_THR else
              "REVIEW" if abs(delta) >= REVIEW_THR else "KEEP")
    return action, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match")
    ap.add_argument("--event-id")
    ap.add_argument("--questions", default="data/submission_sheets/2026-06-24_questions.csv")
    ap.add_argument("--baseline", default="data/submission_sheets/2026-06-24_optimized_submit_sheet.csv")
    ap.add_argument("--regions", default="us,uk,eu")
    ap.add_argument("--no-fetch", action="store_true", help="use cached odds (no credits)")
    args = ap.parse_args()

    q = pd.read_csv(args.questions, dtype=str).fillna("")
    if args.event_id:
        rows = q[q["event_id"] == args.event_id]
    elif args.match:
        rows = q[q["match"].str.lower() == args.match.lower()]
    else:
        ap.error("provide --match or --event-id")
    if rows.empty:
        ap.error("no questions found for that match/event in the questions file")
    eid = rows["event_id"].iloc[0]
    match = rows["match"].iloc[0]

    if args.no_fetch:
        cache = latest_event_cache(eid)
        if cache is None:
            ap.error("no cached odds; drop --no-fetch to fetch")
        print(f"[no-fetch] using cache {cache.name}")
    else:
        print(f"refetching {match} ({eid}) at lock...")
        cache = fetch_event_odds(SPORT, eid, markets=REFRESH_MARKETS, regions=args.regions)

    game = json.loads(Path(cache).read_text())
    game = game if isinstance(game, dict) else game[0]
    home, away = game["home_team"], game["away_team"]
    c = slate.build_consensus([game], market_keys=("h2h", "totals", "btts", "h2h_h1"))
    model = slate.build_model(c, home, away)
    field = FieldMeanEstimator()
    lineup = load_lineup(match)   # None until lineups post -> props shadow (k=0)
    edge_table = (compute_edge_table(build_edge_frame(pd.read_csv(LOG_PATH, dtype=str)))
                  if LOG_PATH.exists() else pd.DataFrame())

    if Path(args.baseline).exists():
        base = pd.read_csv(args.baseline)
        base_by_q = {r["q"]: r for _, r in base[base["match"] == match].iterrows()}
    else:
        print(f"[no baseline yet at {args.baseline}; first run of this slate -> d_base blank]")
        base_by_q = {}

    out = []
    for _, r in rows.iterrows():
        tier, p_hat, mkt = slate.resolve_row(r.to_dict(), c, game, model, lineup=lineup)
        fe = field.estimate(r["question_type"])
        sub = optimize(tier=tier, question_type=r["question_type"],
                       p_hat=p_hat, shadow=fe.q_hat, table=edge_table,
                       lower_bound=is_lower_bound_prop(r["question_type"]))
        new_q = round(sub.q, 3)
        b = base_by_q.get(r["question_number"])
        old_q = round(float(b["SUBMIT"]), 3) if b is not None else float("nan")
        old_tier = b["tier"] if b is not None else ""
        delta = (new_q - old_q) if old_q == old_q else 0.0
        action, _ = decide(old_tier, tier, delta, p_hat)
        out.append({
            "match": match, "q": r["question_number"], "type": r["question_type"],
            "tier": tier, "class": sub.source_class, "k": round(sub.k, 2),
            "c_hat": round(fe.q_hat, 3),
            "p_hat": round(p_hat, 3) if p_hat is not None else "",
            "mode": sub.mode + ("+clamp" if sub.lower_bound_clamped else ""),
            "SUBMIT": new_q, "d_base": round(delta, 3),
            "action": action,
        })

    res = pd.DataFrame(out)
    pd.set_option("display.width", 220, "display.max_colwidth", 36)
    print(f"\n=== PREGAME REFRESH (edge-weighted): {match} ===")
    print(res.drop(columns=["match"]).to_string(index=False))
    print("SUBMIT = c_hat + k*(p_hat - c_hat); d_base = SUBMIT - overnight baseline.")
    slug = match.lower().replace(" ", "_").replace("vs", "v")
    outpath = Path(args.baseline).parent / f"{Path(args.baseline).stem.split('_')[0]}_{slug}_refresh.csv"
    res.to_csv(outpath, index=False)
    acts = res["action"].value_counts().to_dict()
    print(f"\n{acts}  -> review/update the non-KEEP rows, then lock {match}.")
    print(f"wrote {outpath}  (baseline + other matches untouched)")


if __name__ == "__main__":
    main()
