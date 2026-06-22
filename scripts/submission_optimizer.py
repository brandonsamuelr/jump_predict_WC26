"""Produce the final submission for ALL questions (breadth-first).

Resolves each row to (tier, p_hat) via the shared resolver in odds_lib.slate
(market / engine-goals / SOT rate / prop), attaches the field-mean shadow
anchor, and runs the optimizer. Emits a submission for every question — lean
toward p where we have edge, shadow the field mean where we don't.

    python scripts/submission_optimizer.py [--tilt 0.0]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib import slate
from odds_lib.field_model import FieldMeanEstimator
from odds_lib.optimizer import optimize


def ev_cache(eid):
    fs = sorted(glob.glob(f"data/raw/soccer_fifa_world_cup__event-{eid}__*.json"))
    return fs[-1] if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/submission_sheets/2026-06-23_questions.csv")
    ap.add_argument("--tilt", type=float, default=0.0,
                    help="variance tilt (>0 overshoots p_hat to buy variance; EV-negative)")
    ap.add_argument("--out", default="data/submission_sheets/2026-06-23_optimized_submit_sheet.csv")
    args = ap.parse_args()

    q = pd.read_csv(args.questions, dtype=str).fillna("")
    bulk = sorted(glob.glob("data/raw/soccer_fifa_world_cup__h2h-totals__*.json"))[-1]
    bulk_by_id = {g["id"]: g for g in json.loads(Path(bulk).read_text())}
    field = FieldMeanEstimator()

    cons, models, games = {}, {}, {}
    for eid in q["event_id"].unique():
        match = q[q["event_id"] == eid]["match"].iloc[0]
        home, away = match.split(" vs ")
        event_json = json.loads(Path(ev_cache(eid)).read_text())
        games[eid] = event_json
        c = slate.build_consensus([bulk_by_id[eid], event_json])
        cons[eid] = c
        models[eid] = slate.build_model(c, home, away)

    out = []
    for _, r in q.iterrows():
        eid = r["event_id"]
        tier, p_hat, _ = slate.resolve_row(r.to_dict(), cons[eid], games[eid], models[eid])
        fe = field.estimate(r["question_type"])
        sub = optimize(tier=tier, p_hat=p_hat, shadow=fe.q_hat, variance_tilt=args.tilt)
        out.append({"match": r["match"], "q": r["question_number"], "type": r["question_type"],
                    "tier": tier, "p_hat": round(p_hat, 3) if p_hat is not None else "",
                    "shadow": round(fe.q_hat, 3), "mode": sub.mode, "SUBMIT": round(sub.q, 3)})

    res = pd.DataFrame(out)
    pd.set_option("display.width", 200, "display.max_rows", 60)
    print(res.to_string(index=False))
    res.to_csv(args.out, index=False)
    n_lean = (res["mode"] == "lean").sum()
    print(f"\nALL {len(res)} questions answered  |  lean(edge)={n_lean}  shadow(harvest)={len(res)-n_lean}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
