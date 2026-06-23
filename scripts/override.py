"""Record a manual deviation from the pipeline in the measurement log.

Keeps pipeline_submit as the counterfactual and stores final_submitted +
source + override_reason. Requires a reason (the discipline gate) — if you
can't write one sentence, you shouldn't override.

    python scripts/override.py --run-id 2026-06-23 --match "Argentina vs Austria" \\
        --q Q8 --to 0.60 --reason "SOT count near threshold; blowout-driven slope likely overstates a controlled game"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib.measurement import apply_override, LOG_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--q", required=True, help="question_number, e.g. Q8")
    ap.add_argument("--to", type=float, required=True, help="final submitted probability")
    ap.add_argument("--reason", required=True, help="one sentence — the discipline gate")
    ap.add_argument("--source", default="manual", choices=["manual", "llm"])
    ap.add_argument("--allow-soft", action="store_true",
                    help="force a SOFT-category override (disabled by default; net-negative)")
    args = ap.parse_args()

    try:
        n = apply_override(args.run_id, args.match, args.q, args.to,
                           args.reason, source=args.source, allow_soft=args.allow_soft)
    except ValueError as e:
        ap.error(str(e))
    if n == 0:
        print(f"no logged row matched run={args.run_id} match='{args.match}' q={args.q}")
        return
    row = pd.read_csv(LOG_PATH, dtype=str)
    r = row[(row.run_id == args.run_id) & (row.match == args.match)
            & (row.question_number == args.q)].iloc[0]
    print(f"override recorded ({n} row):")
    print(f"  {args.match} {args.q}: pipeline {r['pipeline_submit']} -> final {r['final_submitted']} "
          f"[{args.source}]")
    print(f"  reason: {args.reason}")
    print("  (pipeline value kept as counterfactual; the loop will score whether this override helped.)")


if __name__ == "__main__":
    main()
