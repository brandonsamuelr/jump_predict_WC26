"""Append an optimizer submit sheet to the accumulating measurement log.

    python scripts/log_slate.py --sheet data/submission_sheets/2026-06-23_optimized_submit_sheet.csv \\
        --run-id 2026-06-23 [--multiplier 1]

Optionally add a column `llm_estimate` to the sheet first to compare your old
LLM numbers per tier once results resolve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib.measurement import log_slate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--multiplier", type=int, default=1)
    args = ap.parse_args()
    sheet = pd.read_csv(args.sheet)
    n = log_slate(sheet, run_id=args.run_id, multiplier=args.multiplier)
    print(f"logged {n} rows for run {args.run_id} (multiplier {args.multiplier})")
    print("After games resolve, fill `result` and `actual_rbp` in "
          "data/measurement_log.csv, then run scripts/measure.py")


if __name__ == "__main__":
    main()
