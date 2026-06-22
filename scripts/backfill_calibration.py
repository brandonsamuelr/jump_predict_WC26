"""Fill post-lock columns on pending calibration rows after a match resolves.

The calibration log is the only feedback loop we have on whether
``p_field_est`` is any good. Skipping the backfill makes every future
strategy debate vibes-based. Use this script after each match.

Three modes:

1. ``--list-pending`` — print every pending row, grouped by match, so you
   can see exactly what needs backfilling.

2. ``--from-csv path.csv`` — batch mode. CSV columns (header required)::

       match,question,field_prob,result[,actual_rbp,if_yes_rbp,if_no_rbp,swing]

   ``match`` and ``question`` are case-insensitive substrings. ``field_prob``
   is the locked crowd YES % (integer or float). ``result`` is "Yes" or "No".
   Other columns are optional and can be added once you see them on the
   site post-resolution.

3. one-shot CLI args (``--match X --question Y --field-prob N --result Yes ...``)
   for single-row updates.

The script prints, per update, the field_error so you can spot
miscalibrations immediately.

Examples
--------
  # see what's pending
  python scripts/backfill_calibration.py --list-pending

  # one-shot
  python scripts/backfill_calibration.py \\
      --match "Uruguay" --question "win the match" \\
      --field-prob 64 --result Yes --actual-rbp 2.5

  # batch (the usual workflow after a match)
  python scripts/backfill_calibration.py --from-csv \\
      data/calibration_backfill/2026-06-21_uruguay_capeverde.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib import backfill_calibration_row, CALIBRATION_LOG_PATH


def _list_pending(path: Path) -> None:
    if not path.exists():
        print(f"no calibration log at {path}")
        return
    df = pd.read_csv(path)
    pending = df[df["status"] == "pending"]
    if pending.empty:
        print(f"no pending rows in {path}")
        return
    print(f"{len(pending)} pending row(s) in {path}:\n")
    for match, sub in pending.groupby("match_norm", sort=True):
        print(f"  {match}")
        for _, r in sub.iterrows():
            qt = r["question_type"]
            q = (r["question"] or "")[:60]
            sub_pct = r["submitted_percent"]
            p_field = r.get("p_field_est", "")
            mode = r.get("decision_mode", "")
            print(
                f"    [{mode:>34}] sub={sub_pct:>3} p_field_est={p_field}  "
                f"{qt:25}  {q}"
            )


def _apply_one(
    path: Path,
    *,
    match_substring: str,
    question_substring: str,
    field_prob: float | None,
    result: str | None,
    actual_rbp: float | None,
    if_yes_rbp: float | None,
    if_no_rbp: float | None,
    swing: float | None,
) -> None:
    n = backfill_calibration_row(
        path,
        match_substring=match_substring,
        question_substring=question_substring,
        field_prob=field_prob,
        result=result,
        actual_rbp=actual_rbp,
        if_yes_rbp=if_yes_rbp,
        if_no_rbp=if_no_rbp,
        swing_display=swing,
    )
    if n == 0:
        print(
            f"  NO MATCH: match~{match_substring!r}, q~{question_substring!r}"
        )
        return
    df = pd.read_csv(path)
    mask = (
        df["match_norm"].astype(str).str.lower().str.contains(match_substring.lower(), na=False, regex=False)
        & df["question"].astype(str).str.lower().str.contains(question_substring.lower(), na=False, regex=False)
    )
    for _, r in df[mask].tail(n).iterrows():
        p_est = r.get("p_field_est", "")
        err = r.get("field_error", "")
        sub_pct = r.get("submitted_percent", "")
        try:
            err_str = f"{float(err):+.3f}"
        except (TypeError, ValueError):
            err_str = str(err)
        result_str = r.get("result", "") or "?"
        status = r.get("status", "")
        print(
            f"  [{status}] {r['match_norm']} / {r['question_type']} "
            f"(sub={sub_pct}, p_field_est={p_est}, locked={field_prob}, "
            f"field_error={err_str}, result={result_str})"
        )


def _from_csv(path: Path, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    required = {"match", "question"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"--from-csv: missing required columns: {sorted(missing)}")
    print(f"applying {len(df)} backfill row(s) from {csv_path}:")
    for _, row in df.iterrows():
        # field_prob and result are independently optional so the same CSV
        # format works for "record locked crowd at kickoff" and "record
        # result + rbp post-match" phases.
        result_val = row.get("result") if "result" in row else None
        if isinstance(result_val, str):
            result_val = result_val.strip() or None
        elif result_val is None or (hasattr(pd, 'isna') and pd.isna(result_val)):
            result_val = None
        _apply_one(
            path,
            match_substring=str(row["match"]),
            question_substring=str(row["question"]),
            field_prob=_optf(row.get("field_prob")),
            result=result_val,
            actual_rbp=_optf(row.get("actual_rbp")),
            if_yes_rbp=_optf(row.get("if_yes_rbp")),
            if_no_rbp=_optf(row.get("if_no_rbp")),
            swing=_optf(row.get("swing") if "swing" in row else row.get("swing_display")),
        )


def _optf(v) -> float | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--path",
        default=str(CALIBRATION_LOG_PATH),
        help=f"Calibration log path (default: {CALIBRATION_LOG_PATH}).",
    )
    p.add_argument("--list-pending", action="store_true")
    p.add_argument("--from-csv", type=str, help="Batch backfill from a CSV file.")
    p.add_argument("--match", type=str)
    p.add_argument("--question", type=str)
    p.add_argument("--field-prob", type=float, help="Locked crowd YES%.")
    p.add_argument("--result", type=str, choices=["Yes", "No"], help="Post-match result.")
    p.add_argument("--actual-rbp", type=float)
    p.add_argument("--if-yes-rbp", type=float)
    p.add_argument("--if-no-rbp", type=float)
    p.add_argument("--swing", type=float)
    args = p.parse_args()

    path = Path(args.path)

    if args.list_pending:
        _list_pending(path)
        return

    if args.from_csv:
        _from_csv(path, Path(args.from_csv))
        return

    if not (args.match and args.question and (args.field_prob is not None or args.result)):
        p.error(
            "need --list-pending, --from-csv, or "
            "--match + --question + at least one of --field-prob / --result"
        )

    _apply_one(
        path,
        match_substring=args.match,
        question_substring=args.question,
        field_prob=args.field_prob,
        result=args.result,
        actual_rbp=args.actual_rbp,
        if_yes_rbp=args.if_yes_rbp,
        if_no_rbp=args.if_no_rbp,
        swing=args.swing,
    )


if __name__ == "__main__":
    main()
