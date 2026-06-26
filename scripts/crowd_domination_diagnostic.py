"""Post-hoc 'structural domination' monitor (NOT a tuning target -- diagnostic only).

The failure mode (found on the Senegal-Iraq slate): we and the crowd lean the SAME way, but
the crowd is MORE confident in that lean, so when the shared-expected outcome lands we lose
relative RBP. It concentrated on direction-blind constants (half-corner stopgap, offside floor).

This scans RESOLVED rows in the measurement log and flags where that domination MATERIALIZED:
    same lean as crowd   AND   crowd more confident   AND   we scored worse (Brier vs outcome).

CRITICAL: the crowd number is only visible POST-LOCK, so this is a HEALTH CHECK, never an input
or a gate target (gates use outcomes only -- see feedback_no_crowd_value_pre_lock). Run it after a
slate resolves to confirm the fixes (1H-corner model, offside EB prior) actually reduce domination.

    .venv/bin/python scripts/crowd_domination_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

LOG = Path("data/measurement_log.csv")


def _yn(s):
    s = str(s).strip().lower()
    return 1 if s in ("1", "yes", "y", "true") else (0 if s in ("0", "no", "n", "false") else None)


def main():
    df = pd.read_csv(LOG, dtype=str)
    df["y"] = df["result"].map(_yn)
    df["you"] = pd.to_numeric(df["final_submitted"], errors="coerce")
    df["crowd"] = pd.to_numeric(df["crowd_prob"], errors="coerce") / 100.0
    r = df.dropna(subset=["y", "you", "crowd"]).copy()
    if r.empty:
        print("no resolved rows with crowd + outcome yet."); return

    same_lean = (r.you - 0.5) * (r.crowd - 0.5) > 0
    crowd_more = (r.crowd - 0.5).abs() > (r.you - 0.5).abs()
    we_lost = (r.you - r.y) ** 2 > (r.crowd - r.y) ** 2          # worse Brier than the crowd
    r["exposed"] = same_lean & crowd_more                        # structurally dominated set-up
    r["dominated"] = r["exposed"] & we_lost                      # ... and it materialized

    # v2 split: STRUCTURAL (direction-blind constants/floors -> fixable bug) vs VARIANCE
    # (founded true-P rows where we were merely less extreme than a sometimes-sharper crowd
    # -> expected ~50/50, NOT a bug). Only the structural set is actionable.
    STRUCTURAL = {"CORNER_HALF_STOPGAP", "OFFSIDES_FLOOR", "PENALTY_BASE",
                  "CARDS_2H_FLOOR", "CORNERS_BASE"}
    r["kind"] = r.tier.where(r.tier.isin(STRUCTURAL)).notna().map({True: "STRUCTURAL", False: "variance"})

    n, ne, nd = len(r), int(r.exposed.sum()), int(r.dominated.sum())
    print(f"resolved rows: {n}")
    print(f"  EXPOSED (same lean, crowd more confident): {ne} ({100*ne/n:.0f}%)")
    print(f"  DOMINATED (exposed AND we scored worse):   {nd} ({100*nd/n:.0f}%)\n")

    dom = r[r.dominated].copy()
    for kind in ("STRUCTURAL", "variance"):
        sub = dom[dom.kind == kind]
        rbp = pd.to_numeric(sub["actual_rbp"], errors="coerce").sum()
        label = ("STRUCTURAL (direction-blind constants -> FIXABLE)" if kind == "STRUCTURAL"
                 else "variance (founded rows, less extreme than a sharp crowd -> expected)")
        print(f"  {label}: {len(sub)} rows, {rbp:+.2f} RBP")
        if len(sub):
            print(sub.tier.value_counts().to_string().replace("\n", "  |  "))
        print()

    if len(dom[dom.kind == "STRUCTURAL"]):
        print("ACTIONABLE structural-domination rows:")
        cols = ["match", "question_number", "tier", "you", "crowd", "result", "actual_rbp"]
        print(dom.loc[dom.kind == "STRUCTURAL", cols].to_string(index=False))


if __name__ == "__main__":
    main()
