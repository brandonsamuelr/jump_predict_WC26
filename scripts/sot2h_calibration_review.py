"""SHADOW calibration review for team_more_sot_2h (RATE_SOT_CMP). NOT a production override.

Correction banked (2026-06-27): the right test for the over-confidence is realized-hit-rate vs
SHIPPED probability against OUTCOMES -- never RBP-vs-crowd (that's the scoreboard, not the target).
And the calibration target is CONDITIONAL on the route's own information (market lambdas), so the fix
must preserve within-tail ordering -- a flat ~0.66 would discard real signal (0.82-lambda != 0.72-lambda).

This compares, on resolved rows, the RAW route vs ORDER-PRESERVING recalibrations against outcomes,
in favorite-perspective (pred = max(p,1-p); outcome = did the favorite win the 2H SOT battle). It is
SHADOW ONLY: it changes nothing, it accumulates evidence. Re-run as rows resolve. Replace production
ONLY when (a) enough production-faithful rows exist and (b) a calibrated candidate beats RAW on OUTCOME
Brier out-of-sample, preserving ordering.

    .venv/bin/python scripts/sot2h_calibration_review.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

LOG = Path("data/measurement_log.csv")
SHADOW = Path("data/models/sot2h_shadow_calib.csv")


def main():
    df = pd.read_csv(LOG, dtype=str)
    s = df[(df.question_type == "team_more_sot_2h") & (df.result.fillna("") != "")].copy()
    if s.empty:
        print("no resolved team_more_sot_2h rows yet."); return
    s["p"] = pd.to_numeric(s.final_submitted, errors="coerce")
    s["fav_p"] = np.maximum(s.p, 1 - s.p)                    # the route's favorite-side confidence
    s["fav_won"] = (((s.result == "Yes") & (s.p > 0.5)) | ((s.result == "No") & (s.p < 0.5))).astype(int)
    s = s.dropna(subset=["p"])
    s[["match", "question_number", "p", "fav_p", "result", "fav_won"]].to_csv(SHADOW, index=False)

    y = s.fav_won.values; fp = s.fav_p.values
    print(f"resolved team_more_sot_2h rows: {len(s)}  (SHADOW review -- production unchanged)\n")
    print(f"mean SHIPPED favorite-prob = {fp.mean():.3f}   realized favorite-dominance = {y.mean():.3f}  ({y.sum()}/{len(y)})")

    # calibration curve (is the miscalibration MONOTONE -> recalibrate-preserving-order, or flat?)
    print(f"\ncalibration curve (route favorite-prob vs realized):")
    print(f"  {'bucket':12}{'n':>4}{'mean shipped':>14}{'realized':>10}")
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        m = (fp >= lo) & (fp < hi); n = int(m.sum())
        if n == 0:
            print(f"  {f'{lo:.2f}-{hi:.2f}':12}{0:>4}"); continue
        print(f"  {f'{lo:.2f}-{hi:.2f}':12}{n:>4}{fp[m].mean():>14.3f}{y[m].mean():>10.3f}")

    def brier(p): return float(np.mean((p - y) ** 2))
    # candidate A: order-preserving confidence scale toward 0.5 (Platt-like alpha; alpha<1 = less extreme)
    print(f"\nOUTCOME-Brier (favorite-perspective), candidates ORDER-PRESERVING:")
    print(f"  RAW route                         {brier(fp):.4f}")
    best = (brier(fp), 1.0)
    for a in (0.85, 0.7, 0.55, 0.4):
        cand = 0.5 + (fp - 0.5) * a
        b = brier(cand); best = min(best, (b, a))
        print(f"  alpha={a:.2f} (shrink-confidence)        {b:.4f}   d_vs_raw {brier(fp)-b:+.4f}")
    print(f"  -> best alpha={best[1]:.2f} (alpha<1 = route too extreme; preserves ordering)")
    print(f"\n** n={len(s)} is UNDERPOWERED -- this is the accumulating shadow signal, NOT a ship trigger.")
    print(f"   Gate (n=406) says ~0.66 marginal; ship a recalibration only when faithful rows + OOS beat RAW.")


if __name__ == "__main__":
    main()
