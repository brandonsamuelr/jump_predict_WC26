"""Deliverable 5 (remaining piece) — the total_sot_2h_over SLOPE/DISPERSION audit.

The baseline (c_hat=0.623) and level (recenter, OFFSET=1.2) are settled. The
open question that gates Deliverable 3: should P(4+ total SOT in 2H) fall as
steeply at the tempo extremes as the Poisson plug-in says? Two distinct objects
are easy to conflate:

  - the MEAN slope  : dE[SOT]/dlambda. Empirically fit (team-level OLS:
                      SOT = 1.01 + 3.03*lam, R^2=0.37, n=20). The recenter keeps
                      this slope, so the LEVEL/tilt of mu is data-backed.
  - the TAIL prob   : P(N>=4) read off a single-mu Poisson. Overdispersion
    STEEPNESS          (single-game SOT is noisy: R^2=0.37) FLATTENS the true
                      tail vs the plug-in -> the plug-in is OVER-confident in
                      the tails. k<1 (shrink toward base rate) is the first-order
                      correction for THAT, and is a different lever than the
                      recenter (which moves mu, not the tail curvature).

A PAIRED SOT-vs-lambda regression is NOT reconstructable here: the match-report
SOT counts the team-level fit used are not in the repo (sot_calibration_rows.csv
has lam but an empty sot_for column), and the 9 full-match totals carried in the
handoff have no recoverable match->total mapping. So we run the checks that ARE
rigorous with the data on hand and report honestly what they can and cannot
decide.

    .venv/bin/python scripts/audit_total_sot_2h_slope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from odds_lib.match_engine import H1_SHARE
from odds_lib.rate_layer import (
    SOT_INTERCEPT, SOT_SLOPE, TOTAL_SOT_2H_LEVEL_OFFSET, price_total_sot_2h_over,
)
from odds_lib.field_model import TYPE_BASE_RATE

CALIB = "data/models/sot_calibration_rows.csv"
# Full-match (both teams) total SOT carried from the prior session's match-report
# collection. UNPAIRED to lambda (mapping not recoverable), so usable only for
# distribution-level tests (mean, dispersion), NOT a paired regression.
FULLMATCH_SOT_TOTALS = [10, 18, 10, 10, 6, 7, 15, 18, 3]
C_HAT = TYPE_BASE_RATE["total_sot_2h_over"]   # 0.623
LINE = 3.5                                     # "4 or more" -> threshold 4


def lam_total_distribution() -> np.ndarray:
    df = pd.read_csv(CALIB)
    tot = df.groupby("match")["lam"].sum()
    return tot.to_numpy(dtype=float)


def p_submit(p_model: float, k: float) -> float:
    return min(max(C_HAT + k * (p_model - C_HAT), 0.02), 0.98)


def main():
    lt = lam_total_distribution()
    print("=" * 78)
    print("total_sot_2h_over — SLOPE / DISPERSION AUDIT")
    print("=" * 78)
    print(f"H1_SHARE={H1_SHARE} (2H share={1-H1_SHARE:.2f}); SOT mu = "
          f"{SOT_INTERCEPT} + {SOT_SLOPE}*lam; total-2H offset={TOTAL_SOT_2H_LEVEL_OFFSET}; "
          f"c_hat={C_HAT}")

    # --- (1) real lambda_total distribution ---------------------------------
    print(f"\n[1] lambda_total over {len(lt)} calibration matches (REAL, market-derived)")
    print(f"    mean={lt.mean():.3f}  sd={lt.std(ddof=1):.3f}  "
          f"min={lt.min():.3f}  p25={np.percentile(lt,25):.3f}  "
          f"median={np.median(lt):.3f}  p75={np.percentile(lt,75):.3f}  max={lt.max():.3f}")
    print("    -> the OBSERVED tempo range is narrow; the cagey(1.6)/high(3.6) prints")
    print("       below are SYNTHETIC extrapolations beyond it, by design.")

    # --- (2) LEVEL check: model full-match mean vs observed ------------------
    sot = np.array(FULLMATCH_SOT_TOTALS, dtype=float)
    model_fullmatch_mean = SOT_INTERCEPT * 2 + SOT_SLOPE * lt.mean()  # 2.02 + 3.03*lam_bar
    print(f"\n[2] LEVEL check — full-match TOTAL SOT (both teams), unpaired means")
    print(f"    observed mean (n={len(sot)}): {sot.mean():.2f}")
    print(f"    model E[total]=2*{SOT_INTERCEPT}+{SOT_SLOPE}*lam_bar = {model_fullmatch_mean:.2f}")
    print(f"    -> level agreement is {'GOOD' if abs(model_fullmatch_mean-sot.mean())<2 else 'OFF'} "
          f"(diff {model_fullmatch_mean-sot.mean():+.2f} SOT); the full-match mean is not biased.")

    # --- (3) DISPERSION test (the part we CAN do without pairing) ------------
    var = sot.var(ddof=1)
    disp = var / sot.mean()
    print(f"\n[3] DISPERSION test — full-match total SOT (the decisive unpaired check)")
    print(f"    mean={sot.mean():.2f}  var={var:.2f}  dispersion index var/mean={disp:.2f}")
    print(f"    Poisson predicts var/mean = 1.0. Observed ~{disp:.1f}.")
    print(f"    CAVEAT: this index is INFLATED by lambda-heterogeneity across matches")
    print(f"    (systematic), so it is an UPPER bound on TRUE within-state overdispersion.")
    # subtract the systematic component the model itself attributes to lambda spread:
    sys_var = (SOT_SLOPE ** 2) * lt.var(ddof=1)            # var of 2.02+3.03*lam over matches
    print(f"    model-attributed systematic var (3.03^2 * var(lam_tot)) = {sys_var:.2f}")
    print(f"    => even the systematic part ({sys_var:.1f}) is far below observed var ({var:.1f}),")
    print(f"       so substantial NON-Poisson, non-lambda noise remains. This corroborates")
    print(f"       the team-level R^2=0.37: single-game SOT is overdispersed. The Poisson")
    print(f"       plug-in tail P(>=4) is therefore OVER-confident at the extremes.")

    # --- (4) across-tempo print: what the model & submission actually do -----
    print(f"\n[4] ACROSS-TEMPO PRINT — model P(4+) and p_submit at each k")
    print(f"    (cagey=1.6 and high=3.6 are SYNTHETIC extremes; mid ~ observed median)")
    grid = [("CAGEY*", 1.6), ("low", 2.2), ("obs-median", float(np.median(lt))),
            ("high-obs", 2.9), ("HIGH*", 3.6)]
    print(f"    {'tempo':>11} {'lam_tot':>7} {'mu_adj':>6} {'P_model(4+)':>11} "
          f"{'submit_k=0.5':>12} {'submit_k=0.9':>12} {'submit_k=1.0':>12}")
    for label, ltot in grid:
        lh = la = ltot / 2.0
        rr = price_total_sot_2h_over(lh, la, LINE, H1_SHARE)
        pm = rr.p
        mu_adj = float(rr.detail.split("mu_adj=")[1].split()[0])
        print(f"    {label:>11} {ltot:>7.2f} {mu_adj:>6.2f} {pm:>11.3f} "
              f"{p_submit(pm,0.5):>12.3f} {p_submit(pm,0.9):>12.3f} {p_submit(pm,1.0):>12.3f}")

    # --- (5) the resolution -------------------------------------------------
    print(f"\n[5] READ / DECISION")
    print(f"    - MEAN slope (recenter) is data-backed (OLS 3.03, R^2=0.37) and the")
    print(f"      full-match level matches observed ([2]); KEEP the recenter, KEEP the tilt.")
    print(f"    - TAIL steepness is NOT trustworthy: the data is overdispersed ([3]),")
    print(f"      so the Poisson plug-in P(4+) over-tilts at the extremes. Shrinking toward")
    print(f"      c_hat (k<1) is the DIRECTIONALLY-CORRECT first-order fix for that, at BOTH")
    print(f"      extremes (it pulls the over-confident tail probs back to the base rate).")
    print(f"    - This resolves the apparent contradiction: recenter moves mu (the MEAN);")
    print(f"      k shrinks the TAIL curvature. Different objects -> we can trust the mean")
    print(f"      tilt AND discount the tail steepness without inconsistency.")
    print(f"    - MAGNITUDE is NOT calibratable from this data (no paired SOT-lambda; n=9;")
    print(f"      lambda-heterogeneity confound). So k=0.50 stands as an EXPLICIT, LABELED")
    print(f"      tail-overdispersion discount (NOT 'conservative tempering'), with the")
    print(f"      n_active<4 FREEZE so it cannot drift on single outcomes. Re-fit when")
    print(f"      resolved total_sot_2h_over rows accumulate across matches.")


if __name__ == "__main__":
    main()
