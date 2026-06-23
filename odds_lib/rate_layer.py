"""Shot-volume rate layer: derive shots-on-target (SOT) counts from the
market-anchored expected goals produced by :mod:`match_engine`, then price
SOT-based questions that have no betting line.

Why SOT and not cards/corners/fouls
-----------------------------------
SOT volume tracks expected goals strongly (shots are how goals happen), so a
team's SOT count is a defensible function of its market-implied lambda.
Corners track attacking *dominance* more loosely (separate, weaker sublayer,
later). Cards / fouls / offsides do NOT track expected goals — deriving them
from lambda would be fabrication, so they are deliberately NOT modelled here
and remain shadow rows.

Confidence discipline
---------------------
The lambda inputs are market-derived, but the SOT conversion constants below
are UNCALIBRATED v1 baselines (no per-team SOT-rate data yet). Every output
is tagged ``confidence="model_sot_baseline_uncalibrated"`` — a *repeatable
model*, but a lower tier than market-derived p. The constants are the
explicit calibration target for the next iteration (real per-team SOT rates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- POOLED-CALIBRATED conversion constants --------------------------------
# Team full-match SOT modelled as Poisson(mu), mu = INTERCEPT + SLOPE * lam.
# Calibrated 2026-06-22 by OLS of observed SoT_for on market-implied lambda
# across 20 completed WC26 team-matches (pre-match h2h+totals -> lambda via the
# engine; SoT scraped from match reports). NO per-team qualifier rates, NO xG.
#   fit:            SOT = 1.01 + 3.03*lam   (R^2 = 0.37, n = 20)
#   prior guess:    SOT = 1.50 + 2.20*lam
#   robustness:     dropping the 2 worst residuals -> 0.50 + 3.43 (slope stable
#                   ~3.0-3.4, intercept ~0.5-1.0; DIRECTION robust).
# R^2=0.37 means single-game SoT is noisy: COMPARISON outputs are reliable
# (constant largely divides out), COUNT/threshold outputs are usable-but-noisy
# and should be treated as medium confidence, not market-grade.
SOT_INTERCEPT = 1.01
SOT_SLOPE = 3.03

CONFIDENCE_TAG = "model_sot_pooled_calibrated_r2_0.37"


def _poisson_pmf(lam: float, kmax: int = 25):
    out = []
    for k in range(kmax + 1):
        out.append(math.exp(-lam) * lam**k / math.factorial(k))
    return out


def _p_ge(lam: float, threshold: int) -> float:
    """P(N >= threshold) for N ~ Poisson(lam)."""
    pmf = _poisson_pmf(lam)
    return float(sum(pmf[threshold:]))


def _p_a_gt_b(mu_a: float, mu_b: float) -> float:
    """P(A > B) for independent Poissons."""
    pa, pb = _poisson_pmf(mu_a), _poisson_pmf(mu_b)
    p = 0.0
    for b in range(len(pb)):
        p += pb[b] * sum(pa[b + 1:])
    return float(p)


def team_sot_mu(lam_team: float, share: float = 1.0) -> float:
    """Expected SOT for a team over a match fraction (1.0 = full, else half)."""
    return (SOT_INTERCEPT + SOT_SLOPE * lam_team) * share


@dataclass
class RateResult:
    question_type: str
    p: float
    confidence: str
    detail: str


def _line_to_ge(line: float) -> int:
    """'6 or more' is given as line 5.5 -> threshold 6; integer line N -> N."""
    return math.ceil(line) if line != int(line) else int(line)


def price_team_sot_over(lam_team: float, line: float) -> RateResult:
    mu = team_sot_mu(lam_team)
    thr = _line_to_ge(line)
    return RateResult("team_sot_over", _p_ge(mu, thr), CONFIDENCE_TAG,
                      f"mu={mu:.2f} P(SOT>={thr})")


def price_team_sot_2h_over(lam_team: float, line: float, h1_share: float) -> RateResult:
    mu = team_sot_mu(lam_team, share=1 - h1_share)
    thr = _line_to_ge(line)
    return RateResult("team_sot_2h_over", _p_ge(mu, thr), CONFIDENCE_TAG,
                      f"mu_2h={mu:.2f} P(SOT_2h>={thr})")


def price_team_more_sot_2h(lam_team: float, lam_other: float, h1_share: float) -> RateResult:
    mu_t = team_sot_mu(lam_team, share=1 - h1_share)
    mu_o = team_sot_mu(lam_other, share=1 - h1_share)
    return RateResult("team_more_sot_2h", _p_a_gt_b(mu_t, mu_o), CONFIDENCE_TAG,
                      f"mu_team_2h={mu_t:.2f} vs mu_other_2h={mu_o:.2f}")


# Total-row-specific level correction (2026-06-23). The total_sot_2h_over row
# was the one SOT bucket the calibration check showed systematically high vs
# the field (NOR +0.17, JOR +0.20 above crowd; the model's ~0.80 sits ~1 SOT
# above where the field prices it). We subtract a level offset from the total
# 2H mu so the row recenters from ~0.80 to ~the field level (~0.62-0.65),
# WHILE PRESERVING the per-match tilt (mu still varies with lambda) — i.e. not
# a flat constant, a recentered one. This deliberately does NOT touch the
# shared SOT_INTERCEPT/SLOPE, so single-team / comparison SOT rows (which the
# check showed track the field) are untouched.
#
# SOFT and total-specific: there are ZERO realized rows at the model's raw
# 0.80 (NOR Q5 was overridden to 0.57, JOR Q2 pending), so this is a
# structurally-motivated recenter toward the field, not an outcome-confirmed
# one. Tuned so the two observed cases land ~at the crowd (not below it — this
# is a YES-leaning row). To be hardened by the contest's accumulating hit rate.
#
# SCOPE: this offset corrects the LEVEL of mu (the mean) only. The D5 audit
# (scripts/audit_total_sot_2h_slope.py) confirmed the mean slope is data-backed
# but the single-mu Poisson TAIL is overdispersed (var/mean~2.6) hence too steep
# at the tempo extremes. That tail discount is NOT applied here — it lives in the
# edge layer as k<1 (edge.K_PRIOR RATE_SOT/total_2h), so mu keeps its full tilt.
TOTAL_SOT_2H_LEVEL_OFFSET = 1.2
TOTAL_SOT_CONFIDENCE_TAG = "model_sot_2h_total_level_recentered_lowconf"


def price_total_sot_2h_over(lam_home: float, lam_away: float, line: float,
                            h1_share: float) -> RateResult:
    mu_raw = team_sot_mu(lam_home, share=1 - h1_share) + team_sot_mu(lam_away, share=1 - h1_share)
    mu_adj = max(mu_raw - TOTAL_SOT_2H_LEVEL_OFFSET, 0.1)  # recenter level, keep per-match tilt
    thr = _line_to_ge(line)
    return RateResult("total_sot_2h_over", _p_ge(mu_adj, thr), TOTAL_SOT_CONFIDENCE_TAG,
                      f"mu_raw={mu_raw:.2f} mu_adj={mu_adj:.2f} P(>= {thr})")


__all__ = [
    "SOT_INTERCEPT", "SOT_SLOPE", "CONFIDENCE_TAG", "RateResult",
    "TOTAL_SOT_2H_LEVEL_OFFSET", "TOTAL_SOT_CONFIDENCE_TAG",
    "team_sot_mu", "price_team_sot_over", "price_team_sot_2h_over",
    "price_team_more_sot_2h", "price_total_sot_2h_over",
]
