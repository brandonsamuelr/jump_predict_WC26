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
# R^2=0.37 means single-game SoT is noisy. Honest status (post-revalidation):
#  - full-match COMPARISON was OOS-gate-validated (scripts/validate_sot.py) and is
#    well-calibrated; BUT the DEPLOYED 2H comparison (team_more_sot_2h) rests on
#    the unvalidated H1_SHARE half-split -> treat the 2H version as UNVALIDATED.
#  - COUNT/threshold rows: this Poisson ran +0.138 HIGH and is now REPLACED by the
#    gate-validated logistic (odds_lib/sot_count_model.py); rate_layer count output
#    is only a fallback. 2H count/total rows remain UNVALIDATED (no corpus half-split).
SOT_INTERCEPT = 1.01     # LEGACY linear map (deprecated; kept for provenance/diagnostics)
SOT_SLOPE = 3.03

# CONCAVE (saturating) SOT-mean, replaces the linear map 2026-06-26. The linear form
# over-extrapolated at high lambda (a 3-xG favourite -> ~10 SOT, above the WC team max ~7.5),
# inflating SOT-volume rows for blow-out favourites. Fit mu = A*(1-exp(-B*lambda)) on 12k club
# team-matches (lambda backed out of odds), OOS-GATE-VALIDATED: lower Brier than the linear map at
# P(SOT>=4/5/6/7) on a date-holdout (e.g. >=5: 0.216 vs 0.240; >=6: 0.180 vs 0.195). Saturates so
# it cannot over-extrapolate (lambda 2.95 -> 7.6, bounded). Level matches WC (2x mid-lambda = 8.3
# vs FotMob 8.2). SHAPE+level transfer (SOT-per-goal ~stable club<->WC); used by all RATE_SOT /
# both_sot routes. NOT used by MATCH_SOT (that's a separate gate-validated logistic).
SOT_SAT_A = 13.016
SOT_SAT_B = 0.297

CONFIDENCE_TAG = "model_sot_concave_oos_validated_2026_06_26"


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
    """Expected SOT for a team over a match fraction (1.0 = full, else half). CONCAVE/saturating
    map (OOS-validated, see SOT_SAT_A/B) -- bounded, no high-lambda over-extrapolation."""
    return SOT_SAT_A * (1.0 - math.exp(-SOT_SAT_B * lam_team)) * share


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


def p_both_teams_sot_1h(lam_home: float, lam_away: float, h1_share: float) -> RateResult:
    """P(BOTH teams have >=1 SOT in the 1st half) -- CLOSED FORM.

    Each team's 1H SOT count ~ Poisson(mu_team_1h), mu_team_1h = team_sot_mu(lam_team,
    share=h1_share) (the SAME market-calibrated lambda->SOT conversion as every other
    SOT row; SOT_SLOPE 3.03 == measured SOT-per-goal 3.02). Treating the two teams'
    1H SOT as conditionally independent GIVEN the match (empirically validated on the
    SGO corpus: phi-corr 0.012, product-of-marginals 0.738 == empirical 0.740):

        P(both) = (1 - e^{-mu_home_1h}) * (1 - e^{-mu_away_1h})

    DRIVER = match SHOT VOLUME (captured by the engine lambdas), not strength-gap:
    a high-lambda match -> both mu high -> P(both) high; a lopsided match -> underdog
    mu low -> P(both) lower. OOS-gated vs the pooled base rate (scripts/fit_shadow_routes);
    caller auto-falls-back to the MEASURED base rate (0.74) if the gate fails / engine
    is unavailable -- never a flat 0.50."""
    mu_h = team_sot_mu(lam_home, share=h1_share)
    mu_a = team_sot_mu(lam_away, share=h1_share)
    p = (1 - math.exp(-mu_h)) * (1 - math.exp(-mu_a))
    return RateResult("both_teams_sot_1h", float(p), CONFIDENCE_TAG,
                      f"mu_home_1h={mu_h:.2f} mu_away_1h={mu_a:.2f} P(both)={p:.3f}")


def p_both_teams_sot_2h_1plus(lam_home: float, lam_away: float, h1_share: float) -> RateResult:
    """P(BOTH teams have >=1 SOT in the 2nd half) -- CLOSED FORM, mirror of the 1H route
    with the 2H share. mu_team_2h = team_sot_mu(lam_team, share=1-h1_share):

        P(both) = (1 - e^{-mu_home_2h}) * (1 - e^{-mu_away_2h})

    Shipped RAW (k=1), undistorted -- the true-P forward computation from the match's own
    market-calibrated lambda. NOT anchored/blended toward the base rate. The volume LEVER
    is validated OOS on the SGO corpus (P(both 2H SOT) rises 0.70->0.88->0.94 across
    realized-goal tertiles); 2H teams' SOT are ~independent (phi 0.09). The measured base
    rate (0.81) is the fallback ONLY when no engine lambda exists (no market)."""
    mu_h = team_sot_mu(lam_home, share=1 - h1_share)
    mu_a = team_sot_mu(lam_away, share=1 - h1_share)
    p = (1 - math.exp(-mu_h)) * (1 - math.exp(-mu_a))
    return RateResult("both_teams_sot_2h_1plus", float(p), CONFIDENCE_TAG,
                      f"mu_home_2h={mu_h:.2f} mu_away_2h={mu_a:.2f} P(both)={p:.3f}")


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
# DEPRECATED 2026-06-26 (set to 0.0). This was a hand-tuned -1.2 subtraction that recentered
# the model output down to the field placeholder (~0.62) — an unvalidated PULL TOWARD A CONSTANT
# (the disease). REMOVED: ship the raw market-anchored mean. Two findings drove this:
#   (1) the SOT level was FotMob-validated (WC ~8.2 total/match), so the offset pushed a CORRECT
#       mean too LOW (wrong direction).
#   (2) the earlier "var/mean~2.6 -> needs a tail discount" was the WRONG statistic: TOTAL 2H SOT
#       (both teams summed) measures var/mean = 1.21 in-corpus (mild, ~Poisson). So NO heavy tail
#       discount is warranted and an NB barely differs from Poisson at the contest line (N>=4) ->
#       NB deferred. The 2.6 was a per-team figure that doesn't apply to the summed total.
# RESIDUAL (flagged, NOT a constant-pull): team_sot_mu is LINEAR in lambda, so it OVER-EXTRAPOLATES
# the mean at high lambda (e.g. a 3-xG favourite -> ~10 full SOT). This inflates total_sot_2h_over
# for blow-out-favourite matches. Same root cause as MATCH_SOT; the proper fix is a SATURATING
# (concave) SOT-mean, a dedicated MATCH_SOT+RATE_SOT change -- NOT a field-pull patch.
TOTAL_SOT_2H_LEVEL_OFFSET = 0.0   # deprecated, no longer subtracted


def price_total_sot_2h_over(lam_home: float, lam_away: float, line: float,
                            h1_share: float) -> RateResult:
    # RAW market-anchored mean (no offset). Poisson tail (total 2H SOT ~Poisson: var/mean 1.21).
    # Shipped at k=1 (edge.K_PRIOR RATE_SOT/total_2h = 1.0) -- NO shrink toward c_hat.
    mu = team_sot_mu(lam_home, share=1 - h1_share) + team_sot_mu(lam_away, share=1 - h1_share)
    thr = _line_to_ge(line)
    return RateResult("total_sot_2h_over", _p_ge(mu, thr), CONFIDENCE_TAG,
                      f"mu_2h_total={mu:.2f} P(>= {thr})")


__all__ = [
    "SOT_INTERCEPT", "SOT_SLOPE", "CONFIDENCE_TAG", "RateResult",
    "TOTAL_SOT_2H_LEVEL_OFFSET", "TOTAL_SOT_CONFIDENCE_TAG",
    "team_sot_mu", "price_team_sot_over", "price_team_sot_2h_over",
    "price_team_more_sot_2h", "price_total_sot_2h_over", "p_both_teams_sot_1h",
    "p_both_teams_sot_2h_1plus",
]
