"""Deliverable 1 — per-class/subtype edge measurement.

Estimates the Brier-optimal edge multiplier k for each source class: how far
the model's deviation from the PRE-LOCK field proxy (c_hat) points toward
truth. Fit on c_hat (the predicted/shadow field value available pre-lock),
NEVER on the realized crowd (which is post-lock, diagnostic only).

  d        = p_model - c_hat
  residual = y - c_hat
  k_hat    = sum(d*resid) / sum(d^2)                 # UNCLIPPED diagnostic; may be < 0
  k_shrunk = (sum(d*resid) + lam*k_prior) / (sum(d^2) + lam)
  lam      = m_prior * d_bar_sq                       # squared-deviation units, NOT a row count
  deployed = clip(k_shrunk, 0, 1)                     # only the DEPLOYED k is clipped

A negative k_hat is a five-alarm signal: the class is anti-predictive (truth
moves opposite the model's deviation). We surface it; we don't deploy it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Gate-validated models (team_sot_over, team_more_cards, match_total_sot_over) are
# submitted UNDISTORTED — k=1.0 toward the model, exactly like a market price. They
# passed the out-of-sample gate against actual outcomes, so the model probability IS
# the best truth-estimate. The former k=0.50 "international level-hedge" pulled them
# toward the placeholder shadow, which is the SAME ill-conditioned error we removed
# for corners: the shadow has no claim to being the truth, and club->international
# level error has UNKNOWN direction, so a pull toward the placeholder injects bias
# with no basis. Correct level-uncertainty handling: submit undistorted, LOG
# predictions vs realized (kept), and correct ONLY from measured directional bias.
GATE_MODEL_TRUST_K = 1.00

# FUTURE, currently-INACTIVE correction hook. Maps a gate-validated (class, subtype)
# to a DATA-DERIVED, DIRECTIONAL adjustment, set ONLY from observed international
# miscalibration in the live measurement log (e.g. if logged team_sot_over runs +X
# above realized, correct DOWN by the measured X). Empty = identity (no correction)
# = the current state. This is NOT a hedge toward a placeholder and must never be
# set to one; when populated it is applied in the MEASURED direction only. (Not yet
# wired into the pricing path — it is the documented location for that future fix.)
GATE_MODEL_LEVEL_CORRECTION: dict[tuple[str, str], float] = {}   # inactive: identity

# Structural priors by (source_class, subtype). Independent sharp sources get
# high conviction; odds-independent fallback gets none.
#
# CHANGE 1 (anchor-quality k): rows that carry a REAL market/model PRICE must
# submit AT/NEAR that price, not be blended back toward the field-mean c_hat,
# which is only a PLACEHOLDER predictor of the crowd on these rows. Diluting a
# good price toward that placeholder is what cost the Colombia corners row -7.40
# (hand-priced 0.48 dragged off the ~0.569 market). So the market-priced classes
# get high k (graded by price reliability). SHADOW stays 0.00 (placeholder and
# real-base-rate shadow rows are UNTOUCHED — they have no price to dilute).
# Old -> new shown inline; values are the deliberate, adjustable policy.
# ====================================================================================
# ROUTE-ADDING CHECKLIST (read before wiring a new tier into slate.resolve_row):
#   • If the new route carries a REAL de-vigged MARKET PRICE (or a market-derived read):
#       1. set its K_PRIOR to 1.00,
#       2. add its (class, subtype) to TRUST_PRICE_K (immune to the edge-table fit),
#       3. add the tier to the market lists in BOTH tests/test_no_market_override.py
#          and tests/test_no_pull_to_constant.py (INDEPENDENT_TIERS).
#   • If it's a model/derived estimate (k<1 allowed): add it to CONSTANT_DEPENDENT_TIERS in
#     tests/test_no_pull_to_constant.py so the c_hat-dependence is tracked, and JUSTIFY the
#     k<1 (outcome-fitted + genuinely overconfident — else fix the distribution, don't shrink
#     toward c_hat). A k<1 shrinks the read toward the stale field-mean: that is the bug class.
#   • Never use a TYPE_BASE_RATE / hardcoded constant as a CENTER on the main path — only as a
#     degenerate fallback when the engine/market literally cannot run (and population-matched).
#   See memory: feedback_dont_shrink_liquid_h2h, feedback_ship_true_p_no_anchor.
# ====================================================================================
K_PRIOR = {
    ("MARKET", "market"): 1.00,        # de-vigged sharp book line: submit AT the line, NEVER shrink
                                       # toward c_hat (TRUST_PRICE_K enforces immunity to the fit)
    # ENGINE de-hedged to k=1 (2026-06-26): the goals engine is MARKET-CALIBRATED (lambda from
    # de-vigged 1X2+totals) and beats base OOS (edge table fitted k_hat 2.08 -> deployed 1.0). The
    # 0.92 "small model hedge" was a residual shrink toward the stale c_hat -> removed. Ship raw.
    # EXPLICIT OOS GATE (population re-level review, 2026-06-26): on n=25 RESOLVED WC engine rows
    # (team_score_any/BTTS/2H-goals/etc) Brier model k=1 = 0.1717 vs base(c_hat) 0.2319 vs old
    # k=0.92 0.1752 -> beats base AND beats the 0.92 shrink. k=1 CONFIRMED on outcomes, not asserted.
    ("ENGINE", "engine"): 1.00,
    # ALL confirmed-starter player-prop reads are REAL de-vigged MARKET reads -> ship RAW (k=1).
    # Thin = VARIANCE, not bias; shrinking a prop de-vig toward c_hat is the base-rate-over-specific-
    # info error (live: Sangare shrunk LOST -4.24; Skhiri raw WON +7.5). The liquidity/quality gate
    # in price_player_prop still routes garbage single-book quotes to their fallback; we only removed
    # the c_hat SHRINK. (Benched players are a SEPARATE minutes-scaled path -> _prop_tier never
    # fires for them; see PROP_SUB.) All in TRUST_PRICE_K.
    ("PROP", "confirmed"): 1.00,       # was 0.90 — liquid prop market
    ("PROP", "thin"): 1.00,            # was 0.40 — thin (few-book) prop, variance not bias
    ("PROP", "direct_thin"): 1.00,     # was 0.80 — direct 1-2 book score-or-assist
    ("PROP", "proxy_floor"): 1.00,     # was 0.40 — anytime-goal de-vig as a LOWER BOUND (lower_bound clamp)
    ("PROP", "sub"): 1.00,             # benched-player minutes-scaled closed form (founded), ship raw
    # Gate-validated team_sot_over count model: submit UNDISTORTED (k=1, see above).
    ("SOT_COUNT", "model"): GATE_MODEL_TRUST_K,
    # CHANGE 3 (corrected): corners count rows submit the de-vigged market price
    # UNDISTORTED (k=1). A thin/few-book market is higher-VARIANCE but NOT biased
    # in a known direction, so blending it toward an arbitrary anchor (placeholder
    # shadow or 0.50) would inject a directional bias the data doesn't support —
    # the best estimate of a noisy-but-unbiased price IS that price. The ok/thin
    # split is now a DIAGNOSTIC liquidity flag only (tracked via the tier), it no
    # longer pulls the number. (No real price -> caller falls back to shadow.)
    ("CORNERS", "ok"): 1.00,
    ("CORNERS", "thin"): 1.00,
    # line-gap Poisson-ladder fit = market-DERIVED read -> submit undistorted (k=1).
    ("CORNERS", "ladder"): 1.00,
    # no-ladder MEASURED corner base rate: it IS the estimate (like CORNER_HALF stopgap),
    # submit undistorted (k=1). An honest measured anchor, NOT a pull toward the crowd.
    ("CORNERS", "base"): 1.00,
    # Newly founded shadow families. Closed-form derivations from the validated goals
    # engine (+ the OOS-validated SOT conversion); submit undistorted (k=1) like any
    # gate-validated model. The measured base-rate fallbacks also submit at k=1 (they
    # ARE the estimate where the engine can't run) -- never pulled toward the crowd mean.
    ("BOTH_SOT_1H", "model"): GATE_MODEL_TRUST_K,
    ("BOTH_SOT_1H", "base"): 1.00,
    # 2H both-SOT: RAW true-P shipped undistorted (k=1), NO anchor/damping. Volume lever
    # validated OOS (monotone 0.70->0.94 across realized-goal tertiles). base = degenerate
    # no-engine fallback only (k=1, it IS the estimate then).
    ("BOTH_SOT_2H", "model"): GATE_MODEL_TRUST_K,
    ("BOTH_SOT_2H", "base"): 1.00,
    ("FIRST_GOAL_2H", "model"): GATE_MODEL_TRUST_K,
    ("FIRST_GOAL_2H", "base"): 1.00,
    # Measured anchors (offsides corpus rate; sourced external pen/red): the measured
    # rate IS the estimate -> submit undistorted (k=1), never pulled toward the crowd mean.
    # offsides TEAM: FOUNDED per-team empirical-Bayes measured rate (beat the pooled floor
    # OOS on StatsBomb intl, +0.003 Brier) -> ship the measured rate RAW (k=1). offsides
    # FLOOR: uncovered team -> honest no-edge pooled measured rate, also k=1 (it IS the floor).
    ("OFFSIDES", "team"): 1.00,
    ("OFFSIDES", "floor"): 1.00,
    # 2H cards: market-derived (full-card lambda x 2H share) -> per-match, correctly populated,
    # submit undistorted (k=1). Club-only corpus floor is the last-resort fallback (mis-populated).
    ("CARDS_2H", "market"): 1.00,
    ("CARDS_2H", "floor"): 1.00,
    ("PENALTY", "base"): 1.00,
    # Cards count row (total_cards_over) off alternate_totals_cards — same policy
    # as corners: de-vigged market price UNDISTORTED (k=1); flag is diagnostic only.
    ("CARDS", "ok"): 1.00,
    ("CARDS", "thin"): 1.00,
    # Remaining market-available rows (Track 1): all submit the de-vigged market
    # price UNDISTORTED (k=1); ok/thin is a diagnostic liquidity flag only.
    ("TEAMGOALS", "ok"): 1.00,   ("TEAMGOALS", "thin"): 1.00,   # team_total_goals_over
    ("CORNERS_CMP", "ok"): 1.00, ("CORNERS_CMP", "thin"): 1.00, # team_more_corners_full (corners_1x2 market)
    # corner-comparison MODEL: full-match fallback (no market) + provisional 1H/2H.
    # Undistorted (k=1) — the gate-validated model IS the estimate, not hedged.
    ("CORNERS_CMP", "model"): 1.00,          # full-match model fallback (favorite_gap)
    ("CORNERS_CMP", "provisional_1h"): 1.00, # DEPRECATED old 1H/2H shrink path
    # 1H/2H more-corners STOPGAP: submit the measured per-half base-rate floor
    # UNDISTORTED (k=1, it IS the base-rate estimate). STOPGAP_NOT_TRUE_P — ignores
    # favorite_gap (data wall blocks a conditioned half model). HIGH-PRIORITY UNSOLVED.
    ("CORNER_HALF", "stopgap"): 1.00,
    # OddsPapi-Pinnacle sharp half-corner read (0.0 handicap de-vig); single_book sharp,
    # use-if-plausible (LAW) -> submit UNDISTORTED (k=1); plausibility band is the guard.
    ("CORNER_HALF", "pinnacle"): 1.00,
    ("H2GOALS", "ok"): 1.00,     ("H2GOALS", "thin"): 1.00,     # second_half_goals_over (alternate_totals_h2)
    # 1st-half total goals direct (totals_h1), de-vigged market price UNDISTORTED (k=1);
    # ok/thin is a diagnostic book-count flag only (totals_h1 is often thin/split-line).
    ("H1GOALS", "ok"): 1.00,     ("H1GOALS", "thin"): 1.00,
    # Track 2 gate-validated corpus models (NO market): submit UNDISTORTED (k=1).
    ("MORE_CARDS", "model"): GATE_MODEL_TRUST_K,   # team_more_cards
    ("FOUL_CMP", "model"): GATE_MODEL_TRUST_K,     # team_more_fouls (validated favorite_gap model)
    ("MATCH_SOT", "model"): GATE_MODEL_TRUST_K,    # match_total_sot_over
    # RATE_SOT family DE-SHRUNK to k=1 (2026-06-26, Item 3): the concave SOT-mean (rate_layer,
    # OOS-validated) makes the single-team SOT model BEAT the base rate OOS at every threshold
    # (Brier P(SOT>=4/5/6/7): concave < base); the comparison beats base per the edge table (CUR
    # Q5 raw 0.07 WON +7.38). So ship raw -> NO shrink toward c_hat. (Was 0.60/0.50/0.50 = a
    # c_hat shrink that was an unvalidated overdispersion proxy; the overdispersion is now handled
    # by the concave MEAN + the ~Poisson tail, not by dragging toward the field.)
    ("RATE_SOT", "comparison"): 1.00,
    ("RATE_SOT", "single"): 1.00,
    ("RATE_SOT", "single_2h"): 1.00,
    # total_2h: c_hat is an UNVALIDATED placeholder anchor (~0.62, set from CROWD
    # agreement, NOT gate-validated against outcomes), used as a less-bad fallback
    # than the 0.49 global mean — see field_model.TYPE_BASE_RATE. D5 slope/dispersion
    # audit (scripts/audit_total_sot_2h_slope.py) RESOLVED the open question:
    #   - the MEAN slope is data-backed (team OLS 3.03; full-match model mean
    #     10.34 vs observed 10.78) -> the recenter keeps it; do NOT flatten it.
    #   - the TAIL is overdispersed (full-match SOT var/mean=2.57 vs Poisson 1.0,
    #     not explainable by lambda spread) -> the single-mu Poisson P(4+) is
    #     OVER-confident at the extremes. k<1 (shrink toward c_hat) is the
    #     directionally-correct first-order fix for THAT, at both extremes.
    # So 0.50 is an EXPLICIT tail-overdispersion discount (NOT timid tempering of
    # a trusted tilt): recenter moves mu (mean, trusted), k shrinks tail curvature
    # (over-steep). Magnitude isn't calibratable (no paired SOT-lambda; n=9) -> a
    # neg-binomial tail is the eventual structural fix; until then 0.50 + the
    # n_active<4 freeze. (Was 0.90 when k routed around the bad 0.49 baseline.)
    # total_2h FIXED 2026-06-26: was 0.50 (shrink toward the c_hat=0.623 placeholder) ON TOP of a
    # -1.2 mu offset -- a DOUBLE pull toward a stale constant, both unvalidated. Now k=1.0: ship the
    # raw market-anchored Poisson tail (offset removed; total 2H SOT var/mean=1.21 ~Poisson so no
    # tail discount needed). Residual = the LINEAR SOT-mean over-extrapolation at high lambda
    # (shared w/ MATCH_SOT) -> dedicated saturation fix, NOT a field-pull. single/single_2h/comparison
    # below stay <1 (flagged in test_no_pull_to_constant for review; same overdispersion question).
    ("RATE_SOT", "total_2h"): 1.00,
    ("SHADOW", "shadow"): 0.00,
}
M_PRIOR = 8.0            # pseudo-matches of prior conviction (prior dominates at small n)
D_BAR_SQ_FALLBACK = 0.04  # a typical squared deviation (0.20^2) when none observed
ACTIVE_D = 0.05          # |d| above which the model "took a position"
ACTIVE_FREEZE_N = 4      # below this many active rows, deployed k = k_prior (no drift)
# below this many MATCH clusters, also freeze (correlated rows). Raised 5 -> 10
# after ENGINE auto-unfroze to a fitted k=1.0 off just 5 correlated clusters:
# at this sample k is a MONITORING signal, not a deployed parameter. 10 clusters
# is a meaningful "enough independent matches" bar; fitted k_hat keeps printing
# as a diagnostic, but deployed k stays on the structural prior until then.
MIN_CLUSTERS = 10


EDGE_CLIP_LO, EDGE_CLIP_HI = 0.02, 0.98  # final p_submit bounds


def edge_submit(p_model: float | None, c_hat: float | None, k: float) -> float:
    """The edge-weighted submission: ``c_hat + k*(p_model - c_hat)``, clipped.

    This is the SINGLE submission rule for every row. There is no other
    shrinkage path — all deviation from the field proxy is governed by k:
      * trusted class (high k)  -> lands near the raw model when it disagrees.
      * no-edge / SHADOW (k=0)  -> lands ON c_hat (no manufactured deviation).
      * no model (p_model None) -> lands ON c_hat (nothing to express).
    c_hat is the PRE-LOCK field proxy; never the realized (post-lock) crowd.
    """
    if c_hat is None:
        raise ValueError("edge_submit needs a c_hat (pre-lock field proxy)")
    c = float(c_hat)
    if p_model is None or k == 0.0:
        q = c
    else:
        q = c + float(k) * (float(p_model) - c)
    return min(max(q, EDGE_CLIP_LO), EDGE_CLIP_HI)


# Classes that carry a REAL de-vigged market price (a sharp/liquid book line). These must
# ALWAYS submit AT the price — the edge-table fit must NEVER be allowed to shrink them toward
# the field-mean c_hat. Overriding a sharp book line is the banned behavior that pulled
# 'Turkiye win' from the de-vig 0.29 up to 0.45 (k=0.52 fit toward the high team_win field-mean
# 0.62). The rule was in the prior; this enforces it against the FIT. MODEL classes (ENGINE,
# RATE_SOT, FOUL_CMP, MATCH_SOT, ...) and measured FLOORS stay fit-adjustable. See
# memory feedback_dont_shrink_liquid_h2h / feedback_no_hardcoded_p_truth.
TRUST_PRICE_K = frozenset({
    ("MARKET", "market"),
    ("CORNERS", "ok"), ("CORNERS", "thin"), ("CORNERS", "ladder"),
    ("CARDS", "ok"), ("CARDS", "thin"),
    ("TEAMGOALS", "ok"), ("TEAMGOALS", "thin"),
    ("CORNERS_CMP", "ok"), ("CORNERS_CMP", "thin"),
    ("H2GOALS", "ok"), ("H2GOALS", "thin"),
    ("H1GOALS", "ok"), ("H1GOALS", "thin"),
    ("CORNER_HALF", "pinnacle"),
    ("CARDS_2H", "market"),                 # cards-market lambda x 2H share (market-derived)
    ("OFFSIDES", "team"),                    # founded per-team EB offside rate (OOS-gated) -> k=1
    # All confirmed-starter player-prop reads (real de-vigged markets) -> k=1, ship raw.
    ("PROP", "confirmed"), ("PROP", "thin"), ("PROP", "direct_thin"), ("PROP", "proxy_floor"),
    # Every member of TRUST_PRICE_K MUST have prior == 1.0 (enforced by
    # test_trust_price_classes_have_high_prior).
})


def deployed_k(cls: str, sub: str, table: pd.DataFrame | None = None) -> float:
    """Deployed k for a (class, subtype).

    HARD RULE: direct-market-price classes (TRUST_PRICE_K) are IMMUNE to the edge-table fit —
    they always deploy their structural prior (~1, trust the de-vigged line). A sharp book line
    is never shrunk toward the field-mean, no matter what the fitted k says.

    All other (MODEL / floor) classes use the fitted edge table's ``k_deployed`` (which shrinks
    toward, and freezes on, the structural prior at small samples) when resolved rows exist;
    otherwise the structural prior.
    """
    prior = float(K_PRIOR.get((cls, sub), 0.0))
    if (cls, sub) in TRUST_PRICE_K:
        return prior                                  # market price: hard-pinned, immune to the fit
    if table is not None and not table.empty and (cls, sub) in table.index:
        fitted = float(table.loc[(cls, sub), "k_deployed"])
        # UNIVERSAL GUARD: the fit may only RAISE trust in a read (higher k = closer to the read).
        # It may NEVER lower k below the structural prior — i.e. never increase the shrink toward
        # the field-mean c_hat beyond the deliberate prior. This makes it structurally impossible
        # for a small-sample fit to drag ANY row's read toward c_hat (the Turkiye-win failure mode).
        # If a model is genuinely overconfident, fix the model / lower its prior explicitly — never
        # let the fit silently shrink it. (k clamped to [prior, 1].)
        return min(max(prior, fitted), 1.0)
    return prior


def classify(tier: str, question_type: str) -> tuple[str, str]:
    t = (tier or "").strip()
    qt = (question_type or "").strip().lower()
    if t == "MARKET" or t == "MARKET_INTERP":   # MARKET_INTERP = totals 2.5 used for a non-2.5 line (no alt market)
        return ("MARKET", "market")
    if t.startswith("ENGINE_GOALS"):   # incl. _H1MKT (market half-split) / _H1FALLBACK (constant H1_SHARE)
        return ("ENGINE", "engine")
    if t == "H1GOALS_OK":   return ("H1GOALS", "ok")     # 1H total goals direct (totals_h1)
    if t == "H1GOALS_THIN": return ("H1GOALS", "thin")
    if t == "PROP_ok":
        return ("PROP", "confirmed")
    if t == "PROP_thin":
        return ("PROP", "thin")
    if t == "PROP_direct_thin":
        return ("PROP", "direct_thin")
    if t == "PROP_proxy_floor":
        return ("PROP", "proxy_floor")
    if t == "PROP_SUB":
        return ("PROP", "sub")
    if t == "SOT_COUNT":   # CHANGE 2: validated team_sot_over count-row logistic
        return ("SOT_COUNT", "model")
    if t == "CORNERS_OK":   # CHANGE 3: direct corners market, well-booked line
        return ("CORNERS", "ok")
    if t == "CORNERS_THIN":  # CHANGE 3: direct corners market, thin line
        return ("CORNERS", "thin")
    if t == "CORNERS_LADDER":  # line-gap: Poisson fit of the quoted book ladder (market-derived)
        return ("CORNERS", "ladder")
    if t == "CORNERS_BASE":    # no ladder: MEASURED corner base rate (anchor, not 0.50)
        return ("CORNERS", "base")
    if t == "BOTH_SOT_1H":        return ("BOTH_SOT_1H", "model")   # engine-lambda volume model (OOS-gated)
    if t == "BOTH_SOT_1H_BASE":   return ("BOTH_SOT_1H", "base")    # measured base rate fallback
    if t == "BOTH_SOT_2H":        return ("BOTH_SOT_2H", "model")   # raw closed-form true P (k=1, no anchor)
    if t == "BOTH_SOT_2H_BASE":   return ("BOTH_SOT_2H", "base")    # degenerate no-engine fallback
    if t == "FIRST_GOAL_2H":      return ("FIRST_GOAL_2H", "model") # closed-form race of 2H Poissons
    if t == "FIRST_GOAL_2H_BASE": return ("FIRST_GOAL_2H", "base")  # measured home/away anchor
    if t == "OFFSIDES_TEAM":      return ("OFFSIDES", "team")       # founded per-team EB rate (OOS-gated)
    if t == "OFFSIDES_FLOOR":     return ("OFFSIDES", "floor")      # no-edge floor (uncovered team -> pooled rate)
    if t == "CARDS_2H_MKT":       return ("CARDS_2H", "market")     # full-card market lambda x 2H share (per-match)
    if t == "CARDS_2H_FLOOR":     return ("CARDS_2H", "floor")      # CLUB-only corpus floor (last resort, mis-populated)
    if t == "PENALTY_BASE":       return ("PENALTY", "base")        # sourced external pen/red rate
    if t == "CARDS_OK":     # direct total-cards market, well-booked line
        return ("CARDS", "ok")
    if t == "CARDS_THIN":   # direct total-cards market, thin line
        return ("CARDS", "thin")
    if t == "TEAMGOALS_OK":   return ("TEAMGOALS", "ok")
    if t == "TEAMGOALS_THIN": return ("TEAMGOALS", "thin")
    if t == "CORNERS_CMP_OK":   return ("CORNERS_CMP", "ok")
    if t == "CORNERS_CMP_THIN": return ("CORNERS_CMP", "thin")
    if t == "CORNERS_CMP_MODEL": return ("CORNERS_CMP", "model")        # full-match model fallback
    if t == "CORNERS_CMP_1H":    return ("CORNERS_CMP", "provisional_1h")  # DEPRECATED old shrink path
    if t == "CORNER_HALF_STOPGAP": return ("CORNER_HALF", "stopgap")    # 1H/2H base-rate floor (STOPGAP, not true P)
    if t == "CORNER_HALF_PINNACLE": return ("CORNER_HALF", "pinnacle")  # sharp Pinnacle half-corner read
    if t == "H2GOALS_OK":   return ("H2GOALS", "ok")
    if t == "H2GOALS_THIN": return ("H2GOALS", "thin")
    if t == "MORE_CARDS":   return ("MORE_CARDS", "model")   # Track 2 gate-validated
    if t == "FOUL_CMP":     return ("FOUL_CMP", "model")     # validated favorite_gap foul model
    if t == "MATCH_SOT":    return ("MATCH_SOT", "model")     # Track 2 gate-validated
    if t == "RATE_SOT_CMP":
        return ("RATE_SOT", "comparison")
    if t == "RATE_SOT":
        if "total_sot_2h" in qt:
            return ("RATE_SOT", "total_2h")
        if "2h" in qt:
            return ("RATE_SOT", "single_2h")
        return ("RATE_SOT", "single")
    return ("SHADOW", "shadow")


def _agg(g: pd.DataFrame, k_prior: float) -> pd.Series:
    d = (g["p_model"] - g["c_hat"]).to_numpy(dtype=float)
    resid = (g["y"] - g["c_hat"]).to_numpy(dtype=float)
    sd2 = float(np.sum(d ** 2))
    sd4 = float(np.sum(d ** 4))
    sdr = float(np.sum(d * resid))
    active = np.abs(d) > ACTIVE_D
    eff_n = (sd2 ** 2 / sd4) if sd4 > 0 else 0.0
    d_bar_sq = float(np.median(d[active] ** 2)) if active.any() else D_BAR_SQ_FALLBACK
    lam = M_PRIOR * d_bar_sq
    k_hat = (sdr / sd2) if sd2 > 0 else float("nan")
    k_shrunk = (sdr + lam * k_prior) / (sd2 + lam)
    n_active = int(active.sum())
    n_clusters = int(g["match"].nunique())
    # FREEZE: thin classes sit ON the prior, they don't drift on noise. Two
    # gates, BOTH required to unfreeze, because questions within a match share
    # one game script (correlated): enough active rows AND enough independent
    # MATCH clusters. Without the cluster gate a class with 6 active rows across
    # 3 lucky matches would crank k to its clipped max on ~3 correlated obs —
    # the exact "3 matches look good -> max k" trap. Frozen <=> LOW(prior-
    # dominated), so a prior-dominated class always deploys its structural prior.
    frozen = (n_active < ACTIVE_FREEZE_N) or (n_clusters < MIN_CLUSTERS)
    k_deployed = k_prior if frozen else min(max(k_shrunk, 0.0), 1.0)
    return pd.Series({
        "n": len(g),
        "clusters": g["match"].nunique(),
        "n_active": n_active,                # rows where the model took a position
        "eff_n_k": round(eff_n, 2),          # effective fitting sample for k
        "sum_d2": round(sd2, 4),
        "k_prior": k_prior,
        "k_hat": round(k_hat, 3) if sd2 > 0 else float("nan"),  # UNCLIPPED diagnostic
        "k_shrunk": round(k_shrunk, 3),                          # diagnostic
        "frozen": frozen,
        "k_deployed": round(min(max(k_deployed, 0.0), 1.0), 3),  # prior if frozen, else clipped shrunk
        "mean_rbp_final": round(g["rbp_final"].mean(), 2),
        "mean_rbp_model": round(g["rbp_model_cf"].dropna().mean(), 2) if g["rbp_model_cf"].notna().any() else float("nan"),
        "mean_rbp_base": round(g["rbp_baseline_cf"].dropna().mean(), 2) if g["rbp_baseline_cf"].notna().any() else float("nan"),
    })


def compute_edge_table(df: pd.DataFrame) -> pd.DataFrame:
    """df columns required: source_class, source_subtype, match, p_model, c_hat,
    y, rbp_final, rbp_model_cf, rbp_baseline_cf. Returns one row per
    (class, subtype) with the k estimators and confidence flags."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (cls, sub), g in df.groupby(["source_class", "source_subtype"]):
        kp = K_PRIOR.get((cls, sub), 0.0)
        s = _agg(g, kp)
        s["source_class"], s["source_subtype"] = cls, sub
        # confidence flag from match clustering + active fitting sample. The LOW
        # condition is EXACTLY the freeze condition, so frozen <=> prior-dominated.
        if s["clusters"] < MIN_CLUSTERS or s["n_active"] < ACTIVE_FREEZE_N:
            s["confidence"] = "LOW(prior-dominated)"
        elif s["clusters"] < 2 * MIN_CLUSTERS:
            s["confidence"] = "MED"
        else:
            s["confidence"] = "OK"
        rows.append(s)
    rep = pd.DataFrame(rows).set_index(["source_class", "source_subtype"])
    cols = ["n", "clusters", "n_active", "eff_n_k", "sum_d2", "k_prior",
            "k_hat", "k_shrunk", "frozen", "k_deployed", "confidence",
            "mean_rbp_final", "mean_rbp_model", "mean_rbp_base"]
    return rep[cols].sort_index()


__all__ = ["classify", "compute_edge_table", "edge_submit", "deployed_k",
           "K_PRIOR", "TRUST_PRICE_K", "M_PRIOR", "D_BAR_SQ_FALLBACK", "ACTIVE_D",
           "ACTIVE_FREEZE_N", "MIN_CLUSTERS", "EDGE_CLIP_LO", "EDGE_CLIP_HI",
           "GATE_MODEL_TRUST_K", "GATE_MODEL_LEVEL_CORRECTION"]
