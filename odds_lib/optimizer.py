"""Breadth-first submission optimizer.

Turns a per-question (truth estimate p_hat, confidence tier, shadow anchor)
into a final submission, on EVERY question — because the leaders' ~3 RBP/
question is mostly the convexity harvest Var(q_i), which you only collect by
answering. Skipping forfeits it.

One formula
-----------
    edge rows (we have a trusted p_hat):
        q = 0.5 + (w + tilt) * (p_hat - 0.5)
        - w in [0,1] shrinks an uncertain p_hat toward 0.5 (model-risk hedge);
          w = 1 submits p_hat exactly.
        - tilt >= 0 deliberately overshoots p_hat to BUY VARIANCE when far
          behind in the tournament. tilt is EV-negative beyond p_hat, so it
          defaults to 0 (pure EV). Dial up only as a behind-in-tournament bet.

    shadow rows (no trusted p_hat):
        q = field shadow anchor (q-bar estimate) -> harvest Var(q_i), no view.

Why shrink toward 0.5 and not toward the qt-mean: the qt-mean is role-blind
(e.g. team_win pools 0.18..0.92), so it is a safe *shadow anchor* only on
no-market types, never a shrink target for a real p_hat.
"""

from __future__ import annotations

from dataclasses import dataclass

# A tier is either trusted (submit our estimate as-is) or not (shadow the
# field). No fractional shrink toward 0.5 — that was an arbitrary hard-coded
# fudge. The only principled shrink is one DERIVED from measured calibration
# error, which we don't have yet; until then, trust-or-shadow is honest.
#
# Trusted = market-grounded or market-calibrated.
#  - RATE_SOT_CMP: a SOT *comparison* (team A vs team B). The conversion
#    constant largely divides out, so the result is directionally robust and
#    depends on the market lambda gap (comparison stays 0.12-0.27 across wild
#    a/b while the count swings 0.35-0.72).
#  - RATE_SOT (count/threshold): now uses POOLED-CALIBRATED constants (OLS on
#    20 completed WC team-matches, R^2=0.37). Medium confidence — beats the
#    role-blind shadow on net (esp. extreme teams + totals) but single-game
#    SoT is noisy, so it is NOT market-grade. Trusted, but flagged.
TRUSTED_TIERS = frozenset(
    {"MARKET", "ENGINE_GOALS", "PROP_ok", "PROP_thin", "RATE_SOT_CMP", "RATE_SOT"}
)

CLIP_LO, CLIP_HI = 0.03, 0.97


@dataclass
class Submission:
    q: float
    mode: str          # "lean" | "shadow"
    tier: str
    p_hat: float | None
    shadow: float | None
    weight: float
    note: str


def _clip(x: float) -> float:
    return max(CLIP_LO, min(CLIP_HI, x))


def optimize(
    *,
    tier: str,
    p_hat: float | None,
    shadow: float | None,
    variance_tilt: float = 0.0,
) -> Submission:
    """Compute the submission for one question.

    ``tier`` is the confidence label (keys of :data:`TRUST`, or anything else
    -> treated as no-edge shadow). ``shadow`` is the field-mean anchor
    (required for shadow rows; used as a last-resort fallback otherwise).
    """
    if tier in TRUSTED_TIERS and p_hat is not None:
        # submit our estimate as-is. tilt is an OPT-IN extremize of p_hat away
        # from 50% to buy variance when behind (EV-negative, default 0).
        q = p_hat if not variance_tilt else p_hat + variance_tilt * (p_hat - 0.5)
        return Submission(q=_clip(q), mode="lean", tier=tier, p_hat=p_hat,
                          shadow=shadow, weight=1.0,
                          note="submit p_hat" + (f"+tilt{variance_tilt}" if variance_tilt else ""))
    # not trusted: stick close to the field — submit the field-mean shadow
    # anchor to harvest convexity. NOT 0.5; the anchor is the qt-mean (or the
    # global field mean for thin types). The 0.5 below is an unreachable guard
    # (a real FieldMeanEstimator always returns at least the global mean).
    if shadow is None:
        raise ValueError("shadow row needs a field-mean anchor; none supplied")
    return Submission(q=_clip(shadow), mode="shadow", tier=tier, p_hat=p_hat,
                      shadow=shadow, weight=0.0, note="shadow field mean")


__all__ = ["TRUSTED_TIERS", "Submission", "optimize"]
