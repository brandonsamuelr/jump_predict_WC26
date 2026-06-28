"""Time-window event pricing for hydration-break-bounded R32 questions.

Hydration breaks are VERIFIED mandatory for ALL 2026 World Cup matches at ~22' (1H) and
~67' (2H), with small natural-stoppage jitter (~1-3'). Sources: ESPN, World Soccer Talk,
Reuters (Infantino). So the window boundary EXISTS in every match -- these are well-defined
time-window events, NOT skip-class/undefined ones.

P(>=1 event in window) is a POSTERIOR MEAN over the window's share of full-match event volume:

    P = E_s[ 1 - exp(-lam_event * s) ]        (s ~ the timing-share distribution)

NO hard cap, NO pull-to-middle. Moderation comes ONLY from integrating the share/boundary
uncertainty: 1 - e^{-x} is concave, so a WIDER share distribution yields a LOWER posterior
mean -- by the model, not by a ceiling. A certain share reduces to the deterministic
1 - exp(-lam * s). A genuinely high founded probability ships RAW.

GATE: window_probability refuses to price unless the hydration-break fact is verified
(BREAKS_VERIFIED or an explicit breaks_verified=True) -- no unverified break assumption may
silently produce a number.
"""
from __future__ import annotations

import math

# --- B1 gate: hydration-break fact (verified from primary sources) -----------
BREAKS_VERIFIED = True              # set True ONLY after primary-source verification
FIRST_BREAK_MIN = 22.0
SECOND_BREAK_MIN = 67.0
BREAK_JITTER_MIN = 2.0             # ~1-3' natural-stoppage jitter (sd of the boundary)


class BreakNotVerified(Exception):
    """Raised when a TIME_WINDOW_* row is priced without the break fact verified."""


def share_distribution(s_mean: float, s_sd: float = 0.0, n: int = 5) -> list[tuple[float, float]]:
    """Discrete atoms [(s, weight)] approximating the timing-share distribution.

    s_sd == 0  -> a single atom (DETERMINISTIC: the window's share is known).
    s_sd  > 0  -> Gaussian-weighted atoms in [s_mean +/- 2 sd], clipped to (0,1). This is
                  how boundary jitter / timing-curve uncertainty enters; integrating it is
                  the ONLY source of moderation (no cap)."""
    s_mean = min(max(float(s_mean), 1e-6), 1 - 1e-6)
    if s_sd <= 0:
        return [(s_mean, 1.0)]
    atoms = []
    for k in range(n):
        z = -2.0 + 4.0 * k / (n - 1)               # evenly spaced in [-2, 2] sd
        s = min(max(s_mean + z * s_sd, 1e-6), 1 - 1e-6)
        atoms.append((s, math.exp(-0.5 * z * z)))   # Gaussian weight
    return atoms


def window_probability(lam_event: float, share_atoms: list[tuple[float, float]],
                       breaks_verified: bool | None = None) -> float:
    """Posterior-mean P(>=1 event in window) = sum_i w_i (1 - exp(-lam * s_i)) / sum w_i.

    NO cap. Raises BreakNotVerified if the hydration-break fact is not verified (the B1 gate).
    """
    verified = BREAKS_VERIFIED if breaks_verified is None else breaks_verified
    if verified is not True:
        raise BreakNotVerified(
            "hydration-break fact NOT verified -> TIME_WINDOW_* row is a skip-candidate, "
            "not a silently-priced number")
    if not share_atoms:
        raise ValueError("share_atoms is empty")
    if lam_event < 0:
        raise ValueError("lam_event must be >= 0")
    tot = sum(w for _, w in share_atoms)
    return sum(w * (1.0 - math.exp(-lam_event * s)) for s, w in share_atoms) / tot


def deterministic_window_p(lam_event: float, s: float) -> float:
    """The deterministic component x timing-share calc (the s_sd->0 limit). Exposed so tests
    can assert that a certain share reduces window_probability to exactly this."""
    return 1.0 - math.exp(-lam_event * float(s))


__all__ = ["BREAKS_VERIFIED", "FIRST_BREAK_MIN", "SECOND_BREAK_MIN", "BREAK_JITTER_MIN",
           "BreakNotVerified", "share_distribution", "window_probability",
           "deterministic_window_p"]
