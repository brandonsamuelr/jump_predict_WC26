"""Live evaluator for the gate-validated 1H-CORNER comparison model.

P(target team MORE 1st-half corners than opponent) from favorite_gap (2-way de-vig) ONLY
-- structural/odds-derived, NO team identity, so it transfers club->international. Replaces
the direction-BLIND flat stopgap (0.389) for 1H, which was structurally dominated on lopsided
matches (the symmetric 0.389 sat far above a heavy underdog's true 1H-corner rate). Fit on
favorite_gap vs realized OUTCOMES only -- the crowd is NEVER a fit target, only a post-hoc check.

1H ONLY. The 2nd half is game-state-reversed (the trailing underdog chases and wins late
corners -> favorite_gap is WRONG-signed for 2H), so team_more_corners_2h stays on the
stopgap. Coefficients fit offline (data/models/corner_1h_cmp_model.json); this evaluates a
plain logistic -- no sklearn / no corpus at lock.

GATE (SGO corpus 1H corner counts; predictor = favorite_gap, ZERO overlap with corner
outcomes so no FULL-margin trap): beats the flat 0.389 constant by +0.0150 Brier overall
(n=163) and +0.0204 on the lopsided subset (|fg|>0.2, n=134) -- well-powered, not thin.

SIGN: favorite_gap = (target_winprob - opp_winprob)/(target_winprob + opp_winprob), de-vigged
h2h. coef is POSITIVE -> the favorite controls early territory and takes more 1H corners.
Even matchup (gap 0) -> 0.431 (measured base rate, ties->NO). Heavy underdog (-0.5) -> 0.323;
heavy favorite (+0.5) -> 0.547.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path("data/models/corner_1h_cmp_model.json")


@lru_cache(maxsize=1)
def _load(path_str: str = str(MODEL_PATH)) -> dict:
    return json.loads(Path(path_str).read_text())


def is_available() -> bool:
    try:
        m = _load()
        return "intercept" in m and "favorite_gap" in m.get("coefs", {})
    except Exception:
        return False


def predict_more_corners_1h(favorite_gap) -> float | None:
    """P(target team MORE 1H corners than opponent). favorite_gap = target 2-way de-vig edge."""
    if favorite_gap is None:
        return None
    m = _load()
    z = m["intercept"] + m["coefs"]["favorite_gap"] * float(favorite_gap)
    return 1.0 / (1.0 + math.exp(-z))


__all__ = ["predict_more_corners_1h", "is_available", "MODEL_PATH"]
