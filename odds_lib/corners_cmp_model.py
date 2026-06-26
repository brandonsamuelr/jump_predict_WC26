"""Live evaluator for the gate-validated corner-comparison model.

Predicts P(team has MORE corners than opponent) from favorite_gap (de-vigged odds)
— corners are a byproduct of territorial dominance, which the odds already encode.
Gate-validated OOS on the 84k club corpus (−0.029 Brier vs the flat shadow,
favorite_gap only, consistent). Coefficients fit offline (scripts produced
data/models/corners_cmp_model.json); this evaluates a plain logistic — no sklearn /
no corpus at lock.

TRANSFER: uses ONLY odds-derived features (favorite_gap, total_line_prob) — NO team
identities — so it transfers club->international (odds put both on the same dominance
scale). Log predicted-vs-realized on the first international corner rows; correct the
LEVEL only if a measured directional bias appears, never pre-emptively.

HALF-WINDOW (1H/2H): the full-match DIRECTION is validated; the 1H/2H LEVEL is NOT
(no half-corner data exists to calibrate it). A single half has ~half the corners, so
the comparison is genuinely noisier and sits closer to 0.5 — this is a half-window
conversion (like H1_SHARE for goals), NOT a hedge toward an anchor. We regress the
full-match probability toward 0.5 by CORNER_HALF_SHRINK, a labeled PROVISIONAL,
UNVALIDATED constant — flagged everywhere, calibrated later from logged outcomes.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path("data/models/corners_cmp_model.json")

# DEPRECATED (no longer in the live path). The half-window regression toward 0.5 by
# CORNER_HALF_SHRINK=0.65 was PROVEN WRONG by leakage-free analysis on StatsBomb 72
# intl matches: (1) corner dominance barely persists per-half (1H<->2H carry only
# +0.031), and (2) the half base rate is ~0.40, NOT 0.50 — so a 0.5-centered shrink
# is doubly miscalibrated. The 1H/2H route now submits the measured per-half BASE-RATE
# FLOOR below (a STOPGAP, see CORNER_HALF_BASE_RATE). Symbols kept for backward compat.
CORNER_HALF_SHRINK = 0.65            # DEPRECATED — unused in live pricing
CORNER_HALF_SHRINK_CORRECTION: float | None = None   # DEPRECATED

# --- STOPGAP base-rate floors for 1H/2H more-corners (NOT a model, NOT true P) ----
# STOPGAP_NOT_TRUE_P__HIGH_PRIORITY_UNSOLVED. These are the measured per-half base
# rates (ties->NO, StatsBomb 72 intl matches): the honest current estimate while THE
# DATA WALL (no single dataset has BOTH half-splits AND odds) blocks fitting a
# favorite_gap-conditioned half model. A flat ~0.40 for every match is NOT true P —
# it ignores favorite_gap, which demonstrably matters for full-match corners. Open
# until a joint half-split+odds dataset is sourced/constructed; the route is tagged
# CORNER_HALF_STOPGAP so resolved rows accumulate toward a real fit.
CORNER_HALF_BASE_RATE = {"1h": 0.389, "2h": 0.410}


def half_stopgap_p(question_type: str) -> float:
    """STOPGAP per-half base-rate floor for a more-corners 1H/2H row (NOT a model).
    Ignores favorite_gap (un-fittable per-half due to the data wall) — flagged so the
    bleed accumulates toward a real conditional fit."""
    qt = question_type.lower()
    is_1h = "1h" in qt or "h1" in qt          # accept both spellings (..._1h / ..._h1)
    return CORNER_HALF_BASE_RATE["1h" if is_1h else "2h"]


@lru_cache(maxsize=1)
def _load(path_str: str = str(MODEL_PATH)) -> dict:
    return json.loads(Path(path_str).read_text())


def is_available() -> bool:
    try:
        return bool(_load().get("coefs"))
    except Exception:
        return False


def predict_more_corners(favorite_gap, total_line_prob) -> float | None:
    """P(target team finishes with MORE corners than opponent), FULL match."""
    if favorite_gap is None or total_line_prob is None:
        return None
    m = _load()
    c = m["coefs"]
    z = (m["intercept"] + c["favorite_gap"] * float(favorite_gap)
         + c["total_line_prob"] * float(total_line_prob))
    return 1.0 / (1.0 + math.exp(-z))


def _half_shrink() -> float:
    return CORNER_HALF_SHRINK_CORRECTION if CORNER_HALF_SHRINK_CORRECTION is not None else CORNER_HALF_SHRINK


def apply_half_shrink(p_full: float) -> float:
    """Convert a FULL-match more-corners prob to a 1H/2H value (regress toward 0.5).
    PROVISIONAL — the shrink constant is unvalidated (see module docstring)."""
    return 0.5 + _half_shrink() * (float(p_full) - 0.5)


__all__ = ["predict_more_corners", "apply_half_shrink", "is_available",
           "half_stopgap_p", "CORNER_HALF_BASE_RATE",
           "CORNER_HALF_SHRINK", "CORNER_HALF_SHRINK_CORRECTION", "MODEL_PATH"]
