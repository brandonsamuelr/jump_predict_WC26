"""Live evaluator for the validated team_sot_over count-row model.

Replaces the high-biased live SOT model (rate_layer: SOT~Poisson(1.01+3.03*lam),
which runs +0.138 high on count rows) for the ``team_sot_over`` row type only.
Coefficients are fit offline by scripts/fit_sot_count_model.py and stored in
data/models/sot_count_model.json; this module just evaluates a logistic — no
sklearn, no corpus read at lock time.

Features (BOTH pre-kickoff, odds-derived, computed via the pipeline's own de-vig
exactly as in training — no leakage):
  - favorite_gap    = de-vigged P(team wins) - P(opponent wins)   [from h2h]
  - total_line_prob = de-vigged P(over 2.5 goals)                 [from totals]

This returns the model probability, which is submitted UNDISTORTED (k=1 via the
SOT_COUNT tier) — a gate-validated estimate is a real truth-estimate, not pulled
toward the placeholder shadow. Any future LEVEL correction must be data-derived and
directional (edge.GATE_MODEL_LEVEL_CORRECTION, currently inactive), never a hedge.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path("data/models/sot_count_model.json")


@lru_cache(maxsize=1)
def _load(path_str: str = str(MODEL_PATH)) -> dict:
    return json.loads(Path(path_str).read_text())


def is_available() -> bool:
    try:
        return bool(_load().get("thresholds"))
    except Exception:
        return False


def _line_to_threshold(line: float) -> int:
    """'3 or more SOT' is given as line 2.5 -> threshold 3; integer N -> N."""
    return math.ceil(line) if line != int(line) else int(line)


def predict_team_sot_over(favorite_gap: float, total_line_prob: float,
                          line: float) -> float | None:
    """P(team SOT >= ceil(line)) from the fitted per-threshold logistic.

    Returns None if features are missing or the threshold has no fitted model
    (caller then falls back to the existing path).
    """
    if favorite_gap is None or total_line_prob is None or line is None:
        return None
    thr = _line_to_threshold(float(line))
    th = _load().get("thresholds", {})
    if str(thr) not in th:
        # clamp to nearest available threshold (contest lines are 2.5/3.5/4.5)
        avail = sorted(int(k) for k in th)
        if not avail:
            return None
        thr = min(avail, key=lambda a: abs(a - thr))
    c = th[str(thr)]
    z = (c["intercept"] + c["favorite_gap"] * float(favorite_gap)
         + c["total_line_prob"] * float(total_line_prob))
    return 1.0 / (1.0 + math.exp(-z))


__all__ = ["predict_team_sot_over", "is_available", "MODEL_PATH"]
