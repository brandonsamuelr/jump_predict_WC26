"""Live evaluators for the gate-validated corpus models (Track 2).

Two rows that have NO market but passed the out-of-sample gate on the 84k-match
corpus with meaningful, well-calibrated margins:
  - team_more_cards         (favorite_gap, total_line_prob, is_home)
  - match_total_sot_over     (favorite_gap, total_line_prob; per threshold)

Coefficients are fit offline (scripts/fit_track2_models.py -> corpus_models.json);
this evaluates a plain logistic — no sklearn / no corpus read at lock. ALL features
are pre-kickoff odds-derived (computed via the pipeline de-vig), no leakage.

Like the SOT count model, the model probability is returned here and submitted
UNDISTORTED (k=1 via the MORE_CARDS / MATCH_SOT tiers) — a gate-validated estimate
is not pulled toward the placeholder shadow. Any future LEVEL correction must be
data-derived and directional (edge.GATE_MODEL_LEVEL_CORRECTION, inactive), not a hedge.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path("data/models/corpus_models.json")


@lru_cache(maxsize=1)
def _load(path_str: str = str(MODEL_PATH)) -> dict:
    return json.loads(Path(path_str).read_text())


def _logit(coefs: dict, intercept: float, feats: dict) -> float:
    z = intercept + sum(coefs[k] * feats[k] for k in coefs)
    return 1.0 / (1.0 + math.exp(-z))


def cards_available() -> bool:
    try:
        return "team_more_cards" in _load()
    except Exception:
        return False


def match_sot_available() -> bool:
    try:
        return bool(_load().get("match_total_sot_over", {}).get("thresholds"))
    except Exception:
        return False


def predict_team_more_cards(favorite_gap, total_line_prob, is_home) -> float | None:
    if favorite_gap is None or total_line_prob is None or is_home is None:
        return None
    m = _load().get("team_more_cards")
    if not m:
        return None
    return _logit(m["coefs"], m["intercept"],
                  {"favorite_gap": float(favorite_gap),
                   "total_line_prob": float(total_line_prob),
                   "is_home": float(is_home)})


def _line_to_threshold(line: float) -> int:
    return math.ceil(line) if line != int(line) else int(line)


def predict_match_total_sot_over(favorite_gap, total_line_prob, line) -> float | None:
    if favorite_gap is None or total_line_prob is None or line is None:
        return None
    th = _load().get("match_total_sot_over", {}).get("thresholds", {})
    if not th:
        return None
    k = _line_to_threshold(float(line))
    if str(k) not in th:
        k = min((int(x) for x in th), key=lambda a: abs(a - k))  # nearest available
    c = th[str(k)]
    return _logit(c["coefs"], c["intercept"],
                  {"favorite_gap": float(favorite_gap),
                   "total_line_prob": float(total_line_prob)})


__all__ = ["predict_team_more_cards", "predict_match_total_sot_over",
           "cards_available", "match_sot_available", "MODEL_PATH"]
