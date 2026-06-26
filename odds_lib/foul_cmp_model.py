"""Live evaluator for the gate-validated UNIVERSAL foul-comparison model.

P(target team commits MORE fouls than opponent) from favorite_gap (de-vigged odds)
ONLY — structural/odds-derived, NO team identity, so it transfers club->international.
Gate-validated OOS on the 168k club corpus (beats the flat-0.50 placeholder; coef
-0.71 = underdog fouls more). Coefficients fit offline (data/models/foul_cmp_model.json);
this evaluates a plain logistic — no sklearn / no corpus at lock.

SIGN CONVENTION (verify when wiring): favorite_gap = target_win_prob - opp_win_prob
(de-vigged h2h). coef is NEGATIVE -> as the target becomes MORE of a favorite
(gap up), P(target more fouls) goes DOWN (favorites foul less / underdogs chase & foul
more). Even matchup (gap 0) -> 0.460 (the measured base rate, ties->NO). Heavy underdog
(gap -0.50) -> 0.549; heavy favorite (+0.50) -> 0.374.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path("data/models/foul_cmp_model.json")


@lru_cache(maxsize=1)
def _load(path_str: str = str(MODEL_PATH)) -> dict:
    return json.loads(Path(path_str).read_text())


def is_available() -> bool:
    try:
        m = _load()
        return "intercept" in m and "favorite_gap" in m.get("coefs", {})
    except Exception:
        return False


def predict_more_fouls(favorite_gap) -> float | None:
    """P(target team commits MORE fouls than opponent), full match."""
    if favorite_gap is None:
        return None
    m = _load()
    z = m["intercept"] + m["coefs"]["favorite_gap"] * float(favorite_gap)
    return 1.0 / (1.0 + math.exp(-z))


__all__ = ["predict_more_fouls", "is_available", "MODEL_PATH"]
