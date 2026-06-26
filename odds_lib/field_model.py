"""Field-mean estimator — the SHADOW ANCHOR, not a truth model.

Its only job: estimate the field's average submission q-bar for a question
type, so that on no-edge rows we can submit near the field and harvest the
convexity bonus Var(q_i). It is deliberately NOT used as p_truth (the ML
field model never beat this qt-mean baseline at predicting truth anyway).

The estimate is the historical mean of the crowd's field probability per
question type, with a global-mean fallback for thin/unseen types.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_HISTORY = "data/historical/sportspredict_collected_data.csv"
MIN_COUNT = 4  # below this, fall back (type base rate if known, else global mean)

# UNVALIDATED PLACEHOLDER anchors (pending the outcome gate, odds_lib/validation.py)
# for question types whose qt-mean has too few historical rows to clear MIN_COUNT,
# so they'd otherwise hit the global ~0.49 fallback. These are a less-bad FALLBACK,
# NOT a validated base rate: they were set from agreement with the realized CROWD
# on a tiny sample, which is NOT evidence they predict the OUTCOME (matching the
# crowd earns zero edge by construction). Do not treat as "reliable" — they stay
# placeholders until they pass the gate against actual outcomes.
TYPE_BASE_RATE = {
    # total_sot_2h_over: n=3 crowd-AGREEMENT instances (0.63/0.64/0.60); tight but
    # NOT outcome-gate-validated. Kept only as a less-bad fallback than the 0.49
    # global mean. UNVALIDATED placeholder; value unchanged here (labeling only).
    "total_sot_2h_over": 0.623,
}

# OUTCOME-derived shadow recalibrations (highest precedence — used INSTEAD of the
# crowd qt-mean). Distinct from TYPE_BASE_RATE (crowd-agreement fallback): these are
# set from REALIZED outcomes where the qt-mean is measurably off.
TYPE_SHADOW_OVERRIDE = {
    # team_more_fouls: crowd qt-mean is 0.535, but realized outcomes are 0.500
    # (15/30 historical) and 0.50 is the principled symmetric base for "does THIS
    # team foul more" (each side equally likely, minus ties). The fouls diagnostic
    # confirmed the model can't beat shadow (would lose -7.31 RBP on actual rows),
    # so the only data-supported fix is this VALUE recalibration.
    "team_more_fouls": 0.50,
}


@dataclass
class FieldEstimate:
    q_hat: float
    source: str   # "qt_mean" | "global_mean"
    n: int
    std: float | None


class FieldMeanEstimator:
    def __init__(self, history_path: str = DEFAULT_HISTORY):
        df = pd.read_csv(history_path)
        df = df[df["field_prob"].notna()].copy()
        df["fp"] = df["field_prob"] / 100.0
        self.global_mean = float(df["fp"].mean())
        g = df.groupby("question_type")["fp"].agg(["mean", "std", "size"])
        self._by_type = {
            qt: (float(r["mean"]), (float(r["std"]) if pd.notna(r["std"]) else None),
                 int(r["size"]))
            for qt, r in g.iterrows()
        }

    def estimate(self, question_type: str | None) -> FieldEstimate:
        qt = (question_type or "").strip().lower()
        rec = self._by_type.get(qt)
        if qt in TYPE_SHADOW_OVERRIDE:  # outcome-derived recalibration, highest precedence
            return FieldEstimate(q_hat=TYPE_SHADOW_OVERRIDE[qt], source="outcome_base_rate",
                                 n=(rec[2] if rec else 0), std=None)
        if rec is not None and rec[2] >= MIN_COUNT:
            return FieldEstimate(q_hat=rec[0], source="qt_mean", n=rec[2], std=rec[1])
        if qt in TYPE_BASE_RATE:  # better-sourced fallback than the global ~0.49 mean
            return FieldEstimate(q_hat=TYPE_BASE_RATE[qt], source="type_base_rate",
                                 n=(rec[2] if rec else 0), std=None)
        return FieldEstimate(q_hat=self.global_mean, source="global_mean",
                             n=(rec[2] if rec else 0), std=None)


__all__ = ["FieldMeanEstimator", "FieldEstimate"]
