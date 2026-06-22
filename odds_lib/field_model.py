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
MIN_COUNT = 4  # below this, fall back to the global mean (too few to trust)


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
        if rec is not None and rec[2] >= MIN_COUNT:
            return FieldEstimate(q_hat=rec[0], source="qt_mean", n=rec[2], std=rec[1])
        return FieldEstimate(q_hat=self.global_mean, source="global_mean",
                             n=(rec[2] if rec else 0), std=None)


__all__ = ["FieldMeanEstimator", "FieldEstimate"]
