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

# Computed contest-resolved base rates for question types whose qt-mean has too
# few historical rows to clear MIN_COUNT — so they'd otherwise hit the global
# ~0.49 fallback, which is a BAD anchor for them. A bad c_hat can't be fixed by
# any submission multiplier (the rule is c_hat + k*(p_model-c_hat)), so we fix
# the baseline directly.
TYPE_BASE_RATE = {
    # n=3 field-resolved instances (0.63/0.64/0.60), tight. The full-match x0.55
    # ESPN diagnostic says ~0.84 — the gap is the SOT definition mismatch (ESPN
    # counts more than the contest resolves), so the contest-resolved ~0.62 is
    # the correct anchor and confirms the ESPN-calibrated model ran high. SOFT
    # (n=3); hardens as resolved outcomes on this row accumulate.
    "total_sot_2h_over": 0.623,
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
        if rec is not None and rec[2] >= MIN_COUNT:
            return FieldEstimate(q_hat=rec[0], source="qt_mean", n=rec[2], std=rec[1])
        if qt in TYPE_BASE_RATE:  # better-sourced fallback than the global ~0.49 mean
            return FieldEstimate(q_hat=TYPE_BASE_RATE[qt], source="type_base_rate",
                                 n=(rec[2] if rec else 0), std=None)
        return FieldEstimate(q_hat=self.global_mean, source="global_mean",
                             n=(rec[2] if rec else 0), std=None)


__all__ = ["FieldMeanEstimator", "FieldEstimate"]
