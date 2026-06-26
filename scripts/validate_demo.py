"""Demo of the validation harness (NOT a shipped model).

Proves the gate works by running it once on a trivial candidate: a single
feature logistic regression of P(team has more fouls than opponent) ~
favorite_gap, vs the flat question-type-mean baseline. Reports baseline vs
candidate Brier/log-loss, PASS/FAIL, and calibration buckets.

    python scripts/validate_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from odds_lib.validation import validate_candidate

CORPUS = Path("data/historical/stat_lines.csv")
FEATURE = "favorite_gap"


def target_more_fouls(df: pd.DataFrame) -> pd.Series:
    """Binary: did this team commit MORE fouls than its opponent? (ties -> 0)."""
    return (df["fouls_for"] > df["fouls_against"]).astype(int)


def fit_predict_logit(train: pd.DataFrame, test: pd.DataFrame,
                      y_train: np.ndarray) -> np.ndarray:
    """Single-feature logistic regression on favorite_gap."""
    Xtr = train[[FEATURE]].to_numpy()
    Xte = test[[FEATURE]].to_numpy()
    clf = LogisticRegression()
    clf.fit(Xtr, y_train)
    print(f"  [demo model] logit P(more fouls) ~ {FEATURE}: "
          f"coef={clf.coef_[0][0]:+.3f}  intercept={clf.intercept_[0]:+.3f}")
    return clf.predict_proba(Xte)[:, 1]


def main():
    if not CORPUS.exists():
        sys.exit(f"corpus not found at {CORPUS}; run scripts/build_stat_lines.py first")
    df = pd.read_csv(CORPUS)
    res = validate_candidate(
        df,
        target_fn=target_more_fouls,
        fit_predict_fn=fit_predict_logit,
        required_cols=[FEATURE, "fouls_for", "fouls_against"],
        target_name="more_fouls_than_opp ~ favorite_gap (DEMO)",
    )
    print(res.report())


if __name__ == "__main__":
    main()
