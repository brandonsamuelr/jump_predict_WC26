"""Out-of-sample validation harness — the gate every Bucket-B model must pass.

Deliverable 2 of the Bucket-B foundation. The rule it enforces: **no estimator
ships unless it beats the current baseline (the question-type-mean "shadow")
out-of-sample.** We have repeatedly shipped estimators that were never proven to
beat the simpler alternative (a 20-match SOT fit that ran high; "soft" overrides
that cost RBP). This harness makes that test mandatory and reusable.

What it does
------------
Given the corpus, a binary target derived from it, and a candidate estimator,
it: (1) splits train/test by season (time-based — fit on past, predict future,
no leakage), (2) computes the baseline = the unconditional train-set mean
outcome (the flat "shadow"), (3) scores both on the held-out test set with
Brier score and log-loss (proper scores; lower = better), (4) returns a clear
PASS/FAIL (candidate must beat the baseline on BOTH proper scores) plus a
calibration table so we can see the candidate is well-calibrated, not just
lower-loss.

Reusable by design: ``validate_candidate`` takes any ``target_fn`` (corpus ->
binary y) and any ``fit_predict_fn`` (train_df, test_df, y_train -> p_test), so
every future per-stat model (fouls, cards, SOT-comparison, corners) runs through
the same gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

EPS = 1e-12


# --- proper scores ----------------------------------------------------------
def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of probabilistic prediction. Lower = better."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Binary cross-entropy with clipping. Lower = better."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions into ``n_bins`` and compare predicted vs actual freq."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f})",
            "n": int(m.sum()),
            "pred_mean": float(p[m].mean()),
            "actual_freq": float(y[m].mean()),
        })
    return pd.DataFrame(rows)


# --- splitting --------------------------------------------------------------
def time_split(df: pd.DataFrame, season_col: str = "season",
               test_frac: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: the most recent ``test_frac`` of seasons are the test
    set, the rest are train. Mimics 'fit on past, predict future' and prevents
    leakage of same-season information across the split."""
    seasons = sorted(df[season_col].dropna().unique())
    if len(seasons) < 2:
        raise ValueError("need >= 2 seasons for a time-based split")
    n_test = max(1, round(len(seasons) * test_frac))
    test_seasons = set(seasons[-n_test:])
    test = df[df[season_col].isin(test_seasons)].copy()
    train = df[~df[season_col].isin(test_seasons)].copy()
    return train, test


# --- result container -------------------------------------------------------
@dataclass
class ValidationResult:
    target_name: str
    n_train: int
    n_test: int
    base_rate_train: float
    baseline_brier: float
    baseline_log_loss: float
    candidate_brier: float
    candidate_log_loss: float
    train_seasons: tuple = ()
    test_seasons: tuple = ()
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def brier_delta(self) -> float:        # negative = candidate better
        return self.candidate_brier - self.baseline_brier

    @property
    def log_loss_delta(self) -> float:     # negative = candidate better
        return self.candidate_log_loss - self.baseline_log_loss

    @property
    def passed(self) -> bool:
        """PASS only if the candidate beats the flat baseline on BOTH proper
        scores out-of-sample."""
        return (self.candidate_brier < self.baseline_brier
                and self.candidate_log_loss < self.baseline_log_loss)

    def report(self) -> str:
        v = "PASS" if self.passed else "FAIL"
        tr = f"{self.train_seasons[0]}..{self.train_seasons[-1]}" if self.train_seasons else "?"
        te = f"{self.test_seasons[0]}..{self.test_seasons[-1]}" if self.test_seasons else "?"
        lines = [
            f"=== validation: {self.target_name} ===",
            f"  split: train {self.n_train:,} rows [{tr}]"
            f"  |  test {self.n_test:,} rows [{te}]",
            f"  train base rate (the flat shadow): {self.base_rate_train:.4f}",
            f"  {'metric':<10}{'baseline':>12}{'candidate':>12}{'delta':>12}",
            f"  {'Brier':<10}{self.baseline_brier:>12.5f}{self.candidate_brier:>12.5f}"
            f"{self.brier_delta:>+12.5f}",
            f"  {'log_loss':<10}{self.baseline_log_loss:>12.5f}{self.candidate_log_loss:>12.5f}"
            f"{self.log_loss_delta:>+12.5f}",
            f"  -> {v}  (candidate must beat baseline on BOTH; lower = better)",
            "  calibration (test set):",
        ]
        for _, r in self.calibration.iterrows():
            lines.append(f"    {r['bin']:<12} n={int(r['n']):>6}  "
                         f"pred={r['pred_mean']:.3f}  actual={r['actual_freq']:.3f}")
        return "\n".join(lines)


# --- the gate ---------------------------------------------------------------
def validate_candidate(
    df: pd.DataFrame,
    target_fn: Callable[[pd.DataFrame], pd.Series],
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame, np.ndarray], np.ndarray],
    required_cols: list[str],
    target_name: str = "candidate",
    split_fn: Callable = time_split,
    n_bins: int = 10,
) -> ValidationResult:
    """Run the out-of-sample gate.

    Parameters
    ----------
    df : the corpus (e.g. data/historical/stat_lines.csv loaded).
    target_fn : maps the corpus to a binary 0/1 Series (the outcome to predict).
    fit_predict_fn : ``(train_df, test_df, y_train) -> p_test`` — fits the
        candidate on train and returns probabilities for every test row. Any
        estimator (logistic, GBM, hand-tuned formula) plugs in here.
    required_cols : columns the candidate needs; rows with NaN in any of these
        (or NaN target) are dropped BEFORE the split so baseline and candidate
        are scored on the IDENTICAL test set (fair comparison).

    The baseline is the unconditional train-set mean outcome (the flat
    question-type "shadow"), scored on the same test rows.
    """
    work = df.copy()
    work["_y"] = target_fn(work).astype(float)
    work = work.dropna(subset=required_cols + ["_y"])
    train, test = split_fn(work)

    y_train = train["_y"].to_numpy()
    y_test = test["_y"].to_numpy()
    base_rate = float(y_train.mean())

    # baseline: flat train mean for every test row
    p_base = np.full(len(test), base_rate)
    # candidate
    p_cand = np.asarray(fit_predict_fn(train, test, y_train), float)
    if p_cand.shape[0] != len(test):
        raise ValueError("fit_predict_fn must return one prob per test row")

    return ValidationResult(
        target_name=target_name,
        n_train=len(train), n_test=len(test),
        base_rate_train=base_rate,
        baseline_brier=brier(y_test, p_base),
        baseline_log_loss=log_loss(y_test, p_base),
        candidate_brier=brier(y_test, p_cand),
        candidate_log_loss=log_loss(y_test, p_cand),
        train_seasons=tuple(sorted(train["season"].unique())),
        test_seasons=tuple(sorted(test["season"].unique())),
        calibration=calibration_table(y_test, p_cand, n_bins=n_bins),
    )


__all__ = [
    "brier", "log_loss", "calibration_table", "time_split",
    "ValidationResult", "validate_candidate",
]
