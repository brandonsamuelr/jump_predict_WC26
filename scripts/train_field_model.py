"""Train + validate the supervised p_field model.

Reads ``data/models/field_training_rows.csv``. Trains a Ridge regression on
``z = logit(clip(y_field, 0.01, 0.99))`` with a ColumnTransformer for one-hot
categoricals + median-impute + standardize numerics.

Validation
----------
Grouped by ``match_norm`` (GroupKFold). Compared against three baselines:

1. ``baseline_50``: always predict 0.50.
2. ``baseline_qt_mean``: leave-one-match-out mean of y_field grouped by
   question_type (a fair test of "does the model do anything beyond
   question-type averages").
3. ``baseline_v2_heuristic``: the live decision_engine.estimate_p_field on
   each row using whatever market context is available.

The model is only saved if it beats ``baseline_v2_heuristic`` on overall
grouped-CV MAE. Training rows, metrics JSON, and a per-feature report are
always written for inspection.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from odds_lib.decision_engine import (
    PriorRow,
    MatchContext,
    estimate_p_field,
    load_priors,
)


TRAIN_PATH = Path("data/models/field_training_rows.csv")
MODEL_PATH = Path("data/models/field_model.pkl")
METRICS_PATH = Path("data/models/field_model_metrics.json")
FEATURE_REPORT_PATH = Path("data/models/field_model_feature_report.csv")


CATEGORICAL_FEATURES = [
    "question_type",
    "target_team",
    "target_player_present_str",
    "target_role_bucket",
    "favorite_team",
    "underdog_team",
]
NUMERIC_FEATURES = [
    "line",
    "favorite_win_prob",
    "underdog_win_prob",
    "target_team_win_prob",
    "favorite_gap",
    "match_total_over_2_5_prob",
    "btts_prob",
    "halftime_draw_prob",
    "threshold_line",
]
BINARY_FEATURES = [
    "has_market_context",
    "target_is_favorite",
    "target_is_underdog",
    "target_is_neutral_or_unknown",
    "is_player_prop",
    "is_team_prop",
    "is_match_prop",
    "is_compound_question",
    "is_second_half_question",
    "is_halftime_question",
    "is_threshold_question",
    "is_fouls_question",
    "is_cards_question",
    "is_sot_question",
    "is_corners_question",
    "is_offsides_question",
    "is_goal_question",
    "is_penalty_question",
    "is_favorite_dominance_question",
    "is_underdog_activity_question",
]


def _logit(p: np.ndarray, eps: float = 0.01) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in NUMERIC_FEATURES:
        if c in out.columns:
            out[c] = _coerce_numeric(out[c])
        else:
            out[c] = np.nan
    for c in BINARY_FEATURES:
        if c in out.columns:
            out[c] = _coerce_numeric(out[c]).fillna(0).astype(int)
        else:
            out[c] = 0
    # OneHotEncoder dislikes NaN in strings; coerce.
    for c in CATEGORICAL_FEATURES:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str).str.strip().str.lower()
    # Build a string version of target_player_present so it's categorical-friendly.
    if "target_player_present" in df.columns:
        out["target_player_present_str"] = (
            df["target_player_present"].astype(int).map({0: "no", 1: "yes"})
        )
    else:
        out["target_player_present_str"] = "no"
    return out


def _build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ohe_kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    try:
        # sklearn >=1.2
        ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
    except TypeError:
        ohe = OneHotEncoder(sparse=False, **ohe_kwargs)
    transformer = ColumnTransformer(
        [
            ("cat", ohe, CATEGORICAL_FEATURES),
            ("num", numeric_pipe, NUMERIC_FEATURES + BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("features", transformer),
        ("ridge", Ridge(alpha=2.0, random_state=0)),
    ])


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _baseline_50(df_train: pd.DataFrame, df_val: pd.DataFrame) -> np.ndarray:
    return np.full(len(df_val), 0.5)


def _baseline_qt_mean(df_train: pd.DataFrame, df_val: pd.DataFrame) -> np.ndarray:
    means = df_train.groupby("question_type")["y_field"].mean()
    global_mean = df_train["y_field"].mean()
    return df_val["question_type"].map(means).fillna(global_mean).to_numpy()


def _baseline_role_qt_mean(df_train: pd.DataFrame, df_val: pd.DataFrame) -> np.ndarray:
    """Hierarchical: shrink (qt, role) cell toward (qt) mean toward global."""
    K_ROLE = 5
    K_QT   = 10
    global_mean = df_train["y_field"].mean()
    qt_means = df_train.groupby("question_type")["y_field"].agg(["mean", "count"])
    role_means = (
        df_train.groupby(["question_type", "target_role_bucket"])["y_field"]
        .agg(["mean", "count"])
    )

    def _pred(row: pd.Series) -> float:
        qt = row["question_type"]
        role = row["target_role_bucket"]
        # Shrink qt mean toward global.
        if qt in qt_means.index:
            qm, qn = float(qt_means.loc[qt, "mean"]), int(qt_means.loc[qt, "count"])
            qt_pred = (qn * qm + K_QT * global_mean) / (qn + K_QT)
        else:
            qt_pred = global_mean
        # Shrink (qt, role) mean toward qt_pred.
        key = (qt, role)
        if key in role_means.index:
            rm, rn = float(role_means.loc[key, "mean"]), int(role_means.loc[key, "count"])
            return (rn * rm + K_ROLE * qt_pred) / (rn + K_ROLE)
        return qt_pred

    return df_val.apply(_pred, axis=1).to_numpy()


def _baseline_v2_heuristic(
    df_train: pd.DataFrame, df_val: pd.DataFrame, priors: dict[str, PriorRow]
) -> np.ndarray:
    """Call decision_engine.estimate_p_field per row. Independent of df_train."""
    preds = []
    for _, r in df_val.iterrows():
        ctx = MatchContext(
            favorite_team=str(r.get("favorite_team") or "") or None,
            underdog_team=str(r.get("underdog_team") or "") or None,
            favorite_win_prob=_optf(r.get("favorite_win_prob")),
            underdog_win_prob=_optf(r.get("underdog_win_prob")),
            fav_underdog_gap=_optf(r.get("favorite_gap")),
            total_line=2.5 if _optf(r.get("match_total_over_2_5_prob")) is not None else None,
            p_over=_optf(r.get("match_total_over_2_5_prob")),
            p_btts_yes=_optf(r.get("btts_prob")),
            p_halftime_draw=_optf(r.get("halftime_draw_prob")),
        )
        gap = ctx.fav_underdog_gap
        ctx.is_close_match = gap is not None and gap < 0.20
        ctx.is_lopsided = gap is not None and gap >= 0.40
        qt = str(r.get("question_type") or "").strip().lower()
        prior = priors.get(qt) or PriorRow(0, 0.50, 0.50, 0.0, 0.0)
        # For V2 baseline we must use a prior estimated WITHOUT the val match's
        # contribution to be fair. Cheap approximation: use the priors built
        # from the full historical CSV; the contamination from one match is
        # small (~3% of average qt sample). A leakage-free implementation
        # would rebuild priors per fold — left as a later upgrade.
        p_field, _, _ = estimate_p_field(
            qt,
            {"target_team": str(r.get("target_team") or "") or None},
            ctx,
            prior,
        )
        if p_field is None:
            # No anchor + no prior in V2 — fall back to global mean for the
            # baseline so we still produce a number.
            p_field = 0.5
        preds.append(p_field)
    return np.array(preds)


def _optf(v) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _err_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean_signed_error": float(err.mean()),
        "median_abs_err": float(np.median(abs_err)),
        "max_abs_err": float(abs_err.max()),
        "n": int(len(y_true)),
    }


def _mae_by(df: pd.DataFrame, key: str) -> dict[str, dict[str, float]]:
    out = {}
    for k, sub in df.groupby(key):
        out[str(k)] = _err_stats(sub["y_field"].to_numpy(), sub["pred"].to_numpy())
    return out


def _worst_misses(df: pd.DataFrame, n: int = 20) -> list[dict]:
    df = df.copy()
    df["abs_err"] = (df["pred"] - df["y_field"]).abs()
    cols = [
        "match_norm", "question_number", "question_type", "target_team",
        "target_player", "y_field", "pred", "abs_err",
        "has_market_context", "target_role_bucket",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.sort_values("abs_err", ascending=False).head(n)[cols].to_dict("records")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    df = pd.read_csv(TRAIN_PATH)
    df = _prep_features(df)
    df["y_field"] = pd.to_numeric(df["y_field"], errors="coerce")
    df = df.dropna(subset=["y_field"]).reset_index(drop=True)
    print(f"loaded {len(df)} rows / {df['match_norm'].nunique()} matches")

    priors = load_priors()

    # ---- GroupKFold by match ----
    n_groups = df["match_norm"].nunique()
    n_splits = min(5, n_groups) if n_groups >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["match_norm"].to_numpy()

    # Collect per-row OOF predictions for each strategy.
    df["pred_model"] = np.nan
    df["pred_baseline_50"] = 0.5
    df["pred_baseline_qt_mean"] = np.nan
    df["pred_baseline_role_qt"] = np.nan
    df["pred_baseline_v2"] = np.nan

    for fold_idx, (tr, va) in enumerate(gkf.split(df, groups=groups)):
        df_tr, df_va = df.iloc[tr], df.iloc[va]
        z_tr = _logit(df_tr["y_field"].to_numpy())
        pipe = _build_pipeline()
        pipe.fit(df_tr, z_tr)
        z_pred = pipe.predict(df_va)
        df.loc[df_va.index, "pred_model"] = _sigmoid(z_pred)

        df.loc[df_va.index, "pred_baseline_qt_mean"] = _baseline_qt_mean(df_tr, df_va)
        df.loc[df_va.index, "pred_baseline_role_qt"] = _baseline_role_qt_mean(df_tr, df_va)
        df.loc[df_va.index, "pred_baseline_v2"] = _baseline_v2_heuristic(df_tr, df_va, priors)
        print(
            f"  fold {fold_idx+1}/{n_splits}: train={len(df_tr)} val={len(df_va)} "
            f"({df_va['match_norm'].nunique()} val matches)"
        )

    # ---- aggregate metrics per strategy ----
    y = df["y_field"].to_numpy()
    strategies = {
        "model_ridge_logit":     df["pred_model"].to_numpy(),
        "baseline_50":           df["pred_baseline_50"].to_numpy(),
        "baseline_qt_mean":      df["pred_baseline_qt_mean"].to_numpy(),
        "baseline_role_qt_mean": df["pred_baseline_role_qt"].to_numpy(),
        "baseline_v2_heuristic": df["pred_baseline_v2"].to_numpy(),
    }

    summary = {name: _err_stats(y, preds) for name, preds in strategies.items()}
    print("\n=== overall (grouped CV) ===")
    print(f"{'strategy':<25} {'MAE':>8} {'RMSE':>8} {'bias':>8} {'medAE':>8} {'maxAE':>8}")
    for name, m in summary.items():
        print(f"{name:<25} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
              f"{m['mean_signed_error']:>+8.4f} {m['median_abs_err']:>8.4f} {m['max_abs_err']:>8.4f}")

    # ---- breakdowns for the model ----
    df_model = df.copy()
    df_model["pred"] = df_model["pred_model"]
    mae_by_qt = _mae_by(df_model, "question_type")
    mae_by_role = _mae_by(df_model, "target_role_bucket")
    mae_by_mctx = _mae_by(df_model, "has_market_context")
    worst = _worst_misses(df_model, n=20)

    # ---- decide whether to save the model ----
    model_mae = summary["model_ridge_logit"]["mae"]
    v2_mae = summary["baseline_v2_heuristic"]["mae"]
    beats_v2 = model_mae < v2_mae
    print(f"\nmodel beats baseline_v2_heuristic? {beats_v2} "
          f"(model_mae={model_mae:.4f} vs v2_mae={v2_mae:.4f})")

    # Fit final model on all data so the live engine can use it (only if beats v2).
    if beats_v2:
        z_all = _logit(df["y_field"].to_numpy())
        pipe = _build_pipeline()
        pipe.fit(df, z_all)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "pipeline": pipe,
                "categorical_features": CATEGORICAL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "binary_features": BINARY_FEATURES,
            }, f)
        print(f"saved final model to {MODEL_PATH}")
    else:
        print("model did NOT beat v2 heuristic on grouped CV; not saving live model.")

    # ---- write metrics + feature report ----
    metrics = {
        "n_rows": int(len(df)),
        "n_matches": int(df["match_norm"].nunique()),
        "n_question_types": int(df["question_type"].nunique()),
        "n_with_market_context": int(df["has_market_context"].sum()),
        "cv": {"strategy": "GroupKFold", "n_splits": n_splits, "group": "match_norm"},
        "overall": summary,
        "model_beats_v2": bool(beats_v2),
        "model_mae_by_question_type": mae_by_qt,
        "model_mae_by_role_bucket": mae_by_role,
        "model_mae_by_has_market_context": mae_by_mctx,
        "model_worst_20_misses": worst,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"wrote metrics: {METRICS_PATH}")

    # Per-question-type comparison report.
    rows = []
    for qt, sub in df.groupby("question_type"):
        row = {"question_type": qt, "n": int(len(sub))}
        for name, pred_col in [
            ("model", "pred_model"),
            ("v2", "pred_baseline_v2"),
            ("qt_mean", "pred_baseline_qt_mean"),
            ("role_qt_mean", "pred_baseline_role_qt"),
        ]:
            row[f"mae_{name}"] = float((sub[pred_col] - sub["y_field"]).abs().mean())
        rows.append(row)
    feat_report = pd.DataFrame(rows).sort_values("n", ascending=False)
    feat_report.to_csv(FEATURE_REPORT_PATH, index=False)
    print(f"wrote per-question-type report: {FEATURE_REPORT_PATH}")


if __name__ == "__main__":
    main()
