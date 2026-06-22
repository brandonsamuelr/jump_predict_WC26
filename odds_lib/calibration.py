"""Calibration log — pre-lock predictions + post-lock outcomes in one CSV.

Schema is a superset of ``sportspredict_submitted_scoring_rows.csv`` so the
two files can be concatenated for historical analysis once enough live rows
land. Pre-lock columns are written by ``submit_sheet.py`` at submission time;
post-lock columns (``field_prob``, ``result``, ``outcome_pct``, ``actual_rbp``,
``if_yes_rbp``, ``if_no_rbp``, ``swing_display``) are filled in by hand or
via ``backfill_calibration_row`` after the question resolves.

The point of this file is to measure whether the engine's ``p_field_est`` is
actually close to the locked crowd average. Without that loop we cannot tell
if our shadow submissions are harvesting the Jensen gap or burning RBP on
miscalibrated field estimates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .logs import append_csv


CALIBRATION_LOG_PATH = Path("data/calibration_log.csv")


# Columns shared with sportspredict_submitted_scoring_rows.csv. Order
# matches that file so a concatenation is clean.
HISTORICAL_COMPATIBLE_COLS = [
    "row_id",
    "source_batch",
    "status",
    "game_date",
    "match_raw",
    "match_norm",
    "question_number",
    "question",
    "question_type",
    "field_prob",          # post-lock crowd YES%
    "result",              # post-lock "Yes"/"No"
    "outcome_pct",         # post-lock 0/100
    "submitted",           # 1 if we submitted, 0 if skipped
    "submitted_percent",   # our locked submission (integer %)
    "actual_rbp",          # post-lock
    "if_yes_rbp",          # post-lock
    "if_no_rbp",           # post-lock
    "swing_display",       # post-lock
    "notes",
]

# Engine-side pre-lock columns. These let us reconstruct WHY a submission was
# made and compare engine field estimate vs actual locked field.
ENGINE_PRE_LOCK_COLS = [
    "submitted_at",
    "forecast_run_id",
    "kickoff_time_utc",
    "target_team",
    "target_player",
    "line",
    "market_prob",
    "p_field_est",
    "p_field_source",
    "field_confidence",
    "p_truth_est",
    "p_truth_source",
    "truth_confidence",
    "p_submit",
    "decision_mode",
    "delta_vs_field",
    "estimated_swing",
    "risk_tags",
    "engine_reason",
    "historical_candidate",
    "candidate_n",
    "candidate_raw_bias",
    "candidate_reason",
    "promotion_status",
    "mapping_status",
    "all_num_books",
    "sharp_num_books",
    "liquidity_flag",
    "review_flags",
    "manual_override",
]

# Post-lock derived columns added by backfill_calibration_row.
POST_LOCK_DERIVED_COLS = [
    "field_error",  # p_field_est - field_prob/100
]

CALIBRATION_COLS = (
    HISTORICAL_COMPATIBLE_COLS + ENGINE_PRE_LOCK_COLS + POST_LOCK_DERIVED_COLS
)


def _empty_row() -> dict[str, Any]:
    return {c: "" for c in CALIBRATION_COLS}


def build_prelock_row(
    *,
    forecast_run_id: str,
    sheet_row: dict[str, Any],
    submitted: int,
) -> dict[str, Any] | None:
    """Translate one submit_sheet row into a calibration_log pre-lock row.

    Returns None for rows we did NOT submit (caller filters first). Post-lock
    columns are left blank for later backfill.
    """
    row = _empty_row()

    # Historical-schema columns we can fill at submission time.
    match = sheet_row.get("match", "")
    kickoff_local = sheet_row.get("kickoff_time_local", "")
    game_date = kickoff_local.split(" ")[0] if kickoff_local else ""
    row.update({
        "source_batch": f"live_engine_v2:{forecast_run_id}",
        "status": "pending",
        "game_date": game_date,
        "match_raw": match,
        "match_norm": match,
        "question": sheet_row.get("sports_predict_question", ""),
        "question_type": sheet_row.get("question_type", ""),
        "submitted": int(submitted),
        "submitted_percent": sheet_row.get("submit_percent", ""),
        "notes": "engine_v2_prelock",
    })

    # Engine-side columns.
    row.update({
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "forecast_run_id": forecast_run_id,
        "kickoff_time_utc": sheet_row.get("kickoff_time_utc", ""),
        "target_team": sheet_row.get("target_team", ""),
        "target_player": sheet_row.get("target_player", ""),
        "line": sheet_row.get("mapped_line", ""),
        "market_prob": sheet_row.get("market_prob", ""),
        "p_field_est": sheet_row.get("p_field", ""),
        "p_field_source": sheet_row.get("p_field_source", ""),
        "field_confidence": sheet_row.get("field_confidence", ""),
        "p_truth_est": sheet_row.get("p_truth", ""),
        "p_truth_source": sheet_row.get("p_truth_source", ""),
        "truth_confidence": sheet_row.get("truth_confidence", ""),
        "p_submit": sheet_row.get("p_submit", ""),
        "decision_mode": sheet_row.get("decision_mode", ""),
        "delta_vs_field": sheet_row.get("delta_vs_field", ""),
        "estimated_swing": sheet_row.get("estimated_swing", ""),
        "risk_tags": sheet_row.get("risk_tags", ""),
        "engine_reason": sheet_row.get("reason", ""),
        "historical_candidate": sheet_row.get("historical_candidate", ""),
        "candidate_n": sheet_row.get("candidate_n", ""),
        "candidate_raw_bias": sheet_row.get("candidate_raw_bias", ""),
        "candidate_reason": sheet_row.get("candidate_reason", ""),
        "promotion_status": sheet_row.get("promotion_status", ""),
        "mapping_status": sheet_row.get("mapping_status", ""),
        "all_num_books": sheet_row.get("all_num_books", ""),
        "sharp_num_books": sheet_row.get("sharp_num_books", ""),
        "liquidity_flag": sheet_row.get("liquidity_flag", ""),
        "review_flags": sheet_row.get("review_flags", ""),
        "manual_override": sheet_row.get("manual_override", ""),
    })
    return row


def append_prelock_rows(
    sheet: pd.DataFrame,
    forecast_run_id: str,
    path: Path | str = CALIBRATION_LOG_PATH,
) -> int:
    """Append a pre-lock calibration row for every ``submit_recommendation == 'submit'``.

    Returns the number of rows appended. No-ops (and returns 0) if the sheet
    has no submit rows.
    """
    to_log = sheet[sheet["submit_recommendation"] == "submit"]
    if to_log.empty:
        return 0
    rows = []
    for sheet_row in to_log.to_dict("records"):
        r = build_prelock_row(
            forecast_run_id=forecast_run_id, sheet_row=sheet_row, submitted=1
        )
        if r is not None:
            rows.append(r)
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    # Re-order to canonical column order so the on-disk file stays stable.
    df = df[[c for c in CALIBRATION_COLS if c in df.columns]]
    append_csv(df, Path(path))
    return len(rows)


def backfill_calibration_row(
    path: Path | str,
    *,
    match_substring: str,
    question_substring: str,
    field_prob: float | None = None,
    result: str | None = None,
    actual_rbp: float | None = None,
    if_yes_rbp: float | None = None,
    if_no_rbp: float | None = None,
    swing_display: float | None = None,
) -> int:
    """Fill post-lock columns on matching row(s). Two-phase friendly.

    Realistic timeline:
      1. At kickoff: ``field_prob`` is visible (locked crowd %). Record it.
         Row stays ``status='pending'`` because we don't know ``result`` yet.
      2. After match: ``result`` + ``actual_rbp`` etc. are known. Record
         them; status flips to ``resolved``.

    Either or both phases can be called. A row becomes ``resolved`` only
    when ``result`` is supplied. ``field_error`` is computed whenever
    ``field_prob`` is supplied. Matches both pending and resolved rows so
    you can correct a previous backfill.
    """
    p = Path(path)
    df = pd.read_csv(p)
    # Pre-lock writes leave string columns empty, which pandas reads as NaN
    # float; assignments below would raise. Coerce to object up front.
    for col in ("result", "notes", "status"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    mask = (
        df["match_norm"].astype(str).str.lower().str.contains(match_substring.lower(), na=False, regex=False)
        & df["question"].astype(str).str.lower().str.contains(question_substring.lower(), na=False, regex=False)
    )
    if not mask.any():
        return 0
    if field_prob is not None:
        df.loc[mask, "field_prob"] = field_prob
        p_field_vals = pd.to_numeric(df.loc[mask, "p_field_est"], errors="coerce")
        df.loc[mask, "field_error"] = p_field_vals - (field_prob / 100.0)
    if result is not None:
        df.loc[mask, "result"] = result
        df.loc[mask, "outcome_pct"] = 100.0 if result.strip().lower() == "yes" else 0.0
        df.loc[mask, "status"] = "resolved"
    if actual_rbp is not None:
        df.loc[mask, "actual_rbp"] = actual_rbp
    if if_yes_rbp is not None:
        df.loc[mask, "if_yes_rbp"] = if_yes_rbp
    if if_no_rbp is not None:
        df.loc[mask, "if_no_rbp"] = if_no_rbp
    if swing_display is not None:
        df.loc[mask, "swing_display"] = swing_display
    df.to_csv(p, index=False)
    return int(mask.sum())


__all__ = [
    "CALIBRATION_LOG_PATH",
    "CALIBRATION_COLS",
    "HISTORICAL_COMPATIBLE_COLS",
    "ENGINE_PRE_LOCK_COLS",
    "POST_LOCK_DERIVED_COLS",
    "build_prelock_row",
    "append_prelock_rows",
    "backfill_calibration_row",
]
