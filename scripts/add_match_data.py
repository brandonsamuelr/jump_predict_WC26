"""Idempotently add a fixed list of match rows to the historical + calibration files.

This is the "Part 1" entry point. It is safe to re-run — duplicates are
detected via the ``(match_norm, question_number)`` composite key and existing
rows are overwritten with the canonical values defined below.

Files touched
-------------
1. ``data/historical/sportspredict_collected_data.csv``
   The full historical question pool. Adds new columns
   (``target_team``, ``target_player``, ``line``) if missing.

2. ``data/historical/sportspredict_submitted_scoring_rows.csv``
   The subset Brandon actually submitted, with realized RBP.

3. ``data/calibration_log.csv``
   - Adds a ``row_kind`` column if missing.
   - Marks every existing row as ``row_kind='engine_counterfactual'`` (the
     pre-lock engine retro outputs seeded earlier).
   - Appends new rows for both matches with ``row_kind='actual_submission'``
     carrying Brandon's submitted_percent + locked crowd + post-match RBP.
     Engine-side columns (p_field_est etc.) are left blank — these rows
     record what *was* submitted, not what the engine *would have* submitted.

The dual-row-kind setup is critical: counterfactual rows are valid for
field-error audits but must NOT contribute to RBP/score analysis (Brandon
did not submit those numbers). Audit scripts split on ``row_kind``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.calibration import CALIBRATION_LOG_PATH, CALIBRATION_COLS


COLLECTED_PATH = Path("data/historical/sportspredict_collected_data.csv")
SUBMITTED_PATH = Path("data/historical/sportspredict_submitted_scoring_rows.csv")

# Extra columns needed for field-model feature engineering. Existing rows get
# blank values; new rows populate them from the inline data below.
EXTRA_HISTORICAL_COLS = ["target_team", "target_player", "line"]


# ---------------------------------------------------------------------------
# Canonical row data
# ---------------------------------------------------------------------------
# Schema for the inline lists below (one dict per question). All fields are
# required; use None for empty values.

_BELGIUM_VS_IRAN = [
    dict(qn="Q1", qt="team_more_fouls",         tt="Iran",    tp=None,        line=None, q="Will Iran commit more fouls than Belgium?",
         sub=56, crowd=58, result="Yes", if_yes=0.38, if_no=4.38, actual_rbp=0.38, swing=4.0,  notes="actual result: Iran more fouls"),
    dict(qn="Q2", qt="total_cards_2h_over",     tt=None,      tp=None,        line=1.5,  q="Will there be 2 or more total cards shown in the second half?",
         sub=50, crowd=56, result="No",  if_yes=-3.01, if_no=8.80, actual_rbp=8.80, swing=12.0, notes=""),
    dict(qn="Q3", qt="compound_btts_over_2_5",  tt=None,      tp=None,        line=2.5,  q="Will both teams score AND will the match have 3 or more total goals?",
         sub=33, crowd=38, result="No",  if_yes=-4.41, if_no=5.70, actual_rbp=5.70, swing=10.0, notes=""),
    dict(qn="Q4", qt="team_more_sot_2h",        tt="Belgium", tp=None,        line=None, q="Will Belgium have more shots on target than Iran in the second half?",
         sub=56, crowd=66, result="Yes", if_yes=-5.92, if_no=14.26, actual_rbp=-5.92, swing=20.0, notes=""),
    dict(qn="Q5", qt="total_sot_2h_over",       tt=None,      tp=None,        line=3.5,  q="Will there be 4 or more total shots on target in the second half?",
         sub=58, crowd=63, result="Yes", if_yes=-1.78, if_no=8.21, actual_rbp=-1.78, swing=10.0, notes=""),
    dict(qn="Q6", qt="team_win",                tt="Belgium", tp=None,        line=None, q="Will Belgium win the match?",
         sub=68, crowd=69, result="No",  if_yes=1.32, if_no=2.51, actual_rbp=2.51, swing=1.0,  notes=""),
    dict(qn="Q7", qt="halftime_team_lead",      tt="Belgium", tp=None,        line=None, q="Will Belgium be winning at halftime?",
         sub=43, crowd=50, result="No",  if_yes=-5.49, if_no=8.87, actual_rbp=8.87, swing=14.0, notes=""),
    dict(qn="Q8", qt="player_sot_over",         tt="Belgium", tp="Tielemans", line=0.5,  q="Will Tielemans have 1 or more shots on target?",
         sub=27, crowd=43, result="Yes", if_yes=-18.18, if_no=14.11, actual_rbp=-18.18, swing=32.0, notes=""),
    dict(qn="Q9", qt="player_goal_or_assist",   tt="Iran",    tp="Taremi",    line=None, q="Will Taremi score or assist a goal, excluding own goals?",
         sub=22, crowd=31, result="No",  if_yes=-10.94, if_no=6.59, actual_rbp=6.59, swing=18.0, notes=""),
    dict(qn="Q10",qt="total_corners_over",      tt=None,      tp=None,        line=8.5,  q="Will there be 9 or more total corners?",
         sub=52, crowd=52, result="No",  if_yes=2.72, if_no=2.28, actual_rbp=2.28, swing=0.0,  notes=""),
]

_URUGUAY_VS_CAPE_VERDE = [
    dict(qn="Q1", qt="team_offsides_over",      tt="Uruguay",    tp=None,           line=1.5, q="Will Uruguay be caught offside 2 or more times?",
         sub=48, crowd=52, result="Yes", if_yes=-2.08, if_no=7.00, actual_rbp=-2.08, swing=None, notes="opposite conditional payoff was displayed rounded"),
    dict(qn="Q2", qt="both_teams_sot_2h_1plus", tt=None,         tp=None,           line=None, q="Will both teams have at least 1 shot on target in the second half?",
         sub=68, crowd=60, result="No",  if_yes=8.00, if_no=-7.22, actual_rbp=-7.22, swing=None, notes="opposite conditional payoff was displayed rounded"),
    dict(qn="Q3", qt="team_more_cards",         tt="Cape Verde", tp=None,           line=None, q="Will Cape Verde receive more cards than Uruguay?",
         sub=54, crowd=49, result="No",  if_yes=7.00, if_no=-3.24, actual_rbp=-3.24, swing=None, notes="opposite conditional payoff was displayed rounded"),
    dict(qn="Q4", qt="penalty_awarded",         tt=None,         tp=None,           line=None, q="Will a penalty kick be awarded in the match?",
         sub=31, crowd=30, result="No",  if_yes=4.00, if_no=1.17, actual_rbp=1.17, swing=None, notes="opposite conditional payoff was displayed rounded"),
    dict(qn="Q5", qt="team_more_fouls",         tt="Uruguay",    tp=None,           line=None, q="Will Uruguay commit more fouls than Cape Verde?",
         sub=42, crowd=47, result="Yes", if_yes=-3.17, if_no=7.00, actual_rbp=-3.17, swing=None, notes="opposite conditional payoff was displayed rounded"),
    dict(qn="Q6", qt="team_sot_2h_over",        tt="Cape Verde", tp=None,           line=1.5, q="Will Cape Verde have 2 or more shots on target in the second half?",
         sub=37, crowd=35, result="Yes", if_yes=5.15, if_no=2.00, actual_rbp=5.15, swing=None, notes="opposite conditional payoff was displayed rounded"),
    dict(qn="Q7", qt="team_win",                tt="Uruguay",    tp=None,           line=None, q="Will Uruguay win the match?",
         sub=69, crowd=69, result="No",  if_yes=2.00, if_no=1.81, actual_rbp=1.81, swing=None, notes="opposite conditional payoff was displayed rounded"),
    dict(qn="Q8", qt="team_score_2h",           tt="Cape Verde", tp=None,           line=None, q="Will Cape Verde score in the second half?",
         sub=31, crowd=29, result="Yes", if_yes=4.86, if_no=0.00, actual_rbp=4.86, swing=None, notes="opposite conditional payoff was displayed rounded"),
    dict(qn="Q9", qt="player_goal_or_assist",   tt="Uruguay",    tp="Darwin Núñez", line=None, q="Will Darwin Núñez score or assist a goal, excluding own goals?",
         sub=40, crowd=38, result="No",  if_yes=6.00, if_no=2.09, actual_rbp=2.09, swing=None, notes="opposite conditional payoff was displayed rounded"),
    dict(qn="Q10",qt="team_sot_over",           tt="Uruguay",    tp=None,           line=5.5, q="Will Uruguay have 6 or more shots on target?",
         sub=45, crowd=53, result="No",  if_yes=-4.00, if_no=11.47, actual_rbp=11.47, swing=None, notes="opposite conditional payoff was displayed rounded"),
]

MATCHES = [
    dict(match="Belgium vs Iran", game_date="2026-06-21", source_batch="live_lock_belgium_2026_06_21", rows=_BELGIUM_VS_IRAN),
    dict(match="Uruguay vs Cape Verde", game_date="2026-06-21", source_batch="live_lock_uruguay_capeverde_2026_06_21", rows=_URUGUAY_VS_CAPE_VERDE),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_historical_row(match: str, game_date: str, source_batch: str, r: dict, row_id: int) -> dict:
    """Map the inline schema to the historical CSV row shape."""
    outcome_pct = 100.0 if r["result"].strip().lower() == "yes" else 0.0
    return {
        "row_id": row_id,
        "source_batch": source_batch,
        "status": "resolved",
        "game_date": game_date,
        "match_raw": match,
        "match_norm": match,
        "question_number": r["qn"],
        "question": r["q"],
        "question_type": r["qt"],
        "field_prob": r["crowd"],
        "result": r["result"],
        "outcome_pct": outcome_pct,
        "submitted": 1,
        "submitted_percent": r["sub"],
        "actual_rbp": r["actual_rbp"],
        "if_yes_rbp": r["if_yes"],
        "if_no_rbp": r["if_no"],
        "swing_display": r["swing"] if r["swing"] is not None else "",
        "notes": r["notes"],
        "target_team": r["tt"] if r["tt"] is not None else "",
        "target_player": r["tp"] if r["tp"] is not None else "",
        "line": r["line"] if r["line"] is not None else "",
    }


def _to_calibration_row(match: str, game_date: str, source_batch: str, r: dict) -> dict:
    """Build a calibration_log row marked as actual_submission.

    Engine pre-lock columns are blank — this row records what Brandon
    submitted, not what the engine would have produced. Field error stays
    blank because there is no engine field estimate to compare against.
    """
    row = {c: "" for c in CALIBRATION_COLS}
    row.update({
        "source_batch": source_batch,
        "status": "resolved",
        "game_date": game_date,
        "match_raw": match,
        "match_norm": match,
        "question_number": r["qn"],
        "question": r["q"],
        "question_type": r["qt"],
        "field_prob": r["crowd"],
        "result": r["result"],
        "outcome_pct": 100.0 if r["result"].strip().lower() == "yes" else 0.0,
        "submitted": 1,
        "submitted_percent": r["sub"],
        "actual_rbp": r["actual_rbp"],
        "if_yes_rbp": r["if_yes"],
        "if_no_rbp": r["if_no"],
        "swing_display": r["swing"] if r["swing"] is not None else "",
        "notes": "row_kind=actual_submission",
        "target_team": r["tt"] if r["tt"] is not None else "",
        "target_player": r["tp"] if r["tp"] is not None else "",
        "line": r["line"] if r["line"] is not None else "",
    })
    return row


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series([""] * len(df), dtype=object)
        else:
            # Force object dtype so mixed-type writes (e.g. line=1.5 vs "")
            # don't fail against a previously-inferred str/float dtype.
            df[c] = df[c].astype(object)
    return df


def _upsert(
    df: pd.DataFrame,
    new_rows: list[dict],
    key_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, int, int]:
    """Insert-or-replace by composite key. Returns (df, n_added, n_updated).

    Casts every column the new rows touch to object dtype up front so
    mixed-type writes (string into a previously-float column, etc.) don't
    raise. CSV serialization ignores dtype anyway.
    """
    added = 0
    updated = 0
    touched_cols = {c for r in new_rows for c in r.keys() if c != "row_id"}
    for c in touched_cols:
        if c not in df.columns:
            df[c] = pd.Series([""] * len(df), dtype=object)
        else:
            df[c] = df[c].astype(object)
    # Index existing rows by key tuple for fast lookup.
    existing_keys = {
        tuple(str(df.at[i, k]) for k in key_cols): i
        for i in df.index
    }
    for r in new_rows:
        k = tuple(str(r.get(c, "")) for c in key_cols)
        if k in existing_keys:
            idx = existing_keys[k]
            for col, val in r.items():
                if col == "row_id":
                    continue  # don't change existing row_id
                df.at[idx, col] = val
            updated += 1
        else:
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
            existing_keys[k] = df.index[-1]
            added += 1
    return df, added, updated


# ---------------------------------------------------------------------------
# Per-file workers
# ---------------------------------------------------------------------------

def update_collected(path: Path = COLLECTED_PATH) -> tuple[int, int, int]:
    df = pd.read_csv(path)
    df = _ensure_columns(df, EXTRA_HISTORICAL_COLS)
    n_before = len(df)
    next_id = int(pd.to_numeric(df["row_id"], errors="coerce").max() or 0) + 1

    rows_to_apply = []
    for m in MATCHES:
        for r in m["rows"]:
            rows_to_apply.append(
                _to_historical_row(m["match"], m["game_date"], m["source_batch"], r, next_id)
            )
            next_id += 1

    df, added, updated = _upsert(df, rows_to_apply, key_cols=("match_norm", "question_number"))
    df.to_csv(path, index=False)
    return len(df) - n_before, added, updated


def update_submitted(path: Path = SUBMITTED_PATH) -> tuple[int, int, int]:
    df = pd.read_csv(path)
    df = _ensure_columns(df, EXTRA_HISTORICAL_COLS)
    n_before = len(df)
    next_id = int(pd.to_numeric(df["row_id"], errors="coerce").max() or 0) + 1

    rows_to_apply = []
    for m in MATCHES:
        for r in m["rows"]:
            rows_to_apply.append(
                _to_historical_row(m["match"], m["game_date"], m["source_batch"], r, next_id)
            )
            next_id += 1

    df, added, updated = _upsert(df, rows_to_apply, key_cols=("match_norm", "question_number"))
    df.to_csv(path, index=False)
    return len(df) - n_before, added, updated


def update_calibration(path: Path = CALIBRATION_LOG_PATH) -> tuple[int, int, int, int]:
    df = pd.read_csv(path)
    if "row_kind" not in df.columns:
        df["row_kind"] = ""
    # Tag every pre-existing row as engine_counterfactual exactly once.
    # Rows we add below will be tagged actual_submission. Re-runs are safe
    # because the actual_submission tag never gets overwritten.
    n_tagged = 0
    mask_blank = df["row_kind"].astype(str).isin(["", "nan"])
    if mask_blank.any():
        df.loc[mask_blank, "row_kind"] = "engine_counterfactual"
        n_tagged = int(mask_blank.sum())

    n_before = len(df)
    rows_to_apply = []
    for m in MATCHES:
        for r in m["rows"]:
            cal_row = _to_calibration_row(m["match"], m["game_date"], m["source_batch"], r)
            cal_row["row_kind"] = "actual_submission"
            rows_to_apply.append(cal_row)

    df, added, updated = _upsert(
        df, rows_to_apply, key_cols=("match_norm", "question_number", "row_kind")
    )
    # Persist with the canonical column order, plus row_kind appended.
    canonical = [c for c in CALIBRATION_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in canonical]
    df = df[canonical + extra]
    df.to_csv(path, index=False)
    return len(df) - n_before, added, updated, n_tagged


def main() -> None:
    c_diff, c_add, c_upd = update_collected()
    s_diff, s_add, s_upd = update_submitted()
    k_diff, k_add, k_upd, k_tag = update_calibration()

    print("== Part 1: data add summary ==")
    for label, path, diff, added, updated, extra in [
        ("collected_data.csv",        COLLECTED_PATH,        c_diff, c_add, c_upd, ""),
        ("submitted_scoring_rows.csv", SUBMITTED_PATH,        s_diff, s_add, s_upd, ""),
        ("calibration_log.csv",        CALIBRATION_LOG_PATH,  k_diff, k_add, k_upd,
         f", tagged_engine_counterfactual={k_tag}"),
    ]:
        total = len(pd.read_csv(path))
        print(
            f"  {label}: total={total} (Δ={diff:+d}, added={added}, updated={updated}{extra})"
        )

    # Idempotency sanity check.
    print("\n== idempotency check (re-counting Belgium + Uruguay rows) ==")
    for label, path in [
        ("collected_data",  COLLECTED_PATH),
        ("submitted_rows",  SUBMITTED_PATH),
        ("calibration_log", CALIBRATION_LOG_PATH),
    ]:
        df = pd.read_csv(path)
        col = "match_norm" if "match_norm" in df.columns else "match"
        for m in ("Belgium vs Iran", "Uruguay vs Cape Verde"):
            n = (df[col] == m).sum()
            extra = ""
            if "row_kind" in df.columns:
                kinds = df[df[col] == m]["row_kind"].value_counts().to_dict()
                extra = f"  ({kinds})"
            print(f"  {label:<18} {m:<24} rows={n}{extra}")


if __name__ == "__main__":
    main()
