"""Build the player-prop feature dataset for future truth-model training.

Output: ``data/models/player_prop_feature_rows.csv`` — one row per
player-prop question, with PRE-LOCK feature columns followed by clearly
separated EVALUATION-ONLY columns (outcome, locked crowd %, realized RBP).

Discipline notes
----------------
- Features are strictly pre-lock. Outcome / locked-crowd / realized-RBP
  columns are appended *after* the feature block and are documented as
  evaluation-only. They must never be fed into a model's X.
- Nothing here invents probabilities. Lineup status/role come from lineup
  files when present (else ``unknown``); match-context market features are
  filled only when a consensus cache is supplied (else left blank — we do
  not fabricate implied goals etc.).
- ``p_field_est`` is the engine's heuristic crowd estimate (semantic anchor /
  historical blend). It is a legitimate pre-lock baseline feature, not truth.

Sources (default): the historical SportsPredict collected data + submitted
scoring rows. Player-prop rows there carry real outcomes, so this produces a
usable labelled dataset today. Upcoming candidate player props from the
question inventory can be appended with ``--include-inventory`` (no eval
columns, since they have not resolved).

Usage::

    python scripts/build_player_prop_features.py
    python scripts/build_player_prop_features.py --include-inventory
    python scripts/build_player_prop_features.py --out data/models/foo.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.decision_engine import (
    MatchContext,
    estimate_p_field,
    load_priors,
    _prior_for,
)
from odds_lib.lineups import load_lineup, PlayerContext
from odds_lib.player_features import (
    FEATURE_COLUMNS,
    EVAL_ONLY_COLUMNS,
    PLAYER_PROP_QUESTION_TYPES,
    build_player_prop_features,
)


DEFAULT_OUT = Path("data/models/player_prop_feature_rows.csv")
HISTORICAL_SOURCES = [
    Path("data/historical/sportspredict_collected_data.csv"),
    Path("data/historical/sportspredict_submitted_scoring_rows.csv"),
]
INVENTORY_PATH = Path("data/question_inventory.csv")


def _f(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _build_one(
    row: dict,
    priors: dict,
    *,
    with_eval: bool,
) -> dict:
    """Build a single feature dataset row (features + optional eval cols)."""
    qt = str(row.get("question_type") or "").strip().lower()
    match = row.get("match") or row.get("match_norm") or row.get("match_raw") or ""
    game_date = str(row.get("game_date") or "") or None

    question_row = {
        "question_type": qt,
        "target_player": row.get("target_player"),
        "target_team": row.get("target_team"),
        "line": row.get("line"),
        "match": match,
        "sports_predict_question": row.get("question")
        or row.get("sports_predict_question"),
    }

    # Lineup features: only if a lineup file exists for this match (past
    # matches generally have none -> unknown, which is honest).
    lineup = load_lineup(str(match), game_date=game_date) if match else None
    if lineup is not None:
        player_ctx = lineup.player(question_row["target_player"])
    else:
        player_ctx = PlayerContext()

    # No cached consensus is loaded here (historical matches lack odds caches);
    # match-context market features stay blank rather than fabricated. The
    # field estimate uses an empty MatchContext + question-type prior, which
    # is the engine's anchor-based pre-lock baseline.
    match_ctx = MatchContext()
    prior = _prior_for(priors, qt)
    p_field, p_field_source, _ = estimate_p_field(qt, question_row, match_ctx, prior)

    feats = build_player_prop_features(
        question_row=question_row,
        player_ctx=player_ctx,
        match_ctx=match_ctx,
        p_field_est=p_field,
        p_field_source=p_field_source if p_field is not None else "no_estimate",
        has_direct_market=False,
        direct_market_prob=None,
        game_date=game_date,
    )
    out = feats.to_dict()

    if with_eval:
        out["submitted_percent"] = row.get("submitted_percent", "")
        out["crowd_percent"] = row.get("field_prob", "")
        out["result"] = row.get("result", "")
        out["actual_rbp"] = row.get("actual_rbp", "")
    else:
        for c in EVAL_ONLY_COLUMNS:
            out[c] = ""
    return out


def _collect_historical() -> list[dict]:
    frames = []
    for p in HISTORICAL_SOURCES:
        if p.exists():
            df = pd.read_csv(p)
            df["__src"] = p.name
            frames.append(df)
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    df = df[df["question_type"].astype(str).str.lower().isin(PLAYER_PROP_QUESTION_TYPES)]
    # Dedup by row_id when present (submitted rows overlap collected rows).
    if "row_id" in df.columns:
        df = df.drop_duplicates(subset=["row_id"], keep="first")
    return df.to_dict("records")


def _collect_inventory() -> list[dict]:
    if not INVENTORY_PATH.exists():
        return []
    df = pd.read_csv(INVENTORY_PATH)
    df = df[df["question_type"].astype(str).str.lower().isin(PLAYER_PROP_QUESTION_TYPES)]
    return df.to_dict("records")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--include-inventory",
        action="store_true",
        help="append current candidate player props from the question inventory "
        "(no eval columns; they have not resolved).",
    )
    args = ap.parse_args()

    priors = load_priors()
    rows: list[dict] = []

    for r in _collect_historical():
        rows.append(_build_one(r, priors, with_eval=True))

    if args.include_inventory:
        for r in _collect_inventory():
            rows.append(_build_one(r, priors, with_eval=False))

    columns = FEATURE_COLUMNS + EVAL_ONLY_COLUMNS
    df = pd.DataFrame(rows, columns=columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n_eval = int((df["result"].astype(str).str.strip() != "").sum()) if len(df) else 0
    print(f"wrote {len(df)} player-prop feature row(s) -> {args.out}")
    print(f"  with resolved outcome (eval-ready): {n_eval}")
    if len(df):
        by_qt = df["question_type"].value_counts()
        print("  by question_type:")
        for qt, n in by_qt.items():
            print(f"    {qt:<28} {n}")
        lc = int(df["has_lineup_context"].astype(str).isin(["True", "true"]).sum())
        print(f"  rows with lineup context: {lc}")


if __name__ == "__main__":
    main()
