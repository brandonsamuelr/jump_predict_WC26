"""Seed a player-prop DATA COLLECTION template for an upcoming slate.

This produces a CSV you fill in by hand / from data sources during research.
It is the raw input that will later feed ``build_player_prop_features.py`` and,
eventually, a real data-derived ``p_truth`` model.

Discipline (enforced by leaving cells BLANK, never guessed):
  - Do NOT invent probabilities, expected minutes, or player rates. If a
    value is unknown, leave it blank. A blank means "not collected yet",
    which is honest; a fabricated number is not.
  - Every value that is entered by hand should carry a ``*_source`` and/or
    ``entry_reason`` so we can audit provenance later.
  - Outcome / crowd / RBP columns are EVALUATION-ONLY (post-lock). They are
    grouped at the end and must never be used as model features.

What the seeder fills automatically (these are KNOWN facts, not guesses):
  - identity (match, teams, player, question_type) from lineup files
  - lineup_status / lineup_role / expected_minutes / lineup_source — copied
    verbatim from the lineup JSON (expected_minutes stays blank if the file
    has null, which is the norm)

Everything else is left blank for manual/data collection.

Usage::

    # seed from all lineup files currently in data/lineups/
    python scripts/build_player_prop_collection_template.py

    # seed for one match, custom output path
    python scripts/build_player_prop_collection_template.py \
        --match "New Zealand vs Egypt" --out data/models/nz_egy_collection.csv

    # header-only template (no seed rows)
    python scripts/build_player_prop_collection_template.py --empty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.lineups import LINEUP_DIR, load_lineup, MatchLineup


DEFAULT_OUT = Path("data/models/player_prop_collection_template.csv")

# Player-prop question types we are collecting for first (see
# docs/player_prop_truth_model_plan.md). One row is seeded per (player, qt).
SEED_QUESTION_TYPES = ["player_sot_over", "player_goal_or_assist"]


# ---------------------------------------------------------------------------
# Column schema. Grouped; every column documented in
# docs/player_prop_data_collection_guide.md.
# ---------------------------------------------------------------------------

IDENTITY_COLS = [
    "collection_id",        # stable key, e.g. {game_date}_{slug}_{player}_{qt}
    "game_date",
    "match",
    "competition",          # e.g. World Cup group stage (manual)
    "target_team",
    "opponent_team",
    "target_player",
    "question_type",
    "question",             # exact SportsPredict wording, if known
    "line",
]

# Lineup / availability — auto-filled from lineup JSON where present.
LINEUP_COLS = [
    "lineup_status",        # starter | bench_* | out_of_squad | unknown
    "lineup_role",          # central_attacker | wide_attacker | ... | unknown
    "expected_minutes",     # BLANK unless genuinely known — never invented
    "lineup_source",        # where the XI/status came from
    "lineup_captured_at",   # timestamp the lineup was observed
]

# Recent form — manual / data-derived. Blank if not collected.
RECENT_FORM_COLS = [
    "recent_starts_last5",
    "recent_minutes_last5",
    "recent_form_source",
    "recent_form_reason",
]

# Player rates (per 90 unless noted) — data-derived. Blank if not collected.
PLAYER_RATE_COLS = [
    "shots_per90",
    "sot_per90",
    "goals_per90",
    "assists_per90",
    "xg_per90",
    "xa_per90",
    "is_penalty_taker",     # true/false/blank
    "setpiece_role",        # e.g. corners, free_kicks, none (manual)
    "rates_sample_matches", # how many matches the rates are computed over
    "rates_source",
]

# Match / market context — market-derived. Blank if not collected.
MATCH_CONTEXT_COLS = [
    "target_team_win_prob",
    "team_implied_goals",   # derive from totals+supremacy; do NOT guess
    "match_total_line",
    "match_total_over_2_5_prob",
    "btts_prob",
    "opponent_strength",    # rating/notes (manual or data)
    "market_context_source",
]

# Direct player-prop market (best p_truth when present). Blank if none.
DIRECT_MARKET_COLS = [
    "has_direct_prop_market",   # true/false/blank
    "direct_prop_market_prob",
    "prop_market_source",
]

# Pre-lock field estimate (crowd model). Filled from the engine or a manual
# field estimate before lock. Used by scripts/score_player_prop_edges.py to
# size a recommended submission. Distinct from crowd_percent (post-lock).
FIELD_ESTIMATE_COLS = [
    "p_field_est",
    "p_field_source",
]

# Evaluation-only (post-lock). NEVER features.
EVAL_ONLY_COLS = [
    "crowd_percent",
    "submitted_percent",
    "result",
    "actual_rbp",
    "if_yes_rbp",
    "if_no_rbp",
]

# Free-text provenance for any manual entry.
META_COLS = [
    "entered_by",
    "entry_reason",
    "notes",
]

COLLECTION_COLUMNS = (
    IDENTITY_COLS
    + LINEUP_COLS
    + RECENT_FORM_COLS
    + PLAYER_RATE_COLS
    + MATCH_CONTEXT_COLS
    + DIRECT_MARKET_COLS
    + FIELD_ESTIMATE_COLS
    + EVAL_ONLY_COLS
    + META_COLS
)


def _slugify(match_name: str) -> str:
    return (
        match_name.lower()
        .replace(" vs ", "_vs_")
        .replace(" ", "_")
    )


def _seed_rows_for_lineup(lineup: MatchLineup) -> list[dict]:
    """One blank row per (player, question_type), with KNOWN fields filled."""
    game_date = (lineup.kickoff_utc or "")[:10]
    teams = [t.strip() for t in lineup.match.split(" vs ")] if lineup.match else []
    rows: list[dict] = []
    for player_name, ctx in lineup.players.items():
        opponent = ""
        if ctx.team and len(teams) == 2:
            opponent = teams[1] if ctx.team.strip() == teams[0] else teams[0]
        for qt in SEED_QUESTION_TYPES:
            row = {c: "" for c in COLLECTION_COLUMNS}
            row.update({
                "collection_id": f"{game_date}_{_slugify(lineup.match)}_"
                                 f"{_slugify(player_name)}_{qt}",
                "game_date": game_date,
                "match": lineup.match,
                "target_team": ctx.team or "",
                "opponent_team": opponent,
                "target_player": player_name,
                "question_type": qt,
                # KNOWN lineup facts copied verbatim (not invented):
                "lineup_status": ctx.status,
                "lineup_role": ctx.role,
                "expected_minutes": (
                    "" if ctx.expected_minutes is None else ctx.expected_minutes
                ),
                "lineup_source": ctx.source or lineup.source or "",
                "lineup_captured_at": lineup.captured_at or "",
            })
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--match",
        help="seed only this match (must have a lineup file); default = all "
        "lineup files in data/lineups/",
    )
    ap.add_argument(
        "--empty",
        action="store_true",
        help="write a header-only template with no seed rows",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    if not args.empty:
        if args.match:
            lu = load_lineup(args.match)
            if lu is None:
                print(f"no lineup file found for {args.match!r}; writing header only.")
            else:
                rows.extend(_seed_rows_for_lineup(lu))
        else:
            files = sorted(LINEUP_DIR.glob("*.json")) if LINEUP_DIR.exists() else []
            for p in files:
                try:
                    match_name = json.loads(p.read_text()).get("match", "")
                except Exception:
                    match_name = ""
                lu = load_lineup(match_name) if match_name else None
                if lu is not None:
                    rows.extend(_seed_rows_for_lineup(lu))

    df = pd.DataFrame(rows, columns=COLLECTION_COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to clobber an existing, partially-collected template silently.
    if args.out.exists():
        print(
            f"WARNING: {args.out} already exists and will be OVERWRITTEN with a "
            f"fresh template. Move it aside first if it holds collected data."
        )
    df.to_csv(args.out, index=False)

    print(f"wrote collection template -> {args.out}")
    print(f"  columns: {len(COLLECTION_COLUMNS)}")
    print(f"  seed rows: {len(df)}")
    if len(df):
        print(f"  players seeded: {sorted(df['target_player'].unique())}")
    print(
        "  reminder: blank = not collected. Do NOT fill probabilities, "
        "expected minutes, or rates with guesses; add a *_source/entry_reason "
        "for every manual value."
    )


if __name__ == "__main__":
    main()
