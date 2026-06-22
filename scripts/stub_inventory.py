"""Auto-stub candidate rows into data/question_inventory.csv.

For each match in the cached odds within the look-ahead window, append
candidate rows with include=0 and source=auto_stub. The user flips
include=1 on the questions SportsPredict actually asks.

By default stubs three question types per match:
- team_win (two rows: one per team)
- both_teams_score (one row)
- halftime_draw (one row)

Use ``--question-types`` to narrow or widen. Existing rows are never
modified. Re-running is safe.

Examples
--------
  # Stub all default candidates for the next 24h from cache (free):
  python scripts/stub_inventory.py

  # Only team_win:
  python scripts/stub_inventory.py --question-types team_win

  # Fresh odds pull first (costs credits):
  python scripts/stub_inventory.py --fetch --hours 12
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib import fetch_odds, latest_bulk_cache

INVENTORY_PATH = Path("data/question_inventory.csv")
INVENTORY_COLS = [
    "include",
    "match",
    "question_type",
    "target_team",
    "target_player",
    "line",
    "sports_predict_question",
    "source",
    "notes",
]

SUPPORTED_AUTO_STUBS = (
    "team_win",
    "both_teams_score",
    "halftime_draw",
    "match_total_over",
)


def _load_inventory() -> pd.DataFrame:
    if INVENTORY_PATH.exists():
        df = pd.read_csv(INVENTORY_PATH).fillna("")
        for col in INVENTORY_COLS:
            if col not in df.columns:
                df[col] = ""
        return df[INVENTORY_COLS]
    return pd.DataFrame(columns=INVENTORY_COLS)


def _stub_rows_for_match(g: dict, question_types: list[str]) -> list[dict]:
    match = f"{g['home_team']} vs {g['away_team']}"
    rows: list[dict] = []
    for qt in question_types:
        if qt == "team_win":
            for team in (g["home_team"], g["away_team"]):
                rows.append(
                    {
                        "include": 0,
                        "match": match,
                        "question_type": "team_win",
                        "target_team": team,
                        "target_player": "",
                        "line": "",
                        "sports_predict_question": f"Will {team} win the match?",
                    }
                )
        elif qt == "both_teams_score":
            rows.append(
                {
                    "include": 0,
                    "match": match,
                    "question_type": "both_teams_score",
                    "target_team": "",
                    "target_player": "",
                    "line": "",
                    "sports_predict_question": "Will both teams score?",
                }
            )
        elif qt == "halftime_draw":
            rows.append(
                {
                    "include": 0,
                    "match": match,
                    "question_type": "halftime_draw",
                    "target_team": "",
                    "target_player": "",
                    "line": "",
                    "sports_predict_question": "Will the match be tied at halftime?",
                }
            )
        elif qt == "match_total_over":
            rows.append(
                {
                    "include": 0,
                    "match": match,
                    "question_type": "match_total_over",
                    "target_team": "",
                    "target_player": "",
                    "line": 2.5,
                    "sports_predict_question": (
                        "Will the match have 3 or more total goals?"
                    ),
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-stub inventory candidates.")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--sport", default="soccer_fifa_world_cup")
    p.add_argument("--markets", default="h2h")
    p.add_argument("--regions", default="us,uk,eu")
    p.add_argument(
        "--question-types",
        default=",".join(SUPPORTED_AUTO_STUBS),
        help=(
            "Comma-separated question types to stub. Supported: "
            + ", ".join(SUPPORTED_AUTO_STUBS)
        ),
    )
    p.add_argument(
        "--fetch",
        action="store_true",
        help="Call the API for a fresh bulk cache (costs credits).",
    )
    args = p.parse_args()

    question_types = [
        qt.strip() for qt in args.question_types.split(",") if qt.strip()
    ]
    unknown = [qt for qt in question_types if qt not in SUPPORTED_AUTO_STUBS]
    if unknown:
        raise SystemExit(
            f"unsupported --question-types: {unknown}. "
            f"choose from {list(SUPPORTED_AUTO_STUBS)}"
        )

    if args.fetch:
        cache = fetch_odds(args.sport, markets=args.markets, regions=args.regions)
    else:
        cache = latest_bulk_cache(args.sport, args.regions)
        print(f"using cache: {cache}")

    raw = json.loads(Path(cache).read_text())
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=args.hours)
    in_window = [
        g
        for g in raw
        if now
        <= datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        <= cutoff
    ]
    print(f"games kicking off within next {args.hours}h: {len(in_window)}")

    inventory = _load_inventory()
    # Dedupe key includes target_team (so the two team_win rows per match
    # are distinct while btts/halftime singletons collapse) and line (so
    # match_total_over at 2.5 vs 3.0 stay distinct but identical-line
    # re-runs collapse to one row).
    def _norm_line(v) -> str:
        s = str(v).strip()
        if s in ("", "nan", "None"):
            return ""
        try:
            return f"{float(s)}"
        except ValueError:
            return s

    existing_keys = set(
        zip(
            inventory["match"].astype(str),
            inventory["question_type"].astype(str),
            inventory["target_team"].astype(str),
            inventory["line"].map(_norm_line),
        )
    )

    note = (
        "candidate; set include=1 if this exact question appears on SportsPredict"
    )
    new_rows = []
    for g in in_window:
        for candidate in _stub_rows_for_match(g, question_types):
            key = (
                candidate["match"],
                candidate["question_type"],
                candidate["target_team"],
                _norm_line(candidate["line"]),
            )
            if key in existing_keys:
                continue
            candidate["source"] = "auto_stub"
            candidate["notes"] = note
            new_rows.append(candidate)
            existing_keys.add(key)

    if not new_rows:
        print("no new candidate rows to add (inventory already covers these matches)")
        return

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated = pd.concat([inventory, pd.DataFrame(new_rows)], ignore_index=True)
    updated[INVENTORY_COLS].to_csv(INVENTORY_PATH, index=False)
    print(f"appended {len(new_rows)} candidate row(s) -> {INVENTORY_PATH}")
    print("Edit that file: set include=1 on the questions SportsPredict actually asks.")


if __name__ == "__main__":
    main()
