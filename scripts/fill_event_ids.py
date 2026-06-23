"""Fill real event_ids into a questions file from the FREE /events endpoint.

New-day setup: a freshly built questions file has event_id blank. This matches
each "Home vs Away" to an upcoming event by TEAM SET (order/accent tolerant) and
writes the event_id back, so the optimizer/refresh can find each match's odds.
No odds are fetched here — /events is free.

    python scripts/fill_event_ids.py --questions data/submission_sheets/2026-06-24_questions.csv
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.odds_api import list_events

SPORT = "soccer_fifa_world_cup"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def _teams(match: str) -> set[str]:
    return {_norm(p) for p in match.split(" vs ")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--write", action="store_true",
                    help="write the filled event_ids back (default: dry-run print only)")
    args = ap.parse_args()

    q = pd.read_csv(args.questions, dtype=str).fillna("")
    events = list_events(SPORT)
    # index events by normalized team set
    by_teams = {}
    for e in events:
        key = frozenset({_norm(e.get("home_team", "")), _norm(e.get("away_team", ""))})
        by_teams[key] = e

    print(f"{len(events)} upcoming events from /events (free). Matching {q['match'].nunique()} slate matches:\n")
    mapping = {}
    for mt in q["match"].unique():
        want = frozenset(_teams(mt))
        hit = by_teams.get(want)
        if hit is None:
            # tolerant fallback: both teams' tokens present in some event
            for key, e in by_teams.items():
                if all(any(tok in k for k in key) for tok in [t for t in want]):
                    hit = e
                    break
        if hit is None:
            print(f"  [MISS] {mt:28s} -> no upcoming event found (check team spelling)")
            mapping[mt] = ""
        else:
            mapping[mt] = hit["id"]
            print(f"  [ok]   {mt:28s} -> {hit['id']}  ({hit.get('home_team')} v {hit.get('away_team')}, {hit.get('commence_time')})")

    q["event_id"] = q["match"].map(mapping)
    n_filled = (q["event_id"] != "").sum()
    if args.write:
        q.to_csv(args.questions, index=False)
        print(f"\nWROTE event_ids for {n_filled}/{len(q)} rows -> {args.questions}")
    else:
        print(f"\nDRY-RUN: would fill {n_filled}/{len(q)} rows. Re-run with --write to save.")


if __name__ == "__main__":
    main()
