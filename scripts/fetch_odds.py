"""CLI for The Odds API.

Examples
--------
  python scripts/fetch_odds.py list-sports
  python scripts/fetch_odds.py odds --sport soccer_fifa_world_cup --markets h2h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.odds_api import fetch_odds, list_sports


def main() -> None:
    p = argparse.ArgumentParser(description="The Odds API thin CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-sports", help="List active sports (free; no credits used).")

    f = sub.add_parser("odds", help="Fetch and cache odds for one sport+market.")
    f.add_argument("--sport", required=True)
    f.add_argument("--markets", default="h2h")
    f.add_argument("--regions", default="us,uk,eu")

    args = p.parse_args()

    if args.cmd == "list-sports":
        for s in list_sports():
            key = s["key"]
            title = s.get("title", "")
            group = s.get("group", "")
            if "soccer" in key.lower() or "soccer" in group.lower():
                print(f"  {key:45s}  {title}")
    elif args.cmd == "odds":
        fetch_odds(args.sport, markets=args.markets, regions=args.regions)


if __name__ == "__main__":
    main()
