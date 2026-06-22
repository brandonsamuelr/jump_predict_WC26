"""Daily odds run: fetch (or reuse cached) odds, build consensus, log, print.

Logs every (book, outcome) quote from BOTH the bulk endpoint (h2h, totals)
and the per-event endpoint (btts, h2h_h1, ...) into data/odds_log.csv. This
is the full per-book trace used for later analysis. For the submission
workflow itself, use scripts/submit_sheet.py.

Examples
--------
  # Fresh fetch + log (bulk + per-event):
  python scripts/daily_run.py --forecast-run-id 2026-06-21_morning

  # Re-use the latest cached JSONs (free):
  python scripts/daily_run.py --forecast-run-id 2026-06-21_morning --from-cache

  # Filter to a substring of the match name (repeatable):
  python scripts/daily_run.py --forecast-run-id 2026-06-21_morning --match Spain

  # Print only, do not log to data/odds_log.csv:
  python scripts/daily_run.py --forecast-run-id 2026-06-21_morning --no-log
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from odds_lib import (
    fetch_event_odds,
    fetch_odds,
    json_to_markets,
    latest_bulk_cache,
    latest_event_cache,
    process_match,
    upsert_csv,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Daily odds run.")
    p.add_argument("--forecast-run-id", required=True)
    p.add_argument("--sport", default="soccer_fifa_world_cup")
    p.add_argument(
        "--markets",
        default="h2h,totals",
        help="Comma-separated bulk-endpoint market keys.",
    )
    p.add_argument(
        "--per-event-markets",
        default="btts,h2h_h1",
        help="Comma-separated per-event market keys. Empty string to disable.",
    )
    p.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Per-event look-ahead window in hours (default 24).",
    )
    p.add_argument("--regions", default="us,uk,eu")
    p.add_argument(
        "--from-cache",
        action="store_true",
        help="Reuse the latest cached JSONs instead of calling the API.",
    )
    p.add_argument(
        "--match",
        action="append",
        default=[],
        help="Substring filter on match name; repeatable.",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Skip upsert into data/odds_log.csv.",
    )
    args = p.parse_args()

    # ---- bulk ----
    if args.from_cache:
        bulk_cache = latest_bulk_cache(args.sport, args.regions)
        print(f"using bulk cache: {bulk_cache}")
    else:
        bulk_cache = fetch_odds(args.sport, markets=args.markets, regions=args.regions)

    raw = json.loads(Path(bulk_cache).read_text())
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=args.hours)
    in_window = [
        g
        for g in raw
        if now
        <= datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        <= cutoff
    ]

    bulk_market_keys = tuple(m.strip() for m in args.markets.split(",") if m.strip())
    per_event_market_keys = tuple(
        m.strip() for m in args.per_event_markets.split(",") if m.strip()
    )

    # ---- per-event ----
    per_event_caches: dict[str, Path] = {}
    if per_event_market_keys and in_window:
        per_event_markets = ",".join(per_event_market_keys)
        if args.from_cache:
            for game in in_window:
                cache = latest_event_cache(args.sport, game["id"], args.regions)
                if cache is not None:
                    per_event_caches[game["id"]] = cache
            print(
                f"  per-event caches available: {len(per_event_caches)}/{len(in_window)}"
            )
        else:
            cost = (
                len(in_window)
                * len(per_event_market_keys)
                * len(args.regions.split(","))
            )
            print(
                f"\nfetching per-event markets [{per_event_markets}] for "
                f"{len(in_window)} event(s) — est. {cost} credits..."
            )
            for game in in_window:
                event_id = game["id"]
                label = f"{game['home_team']} vs {game['away_team']}"
                try:
                    per_event_caches[event_id] = fetch_event_odds(
                        args.sport,
                        event_id,
                        markets=per_event_markets,
                        regions=args.regions,
                    )
                except requests.HTTPError as e:
                    print(f"  WARN: per-event fetch failed for {label} ({event_id}): {e}")

    # ---- parse ----
    bulk_markets = json_to_markets(
        bulk_cache,
        forecast_run_id=args.forecast_run_id,
        market_keys=bulk_market_keys,
    )
    per_event_frames = [
        json_to_markets(
            cache,
            forecast_run_id=args.forecast_run_id,
            market_keys=per_event_market_keys,
        )
        for cache in per_event_caches.values()
    ]
    markets = (
        pd.concat([bulk_markets, *per_event_frames], ignore_index=True)
        if per_event_frames
        else bulk_markets
    )

    if args.match:
        needles = [m.lower() for m in args.match]
        mask = markets["match"].str.lower().apply(
            lambda s: any(n in s for n in needles)
        )
        markets = markets[mask].copy()

    if markets.empty:
        print("no matches after filtering — exiting")
        return

    df, consensus, submit_sums = process_match(markets, on_incomplete="drop")

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    for match, sub in consensus.groupby("match", sort=False):
        print()
        print(f"=== {match} ===")
        cols = [
            "market_key",
            "line",
            "outcome",
            "submit_prob",
            "min_prob",
            "max_prob",
            "num_books",
        ]
        print(sub[cols].to_string(index=False))

    print(f"\nmicro-markets processed (each sums to 1.0): {len(submit_sums)}")

    if args.no_log:
        print("\n--no-log set; skipped CSV write")
        return

    to_save = markets.copy()
    to_save["entered_at"] = pd.Timestamp.now(tz="UTC")
    upsert_csv(
        to_save,
        "data/odds_log.csv",
        key_cols=[
            "forecast_run_id",
            "event_id",
            "book",
            "market_key",
            "line",
            "outcome",
        ],
    )
    print(f"\nupserted {len(to_save)} rows into data/odds_log.csv")


if __name__ == "__main__":
    main()
