"""Print the raw market consensus for matches kicking off in the next N hours.

This is the "what do the books say right now" view. It does NOT touch the
SportsPredict question inventory and does NOT write to predictions_log.csv —
that's submit_sheet.py's job. Use this script when you want a quick read on
the markets without going through the inventory/submission workflow.

Combines bulk-endpoint markets (h2h, totals) with per-event markets (btts,
h2h_h1, ...). Defaults to the latest cached files. Pass ``--fetch`` to spend
API credits on a fresh pull (bulk + one per-event call per upcoming event).

Examples
--------
  # Next 24h, from caches (free):
  python scripts/market_sheet.py --forecast-run-id 2026-06-21_pregame

  # Next 6h, fresh API calls:
  python scripts/market_sheet.py --forecast-run-id 2026-06-21_pregame \\
      --hours 6 --fetch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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
)


def filter_by_kickoff_window(
    raw: list[dict], hours: float, now: datetime | None = None
) -> list[dict]:
    """Keep games whose ``commence_time`` is in [now, now + hours] UTC."""
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)
    kept = []
    for game in raw:
        ct = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
        if now <= ct <= cutoff:
            kept.append(game)
    return kept


def _resolve_per_event_caches(
    sport: str,
    regions: str,
    in_window: list[dict],
    per_event_market_keys: tuple[str, ...],
    fetch: bool,
) -> dict[str, Path]:
    if not per_event_market_keys or not in_window:
        return {}

    per_event_markets = ",".join(per_event_market_keys)
    out: dict[str, Path] = {}

    if fetch:
        cost = len(in_window) * len(per_event_market_keys) * len(regions.split(","))
        print(
            f"\nfetching per-event markets [{per_event_markets}] for "
            f"{len(in_window)} event(s) — est. {cost} credits..."
        )
        for game in in_window:
            event_id = game["id"]
            label = f"{game['home_team']} vs {game['away_team']}"
            try:
                out[event_id] = fetch_event_odds(
                    sport, event_id, markets=per_event_markets, regions=regions
                )
            except requests.HTTPError as e:
                print(f"  WARN: per-event fetch failed for {label} ({event_id}): {e}")
        return out

    for game in in_window:
        cache = latest_event_cache(sport, game["id"], regions)
        if cache is not None:
            out[game["id"]] = cache
    print(
        f"  per-event caches available: {len(out)}/{len(in_window)} "
        f"(markets requested: {per_event_markets})"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print market consensus for matches in the next N hours."
    )
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
    p.add_argument("--regions", default="us,uk,eu")
    p.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Look-ahead window in hours (default 24).",
    )
    p.add_argument(
        "--fetch",
        action="store_true",
        help="Call the API (costs credits). Otherwise use latest caches.",
    )
    p.add_argument(
        "--local-tz",
        default="America/New_York",
        help="IANA tz name for the kickoff_time_local column.",
    )
    args = p.parse_args()

    if args.fetch:
        bulk_cache = fetch_odds(args.sport, markets=args.markets, regions=args.regions)
    else:
        bulk_cache = latest_bulk_cache(args.sport, args.regions)
        print(f"using bulk cache: {bulk_cache}")

    raw = json.loads(bulk_cache.read_text())
    print(f"games in cache: {len(raw)}")

    kept = filter_by_kickoff_window(raw, args.hours)
    print(f"games kicking off within next {args.hours}h: {len(kept)}")
    if not kept:
        print("nothing to show.")
        return

    bulk_market_keys = tuple(m.strip() for m in args.markets.split(",") if m.strip())
    per_event_market_keys = tuple(
        m.strip() for m in args.per_event_markets.split(",") if m.strip()
    )

    per_event_caches = _resolve_per_event_caches(
        args.sport, args.regions, kept, per_event_market_keys, args.fetch
    )

    bulk_markets = json_to_markets(
        kept,
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
    all_markets = (
        pd.concat([bulk_markets, *per_event_frames], ignore_index=True)
        if per_event_frames
        else bulk_markets
    )

    df, consensus, submit_sums = process_match(all_markets, on_incomplete="drop")

    kickoff_by_match = {
        f"{g['home_team']} vs {g['away_team']}": g["commence_time"] for g in kept
    }
    consensus["kickoff_time_utc"] = pd.to_datetime(
        consensus["match"].map(kickoff_by_match), utc=True
    )
    local_tz = ZoneInfo(args.local_tz)
    consensus["kickoff_time_local"] = consensus["kickoff_time_utc"].dt.tz_convert(
        local_tz
    )

    consensus = consensus.sort_values(
        ["kickoff_time_utc", "match", "market_key", "line", "submit_prob"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    display_cols = [
        "kickoff_time_local",
        "match",
        "market_key",
        "line",
        "outcome",
        "submit_prob",
        "min_prob",
        "max_prob",
        "num_books",
    ]
    print()
    print("=== market sheet ===")
    print(consensus[display_cols].to_string(index=False))

    print(
        f"\nmicro-markets processed (each sums to 1.0): {len(submit_sums)}"
    )

    print(
        "\n(view-only — to log SportsPredict submissions, "
        "use scripts/submit_sheet.py after curating question_inventory.csv)"
    )


if __name__ == "__main__":
    main()
