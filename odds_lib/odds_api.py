"""The Odds API client + multi-market JSON-to-markets parser.

Quota model: 1 credit per region per market per /odds call. The /sports
endpoint is free. Cache every /odds response so the parser can be iterated
on without spending credits.

Two endpoints are supported:

- bulk ``/sports/{sport}/odds`` — one call returns all upcoming events for a
  small set of markets (h2h, totals). Cache filename does not contain
  ``event-`` so it can be distinguished from per-event caches.
- per-event ``/sports/{sport}/events/{event_id}/odds`` — required for
  markets the bulk endpoint does not surface for soccer (btts, h2h_h1, team
  totals, etc.). Cache filename includes ``event-{event_id}``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

API_BASE = "https://api.the-odds-api.com/v4"
RAW_DIR = Path("data/raw")


def _safe_markets(markets: str) -> str:
    """Filename-safe market segment. Long market lists (now incl. corners/cards/2H)
    would blow past the OS 255-char filename limit, so they collapse to a short
    stable hash. latest_*_cache globs match this segment with ``*`` regardless."""
    s = markets.replace(",", "-")
    if len(s) > 80:
        s = "mkts" + hashlib.md5(markets.encode()).hexdigest()[:12]
    return s


def _persist(payload: str, canonical: Path, fallback: Path) -> Path:
    """Persist a fetched response so a PAID (200) fetch can NEVER be lost.

    Normal path: write the canonical cache file. If that write fails for ANY
    reason (filename length, permissions, disk, ...), write a SHORT deterministic
    fallback (event_id + timestamp only — cannot fail the long-name way) and warn
    to stderr; do NOT re-raise — a successful fetch must not be killed by a write
    problem. Returns the path actually written so the caller can read it back.
    Only re-raises if even the fallback write fails (data is genuinely unwritable).
    """
    try:
        canonical.write_text(payload)
        return canonical
    except Exception as e:
        try:
            fallback.write_text(payload)
        except Exception as e2:
            raise RuntimeError(
                f"paid fetch could not be persisted: canonical failed ({e}) AND "
                f"fallback failed ({e2})") from e2
        print(f"WARNING: canonical cache write failed ({e}); persisted FALLBACK "
              f"{fallback.name} instead — paid fetch preserved.", file=sys.stderr)
        return fallback


def _api_key() -> str:
    load_dotenv()
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set; populate .env")
    return key


def list_sports(only_active: bool = True) -> list[dict]:
    """Return The Odds API sports catalog. Free — does not count against quota."""
    params: dict[str, str] = {"apiKey": _api_key()}
    if not only_active:
        params["all"] = "true"
    resp = requests.get(f"{API_BASE}/sports", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_events(sport: str) -> list[dict]:
    """Free /events endpoint: upcoming events (id, commence_time, home/away).

    Does NOT count against the odds quota — use it to discover event_ids
    without spending credits (e.g. the new-day questions-file setup).
    """
    resp = requests.get(
        f"{API_BASE}/sports/{sport}/events",
        params={"apiKey": _api_key(), "dateFormat": "iso"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_odds(
    sport: str,
    markets: str = "h2h",
    regions: str = "us,uk,eu",
    odds_format: str = "american",
    raw_dir: Path = RAW_DIR,
) -> Path:
    """Make one /odds call, cache the JSON, return the cache path.

    Costs ``len(regions.split(',')) * len(markets.split(','))`` credits.
    """
    resp = requests.get(
        f"{API_BASE}/sports/{sport}/odds",
        params={
            "apiKey": _api_key(),
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
        },
        timeout=30,
    )
    resp.raise_for_status()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_markets = _safe_markets(markets)
    safe_regions = regions.replace(",", "-")
    canonical = raw_dir / f"{sport}__{safe_markets}__{safe_regions}__{ts}.json"
    fallback = raw_dir / f"fallback__{safe_regions}__{ts}.json"
    out = _persist(json.dumps(resp.json(), indent=2), canonical, fallback)

    print(f"wrote {out}")
    print(
        f"  credits used: {resp.headers.get('x-requests-used', '?')}, "
        f"remaining: {resp.headers.get('x-requests-remaining', '?')}, "
        f"this call: {resp.headers.get('x-requests-last', '?')}"
    )
    return out


def fetch_event_odds(
    sport: str,
    event_id: str,
    markets: str = "btts,h2h_h1",
    regions: str = "us,uk,eu",
    odds_format: str = "american",
    raw_dir: Path = RAW_DIR,
) -> Path:
    """Make one per-event /odds call, cache the JSON, return the cache path.

    Costs ``len(regions.split(',')) * len(markets.split(','))`` credits per
    event. Use ``fetch_odds`` for h2h/totals (bulk) and this for markets the
    bulk endpoint does not surface for soccer (btts, h2h_h1, team totals,
    etc.).
    """
    resp = requests.get(
        f"{API_BASE}/sports/{sport}/events/{event_id}/odds",
        params={
            "apiKey": _api_key(),
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
        },
        timeout=30,
    )
    resp.raise_for_status()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_markets = _safe_markets(markets)
    safe_regions = regions.replace(",", "-")
    canonical = (
        raw_dir
        / f"{sport}__event-{event_id}__{safe_markets}__{safe_regions}__{ts}.json"
    )
    # fallback: event_id + timestamp ONLY -> guaranteed short, cannot fail the
    # long-name way that crashed the live lock.
    fallback = raw_dir / f"fallback__event-{event_id}__{ts}.json"
    out = _persist(json.dumps(resp.json(), indent=2), canonical, fallback)

    print(
        f"  wrote {out.name}  "
        f"credits used: {resp.headers.get('x-requests-used', '?')}, "
        f"remaining: {resp.headers.get('x-requests-remaining', '?')}, "
        f"this call: {resp.headers.get('x-requests-last', '?')}"
    )
    return out


def _cache_timestamp(p: Path) -> str:
    """Extract the trailing ``YYYYMMDDTHHMMSSZ`` from a cache filename.

    The cache filename format is
    ``{sport}__...__{regions}__{ts}.json``; the timestamp is always the
    last ``__``-segment before ``.json``. Sorting by full path is unsafe
    because variable-length market segments (e.g. ``h2h`` vs
    ``h2h-totals``) flip the lexicographic order.
    """
    return p.stem.rsplit("__", 1)[-1]


def latest_bulk_cache(sport: str, regions: str, raw_dir: Path = RAW_DIR) -> Path:
    """Return the newest bulk-endpoint cache for this sport+regions.

    Per-event caches (which contain ``__event-`` in the filename) are
    explicitly excluded so they cannot be confused with bulk caches.
    Ordering is by the filename timestamp, not by full path lexicographic
    order.
    """
    safe_regions = regions.replace(",", "-")
    pattern = f"{sport}__*__{safe_regions}__*.json"
    candidates = [
        p for p in raw_dir.glob(pattern) if "__event-" not in p.name
    ]
    # also accept a bulk FALLBACK cache (written when the canonical write failed)
    candidates += list(raw_dir.glob(f"fallback__{safe_regions}__*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no bulk cache matches {pattern} (excluding per-event); "
            "rerun with --fetch"
        )
    candidates.sort(key=_cache_timestamp)
    return candidates[-1]


def latest_event_cache(
    sport: str, event_id: str, regions: str, raw_dir: Path = RAW_DIR
) -> Path | None:
    """Return the newest per-event cache for this event, or None if absent.

    Looks across BOTH the canonical pattern and the short fallback pattern
    (written when a canonical write failed), newest-wins by filename timestamp,
    so a fallback-persisted fetch is still found downstream.
    """
    safe_regions = regions.replace(",", "-")
    pattern = f"{sport}__event-{event_id}__*__{safe_regions}__*.json"
    candidates = list(raw_dir.glob(pattern))
    candidates += list(raw_dir.glob(f"fallback__event-{event_id}__*.json"))
    candidates.sort(key=_cache_timestamp)
    return candidates[-1] if candidates else None


def json_to_markets(
    source: str | Path | list[dict],
    forecast_run_id: str,
    market_keys: tuple[str, ...] | list[str] = ("h2h",),
) -> pd.DataFrame:
    """Parse The Odds API /odds payload into the canonical markets schema.

    Each (game × book × market_key × line × outcome) becomes one row. ``line``
    is the over/under or handicap point (NaN when not applicable, e.g. h2h,
    btts, halftime h2h).

    ``source`` may be a path to cached JSON, an already-parsed list of game
    dicts (so callers can pre-filter by kickoff before parsing), or a single
    game dict (the per-event endpoint's native response shape).

    ``market_keys`` whitelists which API market keys to include. Any other
    keys in the payload are silently ignored.
    """
    if isinstance(source, (str, Path)):
        raw = json.loads(Path(source).read_text())
    elif isinstance(source, dict):
        raw = [source]
    else:
        raw = source

    # The per-event endpoint returns one game object, not a list. Accept it.
    if isinstance(raw, dict):
        raw = [raw]

    keys_wanted = set(market_keys)
    rows: list[dict] = []
    seen: set[tuple] = set()

    for game in raw:
        event_id = game.get("id", "")
        commence_time = game.get("commence_time", "")
        match_date = commence_time[:10] if commence_time else ""
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        match = f"{home_team} vs {away_team}"

        for book in game.get("bookmakers", []):
            book_title = book["title"]
            for market in book.get("markets", []):
                mkey = market["key"]
                if mkey not in keys_wanted:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome["name"]
                    price = int(outcome["price"])
                    point = outcome.get("point")
                    line = float(point) if point is not None else float("nan")
                    line_key = "nan" if pd.isna(line) else f"{line}"

                    # Dedupe: same bookmaker can appear under multiple region
                    # keys (e.g. betfair_ex_uk + betfair_ex_eu both surface as
                    # title "Betfair"). Keep the first quote, drop the rest.
                    dedupe_key = (event_id, book_title, mkey, line_key, name)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)

                    rows.append(
                        {
                            "forecast_run_id": forecast_run_id,
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "match_date": match_date,
                            "home_team": home_team,
                            "away_team": away_team,
                            "match": match,
                            "book": book_title,
                            "market_key": mkey,
                            "line": line,
                            "outcome": name,
                            "american_odds": price,
                            "notes": "",
                        }
                    )
    return pd.DataFrame(rows)
