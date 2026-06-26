"""OddsPapi-Pinnacle corner reads (half-corners constant->founded upgrade).

Pinnacle (sharp) on WC fixtures quotes Corners-Handicap-First-Half and Corners-Handicap
(full). The 0.0 handicap line, de-vigged, IS P(team more corners) — a per-match, sharp,
match-specific read that replaces the CORNER_HALF_STOPGAP constant. SIGN-GATED (verified
Germany>Ecuador on a known-favorite fixture). single_book (Pinnacle) -> use-if-plausible
per the market-quality LAW; the plausibility band is the guard (no agreement cross-check).

NEVER submits Pinnacle's fairOdds; we de-vig the raw two-sided prices ourselves.
Graceful per-fixture fallback: no fixture match / no 0.0 line / implausible -> return None
-> caller uses the existing stopgap (no worse than now).

Data: a cached Pinnacle slate (data/raw/oddspapi/pinnacle_wc_slate.json from
GET /v4/odds-by-tournaments?bookmaker=pinnacle&tournamentIds=16) + the marketId->name map
(markets_map.json). Fixture<->team join: prefers per-fixture team names when the cache
carries them; else a participant-ID->name map (participants_map.json). When neither yields
a confident match, returns None (-> stopgap). Plausibility band from market_quality.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

from . import market_quality as MQ

_LOG = logging.getLogger("oddspapi_pinnacle")

SLATE = Path("data/raw/oddspapi/pinnacle_wc_slate.json")
MKTS = Path("data/raw/oddspapi/markets_map.json")
PMAP = Path("data/raw/oddspapi/participants_map.json")

# --- OddsPapi V4 fetch + refresh (budget-aware; ~250 req/month cap) -----------------
# Name resolution: /fixtures?tournamentId=16 returns participant NAMES + fixtureIds for ALL WC
# fixtures in ONE call (the per-fixture /odds payload does NOT carry names). The bulk
# /odds-by-tournaments?bookmaker=pinnacle carries every fixture's markets (incl Corners-Handicap-
# First/Second-Half) keyed by fixtureId. So a full refresh that makes the sharp half-corner read
# present for EVERY fixture costs just 2 requests (names + odds), cache-first. WC tournamentId=16.
_BASE = "https://api.oddspapi.io/v4"
_UA = "Mozilla/5.0"
_WC_TOURNAMENT = 16


def _api_key() -> str | None:
    k = os.environ.get("ODDSPAPI_KEY")
    if k:
        return k
    try:
        for ln in Path(".env").read_text().splitlines():
            if ln.startswith("ODDSPAPI_KEY="):
                return ln.split("=", 1)[1].strip()
    except Exception:
        pass
    return None


def _get(path: str, **params) -> object:
    key = _api_key()
    if not key:
        raise RuntimeError("ODDSPAPI_KEY not set")
    params["apiKey"] = key
    url = f"{_BASE}{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read())


def _fresh(path: Path, max_age_h: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600.0


def refresh_pinnacle_cache(max_age_h: float = 6.0) -> int:
    """CACHE-FIRST refresh of the two files the half-corner resolver reads:
      1. participants_map.json  <- /fixtures (participantId -> name, ALL WC fixtures)
      2. pinnacle_wc_slate.json <- /odds-by-tournaments?bookmaker=pinnacle (markets per fixtureId)
    Only refetches a file older than ``max_age_h``. Returns the number of OddsPapi requests made
    (0, 1, or 2) so the caller can budget. Never raises into the refresh path -- logs and returns."""
    reqs = 0
    try:
        if not _fresh(PMAP, max_age_h):
            fx = _get("/fixtures", tournamentId=_WC_TOURNAMENT, hasOdds="true"); reqs += 1
            items = fx if isinstance(fx, list) else fx.get("data", [])
            pmap = {}
            for f in items:
                for idk, nmk in (("participant1Id", "participant1Name"), ("participant2Id", "participant2Name")):
                    pid, nm = f.get(idk), f.get(nmk)
                    if pid is not None and nm:
                        pmap[str(pid)] = nm
            if pmap:
                PMAP.write_text(json.dumps(pmap, indent=0))
                _read_json.cache_clear()
                _LOG.info("oddspapi fixtures-name map refreshed: %d participants", len(pmap))
        if not _fresh(SLATE, max_age_h):
            od = _get("/odds-by-tournaments", bookmaker="pinnacle", tournamentIds=_WC_TOURNAMENT); reqs += 1
            SLATE.write_text(json.dumps(od))
            _LOG.info("oddspapi pinnacle slate refreshed")
    except Exception as e:                      # budget/network/WAF issue -> keep the stale cache
        _LOG.warning("oddspapi refresh failed (%r); using existing cache, stopgap may fire", e)
    return reqs


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


@lru_cache(maxsize=2)
def _read_json(path_str):
    try:
        return json.loads(Path(path_str).read_text())
    except Exception:
        return {}


def _markets():
    return _read_json(str(MKTS))


def _pmap():
    return _read_json(str(PMAP))


def _hcap_ids(period):
    """marketIds whose name is the corner-handicap for the given period
    ('1h' -> First Half, '2h' -> Second Half, 'full' -> full match)."""
    out = []
    for mid, v in _markets().items():
        nm = str(v.get("name", "")).lower()
        if "corner" not in nm or "handicap" not in nm:
            continue
        if period == "1h" and "first half" in nm:
            out.append(mid)
        elif period == "2h" and "second half" in nm:
            out.append(mid)
        elif period == "full" and "first half" not in nm and "second half" not in nm:
            out.append(mid)
    return out


def _devig2(o, u):
    if not o or not u:
        return None
    io, iu = 1.0 / o, 1.0 / u
    return io / (io + iu)


def _price(pv):
    if isinstance(pv, list):
        pv = pv[-1] if pv else {}
    return (pv or {}).get("price") if isinstance(pv, dict) else None


def _p_home_more(fixture, period):
    """P(participant1/home more corners) from the 0.0 corner handicap; None if absent."""
    M = ((fixture.get("bookmakerOdds") or {}).get("pinnacle") or {}).get("markets") or {}
    for mid in _hcap_ids(period):
        if mid not in M:
            continue
        legs = {}
        for ov in (M[mid].get("outcomes") or {}).values():
            for pv in (ov.get("players") or {}).values():
                boid = (pv if isinstance(pv, dict) else {}).get("bookmakerOutcomeId", "")
                if "/" in boid:
                    ln, side = boid.split("/")
                    if ln == "0.0":
                        legs[side] = _price(pv)
        if legs.get("home") and legs.get("away"):
            return _devig2(legs["home"], legs["away"])
    return None


def _find_fixture(home, away):
    """Match cached Pinnacle fixture to (home, away) by team name (per-fixture name
    fields if present) or participant-ID->name map. Returns (fixture, target_is_p1)
    where target alignment is by the caller. None if no confident match."""
    try:
        slate = json.loads(SLATE.read_text())
    except Exception:
        return None
    data = slate if isinstance(slate, list) else slate.get("data", [])
    pm = _pmap()
    h, a = _norm(home), _norm(away)
    for fx in data:
        n1 = _norm(fx.get("participant1Name") or pm.get(str(fx.get("participant1Id")), ""))
        n2 = _norm(fx.get("participant2Name") or pm.get(str(fx.get("participant2Id")), ""))
        if not n1 or not n2:
            continue
        if {n1, n2} == {h, a}:
            return fx, (n1 == h)   # is participant1 the home team?
    return None


def more_corners(home_team, away_team, target_team, half):
    """P(target_team more corners) for 'full' or '1h'/'2h'; None -> caller falls back to stopgap.
    half: pass the question_type or '1h'/'2h'/'full'. Logs a NAMED warning whenever the sharp read
    is missed (no fixture match / no half-corner line / implausible) so a missed read is visible."""
    hl = str(half).lower()                     # accept both spellings (..._1h / ..._h1)
    period = "1h" if ("1h" in hl or "h1" in hl) else "2h" if ("2h" in hl or "h2" in hl) else "full"
    m = _find_fixture(home_team, away_team)
    if m is None:
        _LOG.warning("oddspapi half-corner MISS: no Pinnacle fixture match for %s vs %s (%s) -> stopgap",
                     home_team, away_team, period)
        return None
    fx, p1_is_home = m
    p_p1 = _p_home_more(fx, period)            # P(participant1 more corners)
    if p_p1 is None:
        _LOG.warning("oddspapi half-corner MISS: %s vs %s has no Pinnacle %s corner-handicap line "
                     "(real market absence) -> stopgap", home_team, away_team, period)
        return None
    p_home = p_p1 if p1_is_home else (1.0 - p_p1)
    p_away = 1.0 - p_home
    tgt = _norm(target_team)
    p = p_home if tgt == _norm(home_team) else p_away
    if not MQ.in_band(p, "comparison"):        # plausibility guard (sign/degenerate safety)
        _LOG.warning("oddspapi half-corner MISS: %s vs %s %s implausible de-vig %.3f -> stopgap",
                     home_team, away_team, period, p)
        return None
    return round(p, 4)


__all__ = ["more_corners", "refresh_pinnacle_cache", "SLATE", "MKTS", "PMAP"]
