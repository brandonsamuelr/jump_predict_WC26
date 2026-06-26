"""Guards for the OddsPapi-Pinnacle half-corner read + SIGN convention + safe fallback.

    .venv/bin/python tests/test_oddspapi_pinnacle.py

Pinnacle's 0.0 corner-handicap (de-vigged) = P(team more corners). team_more_corners_1h
prices from this sharp read (CORNER_HALF_PINNACLE, k=1) when present; falls back to the
measured base-rate STOPGAP when absent/unmatched/implausible. The SIGN gate is encoded:
on a known-favorite synthetic fixture the favorite must come out as the more-corners side.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib import slate, oddspapi_pinnacle as OP
from odds_lib.edge import classify, K_PRIOR


def _setup(home_odds, away_odds, period_name="Corners - Handicap First Half",
           p1="Spain", p2="Tonga"):
    """Write a synthetic Pinnacle slate + markets map; point OP at them."""
    d = Path(tempfile.mkdtemp())
    (d / "markets.json").write_text(json.dumps({"900": {"name": period_name, "period": "p1"}}))
    fixture = {"fixtureId": "fx1", "participant1Id": 1, "participant2Id": 2,
               "participant1Name": p1, "participant2Name": p2,
               "bookmakerOdds": {"pinnacle": {"markets": {"900": {"outcomes": {
                   "a": {"players": {"0": {"bookmakerOutcomeId": "0.0/home", "price": home_odds}}},
                   "b": {"players": {"0": {"bookmakerOutcomeId": "0.0/away", "price": away_odds}}}}}}}}}
    (d / "slate.json").write_text(json.dumps([fixture]))
    OP.SLATE = d / "slate.json"; OP.MKTS = d / "markets.json"; OP.PMAP = d / "nope.json"
    OP._read_json.cache_clear()   # paths read at call-time -> just clear the cache


def test_refresh_is_cache_first_no_fetch_when_fresh():
    # BUDGET GUARD: if both caches are fresh, refresh_pinnacle_cache makes ZERO OddsPapi requests.
    d = Path(tempfile.mkdtemp())
    (d / "pmap.json").write_text('{"1": "A"}')
    (d / "slate.json").write_text("[]")
    o_pmap, o_slate, o_get = OP.PMAP, OP.SLATE, OP._get
    OP.PMAP, OP.SLATE = d / "pmap.json", d / "slate.json"
    calls = {"n": 0}
    OP._get = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or {}
    try:
        n = OP.refresh_pinnacle_cache(max_age_h=24.0)   # just-written files -> fresh
        assert n == 0 and calls["n"] == 0
    finally:
        OP.PMAP, OP.SLATE, OP._get = o_pmap, o_slate, o_get
        OP._read_json.cache_clear()


def test_refresh_fetches_when_stale_and_never_raises():
    # stale cache -> fetches (mocked); a network/WAF failure must NOT raise into the refresh path.
    d = Path(tempfile.mkdtemp())
    o_pmap, o_slate, o_get = OP.PMAP, OP.SLATE, OP._get
    OP.PMAP, OP.SLATE = d / "pmap.json", d / "slate.json"   # absent -> stale
    OP._get = lambda path, **k: ([{"participant1Id": 1, "participant1Name": "A",
                                   "participant2Id": 2, "participant2Name": "B"}]
                                  if "fixtures" in path else [])
    try:
        n = OP.refresh_pinnacle_cache(max_age_h=0.0)
        assert n == 2 and OP.PMAP.exists() and OP.SLATE.exists()
        # failure path: _get raises -> caught, returns without raising
        OP._get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("WAF 403"))
        (d / "pmap.json").unlink(); (d / "slate.json").unlink()
        assert OP.refresh_pinnacle_cache(max_age_h=0.0) == 0   # both attempts failed, no crash
    finally:
        OP.PMAP, OP.SLATE, OP._get = o_pmap, o_slate, o_get
        OP._read_json.cache_clear()


def test_sign_favorite_more_corners():
    _setup(home_odds=1.50, away_odds=2.60)        # home (Spain) favored to have more corners
    p_home = OP.more_corners("Spain", "Tonga", "Spain", "team_more_corners_1h")
    p_away = OP.more_corners("Spain", "Tonga", "Tonga", "team_more_corners_1h")
    assert p_home is not None and p_home > 0.55 and abs(p_home + p_away - 1.0) < 1e-6  # favorite > 0.5

def test_route_prices_from_pinnacle():
    _setup(1.50, 2.60)
    g = {"home_team": "Spain", "away_team": "Tonga", "bookmakers": []}
    c = pd.DataFrame([{"market_key": "h2h", "line": float("nan"), "outcome": "Spain", "market_prob": 0.7}])
    tier, p, _ = slate.resolve_row({"question_type": "team_more_corners_1h", "target_team": "Spain", "line": ""}, c, g, None)
    assert tier == "CORNER_HALF_PINNACLE" and p > 0.55
    assert classify(tier, "x") == ("CORNER_HALF", "pinnacle") and K_PRIOR[("CORNER_HALF", "pinnacle")] == 1.0

def test_fallback_to_stopgap_when_period_absent():
    # only a FIRST-half handicap exists -> a 2H query finds nothing -> stopgap
    _setup(1.50, 2.60, period_name="Corners - Handicap First Half")
    g = {"home_team": "Spain", "away_team": "Tonga", "bookmakers": []}
    tier, p, _ = slate.resolve_row({"question_type": "team_more_corners_2h", "target_team": "Spain", "line": ""}, None, g, None)
    assert tier == "CORNER_HALF_STOPGAP" and p == 0.410

def test_fallback_when_fixture_unmatched():
    _setup(1.50, 2.60)
    assert OP.more_corners("Narnia", "Atlantis", "Narnia", "team_more_corners_1h") is None

def test_plausibility_band_rejects_degenerate():
    _setup(home_odds=1.001, away_odds=50.0)       # P(home)~0.998 -> outside comparison band
    assert OP.more_corners("Spain", "Tonga", "Spain", "team_more_corners_1h") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
