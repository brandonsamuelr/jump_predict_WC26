"""Guards for the delivery-path hardening: a PAID fetch is never lost.

    .venv/bin/python tests/test_fetch_persist.py

Offline/deterministic (HTTP stubbed). Asserts: canonical-write failure persists a
fallback (no exception, data intact, returned); normal path writes canonical only;
the cache-find glob picks up a fallback (newest-wins); and the coin-flip/missed-lock
override tag classifies as the excluded 'delivery_failure' category.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import odds_lib.odds_api as oa
from odds_lib.measurement import classify_override


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


class _Resp:
    headers: dict = {}
    def __init__(self, d): self._d = d
    def json(self): return self._d
    def raise_for_status(self): return None


_GAME = {"home_team": "H", "away_team": "A", "bookmakers": []}


# --- _persist guard ---------------------------------------------------------

def test_persist_normal_writes_canonical():
    t = _tmp(); can = t / "canon.json"; fb = t / "fb.json"
    out = oa._persist('{"a":1}', can, fb)
    assert out == can and can.exists() and not fb.exists() and can.read_text() == '{"a":1}'

def test_persist_fallback_on_canonical_failure():
    t = _tmp()
    can = t / "missing_dir" / "deep" / "canon.json"   # parent absent -> write raises
    fb = t / "fb.json"
    out = oa._persist('{"a":2}', can, fb)              # must NOT raise
    assert out == fb and fb.exists() and not can.exists() and fb.read_text() == '{"a":2}'


# --- fetch_event_odds end-to-end (HTTP stubbed) -----------------------------

def test_fetch_event_canonical_fail_writes_fallback():
    t = _tmp()
    o_get, o_key, o_sm = oa.requests.get, oa._api_key, oa._safe_markets
    oa.requests.get = lambda *a, **k: _Resp(_GAME)
    oa._api_key = lambda: "k"
    oa._safe_markets = lambda m: "x" * 300            # force a too-long canonical name
    try:
        out = oa.fetch_event_odds("soccer_fifa_world_cup", "EID123", markets="h2h", raw_dir=t)
    finally:
        oa.requests.get, oa._api_key, oa._safe_markets = o_get, o_key, o_sm
    assert out.exists() and out.name.startswith("fallback__event-EID123__")  # (a)+(b)
    assert json.loads(out.read_text())["home_team"] == "H"                    # (c) data intact

def test_fetch_event_normal_writes_canonical():
    t = _tmp()
    o_get, o_key = oa.requests.get, oa._api_key
    oa.requests.get = lambda *a, **k: _Resp(_GAME)
    oa._api_key = lambda: "k"
    try:
        out = oa.fetch_event_odds("soccer_fifa_world_cup", "EID9", markets="h2h", raw_dir=t)
    finally:
        oa.requests.get, oa._api_key = o_get, o_key
    assert out.exists() and out.name.startswith("soccer_fifa_world_cup__event-EID9__")
    assert not list(t.glob("fallback__*"))            # no fallback on the normal path


# --- cache-find glob picks up fallback --------------------------------------

def test_latest_event_cache_finds_fallback_newest_wins():
    t = _tmp()
    (t / "soccer_fifa_world_cup__event-E__h2h__us-uk-eu__20260101T000000Z.json").write_text("{}")
    (t / "fallback__event-E__20260101T010000Z.json").write_text('{"fb":1}')   # newer ts
    got = oa.latest_event_cache("soccer_fifa_world_cup", "E", "us,uk,eu", raw_dir=t)
    assert got is not None and got.name.startswith("fallback__event-E__")     # newest-wins
    t2 = _tmp()
    (t2 / "fallback__event-Z__20260101T000000Z.json").write_text("{}")
    assert oa.latest_event_cache("soccer_fifa_world_cup", "Z", "us,uk,eu", raw_dir=t2) is not None


# --- delivery-failure category ----------------------------------------------

def test_coinflip_and_crash_tags_are_delivery_failure():
    assert classify_override("default_coinflip_refresh_missed_lock") == "delivery_failure"
    assert classify_override("attempted_not_submitted_refresh_crash_at_lock") == "delivery_failure"
    # unchanged: real soft / entry_shift still classify as before
    assert classify_override("split_difference felt high") == "soft"
    assert classify_override("entry_shift_intended_0.288") == "entry_shift"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
