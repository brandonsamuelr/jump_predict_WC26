"""Guard for the per-slate sharp anchor: Pinnacle and exchanges must be reported SEPARATELY (a
pooled 'sharp consensus' can hide the Pinnacle-vs-exchange split that is the real signal).

    .venv/bin/python tests/test_sharp_anchor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.sharp_anchor import favorite_gap_by_source


def _h2h_book(book, a, draw, b, ta="Canada", tb="South Africa"):
    return {"key": book, "markets": [{"key": "h2h", "outcomes": [
        {"name": ta, "price": a}, {"name": "Draw", "price": draw}, {"name": tb, "price": b}]}]}


def test_pinnacle_and_exchange_reported_separately_with_split_flag():
    # Pinnacle prices a modest Canada edge; the exchange runs HOT on Canada; a soft book in between.
    event = {"home_team": "South Africa", "away_team": "Canada", "bookmakers": [
        _h2h_book("pinnacle", -150, +250, +400),       # gap ~0.37
        _h2h_book("betfair_ex_uk", -200, +280, +450),  # gap ~0.44 (hot)
        _h2h_book("draftkings", -140, +260, +380),     # soft, in between
    ]}
    out = favorite_gap_by_source(event, "Canada", "South Africa")
    assert out["pinnacle"] is not None and out["exchange_median"] is not None
    assert out["n_pinnacle"] == 1 and out["n_exchange"] == 1
    # the split is preserved, not pooled away
    assert out["pinnacle"] != out["exchange_median"]
    assert out["pinnacle_minus_exchange"] is not None and abs(out["pinnacle_minus_exchange"]) >= 0.02
    assert any("split" in f for f in out["flags"])
    assert out["anchor_source"] == "pinnacle"           # prefer Pinnacle as the primary sharp


def test_sources_agree_flag_when_aligned():
    event = {"home_team": "X", "away_team": "Y", "bookmakers": [
        _h2h_book("pinnacle", -150, +250, +400, ta="Y", tb="X"),
        _h2h_book("betfair_ex_uk", -150, +250, +400, ta="Y", tb="X"),
    ]}
    out = favorite_gap_by_source(event, "Y", "X")
    assert abs(out["pinnacle_minus_exchange"]) < 0.02
    assert any("agree" in f for f in out["flags"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
