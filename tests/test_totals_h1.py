"""Guards for the totals_h1 market half-split (agreement+plausibility gate) + alternate_totals.

    .venv/bin/python tests/test_totals_h1.py

market_h1_share derives a per-match 1H goal share = E[lambda_1H]/E[lambda_full] from
totals_h1 + totals, inverting ONLY half-integer lines (0.5/1.5) -- whole lines (1.0)
have push semantics that mis-invert (the old share~0.25 bug). The gate is QUALITY, not
count: a few AGREEING (low dispersion) + PLAUSIBLE (share in [0.30,0.60]) books SUPERSEDE
the H1_SHARE constant, regardless of book count. The constant is last-resort ONLY for an
unreliable signal (scattered / implausible / no market), each flagged with the reason.
alternate_totals prices match totals at the EXACT non-2.5 line. No new pricing constant.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib import slate, match_engine as E, market_rows as MR
from odds_lib.edge import classify, K_PRIOR


def _full_book(over=-110, under=-110):
    return {"key": "totals", "outcomes": [
        {"name": "Over", "point": 2.5, "price": over},
        {"name": "Under", "point": 2.5, "price": under}]}

def _h1_book(line, over, under):
    return {"key": "totals_h1", "outcomes": [
        {"name": "Over", "point": line, "price": over},
        {"name": "Under", "point": line, "price": under}]}

def _game(h1_specs=None, extra=None, n_full=6):
    """h1_specs: list of (line, over, under) per book. extra: per-book extra markets."""
    books = []
    for i in range(n_full):
        mks = [_full_book()]
        if h1_specs and i < len(h1_specs):
            mks.append(_h1_book(*h1_specs[i]))
        if extra and i < len(extra):
            mks.append(extra[i])
        books.append({"title": f"B{i}", "markets": mks})
    return {"home_team": "H", "away_team": "A", "bookmakers": books}

# agreeing, plausible 1H quotes (P(1H over 0.5) ~ 0.66 -> share ~ 0.40)
_AGREE = [(0.5, -205, 170), (0.5, -210, 172), (0.5, -200, 168)]

def _consensus():
    return pd.DataFrame([
        {"market_key": "h2h", "line": float("nan"), "outcome": "H", "market_prob": 0.45},
        {"market_key": "h2h", "line": float("nan"), "outcome": "Draw", "market_prob": 0.27},
        {"market_key": "h2h", "line": float("nan"), "outcome": "A", "market_prob": 0.28},
        {"market_key": "totals", "line": 2.5, "outcome": "Over", "market_prob": 0.52},
        {"market_key": "totals", "line": 2.5, "outcome": "Under", "market_prob": 0.48},
    ])


# --- the bug fix: whole-line 1.0 is NOT inverted (only half-integers) --------

def test_only_half_integer_lines_inverted():
    assert MR._H1_INVERT_LINES == (0.5, 1.5)   # 1.0 (push) and 0.75/1.25 (Asian) excluded

def test_share_is_plausible_not_the_025_bug():
    # agreeing 0.5-line quotes -> plausible share in band, NOT the old ~0.25 whole-line bug
    share = MR.market_h1_share(_game(_AGREE))
    assert share is not None
    val, nbk, disp = share
    assert MR.H1_PLAUSIBLE_BAND[0] <= val <= MR.H1_PLAUSIBLE_BAND[1]
    assert nbk == 3 and disp <= MR.H1_DISPERSION_MAX
    assert abs(val - E.H1_SHARE) > 1e-6   # a market read, not the constant


# --- agreement + plausibility GATE (not book count) -------------------------

def test_few_agreeing_books_supersede_constant():
    # only 3 books, but they AGREE and are PLAUSIBLE -> market used (NOT discarded for count)
    mt = slate.build_model(_consensus(), "H", "A", n=20_000, game_json=_game(_AGREE))
    assert mt[4] == "market" and abs(mt[0].h1_share - E.H1_SHARE) > 1e-6

def test_single_agreeing_book_is_used():
    # ONE book, trivially agrees, plausible -> still used (count is not the gate)
    mt = slate.build_model(_consensus(), "H", "A", n=20_000, game_json=_game(_AGREE[:1]))
    assert mt[4] == "market"

def test_scattered_books_fall_back_flagged():
    scatter = [(0.5, -320, 250), (0.5, +150, -180), (0.5, -105, -115)]   # wildly disagree
    sh = MR.market_h1_share(_game(scatter))
    assert sh is not None and sh[2] > MR.H1_DISPERSION_MAX               # high dispersion
    mt = slate.build_model(_consensus(), "H", "A", n=20_000, game_json=_game(scatter))
    assert mt[4] == "constant_scattered_books" and mt[0].h1_share == E.H1_SHARE

def test_implausible_derivation_falls_back_flagged():
    # degenerate 1H quote (P(1H over 0.5)~0.98 -> lambda_1H huge -> share > 0.60)
    impl = [(0.5, -6000, 2000), (0.5, -6000, 2000)]
    sh = MR.market_h1_share(_game(impl))
    assert sh is not None and sh[0] > MR.H1_PLAUSIBLE_BAND[1]
    mt = slate.build_model(_consensus(), "H", "A", n=20_000, game_json=_game(impl))
    assert mt[4] == "constant_implausible_derivation" and mt[0].h1_share == E.H1_SHARE

def test_no_1h_market_falls_back():
    mt = slate.build_model(_consensus(), "H", "A", n=20_000, game_json=_game(None))
    assert mt[4] == "constant_no_market" and mt[0].h1_share == E.H1_SHARE


# --- 1H-total rows route directly off totals_h1 -----------------------------

def test_1h_goals_direct_from_totals_h1():
    g = _game([(0.5, -205, 170)] * 5)
    row = {"question_type": "total_goals_1h_over", "target_team": "", "line": "0.5"}
    tier, p, _ = slate.resolve_row(row, _consensus(), g, None)
    assert tier in ("H1GOALS_OK", "H1GOALS_THIN") and 0.0 < p < 1.0
    assert classify(tier, "x")[0] == "H1GOALS" and K_PRIOR[("H1GOALS", "ok")] == 1.0

def test_1h_goals_shadow_when_no_market():
    row = {"question_type": "total_goals_1h_over", "target_team": "", "line": "0.5"}
    tier, p, _ = slate.resolve_row(row, _consensus(), _game(None), None)
    assert tier == "PENDING" and p is None


# --- 2H precedence: direct market > market-derived(full-1H) > constant -------

def test_2h_derived_from_full_minus_1h_when_no_direct_market():
    g = _game(_AGREE)
    mt = slate.build_model(_consensus(), "H", "A", 20_000, g)
    row = {"question_type": "total_goals_2h_over", "target_team": "", "line": "1.5"}
    tier, p, _ = slate.resolve_row(row, _consensus(), g, mt)
    assert tier == "ENGINE_GOALS_H1MKT" and 0.0 < p < 1.0   # market-derived split

def test_2h_constant_fallback_flagged_when_no_1h_market():
    g = _game(None)
    mt = slate.build_model(_consensus(), "H", "A", 20_000, g)
    row = {"question_type": "total_goals_2h_over", "target_team": "", "line": "1.5"}
    tier, _, _ = slate.resolve_row(row, _consensus(), g, mt)
    assert tier == "ENGINE_GOALS_H1FALLBACK"
    assert classify(tier, "x") == ("ENGINE", "engine")

def test_2h_precedence_direct_market_wins():
    h2 = {"key": "alternate_totals_h2", "outcomes": [
        {"name": "Over", "point": 1.5, "price": 130}, {"name": "Under", "point": 1.5, "price": -160}]}
    g = _game(_AGREE, extra=[h2] * 5)
    mt = slate.build_model(_consensus(), "H", "A", 20_000, g)
    row = {"question_type": "total_goals_2h_over", "target_team": "", "line": "1.5"}
    tier, _, _ = slate.resolve_row(row, _consensus(), g, mt)
    assert tier.startswith("H2GOALS")


# --- alternate_totals: exact non-2.5 line -----------------------------------

def _alt(line, over, under):
    return {"key": "alternate_totals", "outcomes": [
        {"name": "Over", "point": line, "price": over}, {"name": "Under", "point": line, "price": under}]}

def test_alt_totals_exact_line():
    g = _game(None, extra=[_alt(3.5, 150, -180)] * 6)
    row = {"question_type": "match_total_over", "target_team": "", "line": "3.5"}
    tier, p, _ = slate.resolve_row(row, _consensus(), g, None)
    assert tier == "MARKET" and 0.0 < p < 1.0

def test_non25_falls_to_interp_when_no_alt():
    g = _game(None)
    row = {"question_type": "match_total_over", "target_team": "", "line": "3.5"}
    tier, p, _ = slate.resolve_row(row, _consensus(), g, None)
    assert tier == "MARKET_INTERP" and classify(tier, "x") == ("MARKET", "market")

def test_main_25_line_unchanged():
    row = {"question_type": "match_total_over", "target_team": "", "line": "2.5"}
    tier, p, _ = slate.resolve_row(row, _consensus(), {"home_team": "H", "away_team": "A"}, None)
    assert tier == "MARKET"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
