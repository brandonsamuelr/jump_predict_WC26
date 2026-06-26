"""Guards for the remaining 5 market-available rows (Track 1).

    .venv/bin/python tests/test_market_rows.py

team_total_goals_over (per-team O/U), second_half_goals_over (alternate_totals_h2),
team_more_corners_full (corners_1x2 3-way), halftime_team_lead/winning (h2h_h1).
Each wired row: routes to its market tier, submits the de-vigged price UNDISTORTED,
safe shadow fallback when absent, correct line/outcome mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib import slate, market_rows as MR
from odds_lib.optimizer import optimize
from odds_lib.edge import classify, K_PRIOR


def _ou_book(title, key, line, over_px, under_px, team=None):
    oc = [{"name": "Over", "point": line, "price": over_px},
          {"name": "Under", "point": line, "price": under_px}]
    if team is not None:
        for o in oc:
            o["description"] = team
    return {"title": title, "markets": [{"key": key, "outcomes": oc}]}


def _3way_book(title, key, a, draw, b):
    return {"title": title, "markets": [{"key": key, "outcomes": [
        {"name": a[0], "price": a[1]}, {"name": "Draw", "price": draw},
        {"name": b[0], "price": b[1]}]}]}


def _undistorted(tier, qt, p, shadow=0.49):
    s = optimize(tier=tier, question_type=qt, p_hat=p, shadow=shadow)
    return abs(s.q - p) < 1e-9


# --- team_total_goals_over --------------------------------------------------

def test_team_goals_over_prices_off_market_undistorted():
    g = {"bookmakers": [_ou_book(f"B{i}", "alternate_team_totals", 2.5, 120, -150, team="Brazil")
                        for i in range(5)]}
    row = {"question_type": "team_total_goals_over", "target_team": "Brazil", "line": "2.5"}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier == "TEAMGOALS_OK" and classify(tier, "x") == ("TEAMGOALS", "ok")
    assert abs(p - MR.price_team_goals_over(g, "Brazil", 2.5).p_over) < 1e-9
    assert _undistorted(tier, "team_total_goals_over", p)

def test_team_goals_over_fallback_to_team_totals_market():
    # only the narrow team_totals market present, at the line -> still prices
    g = {"bookmakers": [_ou_book(f"B{i}", "team_totals", 1.5, -110, -110, team="Brazil")
                        for i in range(3)]}
    row = {"question_type": "team_total_goals_over", "target_team": "Brazil", "line": "1.5"}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier in ("TEAMGOALS_OK", "TEAMGOALS_THIN") and p is not None


# --- second_half_goals_over -------------------------------------------------

def test_2h_goals_over_reads_h2_market_at_line():
    g = {"bookmakers": [_ou_book(f"B{i}", "alternate_totals_h2", 1.5, 130, -160) for i in range(5)]}
    row = {"question_type": "second_half_goals_over", "target_team": "", "line": "1.5"}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier == "H2GOALS_OK"
    assert abs(p - MR.price_2h_goals_over(g, 1.5).p_over) < 1e-9
    assert _undistorted(tier, "second_half_goals_over", p)


# --- team_more_corners_full (3-way, take team outcome, not draw) -------------

def test_more_corners_takes_team_outcome_not_draw():
    # 3-way: Brazil -150 (fav), Draw +280, Chile +320 -> Brazil de-vig clearly highest
    g = {"bookmakers": [_3way_book(f"B{i}", "corners_1x2", ("Brazil", -150), 280, ("Chile", 320))
                        for i in range(5)]}
    row = {"question_type": "team_more_corners_full", "target_team": "Brazil", "line": ""}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier == "CORNERS_CMP_OK"
    # p is Brazil's MORE-corners prob (not draw, not Chile); favorite -> > 0.5
    assert p > 0.5 and _undistorted(tier, "team_more_corners_full", p)
    # sanity: equals the direct 3-way de-vig for Brazil
    assert abs(p - MR.price_more_corners(g, "Brazil").p_over) < 1e-9


# --- halftime aliases route to h2h_h1 team outcome --------------------------

def test_halftime_aliases_take_team_1h_win_not_draw_or_fulltime():
    c = pd.DataFrame([
        {"market_key": "h2h_h1", "line": float("nan"), "outcome": "Belgium", "market_prob": 0.55},
        {"market_key": "h2h_h1", "line": float("nan"), "outcome": "Draw", "market_prob": 0.30},
        {"market_key": "h2h_h1", "line": float("nan"), "outcome": "Egypt", "market_prob": 0.15},
        {"market_key": "h2h", "line": float("nan"), "outcome": "Belgium", "market_prob": 0.70},  # full-match decoy
    ])
    for qt in ("halftime_team_lead", "halftime_team_winning", "halftime_team_win"):
        row = {"question_type": qt, "target_team": "Belgium", "line": ""}
        tier, p, _ = slate.resolve_row(row, c, {}, None)
        assert tier == "MARKET" and abs(p - 0.55) < 1e-9, f"{qt}: took {p}, want 0.55 (1H team)"


# --- safe fallbacks ---------------------------------------------------------

def test_market_rows_fall_back_to_shadow_when_absent():
    empty = {"bookmakers": []}
    for qt, tgt in [("team_total_goals_over", "Brazil"), ("second_half_goals_over", ""),
                    ("team_more_corners_full", "Brazil")]:
        row = {"question_type": qt, "target_team": tgt, "line": "2.5"}
        tier, p, _ = slate.resolve_row(row, None, empty, None)
        assert tier == "PENDING" and p is None, f"{qt} should shadow when market absent"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
