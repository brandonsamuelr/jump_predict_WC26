"""Guards for the engine half-split recalibration + 2H-goals double-source + fouls shadow.

    .venv/bin/python tests/test_half_split.py

Part A: H1_SHARE is the data-measured 0.44 (not the old guessed 0.45); the 2H-goals
double-source defers to the market when present (so total_goals_2h_over and
second_half_goals_over AGREE), and falls back to the engine half-split only when no
2H market exists. Part B: team_more_fouls shadow recalibrated to 0.50.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib import slate, match_engine as E
from odds_lib.field_model import FieldMeanEstimator


def _2h_market(line=1.5, n=5, over=130, under=-160):
    return {"home_team": "H", "away_team": "A", "bookmakers": [
        {"title": f"B{i}", "markets": [{"key": "alternate_totals_h2", "outcomes": [
            {"name": "Over", "point": line, "price": over},
            {"name": "Under", "point": line, "price": under}]}]} for i in range(n)]}


def _model():
    m = E.calibrate("H", "A", p_home=0.50, p_over=0.55)
    return (m, E.simulate(m, n=40_000, seed=2), "H", "A")


def _row(qt):
    return {"question_type": qt, "target_team": "", "line": "1.5"}


# --- Part A: recalibrated constant -----------------------------------------

def test_h1_share_recalibrated_to_044():
    assert abs(E.H1_SHARE - 0.44) < 1e-9   # was 0.45; measured 0.4394 over 106k matches


# --- Part A.2: double-source precedence ------------------------------------

def test_2h_goals_market_wins_and_both_rows_agree():
    g, mt = _2h_market(), _model()
    t1, p1, _ = slate.resolve_row(_row("total_goals_2h_over"), None, g, mt)
    t2, p2, _ = slate.resolve_row(_row("second_half_goals_over"), None, g, mt)
    assert t1.startswith("H2GOALS") and t2.startswith("H2GOALS")   # both defer to market
    assert abs(p1 - p2) < 1e-9                                      # -> they AGREE

def test_2h_goals_engine_fallback_when_no_market():
    g = {"home_team": "H", "away_team": "A", "bookmakers": []}
    mt = _model()
    t1, p1, _ = slate.resolve_row(_row("total_goals_2h_over"), None, g, mt)
    t2, p2, _ = slate.resolve_row(_row("second_half_goals_over"), None, g, mt)
    # no totals_h1 leg -> engine fallback on the measured H1_SHARE constant, FLAGGED
    assert t1 == "ENGINE_GOALS_H1FALLBACK" and t2 == "ENGINE_GOALS_H1FALLBACK"
    assert abs(p1 - p2) < 1e-9
    assert abs(p1 - round(E.p_total_goals_2h_over(mt[1], 2), 4)) < 1e-9   # uses 0.44 constant split

def test_2h_goals_shadow_when_no_market_and_no_model():
    t, p, _ = slate.resolve_row(_row("total_goals_2h_over"), None, {"bookmakers": []}, None)
    assert t == "PENDING" and p is None


# --- Part B: fouls shadow recalibration ------------------------------------

def test_fouls_shadow_recalibrated_to_050():
    fe = FieldMeanEstimator()
    est = fe.estimate("team_more_fouls")
    assert abs(est.q_hat - 0.50) < 1e-9 and est.source == "outcome_base_rate"

def test_other_types_not_overridden():
    fe = FieldMeanEstimator()
    assert fe.estimate("team_offsides_over").source != "outcome_base_rate"
    assert fe.estimate("team_win").source != "outcome_base_rate"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
