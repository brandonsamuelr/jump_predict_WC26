"""Guards for the corner-comparison model wiring (full-match ship; 1H/2H STOPGAP).

    .venv/bin/python tests/test_corners_cmp_model.py

full-match: corners_1x2 MARKET wins; model is the fallback when no market; shadow only
if neither; undistorted (k=1). 1H/2H: the old CORNER_HALF_SHRINK=0.65 was proven wrong
(0.5-centered + corner dominance barely persists per-half); the route now submits the
measured per-half BASE-RATE FLOOR (1H 0.389 / 2H 0.410), tagged CORNER_HALF_STOPGAP —
a STOPGAP that ignores favorite_gap (data wall), NOT a true conditional P.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib import slate, corners_cmp_model as CC
from odds_lib.optimizer import optimize
from odds_lib.edge import classify, K_PRIOR


def _c(home_p=0.20, draw=0.27, away_p=0.53, over=0.576):
    # home heavy underdog vs away (favgap for home = home_p-away_p ~ -0.33; tune for tests)
    return pd.DataFrame([
        {"market_key": "h2h", "line": float("nan"), "outcome": "Haiti", "market_prob": home_p},
        {"market_key": "h2h", "line": float("nan"), "outcome": "Draw", "market_prob": draw},
        {"market_key": "h2h", "line": float("nan"), "outcome": "Morocco", "market_prob": away_p},
        {"market_key": "totals", "line": 2.5, "outcome": "Over", "market_prob": over},
        {"market_key": "totals", "line": 2.5, "outcome": "Under", "market_prob": 1 - over},
    ])

_GAME = {"home_team": "Haiti", "away_team": "Morocco"}

def _corners_1x2_game(team_a="Haiti", team_b="Morocco"):
    g = dict(_GAME)
    g["bookmakers"] = [{"title": f"B{i}", "markets": [{"key": "corners_1x2", "outcomes": [
        {"name": team_a, "price": 320}, {"name": "Draw", "price": 280}, {"name": team_b, "price": -150}]}]}
        for i in range(5)]
    return g


# --- full-match: market > model > shadow ------------------------------------

def test_full_market_wins_when_present():
    g = _corners_1x2_game()
    row = {"question_type": "team_more_corners_full", "target_team": "Haiti", "line": ""}
    tier, p, _ = slate.resolve_row(row, _c(), g, None)
    assert tier in ("CORNERS_CMP_OK", "CORNERS_CMP_THIN")   # market, not model

def test_full_model_fallback_when_no_market():
    g = dict(_GAME); g["bookmakers"] = []                   # no corners market
    row = {"question_type": "team_more_corners_full", "target_team": "Haiti", "line": ""}
    tier, p, _ = slate.resolve_row(row, _c(), g, None)
    assert tier == "CORNERS_CMP_MODEL" and classify(tier, "x") == ("CORNERS_CMP", "model")
    assert p == round(CC.predict_more_corners(0.20 - 0.53, 0.576), 4)
    # undistorted (k=1): submit == model p
    s = optimize(tier=tier, question_type="team_more_corners_full", p_hat=p, shadow=0.49)
    assert abs(s.q - p) < 1e-9 and K_PRIOR[("CORNERS_CMP", "model")] == 1.0

def test_full_shadow_when_no_market_no_features():
    row = {"question_type": "team_more_corners_full", "target_team": "Haiti", "line": ""}
    tier, p, _ = slate.resolve_row(row, None, {"bookmakers": []}, None)   # no consensus
    assert tier == "PENDING" and p is None


# --- 1H/2H: STOPGAP base-rate floor (NOT a model; favorite_gap ignored) -----
# The old CORNER_HALF_SHRINK=0.65 was proven wrong (0.5-centered + corner dominance
# barely persists per-half, +0.031 leakage-free). The route now submits the measured
# per-half base rate (1H 0.389 / 2H 0.410), tagged CORNER_HALF_STOPGAP. It IGNORES
# favorite_gap (data wall) -> the SAME floor for any matchup, flagged STOPGAP/unsolved.

def test_1h_uses_favorite_gap_model_when_h2h_present():
    # 1H now uses the gate-validated favorite_gap model (+0.015 Brier vs the blind constant):
    # a heavy underdog (Haiti) is pushed BELOW the old flat 0.389, direction-aware, shipped k=1.
    g = dict(_GAME); g["bookmakers"] = []
    row = {"question_type": "team_more_corners_1h", "target_team": "Haiti", "line": ""}
    tier, p1h, _ = slate.resolve_row(row, _c(), g, None)            # _c() = Haiti heavy underdog
    assert tier == "CORNER_HALF_1H_FG" and classify(tier, "x") == ("CORNER_HALF", "model_1h")
    assert p1h < 0.389                                              # underdog below the blind constant
    s = optimize(tier=tier, question_type="team_more_corners_1h", p_hat=p1h, shadow=0.49)
    assert abs(s.q - p1h) < 1e-9 and K_PRIOR[("CORNER_HALF", "model_1h")] == 1.0

def test_1h_falls_to_stopgap_without_h2h():
    # no consensus -> no favorite_gap -> the honest 1H stopgap (unchanged fallback).
    g = dict(_GAME); g["bookmakers"] = []
    row = {"question_type": "team_more_corners_1h", "target_team": "Haiti", "line": ""}
    tier, p1h, _ = slate.resolve_row(row, None, g, None)
    assert tier == "CORNER_HALF_STOPGAP" and p1h == round(CC.CORNER_HALF_BASE_RATE["1h"], 4) == 0.389

def test_2h_returns_base_rate_floor():
    g = dict(_GAME); g["bookmakers"] = []
    row = {"question_type": "team_more_corners_2h", "target_team": "Haiti", "line": ""}
    tier, p, _ = slate.resolve_row(row, _c(), g, None)
    assert tier == "CORNER_HALF_STOPGAP" and p == round(CC.CORNER_HALF_BASE_RATE["2h"], 4) == 0.410

def test_1h_model_responds_to_favorite_gap():
    # THE FIX: 1H corners now REACT to favorite_gap -- the favorite gets a HIGHER P(more 1H
    # corners) than the underdog, no longer the same blind 0.389 constant (which was the bug
    # that got us structurally dominated by the crowd on lopsided games). 2H stays blind by design.
    g = dict(_GAME); g["bookmakers"] = []
    fav = {"question_type": "team_more_corners_1h", "target_team": "Morocco", "line": ""}
    udog = {"question_type": "team_more_corners_1h", "target_team": "Haiti", "line": ""}
    _, p_fav, _ = slate.resolve_row(fav, _c(), g, None)
    _, p_udog, _ = slate.resolve_row(udog, _c(), g, None)
    assert p_fav > 0.389 > p_udog            # favorite up, underdog down -- direction-aware
    # 2H remains the flat stopgap (favorite_gap is wrong-signed there)
    _, p2, _ = slate.resolve_row({"question_type": "team_more_corners_2h", "target_team": "Haiti", "line": ""}, _c(), g, None)
    assert p2 == 0.410


# --- transfer: odds-only features (no identities) ---------------------------

def test_model_features_are_odds_only():
    assert CC._load()["_meta"]["features"] == ["favorite_gap", "total_line_prob"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
