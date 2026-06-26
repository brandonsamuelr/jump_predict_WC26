"""Guards for the Track-2 gate-validated corpus models (team_more_cards, match_sot).

    .venv/bin/python tests/test_track2_models.py

Asserts: each shipped row routes to its gate-validated tier and returns the model
probability (hedged downstream); team_more_fouls STAYS shadow (not shipped — tiny
margin); safe shadow fallback when market context is absent; features pre-kickoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib import slate, corpus_models as CM
from odds_lib.optimizer import optimize
from odds_lib.edge import classify, K_PRIOR, GATE_MODEL_TRUST_K


def _c():
    return pd.DataFrame([
        {"market_key": "h2h", "line": float("nan"), "outcome": "Brazil", "market_prob": 0.62},
        {"market_key": "h2h", "line": float("nan"), "outcome": "Draw", "market_prob": 0.23},
        {"market_key": "h2h", "line": float("nan"), "outcome": "Chile", "market_prob": 0.15},
        {"market_key": "totals", "line": 2.5, "outcome": "Over", "market_prob": 0.55},
        {"market_key": "totals", "line": 2.5, "outcome": "Under", "market_prob": 0.45},
    ])

_GAME = {"home_team": "Brazil", "away_team": "Chile"}


# --- team_more_cards (shipped) ----------------------------------------------

def test_team_more_cards_routes_to_gate_validated_model():
    c = _c()
    row = {"question_type": "team_more_cards", "target_team": "Chile", "line": "", "match": "Brazil vs Chile"}
    tier, p, _ = slate.resolve_row(row, c, _GAME, None)
    assert tier == "MORE_CARDS" and classify(tier, "team_more_cards") == ("MORE_CARDS", "model")
    # Chile is the underdog & away -> model should give > base (underdogs/away get more cards)
    fav, opp, tot = 0.15, 0.62, 0.55
    expect = CM.predict_team_more_cards(fav - opp, tot, is_home=0)
    assert abs(p - round(expect, 4)) < 1e-3
    # submitted UNDISTORTED (k=1), NOT pulled toward the placeholder shadow
    s = optimize(tier=tier, question_type="team_more_cards", p_hat=p, shadow=0.39)
    assert abs(s.q - p) < 1e-9


# --- match_total_sot_over (shipped) -----------------------------------------

def test_match_total_sot_over_routes_to_gate_validated_model():
    c = _c()
    row = {"question_type": "match_total_sot_over", "target_team": "", "line": "7.5"}  # 8+
    tier, p, _ = slate.resolve_row(row, c, _GAME, None)
    assert tier == "MATCH_SOT" and classify(tier, "match_total_sot_over") == ("MATCH_SOT", "model")
    expect = CM.predict_match_total_sot_over(0.62 - 0.15, 0.55, 7.5)
    assert abs(p - round(expect, 4)) < 1e-3
    assert K_PRIOR[("MATCH_SOT", "model")] == GATE_MODEL_TRUST_K == 1.0
    # submitted UNDISTORTED (k=1)
    s = optimize(tier=tier, question_type="match_total_sot_over", p_hat=p, shadow=0.49)
    assert abs(s.q - p) < 1e-9


# --- team_more_fouls NOW shipped via the validated universal foul model ------
# superseded the old "tiny margin -> stays shadow": foul_cmp_model.json was re-validated
# OOS as a favorite_gap-only structural model and is now wired -> FOUL_CMP, k=1.

def test_team_more_fouls_prices_from_foul_model():
    c = _c()
    row = {"question_type": "team_more_fouls", "target_team": "Chile", "line": ""}
    tier, p, _ = slate.resolve_row(row, c, _GAME, None)
    assert tier == "FOUL_CMP" and p is not None and p > 0.46   # Chile heavy underdog -> more fouls


# --- safe fallback ----------------------------------------------------------

def test_track2_falls_back_to_shadow_without_market_context():
    # no consensus (c=None) -> can't compute features -> shadow
    for qt, tgt, line in [("team_more_cards", "Chile", ""), ("match_total_sot_over", "", "7.5")]:
        row = {"question_type": qt, "target_team": tgt, "line": line}
        tier, p, _ = slate.resolve_row(row, None, _GAME, None)
        assert tier == "PENDING" and p is None

def test_track2_falls_back_without_home_away():
    c = _c()
    for qt, tgt, line in [("team_more_cards", "Chile", ""), ("match_total_sot_over", "", "7.5")]:
        row = {"question_type": qt, "target_team": tgt, "line": line}
        tier, p, _ = slate.resolve_row(row, c, {}, None)   # no home/away in game_json
        assert tier == "PENDING" and p is None


# --- leakage: features pre-kickoff odds-derived -----------------------------

def test_corpus_model_features_are_pre_kickoff_only():
    m = CM._load()
    assert m["team_more_cards"]["features"] == ["favorite_gap", "total_line_prob", "is_home"]
    assert m["match_total_sot_over"]["features"] == ["favorite_gap", "total_line_prob"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
