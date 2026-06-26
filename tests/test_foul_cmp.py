"""Guards for the validated universal foul-comparison model wiring (team_more_fouls).

    .venv/bin/python tests/test_foul_cmp.py

team_more_fouls now prices from the gate-validated favorite_gap model (was flat 0.50):
FOUL_CMP tier, k=1 (undistorted). SIGN: underdog fouls more -> a heavy-underdog target
prices ABOVE the even-match base (~0.46); a heavy favorite BELOW. Structural/odds-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib import slate, foul_cmp_model as FC
from odds_lib.optimizer import optimize
from odds_lib.edge import classify, K_PRIOR

_GAME = {"home_team": "Haiti", "away_team": "Morocco"}

def _c(home_p, away_p, draw=0.27):
    return pd.DataFrame([
        {"market_key": "h2h", "line": float("nan"), "outcome": "Haiti", "market_prob": home_p},
        {"market_key": "h2h", "line": float("nan"), "outcome": "Draw", "market_prob": draw},
        {"market_key": "h2h", "line": float("nan"), "outcome": "Morocco", "market_prob": away_p},
    ])


def test_foul_cmp_tier_and_k():
    row = {"question_type": "team_more_fouls", "target_team": "Haiti", "line": ""}
    tier, p, _ = slate.resolve_row(row, _c(0.20, 0.53), _GAME, None)
    assert tier == "FOUL_CMP" and classify(tier, "x") == ("FOUL_CMP", "model")
    # undistorted (k=1): submit == model p
    s = optimize(tier=tier, question_type="team_more_fouls", p_hat=p, shadow=0.50)
    assert abs(s.q - p) < 1e-9 and K_PRIOR[("FOUL_CMP", "model")] == 1.0

def test_sign_underdog_fouls_more():
    # Haiti heavy underdog (gap ~ -0.33) -> ABOVE the even base (~0.46)
    _, p_udog, _ = slate.resolve_row({"question_type": "team_more_fouls", "target_team": "Haiti", "line": ""},
                                     _c(0.20, 0.53), _GAME, None)
    # Haiti heavy favorite (gap ~ +0.33) -> BELOW the even base
    _, p_fav, _ = slate.resolve_row({"question_type": "team_more_fouls", "target_team": "Haiti", "line": ""},
                                    _c(0.53, 0.20), _GAME, None)
    assert p_udog > 0.46 > p_fav        # underdog more fouls; favorite fewer
    assert p_udog == round(FC.predict_more_fouls(0.20 - 0.53), 4)

def test_even_match_at_base_rate():
    _, p, _ = slate.resolve_row({"question_type": "team_more_fouls", "target_team": "Haiti", "line": ""},
                                _c(0.40, 0.40), _GAME, None)
    assert 0.44 < p < 0.48              # ~0.46 measured base rate (ties->NO)

def test_shadow_fallback_when_no_consensus():
    tier, p, _ = slate.resolve_row({"question_type": "team_more_fouls", "target_team": "Haiti", "line": ""},
                                   None, _GAME, None)
    assert tier == "PENDING" and p is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
