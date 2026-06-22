"""Regression guards for the one-sided player-prop pricing handler.

Runnable directly (no pytest needed):
    .venv/bin/python tests/test_player_prop_pricing.py

The critical invariants: prop types are NEVER cross-mapped (anytime-scorer
is not SOT, full-match SOT is not 2H-SOT), and ambiguous player names are
not guessed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.player_prop_pricing import price_player_prop, match_player


def _game_both_markets() -> dict:
    return {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_goal_scorer_anytime", "outcomes": [
            {"name": "Yes", "description": "Riyad Mahrez", "price": 150}]},
        {"key": "player_shots_on_target", "outcomes": [
            {"name": "Over", "description": "Riyad Mahrez", "point": 0.5,
             "price": -120}]},
    ]}]}


def test_2h_sot_never_maps():
    r = price_player_prop("player_sot_2h_over", "Riyad Mahrez", 0.5,
                          _game_both_markets())
    assert not r.mapped
    assert r.status == "unsupported_no_equivalent_market"


def test_player_goal_uses_scorer_market():
    r = price_player_prop("player_goal", "Riyad Mahrez", None,
                          _game_both_markets())
    assert r.mapped and r.api_market == "player_goal_scorer_anytime"


def test_player_sot_uses_sot_market():
    r = price_player_prop("player_sot_over", "Riyad Mahrez", 0.5,
                          _game_both_markets())
    assert r.mapped and r.api_market == "player_shots_on_target"


def test_player_goal_no_fallback_to_sot():
    game = {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"name": "Over", "description": "Riyad Mahrez", "point": 0.5,
             "price": -120}]}]}]}
    r = price_player_prop("player_goal", "Riyad Mahrez", None, game)
    assert not r.mapped and r.status == "unsupported_market_absent"


def test_ambiguous_surname_not_guessed():
    assert match_player("Mohanad Ali",
                        ["Hussein Ali", "Ali Al-Hamadi", "William Saliba"]) is None
    assert match_player("Mohanad Ali",
                        ["Hussein Ali", "Mohanad Ali"]) == "Mohanad Ali"


def test_accent_insensitive_match():
    assert match_player("Sadio Mané", ["Sadio Mane", "Erling Haaland"]) == "Sadio Mane"


def test_shared_first_name_token_not_matched():
    # Regression: "Moussa Al Taamari" must NOT match "Anis Hadj Moussa" (a
    # different player who merely shares the 'Moussa' token), nor the
    # differently-spelled "Musa Al Tamari". This was a non-deterministic
    # cross-player mismatch via a set-ordered surname fallback.
    cands = ["Anis Hadj Moussa", "Musa Al Tamari", "Riyad Mahrez"]
    assert match_player("Moussa Al Taamari", cands) is None


def test_match_is_deterministic():
    cands = ["Anis Hadj Moussa", "Musa Al Tamari", "Mohanad Ali", "Hussein Ali"]
    results = {match_player("Moussa Al Taamari", cands) for _ in range(50)}
    assert results == {None}


def test_vig_adjustment_lowers_prob():
    r = price_player_prop("player_goal", "Riyad Mahrez", None,
                          _game_both_markets())
    assert r.market_prob_vig_adjusted < r.market_prob_raw


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
