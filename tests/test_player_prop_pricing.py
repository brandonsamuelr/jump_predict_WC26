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

from odds_lib.player_prop_pricing import (
    price_player_prop, match_player, GLOBAL_PLAYER_PROP_OVERROUND)


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


def _ga_game(direct_books=(), anytime_books=(), player="Harry Kane",
             direct_price=-105, anytime_price=120):
    bms = []
    for b in direct_books:
        bms.append({"title": b, "markets": [{"key": "player_to_score_or_assist",
                    "outcomes": [{"name": "Yes", "description": player, "price": direct_price}]}]})
    for b in anytime_books:
        bms.append({"title": b, "markets": [{"key": "player_goal_scorer_anytime",
                    "outcomes": [{"name": "Yes", "description": player, "price": anytime_price}]}]})
    return {"bookmakers": bms}


def test_goal_or_assist_uses_direct_market_when_present():
    pr = price_player_prop("player_goal_or_assist", "Harry Kane", None,
                           _ga_game(direct_books=["DraftKings", "FanDuel"], anytime_books=["Pinnacle"]))
    assert pr.mapped and pr.api_market == "player_to_score_or_assist"
    assert pr.source == "direct" and pr.lower_bound is False


def test_goal_or_assist_falls_back_to_anytime_proxy():
    pr = price_player_prop("player_goal_or_assist", "Harry Kane", None,
                           _ga_game(anytime_books=["Pinnacle", "Bovada", "DraftKings"]))  # no direct
    assert pr.mapped and pr.api_market == "player_goal_scorer_anytime"
    assert pr.source == "proxy_floor" and pr.lower_bound is True


def test_goal_or_assist_floored_at_anytime_when_direct_lower():
    # direct priced LOW (+200 -> ~0.30 adj), anytime HIGH (-200 -> ~0.60 adj):
    # the goal-or-assist estimate must be floored at the anytime lower bound.
    pr = price_player_prop("player_goal_or_assist", "Harry Kane", None,
                           _ga_game(direct_books=["DraftKings"], anytime_books=["Pinnacle"],
                                    direct_price=200, anytime_price=-200))
    assert pr.source == "direct"
    assert abs(pr.market_prob_vig_adjusted - pr.floor_prob) < 1e-9   # floored at anytime
    assert pr.market_prob_vig_adjusted > 0.5                          # the floor, not the low direct


def test_vig_adjustment_lowers_prob():
    r = price_player_prop("player_goal", "Riyad Mahrez", None,
                          _game_both_markets())
    assert r.market_prob_vig_adjusted < r.market_prob_raw


# --- tiered de-vig (exact > same-slate prior > global), with provenance -------

def test_tier_a_exact_two_sided_scorer():
    # player quoted BOTH Yes and No -> EXACT two-sided de-vig, not a typed/prior strip
    game = {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_goal_scorer_anytime", "outcomes": [
            {"name": "Yes", "description": "Harry Kane", "price": -110},
            {"name": "No", "description": "Harry Kane", "price": -110}]}]}]}
    r = price_player_prop("player_goal", "Harry Kane", None, game)
    assert r.overround_source == "exact_two_sided" and r.status == "mapped_two_sided"
    assert abs(r.market_prob_vig_adjusted - 0.5) < 1e-3          # symmetric -> 0.5
    assert 1.03 < r.overround_used < 1.06                        # MEASURED booksum, not a constant


def test_tier_b_same_slate_market_prior():
    # target one-sided, but ANOTHER player in the same market is two-sided ->
    # use that player's booksum as the market-specific prior (auditably tagged)
    game = {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"name": "Over", "description": "Target Guy", "point": 0.5, "price": -120},
            {"name": "Over", "description": "Other Guy", "point": 0.5, "price": -110},
            {"name": "Under", "description": "Other Guy", "point": 0.5, "price": -110}]}]}]}
    r = price_player_prop("player_sot_over", "Target Guy", 0.5, game)
    assert r.overround_source == "same_slate_market_prior" and r.overround_prior_n == 1
    assert abs(r.overround_used - 1.0476) < 0.01                 # the OTHER player's booksum


def test_tier_c_global_prior_when_nothing_two_sided():
    game = {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"name": "Over", "description": "Lonely Guy", "point": 0.5, "price": -120}]}]}]}
    r = price_player_prop("player_sot_over", "Lonely Guy", 0.5, game)
    assert r.overround_source == "global_player_prop_prior"
    assert r.overround_used == GLOBAL_PLAYER_PROP_OVERROUND


def test_no_privileged_per_market_constant():
    # SOT and scorer, each one-sided-alone, get the SAME overround (the global
    # measured prior) -- the old 1.06 (SOT) vs 1.12 (scorer) privilege is gone.
    sot = {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"name": "Over", "description": "Common Name", "point": 0.5, "price": 120}]}]}]}
    scorer = {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_goal_scorer_anytime", "outcomes": [
            {"name": "Yes", "description": "Common Name", "price": 120}]}]}]}
    rs = price_player_prop("player_sot_over", "Common Name", 0.5, sot)
    rg = price_player_prop("player_goal", "Common Name", None, scorer)
    assert rs.overround_used == rg.overround_used == GLOBAL_PLAYER_PROP_OVERROUND


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
