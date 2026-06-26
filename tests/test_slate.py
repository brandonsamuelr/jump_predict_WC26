"""Guards for slate.resolve_row routing + the player-prop LINEUP GATE.

    .venv/bin/python tests/test_slate.py

Safety-critical invariant: a player prop expresses edge ONLY on a CONFIRMED
starter. Unknown / no-lineup / bench / out-of-squad all take ZERO position
(shadow, tier PENDING -> k=0), even when the prop market is liquid. This is the
"benched Ramos must not run at k=0.75" gate. PROP_thin (k=0.40) is reserved for
a confirmed starter whose market is merely illiquid.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib import slate, match_engine as E, shadow_routes as SR
from odds_lib.edge import classify
from odds_lib.lineups import MatchLineup, PlayerContext


def _game_sot_market(player="Test Player") -> dict:
    # a LIQUID shots-on-target market for the player (maps in price_player_prop)
    return {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"name": "Over", "description": player, "point": 0.5, "price": -120}]}]},
        {"title": "Bet365", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"name": "Over", "description": player, "point": 0.5, "price": -115}]}]}]}


def _lineup(status: str, player="Test Player") -> MatchLineup:
    return MatchLineup(match="X vs Y", players={player: PlayerContext(status=status)})


def _prop_row(player="Test Player"):
    return {"question_type": "player_sot_over", "target_team": "X",
            "target_player": player, "line": "0.5", "match": "X vs Y"}


# --- the lineup gate --------------------------------------------------------

def test_prop_no_lineup_is_shadow():
    # mapped, liquid market but NO lineup -> must shadow (k=0), NOT PROP_thin.
    tier, p, mp = slate.resolve_row(_prop_row(), None, _game_sot_market(), None, lineup=None)
    assert tier == "PENDING" and p is None


def test_prop_unknown_status_is_shadow():
    lu = MatchLineup(match="X vs Y", players={})  # player absent -> unknown
    tier, p, _ = slate.resolve_row(_prop_row(), None, _game_sot_market(), None, lineup=lu)
    assert tier == "PENDING" and p is None


def test_prop_bench_is_minutes_scaled_sub_not_shadow():
    # Item 4 (2026-06-26): a sub-eligible benched player gets a FOUNDED minutes-scaled true-P
    # (PROP_SUB), not a c_hat shadow. out_of_squad has no appearance -> stays PENDING.
    for st in ("bench_high_usage", "bench_low_usage", "bench_unknown"):
        tier, p, _ = slate.resolve_row(_prop_row(), None, _game_sot_market(), None, lineup=_lineup(st))
        assert tier == "PROP_SUB" and p is not None and 0.0 < p < 1.0, f"{st}: {tier} {p}"
    tier, p, _ = slate.resolve_row(_prop_row(), None, _game_sot_market(), None, lineup=_lineup("out_of_squad"))
    assert tier == "PENDING" and p is None   # no appearance -> no founded read


def test_prop_confirmed_starter_expresses():
    tier, p, _ = slate.resolve_row(_prop_row(), None, _game_sot_market(), None,
                                   lineup=_lineup("starter"))
    assert tier in ("PROP_ok", "PROP_thin") and p is not None


def test_prop_unmapped_market_is_shadow_even_for_starter():
    # no SOT market at all -> unmapped -> shadow regardless of lineup
    empty_game = {"bookmakers": []}
    tier, p, _ = slate.resolve_row(_prop_row(), None, empty_game, None,
                                   lineup=_lineup("starter"))
    assert tier == "PENDING" and p is None


def _c_h2h(fav="Senegal", dog="Iraq", pf=0.78, pd_=0.10):
    return pd.DataFrame([{"market_key": "h2h", "line": float("nan"), "outcome": fav, "market_prob": pf},
                         {"market_key": "h2h", "line": float("nan"), "outcome": dog, "market_prob": pd_}])


def test_1h_corner_underdog_uses_directional_model_not_flat_constant():
    # lopsided: underdog 1H-corners must be pushed BELOW the flat 0.389 by the favorite_gap model.
    g = {"home_team": "Senegal", "away_team": "Iraq", "bookmakers": []}
    tier, p, _ = slate.resolve_row({"question_type": "team_more_corners_h1", "target_team": "Iraq", "line": ""},
                                   _c_h2h(), g, None)
    assert tier == "CORNER_HALF_1H_FG" and p < 0.389          # direction-aware, below the blind constant
    assert classify("CORNER_HALF_1H_FG", "x") == ("CORNER_HALF", "model_1h")
    # favorite side must be the mirror (> 0.5-ish, and above the underdog)
    _, pf, _ = slate.resolve_row({"question_type": "team_more_corners_h1", "target_team": "Senegal", "line": ""},
                                 _c_h2h(), g, None)
    assert pf > p


def test_2h_corner_still_stopgap_not_favorite_tilted():
    # 2H is game-state-reversed -> must NOT use the favorite model; stays on the stopgap.
    g = {"home_team": "Senegal", "away_team": "Iraq", "bookmakers": []}
    tier, p, _ = slate.resolve_row({"question_type": "team_more_corners_2h", "target_team": "Iraq", "line": ""},
                                   _c_h2h(), g, None)
    assert tier == "CORNER_HALF_STOPGAP"


def test_offside_uncovered_uses_eb_pooled_prior_not_old_constant():
    # uncovered team -> EB pooled prior (~0.41), the n=0 limit of the per-team model, not 0.45.
    SR._load_offside_table.cache_clear()
    g = {"home_team": "Senegal", "away_team": "Iraq", "bookmakers": []}
    tier, p, _ = slate.resolve_row({"question_type": "team_offsides_over", "target_team": "Iraq", "line": "2"},
                                   None, g, None)
    assert tier == "OFFSIDES_FLOOR"
    if SR.offside_pooled_prior(2) is not None:           # table present in this env
        assert abs(p - SR.offside_pooled_prior(2)) < 1e-4 and p < 0.44


def _game_team_total(team="Norway", over=-175, under=130):
    # a two-sided team-total Over/Under 0.5 market across two books (maps in price_team_goals_over)
    mk = lambda bk: {"title": bk, "markets": [{"key": "team_totals", "outcomes": [
        {"name": "Over", "description": team, "point": 0.5, "price": over},
        {"name": "Under", "description": team, "point": 0.5, "price": under}]}]}
    return {"home_team": team, "away_team": "Z", "bookmakers": [mk("Pinnacle"), mk("Bet365")]}


def test_team_score_any_prefers_direct_team_total_market():
    # strict-equivalence: team_score_any == team-total Over 0.5 -> the DIRECT market wins over engine.
    tier, p, _ = slate.resolve_row(
        {"question_type": "team_score_any", "target_team": "Norway", "line": ""},
        None, _game_team_total("Norway"), None)
    assert tier in ("TEAMGOALS_OK", "TEAMGOALS_THIN") and 0.5 < p < 0.7   # de-vig of -175/+130


def test_team_score_any_falls_back_to_engine_without_team_total_market():
    # no team-total market -> must NOT be TEAMGOALS; with no engine model -> PENDING (engine path).
    tier, p, _ = slate.resolve_row(
        {"question_type": "team_score_any", "target_team": "Norway", "line": ""},
        None, {"home_team": "Norway", "away_team": "Z", "bookmakers": []}, None)
    assert tier not in ("TEAMGOALS_OK", "TEAMGOALS_THIN")


# --- the two newly wired routes --------------------------------------------

def test_halftime_team_win_routes_to_h2h_h1_market():
    c = pd.DataFrame([
        {"market_key": "h2h_h1", "line": float("nan"), "outcome": "Colombia", "market_prob": 0.42},
        {"market_key": "h2h_h1", "line": float("nan"), "outcome": "Draw", "market_prob": 0.40},
        {"market_key": "h2h_h1", "line": float("nan"), "outcome": "DR Congo", "market_prob": 0.18},
    ])
    row = {"question_type": "halftime_team_win", "target_team": "Colombia", "line": ""}
    tier, p, mp = slate.resolve_row(row, c, {}, None)
    assert tier == "MARKET" and abs(p - 0.42) < 1e-9


def test_total_goals_2h_over_routes_to_engine():
    m = E.calibrate("H", "A", p_home=0.83, p_over=0.57)
    sim = E.simulate(m, n=120_000, seed=4)
    model_tuple = (m, sim, "H", "A")   # legacy 4-tuple, no totals_h1 -> constant fallback
    row = {"question_type": "total_goals_2h_over", "target_team": "", "line": "1.5"}
    tier, p, _ = slate.resolve_row(row, None, {}, model_tuple)
    assert tier == "ENGINE_GOALS_H1FALLBACK" and 0.0 < p < 1.0   # flagged constant fallback
    assert classify(tier, "total_goals_2h_over") == ("ENGINE", "engine")  # still routes as engine
    assert abs(p - E.p_total_goals_2h_over(sim, 2)) < 1e-3   # resolve_row rounds to 4dp


# --- goal-or-assist direct-market routing -----------------------------------

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


def _ga_row():
    return {"question_type": "player_goal_or_assist", "target_team": "England",
            "target_player": "Harry Kane", "line": "", "match": "England vs Ghana"}


def test_goa_direct_thin_is_not_prop_ok():
    # 2-book (no sharp) direct market -> exact but thin -> PROP_direct_thin, NOT PROP_ok
    g = _ga_game(direct_books=["DraftKings", "FanDuel"], anytime_books=["Pinnacle"])
    tier, p, _ = slate.resolve_row(_ga_row(), None, g, None, lineup=_lineup("starter", "Harry Kane"))
    assert tier == "PROP_direct_thin"


def test_goa_direct_liquid_is_prop_ok():
    g = _ga_game(direct_books=["Pinnacle", "DraftKings", "FanDuel"], anytime_books=["Bovada"])
    tier, p, _ = slate.resolve_row(_ga_row(), None, g, None, lineup=_lineup("starter", "Harry Kane"))
    assert tier == "PROP_ok"


def test_goa_proxy_floor_tier_and_clamp_integration():
    # no direct market -> proxy floor; the caller derives the clamp from the tier.
    g = _ga_game(anytime_books=["Pinnacle", "Bovada", "DraftKings"])
    tier, p, _ = slate.resolve_row(_ga_row(), None, g, None, lineup=_lineup("starter", "Harry Kane"))
    assert tier == "PROP_proxy_floor"
    from odds_lib.optimizer import optimize
    s = optimize(tier=tier, question_type="player_goal_or_assist", p_hat=p, shadow=0.30, k=0.40,
                 lower_bound=(tier == "PROP_proxy_floor"))
    assert s.lower_bound_clamped and abs(s.q - p) < 1e-9   # blend pulled below floor -> clamped to p_hat


def test_resolve_distinguishes_direct_vs_floor():
    starter = _lineup("starter", "Harry Kane")
    t_direct, _, _ = slate.resolve_row(_ga_row(), None,
        _ga_game(direct_books=["DraftKings", "FanDuel"], anytime_books=["Pinnacle"]), None, lineup=starter)
    t_floor, _, _ = slate.resolve_row(_ga_row(), None,
        _ga_game(anytime_books=["Pinnacle", "Bovada", "DraftKings"]), None, lineup=starter)
    assert t_direct in ("PROP_ok", "PROP_direct_thin") and t_floor == "PROP_proxy_floor"
    assert t_direct != t_floor


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
