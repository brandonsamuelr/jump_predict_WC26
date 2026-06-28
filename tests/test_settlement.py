"""Guards for the settlement-scope classifier + scope gate.

    .venv/bin/python tests/test_settlement.py

Enforces: a 3-way-with-Draw is REGULATION even under a 'to win' title; advance 2-way is
ADVANCE_ET_PENS; the 'advance to knockout stages' group trap is caught; and the gate refuses to let
an ET+pens advance market feed a regulation row (or vice versa), or a group-qualification market
price an advance-to-R16 row.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.settlement import (
    classify_market, valid_source_for, can_apply_external_advance,
    assert_external_advance_target, target_in_feed)
from odds_lib.r32_routing import ROUTE_CLASSES


def test_three_way_with_draw_is_regulation_even_if_title_says_to_win():
    assert classify_market(["Canada", "Draw", "South Africa"], title="Canada to win") == "REGULATION_3WAY"
    assert classify_market(["South Africa", "Canada", "Draw"]) == "REGULATION_3WAY"


def test_two_way_advance_is_et_pens():
    assert classify_market(["Canada", "South Africa"], title="To Advance") == "ADVANCE_ET_PENS"
    assert classify_market(["Brazil", "Japan"], title="Team to qualify to next round") == "ADVANCE_ET_PENS"


def test_group_stage_qualification_trap_is_caught():
    # Polymarket "advance to knockout STAGES" = group->R32, NOT win-the-R32-tie
    assert classify_market(["Yes", "No"], title="Team to advance to knockout stages") == "GROUP_STAGE_QUALIFICATION"


def test_draw_no_bet_and_two_way_unknown_and_outright():
    assert classify_market(["Canada", "South Africa"], title="Draw No Bet") == "DRAW_NO_BET"
    assert classify_market(["Canada", "South Africa"], title="Match Winner") == "TWO_WAY_WINNER_UNKNOWN"
    assert classify_market(["Brazil"], title="World Cup Outright Winner") == "OUTRIGHT_FUTURE"


def test_gate_blocks_advance_market_feeding_regulation_row():
    ok, _ = valid_source_for("regulation", "ADVANCE_ET_PENS")
    assert ok is False                                   # never leak ET/pens into a regulation row
    assert valid_source_for("regulation", "REGULATION_3WAY")[0] is True
    assert valid_source_for("regulation", "DRAW_NO_BET")[0] is False
    assert valid_source_for("regulation", "TWO_WAY_WINNER_UNKNOWN")[0] is False


def test_gate_blocks_regulation_and_group_qual_feeding_advance_row():
    assert valid_source_for("advance", "ADVANCE_ET_PENS")[0] is True
    assert valid_source_for("advance", "REGULATION_3WAY")[0] is False          # advance != 90' market
    assert valid_source_for("advance", "GROUP_STAGE_QUALIFICATION")[0] is False  # the Polymarket trap


def test_external_advance_quarantined_to_advance_market_only():
    # the HARD rule: external advance price may update ONLY ADVANCE_MARKET, never any other route
    assert can_apply_external_advance("ADVANCE_MARKET") is True
    for rc in ROUTE_CLASSES:
        if rc == "ADVANCE_MARKET":
            continue
        assert can_apply_external_advance(rc) is False, f"{rc} must not accept an external advance price"
    assert_external_advance_target("ADVANCE_MARKET")          # ok, no raise
    for rc in ("MARKET_EXACT", "ENGINE_FOUNDED", "PLAYER_PROP_TIERED_DEVIG",
               "MARKET_COMPONENT_TRANSFORM", "TIME_WINDOW_MARKET_COMPONENT"):
        try:
            assert_external_advance_target(rc)
            assert False, f"external advance must be rejected for regulation route {rc}"
        except ValueError:
            pass


def test_orientation_target_resolves_by_name_not_position():
    # feed is South Africa (home) vs Canada (away) -- flipped vs our 'Canada vs South Africa' label
    game = {"home_team": "South Africa", "away_team": "Canada"}
    assert target_in_feed(game, "Canada") == "Canada"          # found regardless of position
    assert target_in_feed(game, "South Africa") == "South Africa"
    assert target_in_feed(game, "Mexico") is None              # absent -> stale/wrong label, don't price


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
