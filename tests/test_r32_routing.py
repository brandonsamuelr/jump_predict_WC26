"""Guards for the R32 routing table (Deliverable 1 + the skip rule).

    .venv/bin/python tests/test_r32_routing.py

Enforces: provenance completeness (no blank field is a hard failure), advance != 90' 1X2,
win-in-regulation/tie != advance, player props route to tiered de-vig (SOT not legacy 1.06),
hydration market-vs-model split, LOW_CONFIDENCE_PRIOR names a base rate (never a silent 0.50),
no crowd probability as a pricing input, and the skip path keys on FLAGGED directional bias
(not mere uncertainty).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.r32_routing import build_table, validate, REQUIRED_FIELDS, submit_decision

T = build_table()


def test_all_60_rows_and_validation_clean():
    assert len(T) == 60
    assert validate(T) == [], validate(T)


def test_provenance_completeness_no_blank_field():
    for r in T:
        for f in REQUIRED_FIELDS:
            assert str(r.get(f, "")).strip(), f"{r['match']} {r['question_number']} blank {f}"


def test_advance_rows_never_bare_1x2():
    adv = [r for r in T if "advance" in r["question_text"].lower()]
    assert len(adv) == 2
    for r in adv:
        assert r["route_class"] == "ADVANCE_MARKET"
        assert "ET" in r["settlement_scope"] or "penalt" in r["settlement_scope"].lower()
        assert "derived" in r["market_source"].lower() or "to_qualify" in r["market_source"].lower()
        # never sourced from bare 90' 1X2
        assert not (r["market_source"].lower().strip() in ("h2h", "1x2", "h2h(1x2,90')"))


def test_win_in_regulation_and_tie_are_regulation_not_advance():
    hits = [r for r in T if "win in regulation" in r["question_text"].lower()
            or "ends in a tie" in r["question_text"].lower()]
    assert hits
    for r in hits:
        assert r["route_class"] == "MARKET_EXACT"
        assert "ET" not in r["settlement_scope"]


def test_player_props_route_tiered_and_sot_not_legacy_106():
    pr = [r for r in T if r["route_class"] == "PLAYER_PROP_TIERED_DEVIG"]
    assert len(pr) >= 7
    for r in pr:
        assert "tiered_devig" in r["overround_source"]
    sot = [r for r in pr if "SOT" in r["question_text"]]
    assert sot
    for r in sot:
        # one-sided SOT uses the empirical global prior (1.045), explicitly NOT legacy 1.06.
        src = r["overround_source"]
        assert ("global" in src and "1.045" in src) or "model fallback" in src
        assert "NOT 1.06" in src or "1.06" not in src    # never silently the legacy strip


def test_hydration_rows_split_market_vs_model():
    for r in T:
        q = r["question_text"].lower()
        if "hydration break" not in q:
            continue
        if "goal" in q or "corner" in q:
            assert r["route_class"] == "TIME_WINDOW_MARKET_COMPONENT"
        elif "offside" in q or "card" in q:
            assert r["route_class"] == "TIME_WINDOW_MODEL_COMPONENT"


def test_low_confidence_prior_names_base_rate_no_silent_half():
    lcp = [r for r in T if r["route_class"] == "LOW_CONFIDENCE_PRIOR"]
    assert lcp
    for r in lcp:
        assert "base rate" in r["component_source"].lower()
        assert "0.50" not in r["component_source"] and "0.5" not in r["component_source"]


def test_no_crowd_probability_as_pricing_input():
    # the SOURCE fields never use the crowd as an input (submit_reason may mention crowd-quality)
    for r in T:
        for f in ("market_source", "component_source", "overround_source", "time_window_source"):
            assert "crowd" not in str(r[f]).lower()


def test_default_submit_and_skip_path_keys_on_flagged_bias():
    assert all(r["submit_recommendation"] == "submit" for r in T)   # 60/60 live rows submit
    # the skip path EXISTS and keys on flagged directional bias, not uncertainty:
    assert submit_decision("TIME_WINDOW_MODEL_COMPONENT", True)[0] == "skip"
    assert submit_decision("TIME_WINDOW_MODEL_COMPONENT", False)[0] == "submit"   # uncertain != skip
    assert submit_decision("UNSUPPORTED_SKIP", False)[0] == "skip"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
