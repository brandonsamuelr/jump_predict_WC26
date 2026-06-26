"""Guards for the FOUNDED per-team offside rate + its three non-negotiable conditions.

    .venv/bin/python tests/test_offside_team_rates.py

The per-team offside rate is a MEASURED, OOS-gated, refreshable parameter -- same category
as the cards 2H-share or the SOT scalar. These tests pin the conditions the table must hold:
  1. MEASURED-NOT-TYPED: the shipped probability is the Poisson tail of the stored per-team
     lambda; change the lambda -> the output changes (no hardcoded constant).
  2. UNCOVERED -> FLOOR: a team with no measured history routes to the pooled OFFSIDES_FLOOR
     (the founded floor), never a guessed default.
  3. REFRESHES NOT OSSIFIES: the route reads the table live; a new table -> new numbers.
Plus: covered team routes to OFFSIDES_TEAM, classify/prior/trust ship it raw at k=1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib import slate
from odds_lib import shadow_routes as SR
from odds_lib.edge import classify, K_PRIOR, TRUST_PRICE_K

GAME = {"home_team": "France", "away_team": "Curacao", "bookmakers": []}


def _row(team):
    return {"question_type": "team_offsides_over", "target_team": team, "line": "2"}


GATE_OK = {"beats_floor": True}


def _with_table(tbl):
    """Point the offside loader at a synthetic table (restored by the caller)."""
    tbl.setdefault("oos_gate", GATE_OK)
    orig = SR._load_offside_table
    SR._load_offside_table = lambda *a, **k: tbl
    return orig


def test_measured_not_typed_and_refreshes():
    # condition 1 + 3: output is the Poisson tail of the STORED lambda, and tracks the table.
    orig = _with_table({"teams": {"testland": {"display": "Testland", "lambda_hat": 2.0}}})
    try:
        p = SR.offside_team_rate("Testland", 2)
        assert abs(p - SR._poisson_sf_ge(2, 2.0)) < 1e-4      # measured (4dp), not a constant
        # refresh: a different lambda must give a different number (not ossified)
        SR._load_offside_table = lambda *a, **k: {"oos_gate": GATE_OK, "teams": {"testland": {"lambda_hat": 1.0}}}
        p2 = SR.offside_team_rate("Testland", 2)
        assert abs(p2 - SR._poisson_sf_ge(2, 1.0)) < 1e-4 and p2 < p
    finally:
        SR._load_offside_table = orig


def test_failed_gate_self_disables_to_floor():
    # live invariant: a table that did NOT beat the floor OOS -> every team falls to the floor.
    orig = SR._load_offside_table
    SR._load_offside_table = lambda *a, **k: {"oos_gate": {"beats_floor": False},
                                              "teams": {"france": {"lambda_hat": 1.27}}}
    try:
        assert SR.offside_team_rate("France", 2) is None      # gate failed -> no per-team read
        tier, p, _ = slate.resolve_row(_row("France"), None, GAME, None)
        assert tier == "OFFSIDES_FLOOR" and p == round(SR.offsides_rate(2), 4)
    finally:
        SR._load_offside_table = orig


def test_covered_team_routes_to_team_tier():
    orig = _with_table({"teams": {"france": {"display": "France", "lambda_hat": 1.27}}})
    try:
        tier, p, _ = slate.resolve_row(_row("France"), None, GAME, None)
        assert tier == "OFFSIDES_TEAM" and abs(p - SR._poisson_sf_ge(2, 1.27)) < 1e-4
    finally:
        SR._load_offside_table = orig


def test_uncovered_team_falls_to_founded_floor_not_a_guess():
    # condition 2: a team absent from the table -> OFFSIDES_FLOOR == the pooled measured rate.
    orig = _with_table({"teams": {"france": {"lambda_hat": 1.27}}})
    try:
        tier, p, _ = slate.resolve_row(_row("Curacao"), None, GAME, None)
        assert tier == "OFFSIDES_FLOOR"
        assert p == round(SR.offsides_rate(2), 4)             # the founded floor, not a default
    finally:
        SR._load_offside_table = orig


def test_offside_threshold_2_and_1p5_agree_as_p_ge_2():
    # contest "2 or more times" == P(X>=2); line='2' and line='1.5' must NOT diverge (threshold
    # footgun: both representations of "2+" appear across the codebase).
    SR._load_offside_table.cache_clear()
    g = {"home_team": "France", "away_team": "Curacao", "bookmakers": []}
    for tgt in ("France", "Curacao"):          # covered (per-team) + uncovered (EB prior)
        ps = []
        for ln in ("2", "1.5"):
            _, p, _ = slate.resolve_row({"question_type": "team_offsides_over",
                                         "target_team": tgt, "line": ln}, None, g, None)
            ps.append(p)
        assert ps[0] is not None and abs(ps[0] - ps[1]) < 1e-9, f"{tgt}: line2={ps[0]} line1.5={ps[1]}"


def test_team_tier_ships_raw_k1_and_trusted():
    assert classify("OFFSIDES_TEAM", "x") == ("OFFSIDES", "team")
    assert K_PRIOR[("OFFSIDES", "team")] == 1.0
    assert ("OFFSIDES", "team") in TRUST_PRICE_K          # measured founded rate -> never shrunk


def test_real_table_covers_majors_and_floors_minnows():
    # integration against the DEPLOYED table: a major gets a per-team read, a minnow hits the floor.
    SR._load_offside_table.cache_clear()
    if not (SR._load_offside_table().get("teams")):
        return  # table not fit in this env -> skip (fit script generates it)
    tier_major, _, _ = slate.resolve_row(_row("Argentina"), None, GAME, None)
    tier_minnow, p_minnow, _ = slate.resolve_row(_row("Curacao"), None, GAME, None)
    assert tier_major == "OFFSIDES_TEAM"
    # uncovered minnow -> EB pooled prior (n=0 limit of the same model), not the legacy 0.45 floor
    assert tier_minnow == "OFFSIDES_FLOOR" and abs(p_minnow - SR.offside_pooled_prior(2)) < 1e-4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
