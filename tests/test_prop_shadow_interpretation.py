"""Interpretation guard: the flat overround candidates (1.045/1.06/1.10/1.12) are SHADOW
COMPARISON BASELINES ONLY -- to confirm tiered de-vig beats every flat-constant alternative on
outcomes -- and NEVER a production selection.

Enforced invariants:
  1. Production de-vig (devig_tiered) always chooses a TIERED source
     (exact_two_sided | same_slate_market_prior | global_player_prop_prior), never a flat constant
     picked from the candidate list.
  2. The candidates appear in the shadow only as `p@<c>` baseline columns, separate from the
     production `tiered_p` / `overround_source`.
  3. Shadow evaluation is READ-ONLY: even when a flat candidate "wins" the outcome comparison,
     nothing mutates the production constant -- promotion would require a separate, deliberate review.
  4. The shadow module exposes NO promotion/selection API, and the production de-vig code does not
     reference the candidate list at all.
"""
from __future__ import annotations

import csv
import inspect
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib import prop_devig_shadow as PDS
from odds_lib import player_prop_pricing as PPP
from odds_lib.player_prop_pricing import (
    devig_tiered, GLOBAL_PLAYER_PROP_OVERROUND, SHADOW_OVERROUND_CANDIDATES)

TIERED_SOURCES = {"exact_two_sided", "same_slate_market_prior", "global_player_prop_prior"}
SB = ("Pinnacle",)


def _exact_game(p="Test Player"):
    return {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_goal_scorer_anytime", "outcomes": [
            {"name": "Yes", "description": p, "price": -110},
            {"name": "No", "description": p, "price": -110}]}]}]}


def _one_sided_sot(p="Test Player"):
    return {"bookmakers": [{"title": "Pinnacle", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"name": "Over", "description": p, "point": 0.5, "price": -120}]}]}]}


def test_production_chooses_tiered_source_never_a_flat_candidate():
    r_exact, _ = devig_tiered(_exact_game(), "player_goal_scorer_anytime", "Test Player", None, SB)
    assert r_exact["source"] == "exact_two_sided" and r_exact["source"] in TIERED_SOURCES
    r_one, _ = devig_tiered(_one_sided_sot(), "player_shots_on_target", "Test Player", 0.5, SB)
    # one-sided -> global prior selected BY TIER (not by shadow-Brier). Value is the measured prior.
    assert r_one["source"] == "global_player_prop_prior"
    assert r_one["overround"] == GLOBAL_PLAYER_PROP_OVERROUND


def test_candidates_are_baseline_columns_and_production_overround_is_measured():
    for c in SHADOW_OVERROUND_CANDIDATES:
        assert f"p@{c}" in PDS.FIELDS                       # baselines logged as p@<c>
    assert "tiered_p" in PDS.FIELDS and "overround_source" in PDS.FIELDS
    # an exact-two-sided row's PRODUCTION overround is a measured booksum (~1.0476), NOT a flat pick
    r_exact, _ = devig_tiered(_exact_game(), "player_goal_scorer_anytime", "Test Player", None, SB)
    assert round(r_exact["overround"], 4) not in set(SHADOW_OVERROUND_CANDIDATES)


def test_no_promotion_or_selection_api_in_shadow_module():
    banned = ("promote", "select", "winning", "choose", "set_overround", "best_constant",
              "best_candidate", "apply_overround", "pick")
    for name in dir(PDS):
        if name.startswith("_"):
            continue
        if callable(getattr(PDS, name)):
            assert not any(b in name.lower() for b in banned), f"{name}: looks like a promotion path"


def test_evaluate_is_readonly_no_writeback_even_when_a_flat_constant_wins():
    before = PPP.GLOBAL_PLAYER_PROP_OVERROUND
    with tempfile.TemporaryDirectory() as d:
        sh = Path(d) / "shadow.csv"
        # craft rows where a FLAT candidate (p@1.12) predicts outcomes PERFECTLY and tiered is bad,
        # i.e. the flat constant "wins" the comparison -- production must STILL not change.
        rows = []
        for i, y in enumerate([1, 1, 0, 0]):
            row = {f: "" for f in PDS.FIELDS}
            row.update({"match": "T v U", "question_number": f"Q{i}", "target_player": "P",
                        "question_type": "player_goal", "y": y, "raw": 0.5,
                        "overround_source": "exact_two_sided", "overround_used": 1.0476,
                        "overround_prior_n": 1, "tiered_p": 0.5, "prop_band": "mid"})
            for c in SHADOW_OVERROUND_CANDIDATES:
                row[f"p@{c}"] = (float(y) if c == 1.12 else 0.5)   # 1.12 perfect, others poor
            rows.append(row)
        with sh.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PDS.FIELDS); w.writeheader(); w.writerows(rows)
        ret = PDS.evaluate(shadow_path=sh)               # prints "tiered beats every flat? NO"
    assert ret is None                                   # no 'chosen constant' is returned
    assert PPP.GLOBAL_PLAYER_PROP_OVERROUND == before    # NO write-back to production


def test_production_devig_code_does_not_reference_candidate_list():
    src = inspect.getsource(devig_tiered)
    assert "SHADOW_OVERROUND_CANDIDATES" not in src       # candidates never enter the pricing path


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
