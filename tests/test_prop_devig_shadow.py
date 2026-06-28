"""Guards for the auto-capture / auto-harvest prop de-vig shadow.

    .venv/bin/python tests/test_prop_devig_shadow.py

Invariants: capture is lock-time + starter-only + last-write-wins; harvest joins
outcomes idempotently; the shadow row carries provenance + every flat candidate.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib import prop_devig_shadow as PDS


def _scorer_game(player="Harry Kane", yes=-110, no=-110):
    return {"home_team": "England", "away_team": "X", "bookmakers": [
        {"title": "Pinnacle", "markets": [
            {"key": "player_goal_scorer_anytime", "outcomes": [
                {"name": "Yes", "description": player, "price": yes},
                {"name": "No", "description": player, "price": no}]}]}]}


def _row(qn="Q9", player="Harry Kane"):
    return {"question_number": qn, "question_type": "player_goal",
            "target_player": player, "line": ""}


def test_capture_only_starter_market_props():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pending.csv"
        assert PDS.capture_pending("England v X", _row(), _scorer_game(), "PROP_ok", path=p) is True
        assert PDS.capture_pending("England v X", _row("Q1"), _scorer_game(), "PROP_SUB", path=p) is False
        assert PDS.capture_pending("England v X", _row("Q2"), _scorer_game(), "PENDING", path=p) is False
        rows = list(csv.DictReader(p.open()))
        assert len(rows) == 1 and rows[0]["question_number"] == "Q9"
        assert rows[0]["overround_source"] == "exact_two_sided"     # two-sided -> exact
        assert rows[0]["y"] == ""                                   # outcome unknown at lock
        assert rows[0]["p@1.12"] and rows[0]["p@1.045"]            # candidates recorded


def test_capture_last_write_wins():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pending.csv"
        PDS.capture_pending("England v X", _row(), _scorer_game(yes=-110, no=-110), "PROP_ok", path=p)
        PDS.capture_pending("England v X", _row(), _scorer_game(yes=200, no=-260), "PROP_ok", path=p)
        rows = list(csv.DictReader(p.open()))
        assert len(rows) == 1                                       # upsert, not duplicate
        assert float(rows[0]["tiered_p"]) < 0.45                    # refreshed to the new (lower) read


def _fake_log(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["match", "question_number", "result"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_harvest_joins_outcome_and_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        pend = Path(d) / "pending.csv"; log = Path(d) / "log.csv"; sh = Path(d) / "shadow.csv"
        PDS.capture_pending("England v X", _row("Q9"), _scorer_game(), "PROP_ok", path=pend)
        PDS.capture_pending("England v X", _row("Q10", "Bukayo Saka"), _scorer_game("Bukayo Saka"),
                            "PROP_ok", path=pend)
        # only Q9 resolved
        _fake_log(log, [{"match": "England v X", "question_number": "Q9", "result": "Yes"}])
        n = PDS.harvest(log_path=log, pending_path=pend, shadow_path=sh)
        assert n == 1
        rows = list(csv.DictReader(sh.open()))
        assert len(rows) == 1 and rows[0]["question_number"] == "Q9" and rows[0]["y"] == "1"
        # re-run: idempotent (Q9 already harvested, Q10 still unresolved)
        assert PDS.harvest(log_path=log, pending_path=pend, shadow_path=sh) == 0
        # Q10 resolves -> harvested on next pass, Q9 not duplicated
        _fake_log(log, [{"match": "England v X", "question_number": "Q9", "result": "Yes"},
                        {"match": "England v X", "question_number": "Q10", "result": "No"}])
        assert PDS.harvest(log_path=log, pending_path=pend, shadow_path=sh) == 1
        assert len(list(csv.DictReader(sh.open()))) == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
