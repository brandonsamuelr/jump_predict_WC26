"""Guard for the full-name player-identity rule, built around the Luka/Petar Sučić failure.

    .venv/bin/python tests/test_player_identity.py

Rule: a player prop MUST resolve to a unique FULL-NAME match; surname-only, surname-collision, or
no-match must FLAG (never assume). The exact failure: question 'Luka Sučić', XI shows only a
'Sučić' who is actually Petar -> must flag, not price Luka as a starter.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.player_identity import require_full_name


def test_the_sucic_trap_xi_has_a_different_sucic():
    # the actual disaster: question = Luka Sučić; XI lists Petar Sučić (starter), Luka not in the XI
    xi = ["Petar Sučić", "Luka Modrić", "Mateo Kovačić", "Ante Budimir"]
    r = require_full_name("Luka Sučić", xi)
    assert r["ok"] is False and r["flag"] == "SURNAME_ONLY_MATCH"   # do NOT price Luka as the XI 'Sučić'


def test_surname_collision_both_present_flags():
    pool = ["Petar Sučić", "Luka Sučić", "Luka Modrić"]
    # querying just the surname must never resolve
    r = require_full_name("Sučić", pool)
    assert r["ok"] is False and r["flag"] == "QUERY_NOT_FULL_NAME"


def test_unique_full_name_match_ok_but_twin_flagged():
    # both Sučićs in the pool, querying the full name -> matches, but flags the twin for feed-verify
    pool = ["Petar Sučić", "Luka Sučić"]
    r = require_full_name("Luka Sučić", pool)
    assert r["ok"] is True and r["matched"] == "Luka Sučić"
    assert r["flag"] == "SURNAME_TWIN_PRESENT"                      # settlement/data disambiguation risk


def test_clean_unique_full_name_no_twin():
    r = require_full_name("Harry Kane", ["Harry Kane", "Bukayo Saka", "Cole Palmer"])
    assert r["ok"] is True and r["matched"] == "Harry Kane" and r["flag"] is None


def test_not_found_does_not_assume_starter():
    r = require_full_name("José Fajardo", ["Tomás Rodríguez", "Cecilio Waterman"])
    assert r["ok"] is False and r["flag"] == "NOT_FOUND"           # benched/not quoted -> never starter


def test_accent_insensitive():
    r = require_full_name("Luka Sucic", ["Luka Sučić", "Petar Sučić"])
    assert r["ok"] is True and r["flag"] == "SURNAME_TWIN_PRESENT"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
