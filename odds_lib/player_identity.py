"""Player-identity guard: a player prop MUST resolve to a FULL NAME. Never assume from a surname.

The Luka/Petar Sučić disaster (group-stage finale): the contest question said "Luka Sučić", the
pasted XI showed "Sučić", and the squad had BOTH Petar Sučić (STARTING) and Luka Sučić (BENCHED).
Assuming the XI's "Sučić" was Luka priced a benched player as a starter on the biggest-stake row on
the board -- a locked, irreversible loss. The question gave the first name; there was nothing to
assume. This guard turns that into an explicit FLAG, never a silent guess.

Rule (matches the standing instruction): when searching odds / gating a player prop, match on the
FULL name. If the name is a surname only, OR the surname collides with another player in the pool,
OR there is no exact full-name match, FLAG it to the operator -- do NOT pick one. This is a
data-integrity flag (hard-QA), NOT a permission question, so it is always surfaced.
"""
from __future__ import annotations

import re
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z ]", " ", s).split())


def _tokens(s: str) -> list[str]:
    return _norm(s).split()


def _surname(s: str) -> str:
    t = _tokens(s)
    return t[-1] if t else ""


def require_full_name(query: str, candidates) -> dict:
    """Resolve a contest player's FULL NAME against a candidate pool (lineup XI+subs, or the market's
    quoted names). Returns {ok, matched, flag, reason}. ``ok`` is True ONLY on a unique exact
    full-name match; every shortfall is flagged for the operator, never silently resolved.

    flags: QUERY_NOT_FULL_NAME | SURNAME_COLLISION | SURNAME_ONLY_MATCH | NOT_FOUND |
           SURNAME_TWIN_PRESENT (ok=True but a same-surname twin exists -> verify the data feed)."""
    cands = [c for c in (candidates or []) if str(c).strip()]
    if len(_tokens(query)) < 2:
        return {"ok": False, "matched": None, "flag": "QUERY_NOT_FULL_NAME",
                "reason": f"'{query}' is not a full name (need first + last) -> confirm which player, do not assume"}
    qn = _norm(query)
    sur = _surname(query)
    exact = [c for c in cands if _norm(c) == qn]
    same_surname = [c for c in cands if _surname(c) == sur]
    if len(exact) == 1:
        twin = len([c for c in same_surname if _norm(c) != qn]) > 0
        return {"ok": True, "matched": exact[0],
                "flag": "SURNAME_TWIN_PRESENT" if twin else None,
                "reason": (f"full-name match, but another '{sur}' is in the pool {same_surname} -> "
                           f"verify the settlement/data feed disambiguates" if twin
                           else "unique full-name match")}
    if len(same_surname) > 1:
        return {"ok": False, "matched": None, "flag": "SURNAME_COLLISION",
                "reason": f"multiple '{sur}' in pool {same_surname} and no exact full-name match for "
                          f"'{query}' -> confirm which, do NOT assume"}
    if len(same_surname) == 1:
        return {"ok": False, "matched": None, "flag": "SURNAME_ONLY_MATCH",
                "reason": f"only a surname match ('{same_surname[0]}') for '{query}' -> the XI/market "
                          f"may be a DIFFERENT same-surname player; confirm before pricing"}
    return {"ok": False, "matched": None, "flag": "NOT_FOUND",
            "reason": f"'{query}' not in the pool (benched / not quoted) -> do not price as a starter"}


__all__ = ["require_full_name"]
