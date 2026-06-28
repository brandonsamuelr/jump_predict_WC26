"""Per-slate SHARP anchor for the load-bearing regulation favorite_gap.

The regulation 3-way favorite_gap propagates across ~13 favorite-conditional rows (team goals, BTTS,
totals splits, HT lead, first goal, corner/card comparisons, prop context), so its precision matters
more than on any single row. This reports the de-vigged favorite_gap from THREE source groups
SEPARATELY -- all-book median, Pinnacle, and the betting exchanges -- because a pooled "sharp
consensus" can hide the Pinnacle-vs-exchange split that is itself the signal (on CAN-RSA, Pinnacle
agreed with the all-book median while the exchanges ran ~2pts hot).

Anchor policy: prefer Pinnacle; flag if Pinnacle diverges materially (>=0.02) from the all-book
median, and separately flag a Pinnacle-vs-exchange split as a caution. Regulation h2h ONLY -- never
an ET+pens advance market (see settlement.py).
"""
from __future__ import annotations

import statistics

PINNACLE = {"pinnacle"}
EXCHANGES = {"betfair_ex_uk", "betfair_ex_eu", "betfair_ex_au", "smarkets", "matchbook"}
MATERIAL = 0.02


def _implied(american) -> float:
    p = float(american)
    return 100.0 / (p + 100.0) if p > 0 else (-p) / (-p + 100.0)


def _med(xs):
    return round(statistics.median(xs), 3) if xs else None


def favorite_gap_by_source(event: dict, team_a: str, team_b: str) -> dict:
    """De-vigged 3-way favorite_gap = P(team_a) - P(team_b), reported per source group SEPARATELY.

    team_a / team_b must match the FEED outcome labels (orientation-independent). Returns medians +
    pairwise gaps + an anchor recommendation with flags."""
    buckets = {"all": [], "pinnacle": [], "exchange": []}
    for bk in event.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            d = {o.get("name"): _implied(o["price"]) for o in m.get("outcomes", [])}
            if not ({team_a, team_b, "Draw"} <= set(d)):
                continue
            s = sum(d.values())
            gap = d[team_a] / s - d[team_b] / s
            buckets["all"].append(gap)
            if bk.get("key") in PINNACLE:
                buckets["pinnacle"].append(gap)
            if bk.get("key") in EXCHANGES:
                buckets["exchange"].append(gap)
    am, pin, exc = _med(buckets["all"]), _med(buckets["pinnacle"]), _med(buckets["exchange"])
    out = {
        "all_book_median": am, "n_all": len(buckets["all"]),
        "pinnacle": pin, "n_pinnacle": len(buckets["pinnacle"]),
        "exchange_median": exc, "n_exchange": len(buckets["exchange"]),
        "pinnacle_minus_allbook": (round(pin - am, 3) if pin is not None and am is not None else None),
        "exchange_minus_allbook": (round(exc - am, 3) if exc is not None and am is not None else None),
        "pinnacle_minus_exchange": (round(pin - exc, 3) if pin is not None and exc is not None else None),
    }
    flags = []
    if pin is not None and am is not None and abs(pin - am) >= MATERIAL:
        flags.append("Pinnacle diverges from all-book median (>=0.02) -> ANCHOR TO PINNACLE")
    if pin is not None and exc is not None and abs(pin - exc) >= MATERIAL:
        flags.append("Pinnacle-vs-exchange split (>=0.02) -> caution; prefer Pinnacle, note exchanges")
    out["anchor"] = pin if pin is not None else (exc if exc is not None else am)
    out["anchor_source"] = "pinnacle" if pin is not None else ("exchange" if exc is not None else "all_book_median")
    out["flags"] = flags or ["sources agree within 0.02; all-book median fine"]
    return out


__all__ = ["PINNACLE", "EXCHANGES", "MATERIAL", "favorite_gap_by_source"]
