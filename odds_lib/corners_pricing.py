"""Direct-market pricing for corners COUNT rows (over/under), from the Odds API.

In scope (priced off a real market):
  - total_corners_over  <- alternate_totals_corners        (total match corners O/U)
  - team_corners_over   <- alternate_team_totals_corners    (per-team corners O/U)

OUT of scope (NO direct market exists — stay shadow; see market-availability audit):
  - team_more_corners_full / team_more_corners_h1  (comparison)
  - second_half_corners_over / team_more_corners_2h (period)

Line convention: the contest question "N or more corners" is stored with
``line = N - 0.5`` (e.g. "5 or more" -> 4.5), which is exactly the market's
Over N-0.5 line. So the question line maps directly to the market Over line; no
conversion, no interpolation. A book that doesn't quote that exact line does not
contribute; if NO book quotes a complete Over/Under at the line, the row is
unmapped and the caller falls back to shadow (never approximate).

De-vig uses the pipeline's OWN functions (odds_to_prob + remove_vig), per book
then averaged across books — identical to the live consensus and the historical
corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

from .odds import odds_to_prob, remove_vig
from . import market_quality as MQ

TOTAL_CORNERS_MARKET = "alternate_totals_corners"
TEAM_CORNERS_MARKET = "alternate_team_totals_corners"

# QUALITY gate (NOT a count gate). A market is discarded ('low' -> caller falls back
# to shadow) only when there is NO market (0 books) or the books SCATTER (de-vigged
# Over probs disagree by more than DISPERSION_MAX). A thin-but-AGREEING market (even
# 1-2 books) is USABLE ('thin') — a real sportsbook read beats a placeholder. >=THIN_BOOKS
# agreeing books = 'ok'. (book-count is a diagnostic, no longer the discard rule.)
MIN_BOOKS = 3            # retained for back-compat / diagnostics; NOT the discard gate
THIN_BOOKS = 5           # retained for diagnostics; flag now comes from MQ.quality_flag
DISPERSION_MAX = MQ.DISPERSION_SCATTER   # unified (0.10); back-compat alias for tests


@dataclass
class CornersPricing:
    mapped: bool
    p_over: float | None
    n_books: int
    line: float | None
    liquidity_flag: str       # "ok" | "thin" | "low" | "n/a"
    note: str
    dispersion: float = 0.0   # stdev of per-book de-vigged Over probs (agreement signal)


def _empty(line, note) -> CornersPricing:
    return CornersPricing(False, None, 0, line, "n/a", note)


def price_corners_over(game_json: dict, market_key: str,
                       team: str | None, line: float) -> CornersPricing:
    """De-vigged P(corners Over ``line``) for a total (team=None) or team market.

    ``team`` filters on the outcome ``description`` (the per-team field) for the
    team market; None for the total market.
    """
    if game_json is None or line is None:
        return _empty(line, "no game / no line")
    target = round(float(line), 2)
    reads: list[tuple[str, float]] = []          # (book_title, de-vigged P(Over))
    for b in game_json.get("bookmakers", []):
        legs: dict[str, int] = {}
        for m in b.get("markets", []):
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []):
                if team is not None and (o.get("description") or "") != team:
                    continue
                pt = o.get("point")
                if pt is None or round(float(pt), 2) != target:
                    continue
                legs[o["name"]] = int(o["price"])
        if "Over" in legs and "Under" in legs:
            dev = remove_vig(odds_to_prob([legs["Over"], legs["Under"]], "american"))
            reads.append((b.get("title", ""), float(dev[0])))
    n = len(reads)
    if n == 0:                                   # ZERO market is the ONLY count branch
        return _empty(line, f"no two-sided {market_key} at line {line}")
    # SHARPNESS-WEIGHTED estimate + UNIFIED dispersion policy (count never gates).
    p, disp, _ = MQ.weighted(reads)
    flag = MQ.liquidity_flag(disp, n)            # confident->ok, wide/single->thin, scatter->low
    # plausibility: a degenerate/stale de-vigged O/U prob (~0 or ~1) -> distrust ('low')
    if not MQ.in_band(p, "over_under"):
        flag = "low"
    note = (f"{market_key} Over {line} sharp-wt over {n} book(s), "
            f"disp={disp:.3f} q={MQ.quality_flag(disp, n)}")
    return CornersPricing(True, round(p, 4), n, float(line), flag, note, round(disp, 4))


def _poisson_p_ge(lam: float, k: int) -> float:
    """P(N >= k) for N ~ Poisson(lam)."""
    import math
    s, term = 0.0, math.exp(-lam)               # pmf(0)
    for i in range(0, 60):
        if i >= k:
            s += term
        term *= lam / (i + 1)
    return float(s)


def _collect_ladder(game_json: dict, market_key: str, team):
    """{line: sharp-weighted de-vigged P(Over line)} across all quoted lines, keeping
    ONLY lines whose books AGREE (disp <= scatter) and whose price is in band -- so a
    scattered/degenerate line never poisons the Poisson fit (it falls through to base)."""
    by_line: dict[float, list[tuple[str, float]]] = {}
    for b in game_json.get("bookmakers", []):
        legs: dict[float, dict[str, int]] = {}
        for m in b.get("markets", []):
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []):
                if team is not None and (o.get("description") or "") != team:
                    continue
                pt = o.get("point")
                if pt is None:
                    continue
                legs.setdefault(round(float(pt), 2), {})[o["name"]] = int(o["price"])
        for ln, lg in legs.items():
            if "Over" in lg and "Under" in lg:
                dev = remove_vig(odds_to_prob([lg["Over"], lg["Under"]], "american"))
                by_line.setdefault(ln, []).append((b.get("title", ""), float(dev[0])))
    out = {}
    for ln, reads in by_line.items():
        wmean, disp, _ = MQ.weighted(reads)
        if disp <= MQ.DISPERSION_SCATTER and MQ.in_band(wmean, "over_under"):
            out[ln] = wmean
    return out


def _fit_poisson_lambda(points: dict[float, float]) -> float | None:
    """Least-squares Poisson lam fit to {line: P(Over line)} (line L -> P(N>=ceil(L))).
    Monotone P(N>=k) in lam, so a coarse->fine grid is robust. >=1 point required
    (1 point => exact solve, since the objective has a unique minimum)."""
    import math
    pts = [(int(math.ceil(L)), p) for L, p in points.items() if p is not None]
    if not pts:
        return None
    best, grid = None, [0.05 * i for i in range(1, 501)]   # 0.05 .. 25.0
    for lam in grid:
        err = sum((_poisson_p_ge(lam, k) - p) ** 2 for k, p in pts)
        if best is None or err < best[1]:
            best = (lam, err)
    # refine around the grid minimum
    lam0 = best[0]
    for lam in [lam0 + 0.005 * j for j in range(-9, 10)]:
        if lam <= 0:
            continue
        err = sum((_poisson_p_ge(lam, k) - p) ** 2 for k, p in pts)
        if err < best[1]:
            best = (lam, err)
    return best[0]


def price_corners_laddered(game_json: dict, market_key: str, team, target_line: float):
    """Exact-line-GAP fallback: the contest line isn't quoted, but a book ladder is.
    De-vig each quoted line, MLE/LS-fit a Poisson lam to the CDF points, read off
    P(Over target_line) = P(N >= ceil(target_line)). Market-DERIVED (interpolation/
    extrapolation of real prices), not a model fit -> no OOS gate; plausibility-banded.
    Returns CornersPricing (mapped False -> caller falls back to the measured base rate)."""
    import math
    if game_json is None or target_line is None:
        return _empty(target_line, "no game / no line")
    ladder = _collect_ladder(game_json, market_key, team)
    if not ladder:
        return _empty(target_line, f"no {market_key} ladder to fit")
    lam = _fit_poisson_lambda(ladder)
    if lam is None:
        return _empty(target_line, "ladder fit failed")
    p = _poisson_p_ge(lam, int(math.ceil(float(target_line))))
    flag = "thin"                                  # an interpolated read is exact-but-modeled
    if not MQ.in_band(p, "over_under"):
        return _empty(target_line, f"laddered P={p:.3f} out of band")
    note = (f"{market_key} Poisson-ladder fit lam={lam:.2f} over {len(ladder)} line(s) "
            f"{sorted(ladder)} -> P(Over {target_line})")
    return CornersPricing(True, round(float(p), 4), len(ladder), float(target_line), flag, note)


__all__ = ["CornersPricing", "price_corners_over", "price_corners_laddered",
           "TOTAL_CORNERS_MARKET", "TEAM_CORNERS_MARKET", "MIN_BOOKS", "THIN_BOOKS"]
