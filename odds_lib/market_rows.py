"""Direct-market pricing for the remaining market-available rows (Track 1).

Each row's market mapping was scope-checked before wiring:
  - team_total_goals_over   <- alternate_team_totals / team_totals (per-TEAM O/U)
  - second_half_goals_over  <- alternate_totals_h2 (genuine 2H goals O/U)
  - team_more_corners_full  <- corners_1x2 (3-way more-corners; Draw=tie=No)
  (halftime_team_lead / halftime_team_winning reuse the existing h2h_h1 path in
   slate.resolve_row — no pricer needed here.)

All submit the de-vigged market price UNDISTORTED (k=1, set via the tier). The
over/under rows reuse the tested per-book de-vig from corners_pricing; the corners
3-way adds a head-to-head de-vig. Liquidity flag is diagnostic only; no two-sided
/ 3-way quote -> unmapped -> caller falls back to shadow (never approximate).
"""

from __future__ import annotations

import re
import statistics as _stats
import unicodedata

from .odds import odds_to_prob, remove_vig
from .corners_pricing import price_corners_over, CornersPricing, MIN_BOOKS, THIN_BOOKS
from . import match_engine as _E   # Poisson inversion (P(Over) -> lambda), reused not re-derived
from . import market_quality as MQ

# per-team goal totals: prefer the wider alternate ladder, fall back to team_totals
TEAM_GOALS_MARKETS = ("alternate_team_totals", "team_totals")
H2_GOALS_MARKET = "alternate_totals_h2"
H1_GOALS_MARKET = "totals_h1"            # 1st-half total goals O/U (market half-split + odds in one)
FULL_TOTALS_MARKET = "totals"            # full-match total goals O/U (main 2.5 line)
ALT_TOTALS_MARKET = "alternate_totals"   # full-match totals ladder (exact non-2.5 lines)
CORNERS_3WAY_MARKET = "corners_1x2"
# HALF-INTEGER 1H lines ONLY for the Poisson inversion. A half line (0.5/1.5) has clean
# no-push ceil semantics: Over L.5 <=> N >= L+1. WHOLE lines (1.0) carry a PUSH on exactly
# N=1 (Over 1.0 = N>=2, stake returned on 1) so inverting their quote as P(N>=ceil=1) is
# WRONG -- that bug produced the implausible-low cluster (share ~0.25). Quarter lines
# (0.75) are Asian split-stakes, also != P(N>=ceil). So invert ONLY {0.5, 1.5}.
_H1_INVERT_LINES = (0.5, 1.5)

# Gate params now come from the UNIFIED policy in market_quality (one threshold pair for
# every family). Kept as aliases for back-compat; the band is the 106k-match-derived one.
H1_DISPERSION_MAX = MQ.DISPERSION_OK        # 0.05 (unified)
H1_PLAUSIBLE_BAND = MQ.PLAUSIBLE_BANDS["h1_share"]   # (0.30, 0.60)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.strip().lower())


def price_team_goals_over(game_json: dict, team: str, line: float) -> CornersPricing:
    """P(team goals Over ``line``), de-vigged. Tries the alternate ladder first."""
    last = None
    for mk in TEAM_GOALS_MARKETS:
        r = price_corners_over(game_json, mk, team, line)
        if r.mapped:
            return r
        last = r
    return last  # unmapped -> caller falls back to shadow


def price_2h_goals_over(game_json: dict, line: float) -> CornersPricing:
    """P(2nd-half total goals Over ``line``), de-vigged from alternate_totals_h2."""
    return price_corners_over(game_json, H2_GOALS_MARKET, None, line)


def price_1h_goals_over(game_json: dict, line: float) -> CornersPricing:
    """P(1st-half total goals Over ``line``), de-vigged from totals_h1."""
    return price_corners_over(game_json, H1_GOALS_MARKET, None, line)


def price_alt_total_over(game_json: dict, line: float) -> CornersPricing:
    """P(full-match total goals Over ``line``) at the EXACT line, from alternate_totals."""
    return price_corners_over(game_json, ALT_TOTALS_MARKET, None, line)


def _per_book_over(game_json: dict, key: str, line: float) -> list[tuple[str, float]]:
    """Per-book (book_title, de-vigged P(Over ``line``)) for a 2-way total market —
    titles carried so the caller can SHARPNESS-WEIGHT and measure AGREEMENT."""
    target = round(float(line), 2)
    out: list[tuple[str, float]] = []
    for b in (game_json or {}).get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") != key:
                continue
            o = [(x["name"], x["price"]) for x in m.get("outcomes", [])
                 if x.get("point") is not None and round(float(x["point"]), 2) == target
                 and x.get("price") is not None]
            if len(o) == 2:
                dev = remove_vig(odds_to_prob([p for _, p in o], "american"))
                ov = next((dev[i] for i, (n, _) in enumerate(o) if n == "Over"), None)
                if ov is not None:
                    out.append((b.get("title", ""), float(ov)))
    return out


def market_h1_share(game_json: dict):
    """Per-match 1H goal share = E[lambda_1H]/E[lambda_full], BOTH market-derived
    (totals_h1 + totals, Poisson-inverted on HALF-INTEGER lines only). A MARKET READ.

    Returns ``(share, n_books, dispersion)`` or ``None`` (no usable market leg).
    ``dispersion`` = stdev of the per-book share reads = the AGREEMENT signal. The
    caller gates on AGREEMENT + PLAUSIBILITY (NOT book count): a few tightly-agreeing
    plausible books supersede the H1_SHARE constant. NO clamp / NO new pricing constant.
    """
    if game_json is None:
        return None
    full = _per_book_over(game_json, FULL_TOTALS_MARKET, 2.5)
    if not full:
        return None
    full_mean = MQ.weighted(full)[0]                       # sharpness-weighted full P(over 2.5)
    lam_full = _E._bisect(lambda L: _E._p_over(L, 2.5), 0.05, 9.0, full_mean)
    if lam_full <= 0:
        return None
    # pick the best-quoted HALF-INTEGER 1H line (clean no-push inversion); per-book reads
    best, best_line = None, None
    for ln in _H1_INVERT_LINES:
        pb = _per_book_over(game_json, H1_GOALS_MARKET, ln)
        if pb and (best is None or len(pb) > len(best)):
            best, best_line = pb, ln
    if not best:
        return None
    # per-book shares (sharpness-weighted mean for the point estimate; raw stdev = agreement)
    shares = [(title, _E._bisect(lambda L: _E._p_over(L, best_line), 0.01, 6.0, p) / lam_full)
              for title, p in best]
    share, disp, n = MQ.weighted(shares)
    return (round(share, 4), n, round(disp, 4))


def price_more_corners(game_json: dict, team: str) -> CornersPricing:
    """P(team has MORE corners than opponent) from the corners_1x2 3-way.

    De-vig the full outcome set per book, take the target team's outcome (the Draw
    = equal-corners outcome is correctly excluded -> a tie counts as No). Averaged
    across books. Unmapped if no book quotes a corners_1x2 with the team.
    """
    if game_json is None or not team:
        return CornersPricing(False, None, 0, None, "n/a", "no game / no team")
    tnorm = _norm(team)
    reads: list[tuple[str, float]] = []           # (book_title, de-vigged P(team more))
    for b in game_json.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") != CORNERS_3WAY_MARKET:
                continue
            outs = [(o["name"], int(o["price"])) for o in m.get("outcomes", [])
                    if o.get("price") is not None]
            if len(outs) < 2:
                continue
            names = [n for n, _ in outs]
            dev = remove_vig(odds_to_prob([p for _, p in outs], "american"))
            for i, nm in enumerate(names):
                nn = _norm(nm)
                if nn == tnorm or tnorm in nn or nn in tnorm:
                    reads.append((b.get("title", ""), float(dev[i])))
                    break
    n = len(reads)
    if n == 0:                                    # ZERO market is the ONLY count branch
        return CornersPricing(False, None, 0, None, "n/a", "no corners_1x2 quote for team")
    # SHARPNESS-WEIGHTED estimate + UNIFIED dispersion policy (count never gates).
    p, disp, _ = MQ.weighted(reads)
    flag = MQ.liquidity_flag(disp, n)
    if not MQ.in_band(p, "comparison"):           # degenerate/stale 3-way -> distrust
        flag = "low"
    return CornersPricing(True, round(p, 4), n, None, flag,
                          f"corners_1x2 {team} sharp-wt over {n} book(s), "
                          f"disp={disp:.3f} q={MQ.quality_flag(disp, n)}", round(disp, 4))


__all__ = ["price_team_goals_over", "price_2h_goals_over", "price_1h_goals_over",
           "price_alt_total_over", "market_h1_share", "price_more_corners",
           "TEAM_GOALS_MARKETS", "H2_GOALS_MARKET", "H1_GOALS_MARKET",
           "ALT_TOTALS_MARKET", "CORNERS_3WAY_MARKET",
           "H1_DISPERSION_MAX", "H1_PLAUSIBLE_BAND"]
