"""Direct-market pricing for the cards COUNT row (total_cards_over).

Convention (verified before wiring): the Odds API ``alternate_totals_cards``
prices CARD COUNT (lines ~1.5-4.5), NOT booking points (which would be 20-60);
the contest asks "N or more total cards shown" (card count); the contest line is
stored as N-0.5 = the market Over line. Both count cards SHOWN. So the question
line maps directly to the market Over line, no conversion.

In scope: total_cards_over only. OUT of scope (no count-matching market):
team_more_cards (comparison -> corpus-gate candidate), team_card_2h /
total_cards_2h_over (period -> data-blocked). Those stay shadow.

De-vig reuses the exact (tested) per-book over/under de-vig from corners_pricing
(odds_to_prob + remove_vig, averaged across books). Submit the de-vigged P(Over)
UNDISTORTED (k=1, set via the CARDS tier); the liquidity flag is diagnostic only.
Falls back to shadow if no two-sided cards quote exists at the line.
"""

from __future__ import annotations

import math

from .corners_pricing import (price_corners_over, CornersPricing,
                              _collect_ladder, _fit_poisson_lambda, _poisson_p_ge)

CARDS_TOTAL_MARKET = "alternate_totals_cards"

# 2H share of a match's cards. MEASURED 0.71 on club football (the SGO corpus has ZERO
# international per-half card data) -- but it is a RATIO (2H/full), and the late-card skew
# (fatigue, game-state, accumulating fouls) is a near-universal football phenomenon, so the
# RATIO transfers far better than the LEVEL. The LEVEL comes from the live market (correctly
# populated for the actual international match), not the club corpus. The one transferred assumption.
CARDS_2H_SHARE = 0.71


def price_cards_over(game_json: dict, line: float) -> CornersPricing:
    """De-vigged P(total cards Over ``line``) from alternate_totals_cards (full match)."""
    return price_corners_over(game_json, CARDS_TOTAL_MARKET, None, line)


def price_cards_2h_over(game_json: dict, line: float, is_total: bool = True):
    """P(2H cards >= ceil(line)) DERIVED FROM THE FULL-MATCH CARDS MARKET (the variable that
    actually controls cards -- card propensity priced by the market -- NOT favorite_gap, NOT a
    club-corpus floor). Fit lambda_full via the Poisson ladder on alternate_totals_cards, then
    lambda_2h = lambda_full * CARDS_2H_SHARE; a TEAM row * 0.5 (neutral team split, since no
    team-card market exists and the underdog tilt is OOS-weak). Market-derived & correctly
    populated -> P(N >= k) Poisson tail. Returns (mapped, p); mapped=False -> caller floors."""
    if game_json is None or line is None:
        return (False, None)
    ladder = _collect_ladder(game_json, CARDS_TOTAL_MARKET, None)   # agreement-filtered
    if not ladder:
        return (False, None)
    lam = _fit_poisson_lambda(ladder)
    if lam is None:
        return (False, None)
    lam_2h = lam * CARDS_2H_SHARE * (1.0 if is_total else 0.5)
    k = math.ceil(float(line)) if float(line) != int(float(line)) else int(float(line))
    return (True, round(_poisson_p_ge(lam_2h, k), 4))


__all__ = ["price_cards_over", "price_cards_2h_over", "CARDS_TOTAL_MARKET", "CARDS_2H_SHARE"]
