"""Advance / to-qualify pricing for knockout "advance to Round of 16" questions.

CRITICAL settlement distinction (a high-risk routing rule):
  - "advance to R16"      -> INCLUDES extra time + penalties (no draw outcome).
  - "win in regulation"   -> 90' + stoppage only (a draw is a valid NO).
  - "regulation ends in a tie" -> the 90' draw.

ADVANCE PRICING HIERARCHY (use the first that applies):
  1. EXACT to-qualify / advance market (ET+pens inclusive) -- from the feed OR an external book/
     exchange/Kalshi/Polymarket. Feed-absence is NOT market-absence: our Odds API feed lacks a
     to-qualify key, but a sportsbook may quote one (soccer 1X2 = regulation; "to advance/qualify"
     = the ET+pens-inclusive market). If found externally, VERIFY its settlement wording matches
     the question (same tie, ET+pens-inclusive, advancement-not-match-win, not a stale outright)
     and DE-VIG it (enter both sides) before treating it as exact -- a vig-inclusive hand-entered
     number can be WORSE than a clean derivation, so the de-vig step does not disappear.
  2. Else DERIVE: P(advance) = P(win90) + P(draw90) * P(win | drawn after 90'), where the
     conditional is ODDS-DERIVED from team strength (price_advance below).

FLAT 0.5 IS PROHIBITED as a default P(win | drawn): it would understate the favorite and overstate
the underdog. 0.5 is allowed ONLY when the matchup is genuinely symmetric by the odds-derived
win-edge (flagged), or as an explicitly-flagged last-resort. NEVER route an advance question to bare
90-minute 1X2.
"""
from __future__ import annotations

SYMMETRIC_TOL = 0.02      # |win-edge - 0.5| below this == genuinely symmetric (0.5 legit, flagged)


def p_advance(p_win_regulation: float, p_regulation_draw: float,
              p_win_if_drawn: float) -> float:
    """P(advance) = P(win90) + P(draw90) * P(win | drawn after 90').

    Always >= P(win in regulation) (ET/pens only ADD paths). ``p_win_if_drawn`` is REQUIRED and
    must be odds-derived (see :func:`price_advance`) -- there is deliberately NO 0.5 default, so a
    flat 0.5 can never be used implicitly."""
    pw = float(p_win_regulation)
    pd = float(p_regulation_draw)
    pc = min(max(float(p_win_if_drawn), 0.0), 1.0)
    if not (0.0 <= pw <= 1.0) or not (0.0 <= pd <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return pw + pd * pc


def win_if_drawn_from_edge(p_win_regulation: float, p_opp_win_regulation: float,
                           strength: float = 0.5) -> float:
    """Tilt P(win | drawn after 90') toward the stronger side using the de-vigged regulation
    win-edge. strength=1 -> full edge mapping; strength in (0,1] -> partial. The tilt is bounded
    and odds-derived, never a crowd/typed constant. (strength=0 collapses to 0.5 -- price_advance
    forbids that on asymmetric matches.)"""
    pw, po = float(p_win_regulation), float(p_opp_win_regulation)
    denom = pw + po
    if denom <= 0:
        return 0.5
    edge = pw / denom                        # de-vigged 2-way win share
    return 0.5 + strength * (edge - 0.5)


def price_advance(p_win_regulation, p_regulation_draw, p_opp_win_regulation,
                  strength: float = 0.5, symmetric_tol: float = SYMMETRIC_TOL) -> dict:
    """Derive P(advance) with an ODDS-DERIVED P(win | drawn). Returns a provenance dict.

    Enforces the no-flat-0.5 rule:
      - strength must be > 0 (strength=0 would smuggle flat 0.5 back in via the tilt) -> ValueError.
      - no strength signal (missing win/draw odds) -> NOT priced; the row is FLAGGED, never 0.5.
      - genuinely symmetric by the odds win-edge -> conditional 0.5, source 'symmetric_odds_derived'.
      - otherwise -> odds-derived tilt (output strictly off 0.5)."""
    if strength <= 0:
        raise ValueError("advance tilt strength must be > 0 -- flat 0.5 is prohibited as a default")
    if p_win_regulation is None or p_opp_win_regulation is None or p_regulation_draw is None:
        return {"priced": False, "flag": "no_strength_signal_flag_row",
                "p_advance": None, "conditional": None, "conditional_source": "n/a"}
    denom = float(p_win_regulation) + float(p_opp_win_regulation)
    edge = 0.5 if denom <= 0 else float(p_win_regulation) / denom
    if abs(edge - 0.5) <= symmetric_tol:
        cond, src = 0.5, "symmetric_odds_derived"                      # legit, flagged
    else:
        cond = win_if_drawn_from_edge(p_win_regulation, p_opp_win_regulation, strength)
        src = "odds_derived_tilt"
    return {"priced": True, "flag": None, "conditional": cond, "conditional_source": src,
            "p_advance": p_advance(p_win_regulation, p_regulation_draw, cond)}


__all__ = ["SYMMETRIC_TOL", "p_advance", "win_if_drawn_from_edge", "price_advance"]
