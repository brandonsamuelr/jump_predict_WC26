"""Shared per-question resolution: question row -> (tier, p_hat, market_prob).

Single source of truth for how a contest question maps to a probability,
used by both the full overnight optimizer and the per-match pregame refresh
so the two can never drift apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .odds_api import json_to_markets
from .odds import process_match
from . import match_engine as E
from . import rate_layer as R
from .player_prop_pricing import price_player_prop, PROP_EQUIVALENCE
from .lineups import STARTER_STATUSES

GOALS = {"team_score_any", "team_score_1h", "team_score_2h", "second_half_more_goals",
         "team_more_goals_2h", "compound_first_goal_score_2h", "compound_btts_over_2_5",
         "total_goals_2h_over"}
MARKET_DIRECT = {"team_win", "match_total_over", "match_total_under", "halftime_draw",
                 "halftime_team_win"}
SOT = {"team_sot_over", "team_sot_2h_over", "team_more_sot_2h", "total_sot_2h_over"}
PROPS = set(PROP_EQUIVALENCE)
CONSENSUS_KEYS = ("h2h", "totals", "btts", "h2h_h1")


def build_consensus(games: list[dict], market_keys=CONSENSUS_KEYS) -> pd.DataFrame:
    mk = json_to_markets(games, "slate", market_keys=market_keys)
    _, c, _ = process_match(mk, on_incomplete="drop")
    return c


def market_p(c: pd.DataFrame, mkey: str, oc: str, line=None):
    s = c[c["market_key"] == mkey]
    if line is not None:
        s = s[s["line"] == float(line)]
    s = s[s["outcome"].str.lower() == oc.lower()]
    return float(s["market_prob"].iloc[0]) if not s.empty else None


def build_model(c: pd.DataFrame, home: str, away: str, n: int = 120_000):
    ph = market_p(c, "h2h", home)
    po = market_p(c, "totals", "Over", 2.5)
    if ph is None or po is None:
        return None
    m = E.calibrate(home, away, ph, po)
    return (m, E.simulate(m, n=n), home, away)


def _prop_tier(pr) -> str:
    """Map a PropPricing to a tier. goal-or-assist distinguishes the EXACT direct
    market (PROP_ok if liquid, else PROP_direct_thin) from the anytime-goal
    LOWER-BOUND fallback (PROP_proxy_floor). PROP_ok is reserved for broad/liquid
    support — a 1-2 book direct market is exact-but-thin, NOT PROP_ok. Other
    props keep the existing PROP_ok/PROP_thin liquidity split."""
    if pr.source == "proxy_floor":
        return "PROP_proxy_floor"
    if pr.source == "direct":
        return "PROP_ok" if pr.liquidity_flag == "ok" else "PROP_direct_thin"
    return "PROP_ok" if pr.liquidity_flag == "ok" else "PROP_thin"


def resolve_row(row: dict, c: pd.DataFrame, game_json: dict, model_tuple, lineup=None):
    """Return (tier, p_hat, market_prob). market_prob is the underlying
    de-vigged market price where one exists (for the diff), else None.

    ``lineup`` is the match's :class:`~odds_lib.lineups.MatchLineup` (or None).
    Player props express edge ONLY on a CONFIRMED starter; an unknown/no-lineup,
    bench, or out-of-squad player takes ZERO position (shadow) — the prop model
    assumes starter minutes, so any other status is a WRONG input, not a noisier
    one. PROP_thin (k=0.40) is reserved for a confirmed starter with a thin market.
    """
    qt = str(row["question_type"]).strip().lower()
    t = str(row.get("target_team", "")).strip()

    if qt in PROPS:
        line = float(row["line"]) if str(row.get("line", "")).strip() else None
        pr = price_player_prop(qt, (row.get("target_player") or None), line, game_json)
        if not pr.mapped:
            return "PENDING", None, None
        status = lineup.player(row.get("target_player")).status if lineup is not None else "unknown"
        if status in STARTER_STATUSES:
            return _prop_tier(pr), pr.market_prob_vig_adjusted, pr.market_prob_raw
        return "PENDING", None, None   # unconfirmed / bench / out -> shadow (k=0)

    if qt in MARKET_DIRECT:
        p = (market_p(c, "h2h", t) if qt == "team_win" else
             market_p(c, "totals", "Over", 2.5) if qt == "match_total_over" else
             market_p(c, "totals", "Under", 2.5) if qt == "match_total_under" else
             market_p(c, "h2h_h1", "Draw") if qt == "halftime_draw" else
             market_p(c, "h2h_h1", t))  # halftime_team_win: 1H-result market for target team
        return ("MARKET", p, p) if p is not None else ("PENDING", None, None)

    if model_tuple is None:
        return "PENDING", None, None
    m, sim, home, away = model_tuple

    if qt in GOALS:
        if qt == "total_goals_2h_over":
            line = float(row["line"]) if str(row.get("line", "")).strip() else 1.5
            return "ENGINE_GOALS", round(E.p_total_goals_2h_over(sim, math.ceil(line)), 4), None
        fn = {"team_score_any": lambda: E.p_team_score_any(sim, t),
              "team_score_1h": lambda: E.p_team_score_1h(sim, t),
              "team_score_2h": lambda: E.p_team_score_2h(sim, t),
              "second_half_more_goals": lambda: E.p_second_half_more_goals(sim),
              "team_more_goals_2h": lambda: E.p_team_more_goals_2h(sim, t),
              "compound_btts_over_2_5": lambda: E.p_compound_btts_over_2_5(sim),
              "compound_first_goal_score_2h": lambda: E.p_compound_first_goal_score_2h(sim, home, away)}[qt]
        return "ENGINE_GOALS", round(fn(), 4), None

    if qt in SOT:
        other = away if t.lower() == home.lower() else home
        lam = {home.lower(): m.lam_home, away.lower(): m.lam_away}
        line = float(row["line"]) if str(row.get("line", "")).strip() else 0.5
        if qt == "team_sot_over":
            rr = R.price_team_sot_over(lam[t.lower()], line)
        elif qt == "team_sot_2h_over":
            rr = R.price_team_sot_2h_over(lam[t.lower()], line, m.h1_share)
        elif qt == "team_more_sot_2h":
            rr = R.price_team_more_sot_2h(lam[t.lower()], lam[other.lower()], m.h1_share)
            return "RATE_SOT_CMP", round(rr.p, 4), None
        else:
            rr = R.price_total_sot_2h_over(m.lam_home, m.lam_away, line, m.h1_share)
        return "RATE_SOT", round(rr.p, 4), None

    return "PENDING", None, None


__all__ = ["build_consensus", "build_model", "resolve_row", "market_p",
           "GOALS", "MARKET_DIRECT", "SOT", "PROPS", "CONSENSUS_KEYS"]
