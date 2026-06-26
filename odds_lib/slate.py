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
from . import sot_count_model as SC
from . import corners_pricing as CN
from . import cards_pricing as CD
from . import market_rows as MR
from . import market_quality as MQ
from . import corpus_models as CM
from . import corners_cmp_model as CC
from . import foul_cmp_model as FC
from . import oddspapi_pinnacle as OP
from . import shadow_routes as SR
from .player_prop_pricing import price_player_prop, PROP_EQUIVALENCE, minutes_scaled_sub
from .lineups import STARTER_STATUSES

GOALS = {"team_score_any", "team_score_1h", "team_score_2h", "second_half_more_goals",
         "team_more_goals_2h", "compound_first_goal_score_2h", "compound_btts_over_2_5"}
# 2H TOTAL goals over a line — the SAME event priced two ways. Double-source
# precedence (Part A.2): where a real 2H-goals market exists it WINS; the engine
# (on the calibrated half-split) is the fallback only where no 2H market is present.
# Both question strings route through one branch so they can never disagree.
TWO_H_GOALS = {"total_goals_2h_over", "second_half_goals_over"}
# 1st-half TOTAL goals over a line, priced DIRECTLY off the totals_h1 market.
ONE_H_GOALS = {"total_goals_1h_over"}
# GOALS questions whose answer depends on the HALF-SPLIT (so they reflect the
# market-derived per-match H1 share when available, else the H1_SHARE constant —
# tagged so constant-fallback rows are visible). Full-match goal rows are h1-invariant.
HALF_DEP_GOALS = {"team_score_1h", "team_score_2h", "second_half_more_goals",
                  "team_more_goals_2h", "compound_first_goal_score_2h"}
MARKET_DIRECT = {"team_win", "match_total_over", "match_total_under", "halftime_draw",
                 "halftime_team_win",
                 # aliases of halftime_team_win (team leads at half = wins 1H = h2h_h1
                 # team outcome). Same semantic, different contest strings.
                 "halftime_team_lead", "halftime_team_winning"}
SOT = {"team_sot_over", "team_sot_2h_over", "team_more_sot_2h", "total_sot_2h_over"}
# CHANGE 3: corners COUNT rows priced off the direct market. Comparison/period
# corners (team_more_corners_*, second_half_corners_over) are deliberately NOT
# here — no direct market exists, they stay shadow.
CORNERS = {"team_corners_over", "total_corners_over"}
# Cards COUNT total only. team_more_cards (comparison) / team_card_2h /
# total_cards_2h_over (period) are NOT here — no count-matching market, stay shadow.
CARDS_COUNT = {"total_cards_over"}
# Remaining market-available rows (Track 1). Comparison/period variants without a
# direct market are NOT here (stay shadow). halftime_team_lead/winning route via
# MARKET_DIRECT (h2h_h1), so they are not in these sets.
TEAM_GOALS = {"team_total_goals_over"}
CORNERS_CMP = {"team_more_corners_full"}
# Period corner comparisons (NO market) — gate-validated full-match model regressed
# toward 0.5 via a PROVISIONAL half-window constant (see corners_cmp_model).
# team_more_corners_h1 is the SAME semantic as team_more_corners_1h (different contest
# string for "more corners in the first half") -> same Pinnacle/stopgap route.
CORNERS_CMP_1H = {"team_more_corners_1h", "team_more_corners_2h", "team_more_corners_h1"}
# Track 2: gate-validated corpus models (no market). team_more_fouls is NOT here —
# it passed only by a tiny margin and stays shadow (pending decision).
MORE_CARDS = {"team_more_cards"}
MATCH_SOT = {"match_total_sot_over"}
# Validated universal foul-comparison model (favorite_gap only; gate-validated OOS,
# never identity). Supersedes the flat 0.50 team_more_fouls placeholder.
MORE_FOULS = {"team_more_fouls"}
# Newly founded shadow families (closed-form derivations from the validated engine,
# OOS-gated where a count assumption is added). NEVER a flat 0.50; auto-fallback to a
# MEASURED base-rate anchor (shadow_routes.json) when the estimator can't run.
BOTH_SOT_1H = {"both_teams_sot_1h"}      # P(both teams >=1 1H SOT) -- volume (engine lambda)
BOTH_SOT_2H = {"both_teams_sot_2h_1plus"}  # P(both teams >=1 2H SOT) -- raw true P, k=1
FIRST_GOAL_2H = {"team_first_goal_2h"}   # P(team scores first 2H goal) -- race of Poissons
# Measured-anchor families (no market, favorite_gap proven ~0 OOS): ship the MEASURED
# corpus/external rate raw (k=1), matched to the contest line -- never 0.50, never crowd.
OFFSIDES = {"team_offsides_over"}        # P(team offsides >= ceil(line)); contest 'offside 2+'
PENALTY = {"penalty_or_red_card", "penalty_awarded"}  # sourced external pen/red rates
# 2H yellow-card families: favorite_gap is NOT a clean per-match driver (gate: team no-signal,
# total weak signed-fg but clean |fg| fails) -> MEASURED per-threshold FLOORS (not models, not
# crowd-copy). team_card_2h = team >=1; *_over = ceil(line).
CARDS_2H = {"team_card_2h", "team_cards_2h_over", "total_cards_2h_over"}
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


def build_model(c: pd.DataFrame, home: str, away: str, n: int = 120_000,
                game_json: dict | None = None):
    """Calibrate the goals engine. The half-split is taken from the MARKET
    (totals_h1 + totals, per-match) when available, superseding the measured
    H1_SHARE constant; the constant is the last-resort fallback only when the 1H
    market leg is absent/too-thin. Returns (model, sim, home, away, h1_src) where
    h1_src in {'market_ok','market_thin','constant'} so half-dependent rows can be
    flagged when they fall back to the constant."""
    ph = market_p(c, "h2h", home)
    po = market_p(c, "totals", "Over", 2.5)
    if ph is None or po is None:
        return None
    # MARKET half-split supersedes the constant when the signal is RELIABLE — gated on
    # AGREEMENT (books disperse tightly) + PLAUSIBILITY (output in band), NOT book count.
    # A few agreeing, plausible books beat the match-blind constant. The measured
    # H1_SHARE is the LAST RESORT only for an unreliable signal (scattered / implausible
    # / no market) — each flagged with the reason.
    # UNIFIED dispersion policy (market_quality): disp<=0.05 confident, 0.05-0.10 USE+flag
    # wide_agreement, >0.10 scatter->fall back. Plausibility band catches broken derivations.
    # Book count NEVER gates. Constant is last resort only for unreliable signal.
    share = MR.market_h1_share(game_json)
    if share is not None:
        val, _nbk, disp = share
        if not MQ.in_band(val, "h1_share"):
            h1, h1_src = E.H1_SHARE, "constant_implausible_derivation"
        elif disp > MQ.DISPERSION_SCATTER:                 # genuine scatter only (0.10)
            h1, h1_src = E.H1_SHARE, "constant_scattered_books"
        else:
            h1 = val                                       # supersedes constant (any book count)
            h1_src = "market" if disp <= MQ.DISPERSION_OK else "market_wide_agreement"
    else:
        h1, h1_src = E.H1_SHARE, "constant_no_market"
    m = E.calibrate(home, away, ph, po, h1_share=h1)
    return (m, E.simulate(m, n=n), home, away, h1_src)


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


def _corner_model_p(c, game_json, t):
    """P(target team MORE corners) from the gate-validated corner model (favorite_gap
    + total_line_prob — odds-derived only, so it transfers club->international). Returns
    None when odds/teams are unavailable, so the caller falls back to shadow."""
    home = (game_json or {}).get("home_team")
    away = (game_json or {}).get("away_team")
    if c is None or not home or not away or not CC.is_available():
        return None
    opp = away if t.strip().lower() == str(home).strip().lower() else home
    fav = market_p(c, "h2h", t)
    op = market_p(c, "h2h", opp)
    tot = market_p(c, "totals", "Over", 2.5)
    if fav is None or op is None or tot is None:
        return None
    return CC.predict_more_corners(fav - op, tot)


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
        # CONFIRMED BENCH (sub-eligible): founded minutes-scaled true-P = P(appears)*P(event|sub mins),
        # NOT the c_hat shadow. None (out_of_squad / unknown / no-lineup) -> PENDING (no founded read).
        sub = minutes_scaled_sub(pr.market_prob_vig_adjusted, status)
        if sub is not None:
            return "PROP_SUB", round(sub, 4), pr.market_prob_raw
        return "PENDING", None, None   # out_of_squad / unknown / no-lineup -> shadow (k=0)

    if qt in OFFSIDES:
        line = float(row["line"]) if str(row.get("line", "")).strip() else 1.5
        # FOUNDED per-team measured rate FIRST (empirical-Bayes, OOS-gated, k=1): teams with
        # >= min_games of measured intl history get a real per-team P(offsides>=k). This beat
        # the pooled floor OOS (StatsBomb intl); shipped raw.
        tp = SR.offside_team_rate(t, line)
        if tp is not None:
            return "OFFSIDES_TEAM", tp, None
        # UNCOVERED team -> the honest POOLED FLOOR (NOT a guess). The per-match driver search
        # (favorite_gap / home-away / volume) found NO market/structural signal beating the
        # pooled base OOS, so where we lack measured team history we ship the measured pooled rate.
        p = SR.offsides_rate(line)
        return ("OFFSIDES_FLOOR", round(p, 4), None) if p is not None else ("PENDING", None, None)

    if qt in PENALTY:   # MEASURED external pen/red anchors (sourced; no market anywhere).
        p = SR.penalty_anchor(qt)
        return ("PENALTY_BASE", round(p, 4), None) if p is not None else ("PENDING", None, None)

    if qt in CARDS_2H:
        # PROPER controlling variable = card propensity priced by the FULL-MATCH cards market,
        # split to the 2H (lambda_full * 0.71). favorite_gap is a weak distal proxy; the corpus
        # floor is CLUB-ONLY (no intl per-half card data) and runs ~2x high for WC -> market-
        # derived is both correctly-populated AND match-specific. Club floor = flagged last resort.
        is_total = (qt == "total_cards_2h_over")
        if qt == "team_card_2h":            # 'a card in 2H' = >= 1
            line = 0.5
        else:
            line = float(row["line"]) if str(row.get("line", "")).strip() else (1.5 if is_total else 0.5)
        mapped, p = CD.price_cards_2h_over(game_json, line, is_total=is_total)   # 1) market-derived
        if mapped:
            return "CARDS_2H_MKT", p, None
        p = SR.cards_2h_rate("total" if is_total else "team", line)             # 2) club floor (poor)
        return ("CARDS_2H_FLOOR", round(p, 4), None) if p is not None else ("PENDING", None, None)

    if qt in MARKET_DIRECT:
        # match totals at a NON-2.5 line: price the EXACT line off alternate_totals
        # (16-book ladder) instead of interpolating from 2.5; fall back to the 2.5
        # consensus (flagged MARKET_INTERP) only if the exact line isn't quoted.
        if qt in ("match_total_over", "match_total_under"):
            line = float(row["line"]) if str(row.get("line", "")).strip() else 2.5
            side_over = (qt == "match_total_over")
            if abs(line - 2.5) > 1e-9:
                r = MR.price_alt_total_over(game_json, line)
                if r.mapped and r.liquidity_flag != "low" and r.p_over is not None:
                    p = r.p_over if side_over else round(1.0 - r.p_over, 4)
                    return ("MARKET", p, p)
                p = market_p(c, "totals", "Over" if side_over else "Under", 2.5)  # interp fallback
                return ("MARKET_INTERP", p, p) if p is not None else ("PENDING", None, None)
            p = market_p(c, "totals", "Over" if side_over else "Under", 2.5)
            return ("MARKET", p, p) if p is not None else ("PENDING", None, None)
        p = (market_p(c, "h2h", t) if qt == "team_win" else
             market_p(c, "h2h_h1", "Draw") if qt == "halftime_draw" else
             market_p(c, "h2h_h1", t))  # halftime_team_win: 1H-result market for target team
        return ("MARKET", p, p) if p is not None else ("PENDING", None, None)

    if qt in ONE_H_GOALS:  # 1st-half total goals -> totals_h1 direct (market half-split)
        line = float(row["line"]) if str(row.get("line", "")).strip() else 0.5
        r = MR.price_1h_goals_over(game_json, line)
        if r.mapped and r.liquidity_flag != "low":
            return ("H1GOALS_OK" if r.liquidity_flag == "ok" else "H1GOALS_THIN"), r.p_over, r.p_over
        return "PENDING", None, None   # no totals_h1 leg -> shadow (no constant invented)

    if qt in CORNERS:
        # CHANGE 3: direct corners market (over/under count). De-vigged price is
        # the model probability; routed to a market-priced CORNERS tier so the
        # k-policy carries it near full strength (not diluted to the placeholder).
        # Degrades safely to shadow if the market is absent/illiquid for this match.
        line = float(row["line"]) if str(row.get("line", "")).strip() else None
        if line is None:
            return "PENDING", None, None
        is_team = (qt == "team_corners_over")
        mkey = CN.TEAM_CORNERS_MARKET if is_team else CN.TOTAL_CORNERS_MARKET
        team = t if is_team else None
        cp = CN.price_corners_over(game_json, mkey, team, line)            # 1) EXACT quoted line
        if cp.mapped and cp.liquidity_flag != "low":
            return ("CORNERS_OK" if cp.liquidity_flag == "ok" else "CORNERS_THIN"), cp.p_over, cp.p_over
        lp = CN.price_corners_laddered(game_json, mkey, team, line)        # 2) LINE-GAP: Poisson-fit the ladder
        if lp.mapped:
            return "CORNERS_LADDER", lp.p_over, lp.p_over
        br = SR.corner_base_rate("team" if is_team else "total", line)     # 3) MEASURED base rate (never 0.50)
        return ("CORNERS_BASE", round(br, 4), None) if br is not None else ("PENDING", None, None)

    if qt in CARDS_COUNT:
        # total_cards_over off alternate_totals_cards (card-count convention
        # verified). De-vigged market price, submitted UNDISTORTED (k=1). Degrades
        # safely to shadow if no count market at the line.
        line = float(row["line"]) if str(row.get("line", "")).strip() else None
        if line is None:
            return "PENDING", None, None
        cd = CD.price_cards_over(game_json, line)
        if not cd.mapped or cd.liquidity_flag == "low":
            return "PENDING", None, None   # safe fallback to shadow
        tier = "CARDS_OK" if cd.liquidity_flag == "ok" else "CARDS_THIN"
        return tier, cd.p_over, cd.p_over

    if qt in TEAM_GOALS:   # team_total_goals_over -> per-team goal totals market
        line = float(row["line"]) if str(row.get("line", "")).strip() else None
        if line is None:
            return "PENDING", None, None
        r = MR.price_team_goals_over(game_json, t, line)
        if not r.mapped or r.liquidity_flag == "low":
            return "PENDING", None, None
        return ("TEAMGOALS_OK" if r.liquidity_flag == "ok" else "TEAMGOALS_THIN"), r.p_over, r.p_over

    if qt in CORNERS_CMP:  # team_more_corners_full: market -> model fallback -> shadow
        r = MR.price_more_corners(game_json, t)            # 1) corners_1x2 market WINS
        if r.mapped and r.liquidity_flag != "low":
            return ("CORNERS_CMP_OK" if r.liquidity_flag == "ok" else "CORNERS_CMP_THIN"), r.p_over, r.p_over
        pm = _corner_model_p(c, game_json, t)              # 2) gate-validated model fallback (k=1)
        if pm is not None:
            return "CORNERS_CMP_MODEL", round(pm, 4), None
        return "PENDING", None, None                       # 3) shadow (no market, no features)

    if qt in CORNERS_CMP_1H:  # team_more_corners_1h / _2h
        # FOUNDED read: OddsPapi-Pinnacle (sharp) per-fixture half-corner handicap (0.0
        # line de-vigged = P(team more corners), SIGN-GATED). single_book->use-if-plausible
        # (plausibility band guards). Per-fixture graceful fallback to the measured base-
        # rate STOPGAP when Pinnacle is absent/unmatched/implausible (no worse than now).
        home_t = (game_json or {}).get("home_team"); away_t = (game_json or {}).get("away_team")
        if home_t and away_t and t:
            pin = OP.more_corners(home_t, away_t, t, qt)
            if pin is not None:
                return "CORNER_HALF_PINNACLE", pin, None
        # STOPGAP: measured per-half base rate (1H 0.389 / 2H 0.410); NOT true P (no
        # favorite_gap) -- the honest floor where no Pinnacle read exists.
        return "CORNER_HALF_STOPGAP", round(CC.half_stopgap_p(qt), 4), None

    if qt in TWO_H_GOALS:  # 2H total goals (total_goals_2h_over / second_half_goals_over)
        # Part A.2 double-source precedence: the real 2H-goals market WINS; the
        # engine (calibrated half-split) is used ONLY where no 2H market exists.
        # Both strings flow through here, so they agree whenever the market is present.
        line = float(row["line"]) if str(row.get("line", "")).strip() else 1.5
        r = MR.price_2h_goals_over(game_json, line)
        if r.mapped and r.liquidity_flag != "low":
            return ("H2GOALS_OK" if r.liquidity_flag == "ok" else "H2GOALS_THIN"), r.p_over, r.p_over
        # no direct 2H market -> ENGINE fallback. The engine's half-split is the
        # MARKET-derived full-1H split when totals_h1 was present (h1_src='market_*'),
        # i.e. E[2H]=E[full]-E[1H] via the threaded per-match share; else the H1_SHARE
        # constant (flagged _H1FALLBACK). Precedence: direct 2H mkt > derived(full-1H) > constant.
        if model_tuple is None:
            return "PENDING", None, None
        h1_src = model_tuple[4] if len(model_tuple) > 4 else "constant"
        tier = "ENGINE_GOALS_H1MKT" if h1_src.startswith("market") else "ENGINE_GOALS_H1FALLBACK"
        return tier, round(E.p_total_goals_2h_over(model_tuple[1], math.ceil(line)), 4), None

    if qt in MORE_CARDS:   # Track 2 gate-validated: P(team more cards) from corpus model
        home_t = (game_json or {}).get("home_team"); away_t = (game_json or {}).get("away_team")
        if c is None or not home_t or not away_t or not CM.cards_available():
            return "PENDING", None, None
        is_home = 1 if t.strip().lower() == str(home_t).strip().lower() else 0
        other = away_t if is_home else home_t
        fav, opp, tot = market_p(c, "h2h", t), market_p(c, "h2h", other), market_p(c, "totals", "Over", 2.5)
        if fav is None or opp is None or tot is None:
            return "PENDING", None, None
        p = CM.predict_team_more_cards(fav - opp, tot, is_home)
        return ("MORE_CARDS", round(p, 4), None) if p is not None else ("PENDING", None, None)

    if qt in MORE_FOULS:   # validated universal foul-comparison model (favorite_gap only)
        home_t = (game_json or {}).get("home_team"); away_t = (game_json or {}).get("away_team")
        if c is None or not home_t or not away_t or not FC.is_available():
            return "PENDING", None, None
        other = away_t if t.strip().lower() == str(home_t).strip().lower() else home_t
        fav, opp = market_p(c, "h2h", t), market_p(c, "h2h", other)
        if fav is None or opp is None:
            return "PENDING", None, None
        p = FC.predict_more_fouls(fav - opp)        # favorite_gap = target - opp (de-vigged)
        return ("FOUL_CMP", round(p, 4), None) if p is not None else ("PENDING", None, None)

    if qt in MATCH_SOT:    # Track 2 gate-validated: P(total match SOT >= line)
        line = float(row["line"]) if str(row.get("line", "")).strip() else None
        home_t = (game_json or {}).get("home_team"); away_t = (game_json or {}).get("away_team")
        if line is None or c is None or not home_t or not away_t or not CM.match_sot_available():
            return "PENDING", None, None
        fav, awp, tot = market_p(c, "h2h", home_t), market_p(c, "h2h", away_t), market_p(c, "totals", "Over", 2.5)
        if fav is None or awp is None or tot is None:
            return "PENDING", None, None
        p = CM.predict_match_total_sot_over(fav - awp, tot, line)
        return ("MATCH_SOT", round(p, 4), None) if p is not None else ("PENDING", None, None)

    if qt in BOTH_SOT_1H:   # P(both teams >=1 SOT in 1H): engine-lambda volume model,
        # OOS-gated. Auto-fallback to the MEASURED base rate when the engine is absent
        # or the gate failed (never 0.50). Both submit at k=1 (the estimate, not hedged).
        if model_tuple is not None and SR.both_sot_1h_validated():
            m = model_tuple[0]
            rr = R.p_both_teams_sot_1h(m.lam_home, m.lam_away, m.h1_share)
            return "BOTH_SOT_1H", round(rr.p, 4), None
        br = SR.both_sot_1h_base_rate()
        return ("BOTH_SOT_1H_BASE", round(br, 4), None) if br is not None else ("PENDING", None, None)

    if qt in BOTH_SOT_2H:   # P(both teams >=1 2H SOT): ship the RAW closed-form true P (k=1),
        # undistorted -- NO anchor/damping. Volume lever validated OOS (P rises 0.70->0.94
        # across realized-goal tertiles). Measured base rate is the fallback ONLY when no
        # engine lambda exists (no market) -- never blended toward.
        if model_tuple is not None:
            m = model_tuple[0]
            rr = R.p_both_teams_sot_2h_1plus(m.lam_home, m.lam_away, m.h1_share)
            return "BOTH_SOT_2H", round(rr.p, 4), None
        br = SR.both_sot_2h_base_rate()
        return ("BOTH_SOT_2H_BASE", round(br, 4), None) if br is not None else ("PENDING", None, None)

    if qt in FIRST_GOAL_2H:   # P(team scores first 2H goal): closed-form race of 2H Poissons
        # off the validated engine lambdas + half split. Fallback = MEASURED home/away
        # first-2H-goal rate only if the engine is unavailable (never 0.50).
        home_t = (game_json or {}).get("home_team")
        is_home = bool(home_t) and t.strip().lower() == str(home_t).strip().lower()
        if model_tuple is not None:
            p = E.p_team_first_goal_2h(model_tuple[0], t)
            return "FIRST_GOAL_2H", round(p, 4), None
        a = SR.first_goal_2h_anchor(is_home)
        return ("FIRST_GOAL_2H_BASE", round(a, 4), None) if a is not None else ("PENDING", None, None)

    if model_tuple is None:
        return "PENDING", None, None
    m, sim, home, away = model_tuple[:4]          # tolerant: legacy 4-tuple or new 5-tuple
    h1_src = model_tuple[4] if len(model_tuple) > 4 else "constant"

    if qt in GOALS:
        fn = {"team_score_any": lambda: E.p_team_score_any(sim, t),
              "team_score_1h": lambda: E.p_team_score_1h(sim, t),
              "team_score_2h": lambda: E.p_team_score_2h(sim, t),
              "second_half_more_goals": lambda: E.p_second_half_more_goals(sim),
              "team_more_goals_2h": lambda: E.p_team_more_goals_2h(sim, t),
              "compound_btts_over_2_5": lambda: E.p_compound_btts_over_2_5(sim),
              "compound_first_goal_score_2h": lambda: E.p_compound_first_goal_score_2h(sim, home, away)}[qt]
        # half-dependent goal rows reflect the half-split source; full-match rows are
        # h1-invariant so they keep the plain ENGINE_GOALS tier.
        tier = "ENGINE_GOALS"
        if qt in HALF_DEP_GOALS:
            tier = "ENGINE_GOALS_H1MKT" if h1_src.startswith("market") else "ENGINE_GOALS_H1FALLBACK"
        return tier, round(fn(), 4), None

    if qt in SOT:
        other = away if t.lower() == home.lower() else home
        lam = {home.lower(): m.lam_home, away.lower(): m.lam_away}
        line = float(row["line"]) if str(row.get("line", "")).strip() else 0.5
        if qt == "team_sot_over":
            # CHANGE 2: validated count-row logistic replaces the high-biased
            # rate_layer Poisson (which runs +0.138 high on count rows). Features
            # are de-vigged PRE-KICKOFF market quantities (favorite_gap from h2h,
            # total_line_prob from totals) — identical to training, no leakage.
            # Falls back to the old rate_layer if the model/features are absent.
            fav = market_p(c, "h2h", t) if c is not None else None
            opp = market_p(c, "h2h", other) if c is not None else None
            tot = market_p(c, "totals", "Over", 2.5) if c is not None else None
            fav_gap = (fav - opp) if (fav is not None and opp is not None) else None
            p_new = SC.predict_team_sot_over(fav_gap, tot, line) if SC.is_available() else None
            if p_new is not None:
                return "SOT_COUNT", round(p_new, 4), None
            rr = R.price_team_sot_over(lam[t.lower()], line)   # fallback
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
