"""V2 decision engine for the SportsPredict Probability Cup.

Design principles (locked):

- ``p_field`` is the risk anchor — our best estimate of what the locked
  crowd average will be. Built from historical crowd averages, semantic
  anchors, and favorite/underdog context. Never substituted with truth.
- ``p_truth`` is the edge direction — only populated when backed by a real
  source (direct market, derived market, or a manually-promoted historical
  signal). Otherwise None or low-confidence.
- Submission = field-anchored, confidence-weighted move toward truth,
  clamped by a max-deviation that depends on evidence quality::

      p_raw    = p_field + confidence * (p_truth - p_field)
      p_submit = clamp(p_raw, p_field - max_dev, p_field + max_dev)

- Direct sharp markets bypass the blend and submit ``market_prob`` exactly.
- Historical aggregate bias is a hypothesis, never an automatic attack.
  ``strong_historical`` exists as a tier but its allowlist is empty at
  launch; signals are surfaced as candidate diagnostics instead.

Contest framing: this is a chase strategy. Skipping = exactly 0 RBP. A
defensible field shadow has positive expected RBP (Jensen on the field's
per-question Brier variance). Volume on shadow rows is part of the play;
big deviations from field are reserved for direct/derived markets or
manually validated edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .mappings import Mapping, _slice_market
from .lineups import MatchLineup, PlayerContext
from .player_features import (
    PlayerPropFeatures,
    build_player_prop_features,
    player_prop_risk_tags,
)


# ---------------------------------------------------------------------------
# Evidence tiers
# ---------------------------------------------------------------------------

# (confidence, max_dev). ``confidence`` sets how far we move from p_field
# toward p_truth; ``max_dev`` caps absolute deviation from p_field.
# Direct markets bypass the blend entirely (max_dev is effectively
# irrelevant there) — recorded here for diagnostic consistency.
EVIDENCE_TIERS: dict[str, tuple[float, float]] = {
    "direct_market_sharp":              (1.00, 1.00),
    "direct_market_all_book":           (1.00, 1.00),
    "derived_market":                   (0.65, 0.10),
    "strong_historical":                (0.55, 0.12),
    "lean":                             (0.45, 0.07),
    "contextual_shadow_with_bias_hint": (0.30, 0.04),
    "contextual_shadow":                (0.10, 0.015),
    "weak_field":                       (0.00, 0.00),
}

# ---------------------------------------------------------------------------
# Model discipline: production-grade vs heuristic truth sources
# ---------------------------------------------------------------------------
#
# A ``p_truth`` is only allowed to drive a confident, aggressive submission
# when it comes from a PRODUCTION-GRADE source: a real market, a clean market
# derivation, a trained model, an empirically-calibrated table, or an explicit
# manual override. Everything else (semantic anchors, lineup status, role
# multipliers, thin historical priors) is a *heuristic / diagnostic* signal:
# useful for review and as a low-confidence lean, but never permitted to claim
# high truth confidence or a large deviation from p_field.
PRODUCTION_TRUTH_SOURCE_PREFIXES: tuple[str, ...] = (
    "direct_market",
    "derived_market",
    "derived_",            # e.g. derived_btts_over_2_5
    "trained_model",
    "trained_player_prop_model",
    "empirical_calibrated_model",
    "empirical_calibrated",
    "manual_override",
)

# Hard cap on truth_confidence for any non-production-grade source.
HEURISTIC_TRUTH_CONFIDENCE_CAP = 0.35


def is_truth_source_production_grade(source: str | None) -> bool:
    """True only for market/model/manual-override truth sources.

    Heuristic sources (semantic_anchor, lineup_status_only, role_multiplier,
    heuristic_only, historical_thin_prior, shadow_field, etc.) return False.
    Used to gate confident auto-submission — see ``_enforce_truth_discipline``.
    """
    s = (source or "").strip().lower()
    if not s or s in ("none", "none_manual_review", "no_estimate"):
        return False
    return s.startswith(PRODUCTION_TRUTH_SOURCE_PREFIXES)


# Sharp coverage required to call a direct mapping a "sharp" tier (vs.
# all-book). Affects only the source label/confidence — both submit
# ``market_prob`` directly.
SHARP_TIER_MIN_BOOKS = 2

# Crowd-mean fallback when we have no per-question-type information at all.
GLOBAL_CROWD_PRIOR = 0.50

# Sample sizes that govern the historical-only field estimate. K_FIELD is
# the effective sample size we credit semantic anchors with when blending
# them against the empirical crowd average.
K_FIELD_BLEND_GENERIC = 10  # blend weight when anchor is question-type-wide (not role-aware)
K_FIELD_BLEND_ROLE    = 50  # blend weight when anchor is role-specific (favorite/underdog)
                             # — historical samples mix roles, so the role anchor
                             # should dominate unless the historical sample is huge
K_FIELD_SHRINK = 30          # shrink historical crowd mean toward global when n small
K_BIAS_SHRINK  = 30          # shrink raw bias toward zero with K_BIAS_SHRINK pseudo-counts

# Strong-historical allowlist — INTENTIONALLY EMPTY AT LAUNCH.
# Promotion is a manual decision after reviewing calibration data; the
# rough screen ``n * raw_bias^2 >= 2`` is necessary but not sufficient.
# Add entries here (and document the rationale in code review) only after
# (1) statistical signal clears the screen, (2) temporal-stability check
# passes, (3) the signal has a coherent story that applies to the
# realistic submission population for that question type.
STRONG_HISTORICAL_ALLOWLIST: set[str] = set()

# Rough statistical screen for promotion. A type only becomes a candidate
# if it clears this; clearing it does NOT auto-promote.
STRONG_CANDIDATE_NX_BIAS_SQ = 2.0


# ---------------------------------------------------------------------------
# Risk tags
# ---------------------------------------------------------------------------

RISK_TAGS_BY_QT: dict[str, list[str]] = {
    "team_win": [],
    "match_total_over": ["goal_volume"],
    "match_total_under": ["goal_volume"],
    "both_teams_score": ["goal_volume"],
    "halftime_draw": ["halftime", "low_event_match"],
    "halftime_team_lead": ["halftime", "favorite_dominance"],
    "halftime_team_winning": ["halftime", "favorite_dominance"],
    "compound_btts_over_2_5": ["compound", "goal_volume"],
    "player_sot_over": ["player_prop"],
    "player_sot_2h_over": ["player_prop", "second_half_activity"],
    "player_goal_or_assist": ["player_prop"],
    "player_goal": ["player_prop"],
    "team_more_fouls": ["cards_physicality"],
    "team_more_cards": ["cards_physicality"],
    "total_cards_over": ["cards_physicality"],
    "total_cards_2h": ["cards_physicality", "second_half_activity"],
    "second_half_cards_over": ["cards_physicality", "second_half_activity"],
    "team_more_sot_2h": ["second_half_activity"],
    "total_sot_2h": ["second_half_activity"],
    "total_sot_2h_over": ["second_half_activity"],
    "team_more_corners_2h": ["corners_pressure", "second_half_activity"],
    "team_corners_over": ["corners_pressure"],
    "total_corners_over": ["corners_pressure"],
    "team_offsides_over": ["low_event_match"],
    "penalty_or_red_card": ["low_event_match"],
    "penalty_awarded": ["low_event_match"],
    "team_score_2h": ["second_half_activity"],
    "team_score_any": [],
    "team_sot_over": [],
    "team_sot_2h_over": ["second_half_activity"],
    "both_teams_sot_2h_1plus": ["second_half_activity"],
    "both_teams_sot_h1_1plus": [],
}


# ---------------------------------------------------------------------------
# Semantic anchors for field estimation
# ---------------------------------------------------------------------------

# These estimate what the crowd is likely to submit — they are NOT truth
# estimates. They go into ``p_field``, never into ``p_truth``. Lookup is
# tried as ``{qt}_{role}`` first, then plain ``{qt}``. Roles come from
# ``classify_team_role`` (favorite / neutral / underdog).
SEMANTIC_FIELD_ANCHORS: dict[str, float] = {
    # No role differentiation needed:
    "penalty_awarded":         0.32,
    "penalty_or_red_card":     0.40,
    "both_teams_sot_2h_1plus": 0.68,
    "both_teams_sot_h1_1plus": 0.62,
    "halftime_draw":           0.32,
    "team_offsides_over":      0.50,
    "total_cards_2h":          0.55,
    "second_half_cards_over":  0.55,
    "total_sot_2h":            0.62,
    "total_sot_2h_over":       0.62,
    "total_corners_over":      0.52,
    "team_corners_over":       0.45,
    "second_half_goals_over":  0.50,
    "second_half_more_goals":  0.48,

    # Role-aware (target is favorite / neutral / underdog):
    "team_score_2h_favorite":          0.58,
    "team_score_2h_neutral":           0.50,
    "team_score_2h_underdog":          0.35,
    "team_score_any_favorite":         0.85,
    "team_score_any_neutral":          0.70,
    "team_score_any_underdog":         0.45,

    "team_sot_2h_over_favorite":       0.50,
    "team_sot_2h_over_neutral":        0.43,
    "team_sot_2h_over_underdog":       0.34,
    "team_sot_over_favorite":          0.55,
    "team_sot_over_neutral":           0.48,
    "team_sot_over_underdog":          0.38,

    "team_more_fouls_favorite":        0.43,
    "team_more_fouls_neutral":         0.50,
    "team_more_fouls_underdog":        0.57,
    "team_more_cards_favorite":        0.45,
    "team_more_cards_neutral":         0.50,
    "team_more_cards_underdog":        0.55,
    "team_more_sot_2h_favorite":       0.56,
    "team_more_sot_2h_neutral":        0.50,
    "team_more_sot_2h_underdog":       0.42,
    "team_more_corners_2h_favorite":   0.55,
    "team_more_corners_2h_neutral":    0.50,
    "team_more_corners_2h_underdog":   0.42,

    "halftime_team_lead_favorite":     0.42,
    "halftime_team_lead_neutral":      0.30,
    "halftime_team_lead_underdog":     0.18,
    "halftime_team_winning_favorite":  0.45,
    "halftime_team_winning_neutral":   0.34,
    "halftime_team_winning_underdog":  0.22,

    "player_goal_or_assist_favorite":  0.42,
    "player_goal_or_assist_neutral":   0.34,
    "player_goal_or_assist_underdog":  0.26,
    "player_goal_favorite":            0.32,
    "player_goal_neutral":             0.26,
    "player_goal_underdog":            0.20,
    "player_sot_over_favorite":        0.50,
    "player_sot_over_neutral":         0.45,
    "player_sot_over_underdog":        0.38,
    "player_sot_2h_over_favorite":     0.48,
    "player_sot_2h_over_neutral":      0.43,
    "player_sot_2h_over_underdog":     0.36,
}


def get_semantic_anchor(qt: str, role: str) -> float | None:
    """Try role-keyed then plain anchor lookup; None if neither present."""
    keyed = f"{qt}_{role}"
    if keyed in SEMANTIC_FIELD_ANCHORS:
        return SEMANTIC_FIELD_ANCHORS[keyed]
    if qt in SEMANTIC_FIELD_ANCHORS:
        return SEMANTIC_FIELD_ANCHORS[qt]
    return None


# ---------------------------------------------------------------------------
# Match context
# ---------------------------------------------------------------------------

@dataclass
class MatchContext:
    favorite_team: str | None = None
    underdog_team: str | None = None
    favorite_win_prob: float | None = None
    underdog_win_prob: float | None = None
    draw_prob: float | None = None
    fav_underdog_gap: float | None = None  # favorite_win_prob - underdog_win_prob
    total_line: float | None = None
    p_over: float | None = None
    p_under: float | None = None
    p_btts_yes: float | None = None
    p_halftime_draw: float | None = None
    is_close_match: bool = False  # True when fav_underdog_gap < 0.20
    is_lopsided: bool = False     # True when fav_underdog_gap >= 0.40


def extract_match_context(consensus: pd.DataFrame | None) -> MatchContext:
    """Pull favorite/underdog + win probs + totals + BTTS from per-match consensus.

    Tolerates missing markets — fields stay None when their source slice
    isn't present. Used downstream to choose semantic anchors and label
    target_team role.
    """
    ctx = MatchContext()
    if consensus is None or consensus.empty:
        return ctx

    h2h = _slice_market(consensus, "h2h", line=None)
    if not h2h.empty:
        non_draw = h2h[h2h["outcome"].str.lower() != "draw"].copy()
        draw_row = h2h[h2h["outcome"].str.lower() == "draw"]
        if not draw_row.empty:
            ctx.draw_prob = float(draw_row.iloc[0]["market_prob"])
        if not non_draw.empty:
            non_draw = non_draw.sort_values("market_prob", ascending=False)
            ctx.favorite_team = str(non_draw.iloc[0]["outcome"])
            ctx.favorite_win_prob = float(non_draw.iloc[0]["market_prob"])
            if len(non_draw) > 1:
                ctx.underdog_team = str(non_draw.iloc[-1]["outcome"])
                ctx.underdog_win_prob = float(non_draw.iloc[-1]["market_prob"])
            if (
                ctx.favorite_win_prob is not None
                and ctx.underdog_win_prob is not None
            ):
                ctx.fav_underdog_gap = ctx.favorite_win_prob - ctx.underdog_win_prob
                ctx.is_close_match = ctx.fav_underdog_gap < 0.20
                ctx.is_lopsided = ctx.fav_underdog_gap >= 0.40

    totals = consensus[consensus["market_key"] == "totals"]
    if not totals.empty:
        # Pick the line closest to 2.5 (canonical for "3 or more goals").
        totals_with_line = totals.dropna(subset=["line"])
        if not totals_with_line.empty:
            pick = totals_with_line.iloc[
                (totals_with_line["line"] - 2.5).abs().argsort()[:1]
            ]["line"].iloc[0]
            slice_ = totals_with_line[totals_with_line["line"] == pick]
            ctx.total_line = float(pick)
            over = slice_[slice_["outcome"].str.lower() == "over"]
            under = slice_[slice_["outcome"].str.lower() == "under"]
            if not over.empty:
                ctx.p_over = float(over.iloc[0]["market_prob"])
            if not under.empty:
                ctx.p_under = float(under.iloc[0]["market_prob"])

    btts = _slice_market(consensus, "btts", line=None)
    yes = btts[btts["outcome"].str.lower() == "yes"]
    if not yes.empty:
        ctx.p_btts_yes = float(yes.iloc[0]["market_prob"])

    for mkey in ("h2h_h1", "halftime", "first_half_winner"):
        ht = _slice_market(consensus, mkey, line=None)
        ht_draw = ht[ht["outcome"].str.lower() == "draw"]
        if not ht_draw.empty:
            ctx.p_halftime_draw = float(ht_draw.iloc[0]["market_prob"])
            break

    return ctx


TeamRole = Literal["favorite", "underdog", "neutral"]


def classify_team_role(team: str | None, ctx: MatchContext) -> TeamRole:
    """Assign favorite/underdog/neutral. Close matches collapse to neutral.

    A team is favorite/underdog only when the h2h gap is at least 0.20.
    Otherwise both sides read as neutral so we don't over-fit anchors
    to ~50/50 matchups.
    """
    if not team or ctx.favorite_team is None or ctx.fav_underdog_gap is None:
        return "neutral"
    if ctx.fav_underdog_gap < 0.20:
        return "neutral"
    t = team.strip().lower()
    if ctx.favorite_team and t == ctx.favorite_team.strip().lower():
        return "favorite"
    if ctx.underdog_team and t == ctx.underdog_team.strip().lower():
        return "underdog"
    return "neutral"


def classify_player_role(
    target_team: str | None, ctx: MatchContext
) -> TeamRole:
    """Player questions inherit their team's role from the match h2h."""
    return classify_team_role(target_team, ctx)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    p_truth: float | None
    p_truth_source: str
    truth_confidence: float
    p_field: float | None
    p_field_source: str
    field_confidence: float
    p_submit: float | None
    decision_mode: str  # direct_market | derived_market | strong_historical | lean | contextual_shadow_with_bias_hint | contextual_shadow | player_prop_review_required | weak_field | review
    delta_vs_field: float | None
    estimated_swing: float | None
    risk_tags: list[str] = field(default_factory=list)
    reason: str = ""
    # True for any row the caller must hand-resolve before submission
    # (player_prop_review_required, weak_field, review). Direct/derived
    # market and shadow rows leave this False — they are auto-submittable.
    needs_manual_review: bool = False
    # Candidate diagnostics — surfaced even when not promoted so we can
    # learn whether the engine would have made money if we had let it.
    historical_candidate: bool = False
    candidate_n: int = 0
    candidate_raw_bias: float = 0.0
    candidate_reason: str = ""
    promotion_status: str = "no_signal"  # no_signal | below_threshold | not_allowlisted | promoted

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["risk_tags"] = ";".join(self.risk_tags) if self.risk_tags else ""
        return d


# ---------------------------------------------------------------------------
# Historical priors
# ---------------------------------------------------------------------------

HISTORICAL_PATH_DEFAULT = Path("data/historical/sportspredict_collected_data.csv")


@dataclass
class PriorRow:
    n: int
    avg_crowd_prob: float
    actual_yes_rate: float
    raw_bias: float
    shrunk_bias: float


# Conservative hard-coded defaults — used when the historical CSV is missing
# or a question_type has no rows. Small magnitudes by design; the field
# estimator will blend these with semantic anchors anyway.
DEFAULT_PRIORS: dict[str, PriorRow] = {
    "team_win":                   PriorRow(0, 0.55, 0.55, 0.00, 0.00),
    "match_total_over":           PriorRow(0, 0.50, 0.50, 0.00, 0.00),
    "match_total_under":          PriorRow(0, 0.50, 0.50, 0.00, 0.00),
    "both_teams_score":           PriorRow(0, 0.55, 0.55, 0.00, 0.00),
}


def load_priors(path: Path | str | None = None) -> dict[str, PriorRow]:
    """Read SportsPredict historical CSV and build per-question-type priors.

    Falls back to :data:`DEFAULT_PRIORS` if the file is missing or empty.
    ``shrunk_bias = raw_bias * n / (n + K_BIAS_SHRINK)`` keeps thin samples
    from masquerading as strong signals.
    """
    priors: dict[str, PriorRow] = dict(DEFAULT_PRIORS)
    p = Path(path) if path is not None else HISTORICAL_PATH_DEFAULT
    if not p.exists():
        return priors

    try:
        df = pd.read_csv(p)
    except Exception:
        return priors

    df = df[(df.get("status") == "resolved") & df.get("outcome_pct").isin([0, 100])].copy()
    if df.empty:
        return priors

    df["crowd_p"] = pd.to_numeric(df["field_prob"], errors="coerce") / 100.0
    df["yes"] = (df["outcome_pct"] == 100).astype(int)
    df = df.dropna(subset=["crowd_p"])
    if df.empty:
        return priors

    grouped = df.groupby("question_type").agg(
        n=("yes", "size"),
        actual_yes_rate=("yes", "mean"),
        avg_crowd_prob=("crowd_p", "mean"),
    )
    grouped["raw_bias"] = grouped["actual_yes_rate"] - grouped["avg_crowd_prob"]
    grouped["shrunk_bias"] = (
        grouped["raw_bias"] * grouped["n"] / (grouped["n"] + K_BIAS_SHRINK)
    )
    for qt, row in grouped.iterrows():
        priors[str(qt)] = PriorRow(
            n=int(row["n"]),
            avg_crowd_prob=float(row["avg_crowd_prob"]),
            actual_yes_rate=float(row["actual_yes_rate"]),
            raw_bias=float(row["raw_bias"]),
            shrunk_bias=float(row["shrunk_bias"]),
        )
    return priors


def _prior_for(priors: dict[str, PriorRow], qt: str) -> PriorRow:
    return priors.get(qt) or PriorRow(0, GLOBAL_CROWD_PRIOR, GLOBAL_CROWD_PRIOR, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Field estimator
# ---------------------------------------------------------------------------

def estimate_p_field(
    qt: str,
    question_row: dict[str, Any],
    match_ctx: MatchContext,
    prior: PriorRow,
) -> tuple[float | None, str, float]:
    """Estimate the locked-crowd average for this question.

    Returns ``(p_field, source_label, field_confidence)``. ``p_field`` is
    None only when we have neither a semantic anchor nor a historical
    sample — in that case the row should route to ``weak_field`` review.

    Priority:
      1. semantic anchor blended with historical avg_crowd (if both exist)
      2. semantic anchor alone
      3. historical avg_crowd alone (shrunk toward 0.50 if n small)
    """
    role: TeamRole = "neutral"
    if qt.startswith("player_"):
        role = classify_player_role(
            (question_row.get("target_team") or "") or None, match_ctx
        )
    elif qt.startswith("team_") or qt.startswith("halftime_team_"):
        role = classify_team_role(
            (question_row.get("target_team") or "") or None, match_ctx
        )

    role_keyed_anchor = SEMANTIC_FIELD_ANCHORS.get(f"{qt}_{role}")
    generic_anchor = SEMANTIC_FIELD_ANCHORS.get(qt)
    anchor = role_keyed_anchor if role_keyed_anchor is not None else generic_anchor
    is_role_keyed = role_keyed_anchor is not None and role != "neutral"

    if anchor is not None and prior.n > 0:
        # Role-keyed anchors are structurally more informative than the
        # aggregated historical mean (which mixes both team roles), so they
        # deserve heavier weight. Generic anchors stay on the lighter blend.
        k_blend = K_FIELD_BLEND_ROLE if is_role_keyed else K_FIELD_BLEND_GENERIC
        w = prior.n / (prior.n + k_blend)
        p_field = w * prior.avg_crowd_prob + (1 - w) * anchor
        source = (
            f"blend_anchor_{role}+historical_n{prior.n}"
            if is_role_keyed
            else f"blend_anchor_generic+historical_n{prior.n}"
        )
        conf = 0.60
        return p_field, source, conf

    if anchor is not None:
        return anchor, f"semantic_anchor_{role if is_role_keyed else 'generic'}", 0.45

    if prior.n >= 10:
        return prior.avg_crowd_prob, f"historical_avg_crowd_n{prior.n}", 0.55

    if prior.n > 0:
        w = prior.n / (prior.n + 20)
        p_field = w * prior.avg_crowd_prob + (1 - w) * GLOBAL_CROWD_PRIOR
        return p_field, f"weak_historical_shrunk_n{prior.n}", 0.30

    return None, "no_estimate", 0.0


# ---------------------------------------------------------------------------
# Historical-candidate evaluation (diagnostic only at launch)
# ---------------------------------------------------------------------------

def evaluate_historical_candidate(
    qt: str, prior: PriorRow
) -> tuple[bool, str, str]:
    """Decide whether a historical signal qualifies as a strong-tier candidate.

    Returns ``(is_candidate, promotion_status, reason)``. ``is_candidate``
    is purely diagnostic at launch — even True candidates do not auto-promote
    because :data:`STRONG_HISTORICAL_ALLOWLIST` is empty. To promote a type:

      1. Confirm ``n * raw_bias^2 >= STRONG_CANDIDATE_NX_BIAS_SQ``.
      2. Manually run a temporal-stability check (split-half by date).
      3. Write a coherent story about why the bias generalizes to the
         realistic submission population for this question type.
      4. Add to ``STRONG_HISTORICAL_ALLOWLIST``.
    """
    if prior.n <= 0:
        return False, "no_signal", "no historical sample"
    score = prior.n * (prior.raw_bias ** 2)
    if score < STRONG_CANDIDATE_NX_BIAS_SQ:
        return (
            False,
            "below_threshold",
            f"n*raw_bias^2={score:.2f} < {STRONG_CANDIDATE_NX_BIAS_SQ}",
        )
    if qt not in STRONG_HISTORICAL_ALLOWLIST:
        return (
            True,
            "not_allowlisted",
            f"clears stat screen (n={prior.n}, raw_bias={prior.raw_bias:+.3f}, "
            f"n*bias^2={score:.2f}); not in allowlist yet",
        )
    return (
        True,
        "promoted",
        f"allowlisted; n={prior.n}, raw_bias={prior.raw_bias:+.3f}",
    )


# ---------------------------------------------------------------------------
# Core mechanics
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _blend(p_truth: float, p_field: float, confidence: float, max_dev: float) -> float:
    raw = p_field + confidence * (p_truth - p_field)
    return _clamp(raw, p_field - max_dev, p_field + max_dev)


def _risk_tags(qt: str, *, market_prob: float | None = None) -> list[str]:
    tags = list(RISK_TAGS_BY_QT.get(qt, []))
    if qt == "team_win" and market_prob is not None and market_prob >= 0.70:
        tags.append("favorite_dominance")
    return tags


def _pack(
    *,
    p_truth: float | None,
    p_truth_source: str,
    truth_confidence: float,
    p_field: float | None,
    p_field_source: str,
    field_confidence: float,
    p_submit: float | None,
    decision_mode: str,
    risk_tags: list[str],
    reason: str,
    needs_manual_review: bool = False,
    historical_candidate: bool = False,
    candidate_n: int = 0,
    candidate_raw_bias: float = 0.0,
    candidate_reason: str = "",
    promotion_status: str = "no_signal",
) -> Recommendation:
    if p_submit is not None and p_field is not None:
        delta = p_submit - p_field
        swing = 200.0 * abs(delta)
    else:
        delta = None
        swing = None
    return Recommendation(
        p_truth=p_truth,
        p_truth_source=p_truth_source,
        truth_confidence=truth_confidence,
        p_field=p_field,
        p_field_source=p_field_source,
        field_confidence=field_confidence,
        p_submit=p_submit,
        decision_mode=decision_mode,
        delta_vs_field=delta,
        estimated_swing=swing,
        risk_tags=risk_tags,
        reason=reason,
        needs_manual_review=needs_manual_review,
        historical_candidate=historical_candidate,
        candidate_n=candidate_n,
        candidate_raw_bias=candidate_raw_bias,
        candidate_reason=candidate_reason,
        promotion_status=promotion_status,
    )


def _enforce_truth_discipline(rec: Recommendation) -> Recommendation:
    """Final guardrail: heuristics must not masquerade as production truth.

    Applied to every recommendation before it leaves the engine. When the
    truth source is NOT production-grade (market / trained model / empirical
    calibration / manual override):

      - cap ``truth_confidence`` at ``HEURISTIC_TRUTH_CONFIDENCE_CAP``;
      - tag the row ``heuristic_truth_source`` so downstream tooling can see
        the distinction and never auto-submit it as high-confidence truth.

    Production-grade rows pass through untouched. This does not change
    ``p_submit`` (deviation is already bounded by the evidence tiers); it
    enforces the *labelling* discipline so nothing reads a heuristic as fact.
    """
    if is_truth_source_production_grade(rec.p_truth_source):
        return rec
    # Non-production truth source.
    if rec.truth_confidence > HEURISTIC_TRUTH_CONFIDENCE_CAP:
        rec.truth_confidence = HEURISTIC_TRUTH_CONFIDENCE_CAP
    # Only tag rows that actually assert a truth value; pure review rows
    # (p_truth is None) are already clearly non-production.
    if rec.p_truth is not None and "heuristic_truth_source" not in rec.risk_tags:
        rec.risk_tags.append("heuristic_truth_source")
    return rec


# ---------------------------------------------------------------------------
# Derived markets
# ---------------------------------------------------------------------------

def estimate_exact_1_1(match_context: dict[str, Any] | MatchContext | None) -> float:
    """Conservative placeholder for P(final score = 1-1).

    Tunes mildly with the totals line: more total goals -> less 1-1 mass.
    Returns 0.09-0.14. A proper scoreline model would replace this.
    """
    total_line = None
    if isinstance(match_context, MatchContext):
        total_line = match_context.total_line
    elif match_context:
        total_line = match_context.get("total_line")
    if total_line is None or not isinstance(total_line, (int, float)):
        return 0.115
    if total_line <= 2.0:
        return 0.13
    if total_line <= 2.5:
        return 0.12
    if total_line <= 3.0:
        return 0.10
    return 0.09


def _derive_btts_over_2_5(
    consensus: pd.DataFrame | None,
    match_ctx: MatchContext,
) -> tuple[float, str, dict[str, Any]] | None:
    """If BTTS Yes and Over 2.5 are both available, derive P(BTTS and Over2.5).

    P(BTTS=Yes AND Over 2.5) = P(BTTS=Yes) - P(exactly 1-1). 1-1 is the
    only BTTS-yes scoreline with total <= 2.
    """
    if consensus is None or consensus.empty:
        return None
    btts = _slice_market(consensus, "btts", line=None)
    btts_yes = btts[btts["outcome"].str.lower() == "yes"]
    totals = _slice_market(consensus, "totals", line=2.5)
    over = totals[totals["outcome"].str.lower() == "over"]
    if btts_yes.empty or over.empty:
        return None
    p_btts = float(btts_yes.iloc[0]["market_prob"])
    p_over = float(over.iloc[0]["market_prob"])
    p_11 = estimate_exact_1_1(match_ctx)
    p = max(0.0, min(p_btts, p_btts - p_11))
    p = min(p, p_over)
    return p, "derived_btts_over_2_5", {
        "p_btts": p_btts,
        "p_over_2_5": p_over,
        "p_exact_1_1": p_11,
    }


# ---------------------------------------------------------------------------
# Lineup features and player-prop manual-review routing
# ---------------------------------------------------------------------------
#
# Player props with lineup info do NOT receive an automated p_truth here.
# A previous version blended hard-coded role/status/team-role probability
# tables into p_truth; that was removed because the tables were unvalidated
# guesses dressed up as a model. Until a market or an empirically-trained
# prop model exists, lineup context produces FEATURES and RISK TAGS only,
# and the row is flagged ``needs_manual_review`` so a human chooses p_submit.

LINEUP_AWARE_QUESTION_TYPES: set[str] = {
    "player_sot_over",
    "player_sot_2h_over",
    "player_goal_or_assist",
    "player_goal",
}


def _resolve_player_context(
    question_row: dict[str, Any],
    lineup_context: dict[str, Any] | None,
) -> PlayerContext:
    """Look up the question's target_player in the provided lineup.

    ``lineup_context`` is expected to be ``{"lineup": MatchLineup}``.
    Returns an ``unknown/unknown`` PlayerContext when no info is available.
    """
    name = (question_row.get("target_player") or "").strip()
    if not name or not lineup_context:
        return PlayerContext()
    lineup: MatchLineup | None = lineup_context.get("lineup")
    if lineup is None:
        return PlayerContext()
    return lineup.player(name)


def _has_lineup_signal(player_ctx: PlayerContext) -> bool:
    return player_ctx.status != "unknown" or player_ctx.role != "unknown"


def _build_prop_features(
    qt: str,
    question_row: dict[str, Any],
    match_ctx: MatchContext,
    prior: PriorRow,
    player_ctx: PlayerContext,
    mapping: Mapping | None = None,
) -> PlayerPropFeatures:
    """Assemble player-prop features for risk-tagging and diagnostics.

    Delegates to :mod:`odds_lib.player_features` so the engine and the
    offline dataset builder share one extraction path. ``p_field`` is a
    diagnostic crowd estimate only — it never becomes p_truth here.
    """
    p_field, field_source, _ = estimate_p_field(
        qt, question_row, match_ctx, prior
    )
    has_direct = (
        mapping is not None
        and mapping.mapping_status == "mapped_exact"
        and mapping.market_prob is not None
    )
    return build_player_prop_features(
        question_row=question_row,
        player_ctx=player_ctx,
        match_ctx=match_ctx,
        p_field_est=p_field,
        p_field_source=field_source if p_field is not None else "no_estimate",
        has_direct_market=bool(has_direct),
        direct_market_prob=(float(mapping.market_prob) if has_direct else None),
    )


def _player_prop_review_required_rec(
    qt: str,
    question_row: dict[str, Any],
    match_ctx: MatchContext,
    prior: PriorRow,
    player_ctx: PlayerContext,
) -> Recommendation:
    """Player prop with lineup info but no direct/derived market or model.

    Produces no production p_truth or p_submit. Lineup status/role are
    surfaced as structured features and risk tags via
    :mod:`odds_lib.player_features`; ``p_field`` is kept as a diagnostic
    crowd anchor; the row is flagged for manual review.
    """
    feats = _build_prop_features(qt, question_row, match_ctx, prior, player_ctx)
    p_field = feats.p_field_est
    field_source = feats.p_field_source or "no_estimate"
    _, _, field_conf = estimate_p_field(qt, question_row, match_ctx, prior)

    reason_bits = [
        f"Player prop {qt} with lineup info "
        f"(status={feats.lineup_status}, role={feats.lineup_role}"
        + (
            f", expected_minutes={feats.expected_minutes_if_known}"
            if feats.expected_minutes_if_known is not None
            else ""
        )
        + ")",
        "no direct market and no trained player-prop model — manual review "
        "required (lineup is a feature/risk signal, not a truth model)",
    ]
    if p_field is not None:
        reason_bits.append(f"diagnostic p_field={p_field:.3f} ({field_source})")

    return _pack(
        p_truth=None,
        p_truth_source="none_manual_review",
        truth_confidence=0.0,
        p_field=p_field,
        p_field_source=field_source,
        field_confidence=field_conf,
        p_submit=None,
        decision_mode="player_prop_review_required",
        risk_tags=feats.risk_tags,
        reason="; ".join(reason_bits) + ".",
        needs_manual_review=True,
    )


# ---------------------------------------------------------------------------
# Per-mode builders
# ---------------------------------------------------------------------------

def _direct_market_rec(
    mp: Mapping,
    qt: str,
    match_ctx: MatchContext,
    prior: PriorRow,
) -> Recommendation:
    """Liquid direct market — submit market_prob exact, bypass blend entirely.

    p_field is still estimated (semantic + historical) so the diagnostic
    delta_vs_field / estimated_swing remain meaningful for downstream
    calibration analysis.
    """
    has_sharp = (mp.sharp_num_books or 0) >= SHARP_TIER_MIN_BOOKS
    tier = "direct_market_sharp" if has_sharp else "direct_market_all_book"
    confidence, _ = EVIDENCE_TIERS[tier]

    p_truth = float(mp.market_prob)
    # Best available field estimate for this question (diagnostic only —
    # we are not using it to shape p_submit for direct markets).
    p_field, field_source, field_conf = estimate_p_field(
        qt, {"target_team": getattr(mp, "mapped_outcome", None)}, match_ctx, prior
    )
    if p_field is None:
        # Direct markets don't need a field estimate to act, but we want
        # the diagnostic. Fall back to a market-anchored crowd guess.
        p_field = 0.7 * p_truth + 0.3 * GLOBAL_CROWD_PRIOR
        field_source = "market_anchored_field_fallback"
        field_conf = 0.30

    reason_bits = [f"Direct {mp.mapped_market} market"]
    if has_sharp:
        reason_bits.append(
            f"sharp consensus across {mp.sharp_num_books} sharp book(s)"
        )
    else:
        reason_bits.append(f"all-book consensus across {mp.all_num_books} book(s)")
    if mp.abs_diff_all_vs_sharp is not None:
        reason_bits.append(f"|all-sharp|={mp.abs_diff_all_vs_sharp:.3f}")
    reason = "; ".join(reason_bits) + "."

    return _pack(
        p_truth=p_truth,
        p_truth_source=tier,
        truth_confidence=confidence,
        p_field=p_field,
        p_field_source=field_source,
        field_confidence=field_conf,
        p_submit=p_truth,
        decision_mode="direct_market",
        risk_tags=_risk_tags(qt, market_prob=p_truth),
        reason=reason,
    )


def _derived_market_rec(
    qt: str,
    p_truth: float,
    source_label: str,
    derive_context: dict[str, Any],
    match_ctx: MatchContext,
    prior: PriorRow,
    question_row: dict[str, Any],
) -> Recommendation:
    confidence, max_dev = EVIDENCE_TIERS["derived_market"]
    p_field, field_source, field_conf = estimate_p_field(
        qt, question_row, match_ctx, prior
    )
    if p_field is None:
        # No field anchor at all — fall back to truth so submit isn't None,
        # but mark low field confidence so the user sees the gap.
        p_field, field_source, field_conf = p_truth, "fallback_to_truth", 0.20
    p_submit = _blend(p_truth, p_field, confidence, max_dev)

    inputs = ", ".join(f"{k}={v:.3f}" for k, v in derive_context.items())
    reason = (
        f"Derived {qt} from legs ({inputs}); blended "
        f"{confidence:.2f}·truth + {1-confidence:.2f}·field, "
        f"capped ±{max_dev:.2f} from field."
    )
    return _pack(
        p_truth=p_truth,
        p_truth_source=source_label,
        truth_confidence=confidence,
        p_field=p_field,
        p_field_source=field_source,
        field_confidence=field_conf,
        p_submit=p_submit,
        decision_mode="derived_market",
        risk_tags=_risk_tags(qt),
        reason=reason,
    )


def _shadow_rec(
    qt: str,
    question_row: dict[str, Any],
    match_ctx: MatchContext,
    prior: PriorRow,
) -> Recommendation:
    """Field-shadow row — no truth model.

    Two flavors:
      - ``contextual_shadow_with_bias_hint`` when we have a historical
        shrunk_bias with enough sample (|shrunk_bias| >= 0.03 and n >= 8)
        to justify a small directional nudge from field.
      - ``contextual_shadow`` otherwise — submit at p_field exactly.

    Either way, ``p_truth`` is reported as p_field + shrunk_bias to keep
    the diagnostic columns coherent; the actual submission is clamped tight.
    """
    p_field, field_source, field_conf = estimate_p_field(
        qt, question_row, match_ctx, prior
    )
    if p_field is None:
        return _weak_field_rec(qt, prior)

    is_candidate, promo_status, candidate_reason = evaluate_historical_candidate(
        qt, prior
    )

    has_bias_hint = (
        prior.n >= 8 and abs(prior.shrunk_bias) >= 0.03
    )

    if has_bias_hint:
        tier = "contextual_shadow_with_bias_hint"
        confidence, max_dev = EVIDENCE_TIERS[tier]
        p_truth = _clamp(p_field + prior.shrunk_bias, 0.02, 0.98)
        p_submit = _blend(p_truth, p_field, confidence, max_dev)
        reason = (
            f"No market for {qt}; field estimate from {field_source}; "
            f"small bias hint (shrunk_bias={prior.shrunk_bias:+.3f}, n={prior.n}), "
            f"capped ±{max_dev:.2f}."
        )
        truth_source = f"shrunk_historical_bias_n{prior.n}"
        mode = tier
    else:
        tier = "contextual_shadow"
        confidence, max_dev = EVIDENCE_TIERS[tier]
        p_truth = p_field  # no truth signal — sit on field
        p_submit = p_field
        reason = (
            f"No market and no actionable bias for {qt}; shadow field "
            f"({field_source})."
        )
        truth_source = "shadow_field"
        mode = tier

    return _pack(
        p_truth=p_truth,
        p_truth_source=truth_source,
        truth_confidence=confidence,
        p_field=p_field,
        p_field_source=field_source,
        field_confidence=field_conf,
        p_submit=p_submit,
        decision_mode=mode,
        risk_tags=_risk_tags(qt),
        reason=reason,
        historical_candidate=is_candidate,
        candidate_n=prior.n,
        candidate_raw_bias=prior.raw_bias,
        candidate_reason=candidate_reason,
        promotion_status=promo_status,
    )


def _strong_historical_rec(
    qt: str,
    question_row: dict[str, Any],
    match_ctx: MatchContext,
    prior: PriorRow,
) -> Recommendation:
    """Apply a real fade based on promoted historical signal.

    Only reached when ``qt`` is in :data:`STRONG_HISTORICAL_ALLOWLIST` AND
    the statistical screen has cleared. The allowlist is empty at launch
    so this is dead code until promotion is justified by calibration data.
    """
    p_field, field_source, field_conf = estimate_p_field(
        qt, question_row, match_ctx, prior
    )
    if p_field is None:
        return _weak_field_rec(qt, prior)
    confidence, max_dev = EVIDENCE_TIERS["strong_historical"]
    p_truth = _clamp(p_field + prior.shrunk_bias, 0.02, 0.98)
    p_submit = _blend(p_truth, p_field, confidence, max_dev)
    reason = (
        f"Strong-historical promoted: {qt} (n={prior.n}, "
        f"raw_bias={prior.raw_bias:+.3f}); blended {confidence:.2f}·truth + "
        f"{1-confidence:.2f}·field, capped ±{max_dev:.2f}."
    )
    return _pack(
        p_truth=p_truth,
        p_truth_source=f"promoted_historical_bias_n{prior.n}",
        truth_confidence=confidence,
        p_field=p_field,
        p_field_source=field_source,
        field_confidence=field_conf,
        p_submit=p_submit,
        decision_mode="strong_historical",
        risk_tags=_risk_tags(qt),
        reason=reason,
        historical_candidate=True,
        candidate_n=prior.n,
        candidate_raw_bias=prior.raw_bias,
        candidate_reason="allowlisted promotion",
        promotion_status="promoted",
    )


def _weak_field_rec(qt: str, prior: PriorRow) -> Recommendation:
    """No defensible field estimate — route to review, no submission."""
    return _pack(
        p_truth=None,
        p_truth_source="none",
        truth_confidence=0.0,
        p_field=None,
        p_field_source="no_estimate",
        field_confidence=0.0,
        p_submit=None,
        decision_mode="weak_field",
        risk_tags=_risk_tags(qt),
        reason=(
            f"No semantic anchor and no historical sample for {qt!r}; "
            f"cannot estimate field. Route to manual review."
        ),
        needs_manual_review=True,
        historical_candidate=False,
        candidate_n=prior.n,
        candidate_raw_bias=prior.raw_bias,
        candidate_reason="",
        promotion_status="no_signal",
    )


def _review_rec(qt: str, reason: str, prior: PriorRow) -> Recommendation:
    """Genuinely unparseable question — distinct from weak_field for clarity."""
    return _pack(
        p_truth=None,
        p_truth_source="none",
        truth_confidence=0.0,
        p_field=None,
        p_field_source="none",
        field_confidence=0.0,
        p_submit=None,
        decision_mode="review",
        risk_tags=_risk_tags(qt),
        reason=reason,
        needs_manual_review=True,
        promotion_status="no_signal",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recommend_submission(
    question_row: dict[str, Any] | pd.Series,
    market_context: dict[str, Any],
    historical_context: dict[str, PriorRow] | None = None,
    lineup_context: dict[str, Any] | None = None,
) -> Recommendation:
    """Build a single Recommendation.

    Routing order:
      1. Direct liquid market -> ``direct_market`` (submit market_prob).
      2. Derived market (BTTS+Over2.5 today) -> ``derived_market``.
      3. Strong-historical promoted qt -> ``strong_historical`` (empty allowlist).
      4. Defensible field estimate available -> ``contextual_shadow*``.
      5. No estimate possible -> ``weak_field`` / ``review``.

    Default-to-submit is the caller's responsibility — the engine never sets
    a recommendation; it produces ``p_submit`` and ``decision_mode`` and lets
    ``submit_sheet`` decide. ``decision_mode == "weak_field"`` and ``"review"``
    are the only modes the caller should refuse to auto-submit on.
    """
    priors = historical_context if historical_context is not None else load_priors()
    if isinstance(question_row, pd.Series):
        question_row = question_row.to_dict()
    qt = str(question_row.get("question_type") or "").strip().lower()
    mapping: Mapping | None = market_context.get("mapping")
    consensus: pd.DataFrame | None = market_context.get("consensus")
    match_ctx = extract_match_context(consensus)
    prior = _prior_for(priors, qt)

    # Resolve lineup context once for player-prop questions; used for
    # routing (step 3) and to append lineup features as risk tags on the
    # final recommendation regardless of which path it took.
    player_ctx: PlayerContext | None = None
    if qt in LINEUP_AWARE_QUESTION_TYPES:
        player_ctx = _resolve_player_context(question_row, lineup_context)

    def _finalize(rec: Recommendation) -> Recommendation:
        """Attach lineup feature tags (when applicable) and enforce discipline.

        Lineup status/role belong on a player-prop row as features even when
        it maps to a market or shadow. Discipline enforcement runs on EVERY
        recommendation, not just player props, so heuristic truth sources are
        always capped and labelled.
        """
        if player_ctx is not None and _has_lineup_signal(player_ctx):
            feats = _build_prop_features(
                qt, question_row, match_ctx, prior, player_ctx, mapping=mapping
            )
            for tag in feats.risk_tags:
                if tag not in rec.risk_tags:
                    rec.risk_tags.append(tag)
        return _enforce_truth_discipline(rec)

    # 1. Direct market
    if (
        mapping is not None
        and mapping.mapping_status == "mapped_exact"
        and mapping.market_prob is not None
    ):
        return _finalize(_direct_market_rec(mapping, qt, match_ctx, prior))

    # 2. Derived market — currently only compound_btts_over_2_5.
    if qt == "compound_btts_over_2_5":
        derived = _derive_btts_over_2_5(consensus, match_ctx)
        if derived is not None:
            p_truth, source_label, derive_context = derived
            return _finalize(_derived_market_rec(
                qt, p_truth, source_label, derive_context, match_ctx, prior, question_row
            ))

    # 3. Player props with lineup info but no direct/derived market or
    #    trained model — route to manual review. Lineup status/role is
    #    surfaced as structured risk tags by the builder; it never feeds
    #    into an automated p_truth (hard-coded role/status tables are
    #    deliberately not used here).
    if player_ctx is not None and _has_lineup_signal(player_ctx):
        return _finalize(
            _player_prop_review_required_rec(
                qt, question_row, match_ctx, prior, player_ctx
            )
        )

    # 4. Strong-historical (gated on allowlist; empty at launch).
    if qt in STRONG_HISTORICAL_ALLOWLIST:
        is_candidate, _, _ = evaluate_historical_candidate(qt, prior)
        if is_candidate:
            return _finalize(
                _strong_historical_rec(qt, question_row, match_ctx, prior)
            )

    # 5. Contextual shadow (with or without bias hint).
    p_field_check, _, _ = estimate_p_field(qt, question_row, match_ctx, prior)
    if p_field_check is not None:
        return _finalize(_shadow_rec(qt, question_row, match_ctx, prior))

    # 6. Truly no estimate — either the question type is unknown to the
    #    semantic table and we have no prior, or the mapping is broken
    #    (ambiguous team spelling etc.).
    if mapping is not None and mapping.mapping_status in (
        "ambiguous_review",
        "low_liquidity_review",
        "needs_model",
    ):
        return _finalize(_review_rec(
            qt,
            reason=(
                f"Mapping status={mapping.mapping_status} for {qt}; no "
                f"semantic anchor or historical prior to estimate field. "
                f"{mapping.mapped_bet_description or ''}".strip()
            ),
            prior=prior,
        ))
    return _finalize(_review_rec(
        qt,
        reason=f"Unsupported question_type={qt!r}; no market, no anchor, no prior.",
        prior=prior,
    ))


__all__ = [
    "Recommendation",
    "PriorRow",
    "MatchContext",
    "load_priors",
    "recommend_submission",
    "extract_match_context",
    "classify_team_role",
    "classify_player_role",
    "estimate_p_field",
    "evaluate_historical_candidate",
    "estimate_exact_1_1",
    "get_semantic_anchor",
    "is_truth_source_production_grade",
    "EVIDENCE_TIERS",
    "RISK_TAGS_BY_QT",
    "SEMANTIC_FIELD_ANCHORS",
    "STRONG_HISTORICAL_ALLOWLIST",
    "STRONG_CANDIDATE_NX_BIAS_SQ",
    "HISTORICAL_PATH_DEFAULT",
    "PRODUCTION_TRUTH_SOURCE_PREFIXES",
    "HEURISTIC_TRUTH_CONFIDENCE_CAP",
]
