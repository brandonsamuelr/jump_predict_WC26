"""R32 true-P routing table: assign every live Round-of-32 question a route_class +
full provenance + a submit/skip decision, auditable per row (Deliverable 1).

Discipline encoded here (not vibes):
  - advance != 90' 1X2 (settlement_scope makes the ET+pens vs regulation split explicit);
  - player props route to the TIERED de-vig (exact two-sided > same-slate prior > global
    ~1.045), NEVER legacy 1.06 by status quo;
  - hydration/time-window rows split MARKET-component (goal/corner volume from a market) vs
    MODEL-component (offside/card/sub/stoppage) -- the latter are the only skip-candidates,
    and ONLY if a known directional bias is flagged (uncertainty alone is never a skip);
  - LOW_CONFIDENCE_PRIOR rows name a measured base rate, never a silent 0.50;
  - default = SUBMIT (volume is valuable under relative-Brier); skip keys on
    expected-worse-than-crowd, not on uncertainty.

This module assigns ROUTES + PROVENANCE only. Probabilities are produced at lock by the
existing pipeline (slate/engine/devig_tiered/time_window) per the route_class.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROUTE_CLASSES = (
    "MARKET_EXACT", "ADVANCE_MARKET", "MARKET_COMPONENT_TRANSFORM", "ENGINE_FOUNDED",
    "PLAYER_PROP_TIERED_DEVIG", "TIME_WINDOW_MARKET_COMPONENT", "TIME_WINDOW_MODEL_COMPONENT",
    "LOW_CONFIDENCE_PRIOR", "UNSUPPORTED_SKIP",
)

REQUIRED_FIELDS = (
    "route_class", "confidence_bucket", "market_source", "component_source",
    "overround_source", "time_window_source", "route_reason", "settlement_scope",
    "submit_recommendation", "submit_reason",
)

REG = "regulation_90+stoppage"
_N = "n/a"


def _r(route_class, confidence, settlement, route_reason, known_failure_modes,
       market_source=_N, component_source=_N, overround_source=_N, time_window_source=_N,
       directional_bias=False):
    return dict(route_class=route_class, confidence_bucket=confidence, settlement_scope=settlement,
                route_reason=route_reason, known_failure_modes=known_failure_modes,
                market_source=market_source, component_source=component_source,
                overround_source=overround_source, time_window_source=time_window_source,
                directional_bias=directional_bias)


# qtype -> route fields. settlement lives here (qtype encodes scope).
ROUTE_BY_QTYPE: dict[str, dict] = {
    # --- MARKET_EXACT (2-sided market of the identical event) ---
    "btts": _r("MARKET_EXACT", "high", REG, "identical event = liquid 2-sided BTTS market",
               "none material", market_source="btts", overround_source="two_sided_normalization"),
    "match_total_over": _r("MARKET_EXACT", "high", REG, "match totals 2-sided at the contest line",
               "line!=quoted -> Poisson interpolation", market_source="totals(over)",
               overround_source="two_sided_normalization"),
    "match_total_under": _r("MARKET_EXACT", "high", REG, "match totals 2-sided (under side)",
               "line!=quoted -> Poisson interpolation", market_source="totals(under)",
               overround_source="two_sided_normalization"),
    "team_total_over": _r("MARKET_EXACT", "high", REG, "team-totals 2-sided market",
               "thin team-totals -> wider de-vig", market_source="team_totals(over)",
               overround_source="two_sided_normalization"),
    "team_win": _r("MARKET_EXACT", "high", REG, "WIN IN REGULATION = 90' 1X2; explicitly NOT advance",
               "must never be read as advance/to-qualify", market_source="h2h(1X2,90')",
               overround_source="two_sided_normalization(3way)"),
    "team_win_margin": _r("MARKET_EXACT", "medium", REG, "win-by-2 = -1.5 handicap, regulation only",
               "alt-handicap line interpolation", market_source="spreads(-1.5,90')",
               overround_source="two_sided_normalization"),
    "match_draw": _r("MARKET_EXACT", "high", REG, "regulation tie = 90' draw; NOT an advance question",
               "must never be read as advance", market_source="h2h(draw,90')",
               overround_source="two_sided_normalization(3way)"),
    "halftime_team_lead": _r("MARKET_EXACT", "medium", "first_half", "ahead at HT = 1H result market",
               "1H market thinner", market_source="h2h_h1", overround_source="two_sided_normalization(3way)"),
    "first_half_total_over": _r("MARKET_EXACT", "medium", "first_half", "1H 2+ goals = 1H totals market",
               "1H totals thin", market_source="totals_h1(over1.5)", overround_source="two_sided_normalization"),
    # --- ADVANCE (ET+pens; derived, no exact market in feed) ---
    "advance": _r("ADVANCE_MARKET", "medium", "ET+penalties_inclusive",
               "advance INCLUDES ET+pens. HIERARCHY: (1) EXACT to-qualify market -- feed OR external "
               "book/Kalshi/Polymarket, settlement-verified + DE-VIGGED -> use as exact; (2) else DERIVE "
               "P(win90)+P(draw90)*P(win|draw) with ODDS-DERIVED P(win|draw). feed-absence != "
               "market-absence (check external at lock); NEVER flat 0.5, NEVER bare 1X2",
               "external line must be settlement-verified (same tie, ET+pens, advancement-NOT-match-win, "
               "not a stale outright) + de-vigged both sides; if derived, odds-derived P(win|draw) tilt + pens variance",
               market_source="exact_to_qualify if accessible (feed/external, de-vigged) ELSE derived_ET_pens",
               component_source="h2h 90' win+draw + odds-derived P(win|draw) tilt (NOT flat 0.5)",
               overround_source="two_sided_normalization(3way) on 90' legs; external line -> de-vig both sides before use"),
    # --- MARKET_COMPONENT_TRANSFORM (market volume -> threshold/comparison) ---
    "total_corners_over": _r("MARKET_COMPONENT_TRANSFORM", "medium", REG,
               "corner volume from market -> Poisson tail P(>=line)",
               "line!=quoted -> Poisson tail; corner-provider settlement",
               market_source="alternate_totals_corners/Pinnacle", component_source="corner total lambda",
               overround_source="two_sided_normalization(corners)"),
    "team_corners_over": _r("MARKET_COMPONENT_TRANSFORM", "medium", REG,
               "team corner volume from market -> Poisson P(>=line)", "thin team-corner market",
               market_source="alternate_team_totals_corners", component_source="team corner lambda",
               overround_source="two_sided_normalization(corners)"),
    "team_more_corners": _r("MARKET_COMPONENT_TRANSFORM", "medium", REG,
               "corner comparison from market / gate-validated favorite_gap corner model",
               "full-match (no 2H game-state reversal); thin corners",
               market_source="Pinnacle more-corners/team corners", component_source="corner_cmp(favorite_gap)"),
    # --- ENGINE_FOUNDED (model from market-anchored lambdas; no direct market) ---
    "total_sot_over": _r("ENGINE_FOUNDED", "low", REG, "NO total-SOT market anywhere -> concave SOT map (sum mu)",
               "SOT level model; tail dispersion (shadow-tracked)", component_source="SOT concave map(sum team mu)"),
    "team_sot_over": _r("ENGINE_FOUNDED", "low", REG, "NO team-SOT market -> concave SOT map",
               "SOT count tail calibration (shadow-tracked)", component_source="SOT concave map(team mu)"),
    "total_cards_over": _r("ENGINE_FOUNDED", "low", REG, "card count from cards model lambda -> Poisson",
               "cards model not market-anchored; ref/tournament variance", component_source="cards model lambda"),
    "both_teams_card": _r("ENGINE_FOUNDED", "medium", REG, "both-teams-carded from cards model",
               "cards model uncertainty", component_source="cards model (both >=1)"),
    "team_more_cards": _r("ENGINE_FOUNDED", "low", REG, "card comparison from gate-validated favorite_gap model",
               "weak transfer; low base separation", component_source="more_cards model(favorite_gap)"),
    "total_offsides_over": _r("ENGINE_FOUNDED", "low", REG, "offside EB model (team rates) -> Poisson sum, P(>=4)",
               "EB shrinkage; club->intl transfer (population-anchored)", component_source="offside EB model"),
    "second_half_more_goals": _r("ENGINE_FOUNDED", "medium", REG,
               "P(2H>1H) from market-anchored goal engine + measured half-share",
               "half-share constant (market-overridden when available)", component_source="goal engine half-split"),
    "team_score_both_halves": _r("ENGINE_FOUNDED", "medium", REG, "team scores each half from joint goal dist",
               "half-share; half independence approx", component_source="goal engine"),
    "team_first_goal": _r("ENGINE_FOUNDED", "medium", REG, "P(team scores first) from market-anchored lambdas",
               "in-play dynamics not captured", component_source="goal engine (first-goal)"),
    # --- LOW_CONFIDENCE_PRIOR (honest measured base rate; submit, neutral quality) ---
    "penalty_awarded": _r("LOW_CONFIDENCE_PRIOR", "low", REG, "no penalty market -> measured population base rate (~0.30)",
               "ref/VAR variance", component_source="penalty population base rate"),
    "red_card": _r("LOW_CONFIDENCE_PRIOR", "low", REG, "no market -> measured red-card base rate (~0.22)",
               "high variance", component_source="red-card population base rate"),
    "penalty_or_red": _r("LOW_CONFIDENCE_PRIOR", "low", REG, "no market -> measured penalty-or-red base-rate union",
               "variance", component_source="penalty-or-red population base rate"),
    "any_player_2plus_goals": _r("LOW_CONFIDENCE_PRIOR", "low", REG,
               "any-player brace -> measured base rate (no clean market)", "scorer concentration varies",
               component_source="brace population base rate"),
    "any_player_2plus_sot": _r("LOW_CONFIDENCE_PRIOR", "low", REG,
               "any-player 2+ SOT -> measured base rate (no 2+ SOT market pull)", "noisy",
               component_source="any-player 2+SOT base rate"),
    "substitute_scores": _r("LOW_CONFIDENCE_PRIOR", "low", REG,
               "no market -> measured substitute-scores base rate (~0.15-0.20)", "usage-dependent",
               component_source="sub-scores population base rate"),
    "total_shots_over": _r("LOW_CONFIDENCE_PRIOR", "low", REG,
               "no total-shots market; shots~SOT transform too weak -> honest base rate", "base rate noisy",
               component_source="total-shots population base rate"),
    # --- PLAYER_PROP_TIERED_DEVIG (lineup-gated at lock) ---
    "player_goal": _r("PLAYER_PROP_TIERED_DEVIG", "medium", REG,
               "scorer prop via TIERED de-vig; exact two-sided when Yes&No present", "lineup-gated; thin market",
               market_source="player_goal_scorer_anytime",
               overround_source="tiered_devig@lock: exact_two_sided | else global~1.045"),
    "player_goal_or_assist": _r("PLAYER_PROP_TIERED_DEVIG", "medium", REG,
               "direct score-or-assist (lower-bounded by anytime-goal) via tiered de-vig",
               "lineup-gated; proxy-floor if no direct market", market_source="player_to_score_or_assist(else anytime floor)",
               overround_source="tiered_devig@lock: exact_two_sided | else global~1.045"),
    "player_sot_over": _r("PLAYER_PROP_TIERED_DEVIG", "medium", REG,
               "1+ SOT prop; one-sided in feed -> empirical global prior, NOT legacy 1.06",
               "lineup-gated; thin; one-sided vig", market_source="player_shots_on_target@0.5",
               overround_source="tiered_devig@lock: one-sided->global_player_prop_prior~1.045 (NOT 1.06)"),
    "player_2plus_sot": _r("PLAYER_PROP_TIERED_DEVIG", "low", REG,
               "2+ SOT prop if 1.5 line quoted; else model fallback", "1.5 line often absent->fallback; lineup-gated",
               market_source="player_shots_on_target@1.5(if quoted)",
               overround_source="tiered_devig@lock (else model fallback)"),
    # --- TIME_WINDOW_MARKET_COMPONENT (volume from a market) ---
    "tw_goal_before_break": _r("TIME_WINDOW_MARKET_COMPONENT", "medium", "regulation_before_~22'",
               "goal volume (market-anchored engine) x first-22' share -> posterior mean (no cap)",
               "timing-share uncertainty (integrated); ~1-3' boundary jitter",
               component_source="goal lambda (market/engine)",
               time_window_source="break1~22'(VERIFIED)+measured goal-timing share, jitter~2'"),
    "tw_goal_after_break": _r("TIME_WINDOW_MARKET_COMPONENT", "medium", "regulation_after_~67'",
               "goal volume x post-67' share -> posterior mean (no cap)", "timing-share uncertainty",
               component_source="goal lambda (market/engine)",
               time_window_source="break2~67'(VERIFIED)+late-goal share, jitter~2'"),
    "tw_corners_before_break": _r("TIME_WINDOW_MARKET_COMPONENT", "medium", "regulation_before_~22'",
               "corner volume (market) x first-22' share -> P(>=2) posterior mean", "timing-share uncertainty",
               component_source="corner lambda (market)",
               time_window_source="break1~22'(VERIFIED)+corner-timing share, jitter~2'"),
    # --- TIME_WINDOW_MODEL_COMPONENT (volume itself model-derived; only possible skip-candidates) ---
    "tw_offside_before_break": _r("TIME_WINDOW_MODEL_COMPONENT", "low", "regulation_before_~22'",
               "offside volume MODEL-derived x first-22' share -> posterior mean", "doubly-derived (model x timing); no offside market",
               component_source="offside EB model lambda",
               time_window_source="break1~22'(VERIFIED)+offside-timing share, jitter~2'"),
    "tw_card_after_break": _r("TIME_WINDOW_MODEL_COMPONENT", "low", "regulation_after_~67'_INCL_extra_time",
               "card volume MODEL-derived x post-67'(+ET) share -> posterior mean", "doubly-derived; ET-conditional mass",
               component_source="cards model lambda (late, +ET)",
               time_window_source="break2~67'->end +ET(VERIFIED)+late-card share"),
    "tw_goal_1h_stoppage": _r("TIME_WINDOW_MODEL_COMPONENT", "low", "first_half_stoppage",
               "narrow 1H-stoppage window; goal lambda x model micro-share -> posterior mean", "very narrow window; timing model",
               component_source="goal lambda x 1H-stoppage micro-share(model)", time_window_source="1H stoppage ~45'-45+x"),
    "tw_goal_2h_stoppage": _r("TIME_WINDOW_MODEL_COMPONENT", "low", "second_half_stoppage",
               "narrow 2H-stoppage window; goal lambda x model micro-share -> posterior mean", "narrow window; timing model",
               component_source="goal lambda x 2H-stoppage micro-share(model)", time_window_source="2H stoppage ~90'-90+x"),
    "tw_sub_before_ht": _r("TIME_WINDOW_MODEL_COMPONENT", "low", "first_half",
               "early substitution (injury-driven) measured base rate before 45'", "rare event; doubly-derived",
               component_source="early-sub base rate (model)", time_window_source="before 45'"),
    "tw_card_first_half": _r("TIME_WINDOW_MODEL_COMPONENT", "low", "first_half",
               "card volume (model) x 1H share -> P(>=1) posterior mean", "doubly-derived (cards model x half share)",
               component_source="cards model lambda x 1H share", time_window_source="first half 0-45'"),
}


# (match, question_number, question_text, qtype) for the 60 LIVE R32 rows.
ROWS: list[tuple] = [
    # CANADA vs SOUTH AFRICA
    ("Canada vs South Africa", "Q1", "card after second hydration break, including extra time", "tw_card_after_break"),
    ("Canada vs South Africa", "Q2", "both teams score in regulation", "btts"),
    ("Canada vs South Africa", "Q3", "offside before first hydration break", "tw_offside_before_break"),
    ("Canada vs South Africa", "Q4", "substitute scores in regulation", "substitute_scores"),
    ("Canada vs South Africa", "Q5", "goal before first hydration break", "tw_goal_before_break"),
    ("Canada vs South Africa", "Q6", "9+ corners in regulation", "total_corners_over"),
    ("Canada vs South Africa", "Q7", "regulation ends in a tie", "match_draw"),
    ("Canada vs South Africa", "Q8", "Canada ahead at halftime", "halftime_team_lead"),
    ("Canada vs South Africa", "Q9", "South Africa advance to R16", "advance"),
    ("Canada vs South Africa", "Q10", "penalty awarded in regulation", "penalty_awarded"),
    ("Canada vs South Africa", "Q11", "2H more goals than 1H, excluding extra time", "second_half_more_goals"),
    ("Canada vs South Africa", "Q12", "any player 2+ goals in regulation", "any_player_2plus_goals"),
    ("Canada vs South Africa", "Q13", "Cyle Larin scores in regulation", "player_goal"),
    ("Canada vs South Africa", "Q14", "4+ total cards in regulation", "total_cards_over"),
    ("Canada vs South Africa", "Q15", "Iqraam Rayners 1+ SOT in regulation", "player_sot_over"),
    # BRAZIL vs JAPAN
    ("Brazil vs Japan", "Q1", "Brazil 2+ goals in regulation", "team_total_over"),
    ("Brazil vs Japan", "Q2", "8+ total SOT in regulation", "total_sot_over"),
    ("Brazil vs Japan", "Q3", "any player 2+ SOT in regulation", "any_player_2plus_sot"),
    ("Brazil vs Japan", "Q4", "Japan advance to R16", "advance"),
    ("Brazil vs Japan", "Q5", "9+ corners in regulation", "total_corners_over"),
    ("Brazil vs Japan", "Q6", "goal before first hydration break", "tw_goal_before_break"),
    ("Brazil vs Japan", "Q7", "4+ total cards in regulation", "total_cards_over"),
    ("Brazil vs Japan", "Q8", "substitute scores in regulation", "substitute_scores"),
    ("Brazil vs Japan", "Q9", "Matheus Cunha scores in regulation", "player_goal"),
    ("Brazil vs Japan", "Q10", "any player 2+ goals in regulation", "any_player_2plus_goals"),
    ("Brazil vs Japan", "Q11", "offside before first hydration break", "tw_offside_before_break"),
    ("Brazil vs Japan", "Q12", "3+ total goals in regulation", "match_total_over"),
    ("Brazil vs Japan", "Q13", "Brazil scores in both halves of regulation", "team_score_both_halves"),
    ("Brazil vs Japan", "Q14", "Ayase Ueda 1+ SOT in regulation", "player_sot_over"),
    ("Brazil vs Japan", "Q15", "regulation ends in a tie", "match_draw"),
    # MOROCCO vs NETHERLANDS
    ("Morocco vs Netherlands", "Q1", "Morocco ahead at halftime", "halftime_team_lead"),
    ("Morocco vs Netherlands", "Q2", "card after second hydration break, including extra time", "tw_card_after_break"),
    ("Morocco vs Netherlands", "Q3", "goal in first-half stoppage time", "tw_goal_1h_stoppage"),
    ("Morocco vs Netherlands", "Q4", "goal after second hydration break", "tw_goal_after_break"),
    ("Morocco vs Netherlands", "Q5", "red card shown", "red_card"),
    ("Morocco vs Netherlands", "Q6", "2H more goals than 1H, excluding extra time", "second_half_more_goals"),
    ("Morocco vs Netherlands", "Q7", "Netherlands score first", "team_first_goal"),
    ("Morocco vs Netherlands", "Q8", "Brian Brobbey 2+ SOT in regulation", "player_2plus_sot"),
    ("Morocco vs Netherlands", "Q9", "Netherlands more corners than Morocco in regulation", "team_more_corners"),
    ("Morocco vs Netherlands", "Q10", "Netherlands win in regulation", "team_win"),
    ("Morocco vs Netherlands", "Q11", "Ismael Saibari score-or-assist in regulation", "player_goal_or_assist"),
    ("Morocco vs Netherlands", "Q12", "both teams score in regulation", "btts"),
    ("Morocco vs Netherlands", "Q13", "both teams receive 1+ card in regulation", "both_teams_card"),
    ("Morocco vs Netherlands", "Q14", "first half 2+ goals", "first_half_total_over"),
    ("Morocco vs Netherlands", "Q15", "penalty awarded in regulation", "penalty_awarded"),
    # UNITED STATES vs BOSNIA & HERZEGOVINA
    ("United States vs Bosnia & Herzegovina", "Q1", "goal in second-half stoppage time", "tw_goal_2h_stoppage"),
    ("United States vs Bosnia & Herzegovina", "Q2", "2+ corners before first hydration break", "tw_corners_before_break"),
    ("United States vs Bosnia & Herzegovina", "Q3", "USA 6+ corners in regulation", "team_corners_over"),
    ("United States vs Bosnia & Herzegovina", "Q4", "substitution before halftime", "tw_sub_before_ht"),
    ("United States vs Bosnia & Herzegovina", "Q5", "5+ total cards in regulation", "total_cards_over"),
    ("United States vs Bosnia & Herzegovina", "Q6", "penalty OR red card in regulation", "penalty_or_red"),
    ("United States vs Bosnia & Herzegovina", "Q7", "USA win by 2+ in regulation", "team_win_margin"),
    ("United States vs Bosnia & Herzegovina", "Q8", "Bosnia more cards than USA in regulation", "team_more_cards"),
    ("United States vs Bosnia & Herzegovina", "Q9", "Folarin Balogun scores in regulation", "player_goal"),
    ("United States vs Bosnia & Herzegovina", "Q10", "2 or fewer total goals in regulation", "match_total_under"),
    ("United States vs Bosnia & Herzegovina", "Q11", "4+ offside calls in regulation", "total_offsides_over"),
    ("United States vs Bosnia & Herzegovina", "Q12", "Ermedin Demirović 1+ SOT in regulation", "player_sot_over"),
    ("United States vs Bosnia & Herzegovina", "Q13", "card in first half", "tw_card_first_half"),
    ("United States vs Bosnia & Herzegovina", "Q14", "22+ total shots in regulation", "total_shots_over"),
    ("United States vs Bosnia & Herzegovina", "Q15", "USA 6+ SOT in regulation", "team_sot_over"),
]

TABLE_COLUMNS = (
    "match", "question_number", "question_text", "settlement_scope", "route_class",
    "component_source", "market_source", "overround_source", "time_window_source",
    "confidence_bucket", "submit_recommendation", "submit_reason", "route_reason",
    "known_failure_modes",
)


def submit_decision(route_class: str, directional_bias: bool) -> tuple[str, str]:
    """The skip rule: DEFAULT submit. Skip ONLY if the route is unsupported OR the row carries a
    FLAGGED known directional bias (i.e. positive reason it is worse than the crowd). Uncertainty
    alone is NEVER a skip reason."""
    if route_class == "UNSUPPORTED_SKIP":
        return "skip", "route unsupported / settlement unclear"
    if directional_bias:
        return "skip", "model-component with FLAGGED known directional bias (expected worse than crowd)"
    return "submit", "at least crowd-quality; volume valuable under relative-Brier (uncertainty is not a skip reason)"


def classify(match: str, question_number: str, question_text: str, qtype: str) -> dict:
    spec = ROUTE_BY_QTYPE.get(qtype)
    if spec is None:
        row = _r("UNSUPPORTED_SKIP", "skip", "unclear", f"no route mapping for qtype {qtype!r}",
                 "unmapped question")
    else:
        row = dict(spec)
    sub_rec, sub_reason = submit_decision(row["route_class"], row.get("directional_bias", False))
    return {
        "match": match, "question_number": question_number, "question_text": question_text,
        "settlement_scope": row["settlement_scope"], "route_class": row["route_class"],
        "component_source": row["component_source"], "market_source": row["market_source"],
        "overround_source": row["overround_source"], "time_window_source": row["time_window_source"],
        "confidence_bucket": row["confidence_bucket"], "submit_recommendation": sub_rec,
        "submit_reason": sub_reason, "route_reason": row["route_reason"],
        "known_failure_modes": row["known_failure_modes"],
    }


def build_table() -> list[dict]:
    return [classify(m, qn, txt, qt) for (m, qn, txt, qt) in ROWS]


def validate(table: list[dict]) -> list[str]:
    """Return a list of provenance/routing violations (empty == clean). Hard checks."""
    errs = []
    for r in table:
        tag = f"{r['match']} {r['question_number']}"
        for f in REQUIRED_FIELDS:
            if not str(r.get(f, "")).strip() or str(r.get(f)).strip() == _N and f in ("route_class", "route_reason"):
                errs.append(f"{tag}: blank/invalid required field {f}")
        if r["route_class"] not in ROUTE_CLASSES:
            errs.append(f"{tag}: unknown route_class {r['route_class']}")
        # advance must be ET+pens and never bare 1X2
        if "advance" in r["question_text"].lower():
            if r["route_class"] != "ADVANCE_MARKET":
                errs.append(f"{tag}: advance row not routed to ADVANCE_MARKET")
            if "ET" not in r["settlement_scope"] and "penalt" not in r["settlement_scope"].lower():
                errs.append(f"{tag}: advance row settlement not ET+pens")
            if "1x2" in r["market_source"].lower() and "derived" not in r["market_source"].lower():
                errs.append(f"{tag}: advance row sourced from bare 1X2")
        # win-in-regulation / regulation-tie must be regulation 90', not advance
        if ("win in regulation" in r["question_text"].lower() or "ends in a tie" in r["question_text"].lower()):
            if r["route_class"] == "ADVANCE_MARKET" or "ET" in r["settlement_scope"]:
                errs.append(f"{tag}: regulation row mis-routed to advance/ET")
        # LOW_CONFIDENCE_PRIOR must name a base rate, never a silent 0.50/default
        if r["route_class"] == "LOW_CONFIDENCE_PRIOR":
            if "base rate" not in r["component_source"].lower():
                errs.append(f"{tag}: LOW_CONFIDENCE_PRIOR without a named base rate")
            if any(x in (r["component_source"] + r["route_reason"]).lower() for x in ("0.50", "coinflip", "default")):
                errs.append(f"{tag}: LOW_CONFIDENCE_PRIOR smells like a silent 0.50/default")
    return errs


def to_csv(path: str | Path = "data/submission_sheets/2026-06-28_r32_routing_table.csv") -> Path:
    table = build_table()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        w.writeheader()
        for r in table:
            w.writerow({k: r[k] for k in TABLE_COLUMNS})
    return p


__all__ = ["ROUTE_CLASSES", "REQUIRED_FIELDS", "ROUTE_BY_QTYPE", "ROWS", "TABLE_COLUMNS",
           "submit_decision", "classify", "build_table", "validate", "to_csv"]


if __name__ == "__main__":
    t = build_table()
    errs = validate(t)
    print(f"R32 routing table: {len(t)} rows; validation {'CLEAN' if not errs else 'FAILED'}")
    for e in errs:
        print("  !!", e)
    from collections import Counter
    print("\nroute_class distribution:")
    for rc, n in Counter(r["route_class"] for r in t).most_common():
        print(f"  {rc:30} {n}")
    print("\nconfidence:", dict(Counter(r["confidence_bucket"] for r in t)))
    print("submit:", dict(Counter(r["submit_recommendation"] for r in t)))
