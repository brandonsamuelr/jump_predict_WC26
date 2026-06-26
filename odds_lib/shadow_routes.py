"""Live access to the OOS-gate verdicts + MEASURED base-rate anchors for the
newly-founded shadow families (data/models/shadow_routes.json).

The anchors here are MEASURED corpus rates (e.g. both_teams_sot_1h 0.74, team
corners >=5 0.50) -- the honest fallback floor when the founded estimator can't
run. They are NEVER 0.50-by-default; a constant only ever equals a number read
off the corpus. Fit/persist via scripts/fit_shadow_routes.py.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path("data/models/shadow_routes.json")
OFFSIDE_TABLE_PATH = Path("data/models/offside_team_rates.json")


@lru_cache(maxsize=1)
def _load(path_str: str = str(MODEL_PATH)) -> dict:
    try:
        return json.loads(Path(path_str).read_text())
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_offside_table(path_str: str = str(OFFSIDE_TABLE_PATH)) -> dict:
    try:
        return json.loads(Path(path_str).read_text())
    except Exception:
        return {}


def _norm_team(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _poisson_sf_ge(k: int, lam: float) -> float:
    """P(X >= k) for X~Poisson(lam), computed exactly (small k only)."""
    cdf = sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(int(k)))
    return max(0.0, min(1.0, 1.0 - cdf))


def offside_team_rate(team: str, line: float) -> float | None:
    """FOUNDED per-team offside rate: P(team offsides >= ceil(line)) from the empirical-Bayes
    MEASURED per-team lambda (offside_team_rates.json, OOS-gated, k=1 ship-raw). COVERED teams
    (>= min_games of measured history) only; an UNCOVERED team returns None so the caller falls
    to the pooled FLOOR (never a guessed default). Refreshes with the table (re-run the fit)."""
    tbl = _load_offside_table()
    # LIVE gate invariant: deploy per-team rates ONLY if the table beat the floor OOS on its
    # last fit. A refresh that fails the gate self-disables -> every team falls to the floor.
    if not (tbl.get("oos_gate") or {}).get("beats_floor"):
        return None
    rec = (tbl.get("teams") or {}).get(_norm_team(team))
    if not rec or "lambda_hat" not in rec:
        return None
    return round(_poisson_sf_ge(_line_to_ge(float(line)), float(rec["lambda_hat"])), 4)


def offside_pooled_prior(line: float) -> float | None:
    """The EB POOLED PRIOR P(team offsides >= ceil(line)) -- i.e. the n=0 limit of the per-team
    offside model (offside_team_rates.json pooled_rate). An UNCOVERED team is just a team with
    no measured history -> the prior IS its honest estimate. This UNIFIES the offside route into
    one EB model (covered = shrunk per-team; uncovered = pooled prior), retiring the separate
    hand-set 0.45 floor and self-refreshing as the table updates. None if the table is absent."""
    lam = (_load_offside_table() or {}).get("pooled_rate")
    if lam is None:
        return None
    return round(_poisson_sf_ge(_line_to_ge(float(line)), float(lam)), 4)


def both_sot_1h_validated() -> bool:
    """True iff the volume model beat the base rate OOS (else caller uses base rate)."""
    return _load().get("both_teams_sot_1h", {}).get("verdict") == "validated"


def both_sot_1h_base_rate() -> float | None:
    return _load().get("both_teams_sot_1h", {}).get("base_rate")


def both_sot_2h_base_rate() -> float | None:
    """Measured P(both teams >=1 2H SOT) -- degenerate fallback ONLY (no engine lambda).
    The live route ships the RAW closed-form true P (k=1); this is never blended toward."""
    return _load().get("both_teams_sot_2h_1plus", {}).get("base_rate")


def offsides_is_floor_no_edge() -> bool:
    """True: the per-match driver search found NO signal beating base OOS -> the offsides
    rate is an honest last-resort FLOOR (no edge), not a founded per-match model."""
    return bool(_load().get("team_offsides_over_ge2", {}).get("is_floor_no_edge"))


def offsides_rate(line: float) -> float | None:
    """MEASURED P(team offsides >= ceil(line)) -- the POOLED rate (the home/away split is
    in-sample noise that does NOT survive OOS, so it is deliberately NOT used). This is a
    no-edge LAST-RESORT FLOOR: no per-match driver beat the pooled base OOS."""
    tbl = _load().get("team_offsides_over_ge2", {}).get("team_ge", {})
    if not tbl:
        return None
    k = _line_to_ge(float(line))
    keys = sorted(int(x) for x in tbl)
    k = min(max(k, keys[0]), keys[-1])
    return float(tbl[str(k)])


def cards_2h_rate(kind: str, line: float) -> float | None:
    """MEASURED P(2H yellow cards >= ceil(line)) floor. kind in {'team','total'}. favorite_gap
    is NOT a clean per-match driver for 2H cards (see fit gate) -> honest measured floor, not a
    model, not crowd-copy. Clamps to the measured threshold range."""
    tbl = _load().get("cards_2h", {}).get("team_ge" if kind == "team" else "total_ge", {})
    if not tbl:
        return None
    k = _line_to_ge(float(line))
    keys = sorted(int(x) for x in tbl)
    k = min(max(k, keys[0]), keys[-1])
    return float(tbl[str(k)])


def cards_2h_is_floor() -> bool:
    return bool(_load().get("cards_2h", {}).get("is_floor_no_clean_signal"))


def penalty_anchor(question_type: str) -> float | None:
    """MEASURED external anchor for penalty_or_red_card / penalty_awarded (sourced rates,
    Poisson P(>=1) + independence union; see shadow_routes.json _source). Shipped raw k=1."""
    p = _load().get("penalties", {})
    qt = (question_type or "").strip().lower()
    if "red" in qt:                          # penalty_or_red_card
        return p.get("penalty_or_red_card")
    return p.get("penalty_awarded")          # penalty_awarded (penalty only)


def first_goal_2h_anchor(is_home: bool) -> float | None:
    """Measured P(team scores the first 2H goal) -- the honest fallback used only
    when the engine lambdas are unavailable (else the route uses the closed form)."""
    m = _load().get("first_goal_2h", {})
    return m.get("home") if is_home else m.get("away")


def _line_to_ge(line: float) -> int:
    """'N or more' is stored as line N-0.5 -> threshold N; integer line N -> N."""
    return math.ceil(line) if line != int(line) else int(line)


def corner_base_rate(kind: str, line: float) -> float | None:
    """Measured P(corners >= ceil(line)) fallback. kind in {'team','total'}.
    Used only when no market ladder is available (never a flat constant)."""
    tbl = _load().get("corners", {}).get("team_ge" if kind == "team" else "total_ge", {})
    if not tbl:
        return None
    k = _line_to_ge(float(line))
    if str(k) in tbl:
        return float(tbl[str(k)])
    keys = sorted(int(x) for x in tbl)                       # clamp to the measured range
    k = min(max(k, keys[0]), keys[-1])
    return float(tbl[str(k)])


__all__ = ["both_sot_1h_validated", "both_sot_1h_base_rate", "both_sot_2h_base_rate",
           "offsides_rate", "offside_team_rate", "offside_pooled_prior", "offsides_is_floor_no_edge", "cards_2h_rate",
           "cards_2h_is_floor", "penalty_anchor", "first_goal_2h_anchor", "corner_base_rate",
           "MODEL_PATH", "OFFSIDE_TABLE_PATH"]
