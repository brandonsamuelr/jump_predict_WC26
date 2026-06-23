"""Market-anchored goals model: calibrate team scoring rates to the market,
then derive probabilities for goals-based questions that have no direct line.

Discipline
----------
The ONLY inputs are de-vigged market quantities (1X2 + total goals). We do
not invent team strength — we back it out of the market, so every derived
probability is anchored to the same sharp information the books price. This
covers ONLY goals-based questions; shots/corners/cards need a separate rate
layer (not here) and must not be faked from this model.

Calibration (independent-Poisson, two equations / two unknowns)
---------------------------------------------------------------
1. Total: match goals N ~ Poisson(lam_total). Solve lam_total so that
   P(N >= ceil(line)+? ) reproduces the market Over probability.
2. Split: lam_home = s*lam_total, lam_away = (1-s)*lam_total. P(home win)
   under independent Poissons is monotonic in s; solve s to reproduce the
   market home-win probability.

Half split: goals are ~45% first half / ~55% second half empirically
(``H1_SHARE``, documented & configurable). Within a half, goal times are
exchangeable, so "who scored first" is Rao-Blackwellised from the counts
(no need to draw times).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Empirical share of match goals scored in the first half. ~44-46% across
# large samples; 0.45 is the documented default. Override per call.
H1_SHARE = 0.45

_KMAX = 15  # goal grid cap for analytic Poisson comparisons (P(N>=16)~0)


def _poisson_pmf(lam: float, kmax: int = _KMAX) -> np.ndarray:
    k = np.arange(0, kmax + 1)
    return np.exp(-lam) * np.power(lam, k) / np.array([math.factorial(i) for i in k])


def _p_over(lam_total: float, line: float) -> float:
    """P(total goals strictly exceeds ``line``) for Poisson(lam_total).

    For a half-integer line L (e.g. 2.5), Over means N >= ceil(L) = L+0.5.
    """
    threshold = math.ceil(line)  # 2.5 -> 3
    pmf = _poisson_pmf(lam_total)
    return float(pmf[threshold:].sum())


def _p_home_win(lam_h: float, lam_a: float) -> float:
    ph = _poisson_pmf(lam_h)
    pa = _poisson_pmf(lam_a)
    # P(X > Y) = sum_y pa[y] * P(X > y)
    p = 0.0
    for y in range(len(pa)):
        p += pa[y] * ph[y + 1:].sum()
    return float(p)


def _bisect(f, lo: float, hi: float, target: float, tol: float = 1e-6, it: int = 80):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


@dataclass
class MatchModel:
    home_team: str
    away_team: str
    lam_home: float
    lam_away: float
    h1_share: float
    # calibration diagnostics (market vs reproduced)
    market_p_home: float
    market_p_over: float
    over_line: float
    fit_p_home: float = field(default=float("nan"))
    fit_p_over: float = field(default=float("nan"))


def calibrate(
    home_team: str,
    away_team: str,
    p_home: float,
    p_over: float,
    over_line: float = 2.5,
    h1_share: float = H1_SHARE,
) -> MatchModel:
    """Back out (lam_home, lam_away) from de-vigged 1X2 + total."""
    lam_total = _bisect(lambda L: _p_over(L, over_line), 0.05, 9.0, p_over)
    s = _bisect(
        lambda s: _p_home_win(s * lam_total, (1 - s) * lam_total),
        0.01, 0.99, p_home,
    )
    lam_h, lam_a = s * lam_total, (1 - s) * lam_total
    return MatchModel(
        home_team=home_team, away_team=away_team,
        lam_home=lam_h, lam_away=lam_a, h1_share=h1_share,
        market_p_home=p_home, market_p_over=p_over, over_line=over_line,
        fit_p_home=_p_home_win(lam_h, lam_a),
        fit_p_over=_p_over(lam_h + lam_a, over_line),
    )


@dataclass
class Sim:
    """Vectorised simulated goal counts (per half, per team)."""
    n1h: np.ndarray
    n2h: np.ndarray
    n1a: np.ndarray
    n2a: np.ndarray
    home_team: str
    away_team: str

    @property
    def nh(self) -> np.ndarray:
        return self.n1h + self.n2h

    @property
    def na(self) -> np.ndarray:
        return self.n1a + self.n2a

    def _half(self, team: str, half: int) -> np.ndarray:
        is_home = team.strip().lower() == self.home_team.strip().lower()
        if half == 1:
            return self.n1h if is_home else self.n1a
        return self.n2h if is_home else self.n2a

    def _team_total(self, team: str) -> np.ndarray:
        is_home = team.strip().lower() == self.home_team.strip().lower()
        return self.nh if is_home else self.na

    def _first_goal_prob(self, team: str) -> np.ndarray:
        """Per-sim conditional P(team scores match's first goal | counts)."""
        is_home = team.strip().lower() == self.home_team.strip().lower()
        tot1 = self.n1h + self.n1a
        tot2 = self.n2h + self.n2a
        my1 = self.n1h if is_home else self.n1a
        my2 = self.n2h if is_home else self.n2a
        p = np.zeros_like(tot1, dtype=float)
        in1 = tot1 > 0
        p[in1] = my1[in1] / tot1[in1]
        in2 = (tot1 == 0) & (tot2 > 0)
        p[in2] = my2[in2] / tot2[in2]
        return p


def simulate(model: MatchModel, n: int = 200_000, seed: int = 7) -> Sim:
    rng = np.random.default_rng(seed)
    h1 = model.h1_share
    return Sim(
        n1h=rng.poisson(model.lam_home * h1, n),
        n2h=rng.poisson(model.lam_home * (1 - h1), n),
        n1a=rng.poisson(model.lam_away * h1, n),
        n2a=rng.poisson(model.lam_away * (1 - h1), n),
        home_team=model.home_team, away_team=model.away_team,
    )


# --- question evaluators (goals-based ONLY) --------------------------------

def p_team_score_any(sim: Sim, team: str) -> float:
    return float((sim._team_total(team) >= 1).mean())


def p_team_score_1h(sim: Sim, team: str) -> float:
    return float((sim._half(team, 1) >= 1).mean())


def p_team_score_2h(sim: Sim, team: str) -> float:
    return float((sim._half(team, 2) >= 1).mean())


def p_second_half_more_goals(sim: Sim) -> float:
    g1 = sim.n1h + sim.n1a
    g2 = sim.n2h + sim.n2a
    return float((g2 > g1).mean())


def p_team_more_goals_2h(sim: Sim, team: str) -> float:
    mine = sim._half(team, 2)
    other_is_home = team.strip().lower() != sim.home_team.strip().lower()
    other = sim.n2h if other_is_home else sim.n2a
    return float((mine > other).mean())


def p_compound_first_goal_score_2h(sim: Sim, first_team: str, score_team: str) -> float:
    """P(first_team scores match's first goal AND score_team scores in 2H)."""
    hp = sim._first_goal_prob(first_team)
    score2h = (sim._half(score_team, 2) >= 1).astype(float)
    return float((hp * score2h).mean())


def p_compound_btts_over_2_5(sim: Sim) -> float:
    return float(((sim.nh >= 1) & (sim.na >= 1) & (sim.nh + sim.na >= 3)).mean())


def p_total_goals_2h_over(sim: Sim, threshold: int) -> float:
    """P(total 2H goals, both teams, >= ``threshold``). "2 or more" -> thr=2."""
    return float((sim.n2h + sim.n2a >= threshold).mean())


# cross-checks against the market (calibration validation)
def p_home_win(sim: Sim) -> float:
    return float((sim.nh > sim.na).mean())


def p_over_2_5(sim: Sim) -> float:
    return float((sim.nh + sim.na >= 3).mean())


def p_btts(sim: Sim) -> float:
    return float(((sim.nh >= 1) & (sim.na >= 1)).mean())


def p_halftime_draw(sim: Sim) -> float:
    return float((sim.n1h == sim.n1a).mean())


__all__ = [
    "H1_SHARE", "MatchModel", "Sim", "calibrate", "simulate",
    "p_team_score_any", "p_team_score_1h", "p_team_score_2h",
    "p_second_half_more_goals", "p_team_more_goals_2h",
    "p_compound_first_goal_score_2h", "p_compound_btts_over_2_5",
    "p_total_goals_2h_over",
    "p_home_win", "p_over_2_5", "p_btts", "p_halftime_draw",
]
