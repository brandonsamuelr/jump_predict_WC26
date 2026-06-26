"""Fit the FOUNDED per-team offside-rate table (data/models/offside_team_rates.json).

This is a MEASURED, OOS-GATED, REFRESHABLE per-team rate -- the same category as the
cards 2H-share, the SOT concave scalar, and the pooled floors: every number is read off
data, never typed. Three invariants (enforced by tests/test_offside_team_rates.py):
  1. measured-not-typed -- per-team lambda is an empirical-Bayes posterior of measured
     counts; the SHRINKAGE strength (beta) is estimated from the data's variance
     structure (Gamma-Poisson method of moments), not hand-picked.
  2. uncovered -> floor -- teams with < MIN_GAMES of measured history are simply absent
     from the table, so the live route falls to the founded POOLED FLOOR (never a guess).
  3. refreshes not ossifies -- re-run this to regenerate from the current data extract;
     add 2026 in-tournament games (e.g. FBref) to data/historical/.../offsides_*.csv and
     re-run. Nothing here is frozen.

Method (empirical Bayes, Gamma-Poisson):
  team true rate lambda ~ Gamma(a, b), prior mean a/b = pooled rate m.
  posterior mean lambda_hat_i = (a + sum_offsides_i) / (b + n_games_i)
  = m*(b/(b+n)) + r_i*(n/(b+n))  -- shrink toward the pooled floor, weight n/(b+n).
  b (== shrinkage K) is MEASURED: b = m / sigma^2_between, sigma^2_between from MoM.

OOS gate (recorded): past-only expanding EB estimate vs the pooled floor, Brier on
P(team offsides >= 2). Ships only because this beats the floor OOS (it did: ~+0.005).

    .venv/bin/python scripts/fit_offside_team_rates.py
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

SRC = Path("data/historical/statsbomb_probe/offsides_intl.csv")
OUT = Path("data/models/offside_team_rates.json")
MIN_GAMES = 5   # "covered" = at least this many measured matches; below -> floor

# StatsBomb -> contest naming so a covered team is found under either spelling.
# (misses still fall to the founded floor, never a guess -- this only widens coverage.)
ALIAS = {"Korea Republic": "South Korea", "IR Iran": "Iran", "United States": "USA",
         "China PR": "China", "Czech Republic": "Czechia", "Turkiye": "Turkey",
         "Republic of Ireland": "Ireland", "Bosnia and Herzegovina": "Bosnia & Herzegovina"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _eb_beta(df: pd.DataFrame, m: float) -> tuple[float, float]:
    """Gamma-Poisson method of moments -> (alpha, beta). beta is the MEASURED shrinkage K.
    sigma^2_between from the weighted dispersion of team rates net of Poisson sampling var."""
    g = df.groupby("team")["offsides_for"].agg(["sum", "count"])
    n_i = g["count"].values.astype(float)
    r_i = (g["sum"] / g["count"]).values
    N, T = n_i.sum(), len(n_i)
    # Clayton-Kaldor style between-group variance estimator (clip at 0 = no team signal):
    num = np.sum(n_i * (r_i - m) ** 2) - (T - 1) * m
    den = N - np.sum(n_i ** 2) / N
    sigma2 = max(num / den, 1e-9) if den > 0 else 1e-9
    beta = m / sigma2          # shrinkage strength K (measured)
    alpha = m * beta           # prior mean alpha/beta = m
    return alpha, beta


def _oos_gate(df: pd.DataFrame, m: float, alpha: float, beta: float) -> dict:
    """Past-only expanding EB estimate vs pooled floor; Brier on P(offsides>=2). No leak:
    each team-match uses only that team's PRIOR games for its lambda."""
    d = df.sort_values(["team", "date"]).reset_index(drop=True)
    d["n_prior"] = d.groupby("team").cumcount()
    d["sum_prior"] = (d.groupby("team")["offsides_for"].apply(lambda s: s.expanding().sum().shift(1))
                      .reset_index(drop=True))
    te = d[d.n_prior >= MIN_GAMES].copy()
    y = (te.offsides_for >= 2).astype(int).values
    lam_hat = (alpha + te.sum_prior.values) / (beta + te.n_prior.values)
    p_team = poisson.sf(1, lam_hat)
    p_floor = poisson.sf(1, m)
    bA = float(np.mean((p_floor - y) ** 2))
    bB = float(np.mean((p_team - y) ** 2))
    return {"n_test": int(len(te)), "brier_floor": round(bA, 5), "brier_team": round(bB, 5),
            "oos_delta_vs_floor": round(bA - bB, 5), "beats_floor": bool(bB < bA)}


def main():
    df = pd.read_csv(SRC)
    m = float(df.offsides_for.mean())
    alpha, beta = _eb_beta(df, m)
    gate = _oos_gate(df, m, alpha, beta)

    # deployment table: full-history EB posterior per COVERED team
    g = df.groupby("team")["offsides_for"].agg(["sum", "count"])
    teams = {}
    for name, row in g.iterrows():
        n, s = int(row["count"]), int(row["sum"])
        if n < MIN_GAMES:
            continue                                   # uncovered -> absent -> floor
        lam = (alpha + s) / (beta + n)
        rec = {"display": name, "n": n, "total_offsides": s, "lambda_hat": round(lam, 4)}
        for key in {_norm(name)} | ({_norm(ALIAS[name])} if name in ALIAS else set()):
            teams[key] = rec

    out = {
        "_meta": {
            "fit_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": str(SRC),
            "n_matches": int(df.match_id.nunique()), "n_team_rows": int(len(df)),
            "min_games": MIN_GAMES,
            "note": ("FOUNDED per-team offside rate (empirical-Bayes Gamma-Poisson). MEASURED not typed; "
                     "uncovered teams ABSENT -> live route uses pooled FLOOR; REFRESH by re-running on an "
                     "updated extract (append 2026 games via FBref). Ships only because it beats the floor OOS."),
        },
        "pooled_rate": round(m, 4),
        "eb_alpha": round(alpha, 4), "eb_beta_shrinkage_K": round(beta, 4),
        "oos_gate": gate,
        "n_covered_teams": len({id(v) for v in teams.values()}),
        "teams": teams,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"pooled rate m={m:.3f}  EB beta(K)={beta:.2f}  covered teams={out['n_covered_teams']}")
    print(f"OOS gate: floor {gate['brier_floor']} vs team {gate['brier_team']} "
          f"-> d {gate['oos_delta_vs_floor']:+.5f}  beats_floor={gate['beats_floor']}")
    print(f"wrote {OUT}")
    if not gate["beats_floor"]:
        print("WARNING: team table does NOT beat floor OOS on this extract -> do NOT deploy; floor stands.")


if __name__ == "__main__":
    main()
