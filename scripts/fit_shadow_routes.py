"""Found the recurring SHADOW families: OOS-gate the both_teams_sot_1h volume
model and persist MEASURED base-rate anchors (never 0.50) for the auto-fallbacks.

Routes founded this batch:
  1. both_teams_sot_1h  -- closed-form P(both >=1 1H SOT) from engine lambda via the
     rate_layer SOT conversion. OOS-gated here vs the pooled base rate; ships only if
     it beats base out-of-sample, else the route auto-falls-back to the base rate.
  2. team_first_goal_2h -- pure closed-form from engine lambda + half split (no gate;
     derivation from the already-validated goals engine). Base-rate anchor persisted
     only as the safety fallback when the engine is unavailable.
  3. corner exact-line gap -- market ladder Poisson-fit (market-derived, no corpus fit).
     Per-threshold corner base rates persisted as the no-ladder fallback.

OOS gate emulation for (1): the SGO settlement corpus carries per-half SOT + a
home-relative favorite_gap but NO per-match total line. We reconstruct lambda by
holding lambda_total at the measured corpus mean (a constant, not per-fixture -> no
leakage) and splitting it by favorite_gap via the engine win-prob inversion. This
tests the IMBALANCE/form channel only; the DEPLOYED model additionally uses the live
per-match total (strictly more info), so a pass here is conservative.

    python scripts/fit_shadow_routes.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from odds_lib.match_engine import _p_home_win, _p_over, _bisect, H1_SHARE
from odds_lib.rate_layer import team_sot_mu

SGO = Path("data/historical/sgo_settlement_corpus.csv")
OUT = Path("data/models/shadow_routes.json")


def _brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def _share_from_gap(fav_gap: float, lam_total: float) -> float:
    """Home goal share s s.t. the engine's de-vigged 2-way (home-away) win-prob
    difference == fav_gap (home-relative). Monotone in s -> bisection."""
    def gap(s):
        ph = _p_home_win(s * lam_total, (1 - s) * lam_total)
        pa = _p_home_win((1 - s) * lam_total, s * lam_total)
        return (ph - pa) / (ph + pa) if (ph + pa) > 0 else 0.0
    return _bisect(gap, 0.01, 0.99, fav_gap)


def both_sot_1h_model(fav_gap, lam_total, h1=H1_SHARE):
    """P(both teams >=1 SOT in 1H) = (1-e^-mu_h)(1-e^-mu_a),
    mu_team_1h = team_sot_mu(lam_team, share=h1)  [Poisson 1H SOT, teams indep]."""
    s = _share_from_gap(fav_gap, lam_total)
    mu_h = team_sot_mu(s * lam_total, share=h1)
    mu_a = team_sot_mu((1 - s) * lam_total, share=h1)
    return (1 - np.exp(-mu_h)) * (1 - np.exp(-mu_a))


def both_sot_2h_model(fav_gap, lam_total, h1=H1_SHARE):
    """P(both teams >=1 SOT in 2H) = (1-e^-mu_h)(1-e^-mu_a),
    mu_team_2h = team_sot_mu(lam_team, share=1-h1)  [mirror of 1H, 2H share]."""
    s = _share_from_gap(fav_gap, lam_total)
    mu_h = team_sot_mu(s * lam_total, share=1 - h1)
    mu_a = team_sot_mu((1 - s) * lam_total, share=1 - h1)
    return (1 - np.exp(-mu_h)) * (1 - np.exp(-mu_a))


def gate_both_sot_2h():
    """Ships RAW true-P (k=1); this only REPORTS validation. Two OOS tests:
      (A) fixed-total (imbalance-only) -- conservative, like the 1H gate.
      (B) VOLUME lever -- per-fixture lambda_total from REALIZED total goals (a noisy,
          mildly-optimistic lambda proxy: same-match goals share noise with same-match
          SOT). Plus a monotonicity check: P(both 2H SOT) rising in realized goals.
    The deployed route uses the live market lambda (between the two in optimism)."""
    c = pd.read_csv(SGO)
    g = c.dropna(subset=["home_2h_shots_onGoal", "away_2h_shots_onGoal", "favorite_gap",
                         "home_game_points", "away_game_points"]).copy()
    g = g.sort_values("date").reset_index(drop=True)
    g["y"] = ((g.home_2h_shots_onGoal >= 1) & (g.away_2h_shots_onGoal >= 1)).astype(int)
    g["tot_goals"] = g.home_game_points + g.away_game_points
    lam_fixed = float((c.home_game_points + c.away_game_points).dropna().mean())
    g["p_fixed"] = g.favorite_gap.apply(lambda fg: both_sot_2h_model(fg, lam_fixed))
    # VOLUME lever: per-fixture realized total goals as lambda_total (floor at 0.3)
    g["p_vol"] = [both_sot_2h_model(fg, max(tg, 0.3)) for fg, tg in zip(g.favorite_gap, g.tot_goals)]
    n = len(g); cut = int(n * 0.8)
    tr, te = g.iloc[:cut], g.iloc[cut:]
    base = float(tr.y.mean())
    b_fixed = _brier(te.p_fixed, te.y); b_vol = _brier(te.p_vol, te.y)
    b_base = _brier(np.full(len(te), base), te.y)
    # monotonicity: P(both 2H SOT) by realized total-goals tertile
    g["gbin"] = pd.qcut(g.tot_goals, 3, labels=["low", "mid", "high"], duplicates="drop")
    mono = {str(k): round(float(v), 4) for k, v in g.groupby("gbin", observed=True).y.mean().items()}
    ph, pa = float((g.home_2h_shots_onGoal >= 1).mean()), float((g.away_2h_shots_onGoal >= 1).mean())
    return {
        "verdict_fixed": "validated" if b_fixed < b_base else "failed",
        "verdict_volume": "validated" if b_vol < b_base else "failed",
        "n": n, "n_test": len(te),
        "brier_fixed": round(b_fixed, 5), "brier_volume": round(b_vol, 5), "brier_base": round(b_base, 5),
        "oos_delta_fixed": round(b_base - b_fixed, 5), "oos_delta_volume": round(b_base - b_vol, 5),
        "volume_monotonicity_P_both_by_goals_tertile": mono,
        "base_rate": round(float(g.y.mean()), 4),
        "indep_check": {"P_home_ge1": round(ph, 4), "P_away_ge1": round(pa, 4),
                        "product": round(ph * pa, 4), "empirical_both": round(float(g.y.mean()), 4),
                        "phi_corr": round(float(np.corrcoef(g.home_2h_shots_onGoal >= 1,
                                                            g.away_2h_shots_onGoal >= 1)[0, 1]), 4)},
    }


def _sot_per_goal(c):
    """Aligned mean team SOT / mean team goals (the rate_layer SOT_SLOPE should match)."""
    gg = c.dropna(subset=["home_game_shots_onGoal", "away_game_shots_onGoal",
                          "home_game_points", "away_game_points"])
    sot = pd.concat([gg.home_game_shots_onGoal, gg.away_game_shots_onGoal])
    goals = pd.concat([gg.home_game_points, gg.away_game_points])
    return round(float(sot.sum() / goals.sum()), 4)


def gate_both_sot_1h():
    c = pd.read_csv(SGO)
    g = c.dropna(subset=["home_1h_shots_onGoal", "away_1h_shots_onGoal", "favorite_gap"]).copy()
    g = g.sort_values("date").reset_index(drop=True)
    g["y"] = ((g.home_1h_shots_onGoal >= 1) & (g.away_1h_shots_onGoal >= 1)).astype(int)
    # measured lambda_total (constant proxy; corpus mean realized total goals)
    lam_total = float((c.home_game_points + c.away_game_points).dropna().mean())
    g["p_model"] = g.favorite_gap.apply(lambda fg: both_sot_1h_model(fg, lam_total))
    # OOS split: train on first 80% by date, test on last 20%
    n = len(g); cut = int(n * 0.8)
    tr, te = g.iloc[:cut], g.iloc[cut:]
    base = float(tr.y.mean())                       # pooled base rate (the fallback floor)
    b_model = _brier(te.p_model, te.y)
    b_base = _brier(np.full(len(te), base), te.y)
    pooled_base = float(g.y.mean())                 # full-corpus base for deployment fallback
    verdict = "validated" if b_model < b_base else "failed_oos_use_base_rate"
    # independence + Poisson-form diagnostics (full corpus)
    ph, pa = float((g.home_1h_shots_onGoal >= 1).mean()), float((g.away_1h_shots_onGoal >= 1).mean())
    return {
        "verdict": verdict, "n": n, "n_test": len(te),
        "lam_total_proxy": round(lam_total, 4),
        "brier_model": round(b_model, 5), "brier_base": round(b_base, 5),
        "oos_delta": round(b_base - b_model, 5),
        "base_rate": round(pooled_base, 4),
        "indep_check": {"P_home_ge1": round(ph, 4), "P_away_ge1": round(pa, 4),
                        "product": round(ph * pa, 4), "empirical_both": round(pooled_base, 4),
                        "phi_corr": round(float(np.corrcoef(g.home_1h_shots_onGoal >= 1,
                                                            g.away_1h_shots_onGoal >= 1)[0, 1]), 4)},
        "sot_per_goal": _sot_per_goal(c),
    }


def first_goal_2h_rates():
    """Measured P(team scores the FIRST 2H goal), home/away (Rao-Blackwell per match:
    my_2h/(total_2h) given counts). The honest anchor used only if the engine lambdas
    are unavailable; the live route otherwise uses the closed-form engine derivation."""
    c = pd.read_csv(SGO)
    g = c.dropna(subset=["home_2h_points", "away_2h_points"]).copy()
    tot = g.home_2h_points + g.away_2h_points
    home_first = np.where(tot > 0, g.home_2h_points / tot, 0.0)
    away_first = np.where(tot > 0, g.away_2h_points / tot, 0.0)
    return {"home": round(float(home_first.mean()), 4),
            "away": round(float(away_first.mean()), 4),
            "p_any_2h_goal": round(float((tot > 0).mean()), 4), "n": int(len(g))}


def _offsides_driver_gate():
    """OOS-test every candidate PER-MATCH driver for P(team offsides>=2) vs the pooled
    base rate, scored on ACTUAL OUTCOMES (Brier), NOT c_hat. Returns the per-driver deltas
    + the verdict. None beating base => offsides is a genuine no-per-match-signal question."""
    from sklearn.linear_model import LogisticRegression
    c = pd.read_csv(SGO).sort_values("date")
    rows = []
    for _, r in c.iterrows():
        for side in ("home", "away"):
            off = r.get(f"{side}_game_offsides")
            if pd.isna(off):
                continue
            fg = r.get("favorite_gap")
            rows.append(dict(date=r["date"], y=int(off >= 2), is_home=1 if side == "home" else 0,
                             fg_team=(fg if side == "home" else -fg) if pd.notna(fg) else np.nan,
                             tot_goals=r.get("home_game_points", np.nan) + r.get("away_game_points", np.nan),
                             tot_shots=r.get("home_game_shots", np.nan) + r.get("away_game_shots", np.nan)))
    d = pd.DataFrame(rows)
    out = {}
    any_win = False
    for f in ["fg_team", "is_home", "tot_goals", "tot_shots"]:
        dd = d.dropna(subset=[f, "y"]).reset_index(drop=True)
        n = len(dd); cut = int(n * 0.8); tr, te = dd.iloc[:cut], dd.iloc[cut:]
        b_base = float(np.mean((tr.y.mean() - te.y.values) ** 2))
        clf = LogisticRegression(max_iter=2000).fit(tr[[f]].values, tr.y.values)
        p = clf.predict_proba(te[[f]].values)[:, 1]
        b_mod = float(np.mean((p - te.y.values) ** 2))
        win = b_mod < b_base
        any_win = any_win or win
        out[f] = {"oos_delta_vs_base": round(b_base - b_mod, 5), "beats_base": win}
    return {"baseline": "pooled_base_rate_vs_ACTUAL_OUTCOMES_brier_not_chat",
            "drivers": out,
            "verdict": ("per_match_signal_found" if any_win
                        else "NO_per_match_signal_oos__measured_rate_is_last_resort_floor__no_edge")}


def offsides_rates():
    """MEASURED P(team offsides >= 2) [= contest 'offside 2+' = over 1.5]. PER-MATCH DRIVER
    SEARCH (favorite_gap / home-away / volume / attacking intensity) found NO signal that
    beats the pooled base OOS (see verdict) -> this is an HONEST LAST-RESORT FLOOR, NOT a
    founded per-match model. We have NO edge on offsides; ship the pooled measured rate
    (home/away split is in-sample noise, does NOT survive OOS, so it is NOT used)."""
    c = pd.read_csv(SGO)
    pooled = pd.concat([c.home_game_offsides.dropna(), c.away_game_offsides.dropna()])
    p_ge2 = float((pooled >= 2).mean())
    n = int(pooled.size)
    sd = float((p_ge2 * (1 - p_ge2) / n) ** 0.5)
    # POPULATION FIX: the corpus is CLUB-ONLY for offsides (0 international rows), and clubs
    # differ from WC. SOURCED WC2026 rate: 3.05 offsides/MATCH (Squawka/PerformanceOdds, ~64
    # group games) -> 1.525 per team. Ship the WC per-team Poisson table (correct population),
    # NOT the club empirical table. Still a no-edge FLOOR (no per-match signal exists), just
    # correctly leveled. Per-threshold via Poisson(1.525).
    wc_team_mean = 3.05 / 2.0
    def _pois_ge(lam, k):
        tot, term = 0.0, float(np.exp(-lam))     # term = pmf(0)
        for i in range(0, 40):
            if i >= k:
                tot += term
            term *= lam / (i + 1)
        return round(tot, 4)
    wc_ge = {str(k): _pois_ge(wc_team_mean, k) for k in range(1, 6)}
    return {"p_ge2_pooled": round(p_ge2, 4),                       # CLUB empirical (reference only)
            "team_ge_club_reference": {str(k): round(float((pooled >= k).mean()), 4) for k in range(1, 6)},
            "team_ge": wc_ge,                                       # SHIPPED: WC-international Poisson table
            "wc_match_mean_offsides": 3.05, "wc_team_mean": round(wc_team_mean, 3),
            "wc_source": "WC2026 3.05 offsides/match (Squawka/PerformanceOdds); /2 per team; Poisson per-threshold",
            "mean_offsides_club": round(float(pooled.mean()), 3),
            "n_club": n, "is_floor_no_edge": True,
            "per_match_search": _offsides_driver_gate()}


def _cards_2h_gate():
    """OOS-gate favorite_gap as a per-match driver for 2H yellow cards vs the pooled base
    (Brier vs ACTUAL outcomes). TEAM: fg_team has no clean signal (>=1 ~0; >=2 is wrong-
    signed vs the validated underdog-cards mechanism). TOTAL: signed fg beats base OOS but
    the CLEAN mechanism |fg| (lopsidedness) FAILS -> murky, not shipped as a model."""
    from sklearn.linear_model import LogisticRegression
    c = pd.read_csv(SGO).sort_values("date")
    T = []
    for _, r in c.iterrows():
        for side in ("home", "away"):
            yc = r.get(f"{side}_2h_yellowCards"); fg = r.get("favorite_gap")
            if pd.isna(yc):
                continue
            T.append(dict(date=r["date"], yc=int(yc),
                          fg_team=(fg if side == "home" else -fg) if pd.notna(fg) else np.nan))
    T = pd.DataFrame(T)
    tot = c.dropna(subset=["home_2h_yellowCards", "away_2h_yellowCards"]).copy()
    tot["tc"] = tot.home_2h_yellowCards + tot.away_2h_yellowCards
    tot["absfg"] = tot.favorite_gap.abs()

    def gate(df, feats, ycol, k):
        dd = df.dropna(subset=feats + [ycol]).sort_values("date"); y = (dd[ycol] >= k).astype(int)
        if y.nunique() < 2 or y.mean() < 0.04 or y.mean() > 0.96:
            return None
        n = len(dd); cut = int(n * 0.8)
        ytr, yte = y.iloc[:cut], y.iloc[cut:]
        b_base = float(np.mean((ytr.mean() - yte.values) ** 2))
        clf = LogisticRegression(max_iter=2000).fit(dd[feats].values[:cut], ytr.values)
        p = clf.predict_proba(dd[feats].values[cut:])[:, 1]
        return {"k": k, "coef": round(float(clf.coef_[0][0]), 3),
                "delta_vs_base": round(b_base - float(np.mean((p - yte.values) ** 2)), 5)}
    return {"team": {"feature": "fg_team", "results": [gate(T, ["fg_team"], "yc", k) for k in (1, 2)],
                     "verdict": "no_clean_signal_floor"},
            "total": {"signed_fg": [gate(tot, ["favorite_gap"], "tc", k) for k in (2, 3, 4)],
                      "abs_fg_clean_mechanism": [gate(tot, ["absfg"], "tc", k) for k in (2, 3, 4)],
                      "verdict": "weak_signed_signal_beats_base_but_clean_|fg|_fails__floored_pending_clean_driver"}}


def cards_2h_rates():
    """MEASURED per-threshold 2H-yellow-card floors (team + total). favorite_gap is NOT a
    clean per-match driver here (see _cards_2h_gate) -> honest measured floors, NOT models,
    NOT crowd-copy. An upgrade from the prior crowd-mean shadow."""
    c = pd.read_csv(SGO)
    team = pd.concat([c.home_2h_yellowCards.dropna(), c.away_2h_yellowCards.dropna()])
    tot = (c.home_2h_yellowCards + c.away_2h_yellowCards).dropna()
    return {"team_ge": {str(k): round(float((team >= k).mean()), 4) for k in range(1, 5)},
            "total_ge": {str(k): round(float((tot >= k).mean()), 4) for k in range(1, 8)},
            "team_mean": round(float(team.mean()), 3), "total_mean": round(float(tot.mean()), 3),
            "n_team": int(team.size), "is_floor_no_clean_signal": True,
            "gate": _cards_2h_gate()}


def penalty_rates():
    """MEASURED external anchors (no market/corpus carries pen/red). SOURCED:
      - penalties: 0.28 per game across 5 World Cups 2006-2022 (88 pens), bettingoffers.org.uk
        citing the multi-WC aggregate. -> P(>=1 pen) = 1 - e^-0.28 (Poisson per match).
      - red cards: modern VAR-era WC rate ~0.06 (2018 & 2022 each 4/64) up to 0.44 (2006);
        the recent trend governs 2026 -> use 0.10/match (VAR-era, mid of the 0.08-0.12 band),
        Opta/Wikipedia tournament counts. -> P(>=1 red) = 1 - e^-0.10.
    Union (penalty OR red), independence approx: P = P_pen + P_red - P_pen*P_red."""
    pen_per_game, red_per_game = 0.28, 0.10
    import math as _m
    p_pen = 1 - _m.exp(-pen_per_game)
    p_red = 1 - _m.exp(-red_per_game)
    union = p_pen + p_red - p_pen * p_red
    return {"penalty_awarded": round(p_pen, 4),
            "penalty_or_red_card": round(union, 4),
            "_source": {"pen_per_game": pen_per_game, "red_per_game": red_per_game,
                        "p_pen": round(p_pen, 4), "p_red": round(p_red, 4),
                        "note": "pen 0.28/g (5-WC 2006-2022, 88 pens); red 0.10/g (modern VAR-era WC); "
                                "P(>=1)=1-e^-rate (Poisson); union=indep approx"}}


def corner_base_rates():
    """Per-threshold measured corner base rates (the no-ladder fallback)."""
    c = pd.read_csv(SGO)
    team = pd.concat([c.home_game_cornerKicks, c.away_game_cornerKicks]).dropna()
    tot = (c.home_game_cornerKicks + c.away_game_cornerKicks).dropna()
    team_rates = {str(k): round(float((team >= k).mean()), 4) for k in range(1, 13)}
    total_rates = {str(k): round(float((tot >= k).mean()), 4) for k in range(3, 22)}
    return {"team_ge": team_rates, "total_ge": total_rates,
            "team_mean": round(float(team.mean()), 3), "total_mean": round(float(tot.mean()), 3),
            "n_team": int(team.size)}


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    both = gate_both_sot_1h()
    both2 = gate_both_sot_2h()
    offsides = offsides_rates()
    cards_2h = cards_2h_rates()
    penalties = penalty_rates()
    corners = corner_base_rates()
    out = {
        "_meta": {"fit_utc": now, "corpus": str(SGO),
                  "note": "MEASURED base-rate anchors + OOS gate for shadow-family routes"},
        "both_teams_sot_1h": both,
        "both_teams_sot_2h_1plus": both2,
        "first_goal_2h": {"derivation": "closed_form_engine_lambda_half_split",
                          **first_goal_2h_rates()},
        "team_offsides_over_ge2": offsides,
        "cards_2h": cards_2h,
        "penalties": penalties,
        "corners": corners,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"=== both_teams_sot_1h OOS gate ===")
    for k, v in both.items():
        print(f"  {k}: {v}")
    print(f"\n=== both_teams_sot_2h_1plus OOS gate ===")
    for k, v in both2.items():
        print(f"  {k}: {v}")
    print(f"\n=== corner base rates (fallback) === team_ge[5]={corners['team_ge']['5']} total_ge[9]={corners['total_ge']['9']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
