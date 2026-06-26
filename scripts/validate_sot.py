"""Rebuild + validate the SOT-family models on the historical corpus.

Ship-decision is the harness, NOT in-sample R^2 or intuition. We fit simple,
interpretable logistic models for the SOT-family contest targets, run them
through ``validate_candidate`` (time-split, flat-shadow baseline, Brier +
log-loss, calibration), and ALSO score the current live SOT model
(rate_layer: SOT ~ Poisson(1.01 + 3.03*lambda)) on the same held-out rows so we
can answer the real question: is the rebuilt version better than what's bleeding
us now, and does it fix the high-bias?

Restricted to rows with totals odds (~55%) so every feature set AND the
lambda-based live model are scored on the IDENTICAL test rows.

NOT wired to live pricing. Analysis only.

    python scripts/validate_sot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from odds_lib.validation import validate_candidate, brier, log_loss, time_split
from odds_lib import match_engine as E
from odds_lib import rate_layer as R

CORPUS = Path("data/historical/stat_lines.csv")


# --- market-implied lambda via the SAME engine the live pipeline uses -------
def add_lambda(df: pd.DataFrame) -> pd.DataFrame:
    """Back out (lam_home, lam_away) from de-vigged home-win prob + P(over2.5)
    using match_engine.calibrate (cached by rounded inputs)."""
    cache: dict = {}

    def pair(ph: float, po: float):
        if not (np.isfinite(ph) and np.isfinite(po)):
            return (np.nan, np.nan)
        key = (round(ph, 3), round(po, 3))
        if key not in cache:
            m = E.calibrate("h", "a", key[0], key[1])
            cache[key] = (m.lam_home, m.lam_away)
        return cache[key]

    lh = np.empty(len(df))
    la = np.empty(len(df))
    for i, (ph, po) in enumerate(zip(df["home_win_prob"].to_numpy(),
                                     df["total_line_prob"].to_numpy())):
        lh[i], la[i] = pair(ph, po)
    df = df.copy()
    df["lam_home"], df["lam_away"] = lh, la
    is_h = df["is_home"].to_numpy() == 1
    df["lam_for"] = np.where(is_h, lh, la)
    df["lam_against"] = np.where(is_h, la, lh)
    return df


# --- candidate factory (simple, interpretable logistic) ---------------------
def make_logit(features: list[str]):
    def fn(train, test, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(train[features].to_numpy(), y)
        fn.coefs = {f: round(float(c), 3) for f, c in zip(features, clf.coef_[0])}
        fn.intercept = round(float(clf.intercept_[0]), 3)
        return clf.predict_proba(test[features].to_numpy())[:, 1]
    fn.coefs, fn.intercept = {}, 0.0
    return fn


def calib_bias(res) -> float:
    """Weighted mean (pred - actual) on the test set. >0 => model runs HIGH."""
    c = res.calibration
    w = c["n"].sum()
    return float((c["pred_mean"] * c["n"]).sum() / w - (c["actual_freq"] * c["n"]).sum() / w)


SETS = {
    "a: gap":            ["favorite_gap"],
    "b: +total":         ["favorite_gap", "total_line_prob"],
    "c: +home":          ["favorite_gap", "total_line_prob", "is_home"],
    "d: +lambda":        ["favorite_gap", "total_line_prob", "is_home", "lam_for"],
}

rows = []   # summary table accumulator


def run_target(df, target_fn, name, stat_cols, sets=SETS, print_coefs_for=("d: +lambda",)):
    print(f"\n################  {name}  ################")
    best = None
    for setname, feats in sets.items():
        fn = make_logit(feats)
        res = validate_candidate(
            df, target_fn, fn,
            required_cols=feats + stat_cols + ["total_line_prob"],
            target_name=f"{name} [{setname}]",
        )
        bias = calib_bias(res)
        rows.append({
            "target": name, "set": setname, "n_test": res.n_test,
            "base": round(res.base_rate_train, 4),
            "base_brier": round(res.baseline_brier, 5),
            "cand_brier": round(res.candidate_brier, 5),
            "dBrier": round(res.brier_delta, 5),
            "dLogLoss": round(res.log_loss_delta, 5),
            "PASS": res.passed, "calib_bias": round(bias, 4),
        })
        flag = "HIGH" if bias > 0.01 else ("low" if bias < -0.01 else "ok")
        print(f"  [{setname:12}] PASS={res.passed!s:5} dBrier={res.brier_delta:+.5f} "
              f"dLL={res.log_loss_delta:+.5f}  calib={flag}({bias:+.4f})  coefs={fn.coefs}")
        if setname in print_coefs_for:
            best = res
    if best is not None:
        print(f"  calibration (test) for best set:")
        for _, r in best.calibration.iterrows():
            print(f"    {r['bin']:<12} n={int(r['n']):>6}  pred={r['pred_mean']:.3f}  "
                  f"actual={r['actual_freq']:.3f}")


def three_way(df, target_fn, live_pred, name, features_d=SETS["d: +lambda"]):
    """flat-shadow vs current live-SOT vs new-logit(d), same test rows."""
    work = df.copy()
    work["_y"] = target_fn(work).astype(float)
    work = work.dropna(subset=features_d + ["_y", "lam_for", "lam_against"])
    train, test = time_split(work)
    yte = test["_y"].to_numpy()
    p_flat = np.full(len(test), float(train["_y"].mean()))
    p_live = np.asarray(live_pred(test), float)
    clf = LogisticRegression(max_iter=2000).fit(train[features_d].to_numpy(),
                                                train["_y"].to_numpy())
    p_new = clf.predict_proba(test[features_d].to_numpy())[:, 1]
    print(f"\n==== 3-way: {name}  (n_test={len(test):,}, actual_rate={yte.mean():.3f}) ====")
    print(f"  {'model':<26}{'Brier':>10}{'logloss':>10}{'mean_pred':>11}{'bias':>9}")
    for label, p in [("flat-shadow", p_flat),
                     ("live-SOT 1.01+3.03λ", p_live),
                     ("new-logit(d)", p_new)]:
        print(f"  {label:<26}{brier(yte, p):>10.5f}{log_loss(yte, p):>10.5f}"
              f"{p.mean():>11.3f}{p.mean() - yte.mean():>+9.4f}")


def main():
    if not CORPUS.exists():
        sys.exit("corpus missing; run scripts/build_stat_lines.py")
    df = pd.read_csv(CORPUS)
    # restrict to rows with totals odds so all feature sets + the live lambda
    # model score on identical rows
    df = df[df["total_line_prob"].notna() & df["favorite_gap"].notna()].copy()
    print(f"corpus (totals-available subset): {len(df):,} team-rows, "
          f"{df['match_id'].nunique():,} matches, seasons {df['season'].min()}..{df['season'].max()}")
    df = add_lambda(df)

    # ---- Target 1: team_sot_over (line 3.5 -> >=4 primary) ----
    run_target(df, lambda d: (d["sot_for"] >= 4).astype(int),
               "team_sot_over (>=4, line 3.5)", ["sot_for"])
    three_way(df, lambda d: (d["sot_for"] >= 4).astype(int),
              lambda t: [R._p_ge(R.team_sot_mu(l, 1.0), 4) for l in t["lam_for"]],
              "team_sot_over >=4")
    # line sensitivity at set (d)
    print("\n  line sensitivity (set d):")
    for line, k in [(2.5, 3), (3.5, 4), (4.5, 5)]:
        fn = make_logit(SETS["d: +lambda"])
        res = validate_candidate(df, (lambda kk: (lambda d: (d["sot_for"] >= kk).astype(int)))(k),
                                 fn, required_cols=SETS["d: +lambda"] + ["sot_for", "total_line_prob"],
                                 target_name=f">= {k}")
        print(f"    line {line} (>= {k}): base={res.base_rate_train:.3f} PASS={res.passed!s:5} "
              f"dBrier={res.brier_delta:+.5f} calib_bias={calib_bias(res):+.4f}")

    # ---- Target 2: team_more_sot (comparison; high-bias-prone) ----
    run_target(df, lambda d: (d["sot_for"] > d["sot_against"]).astype(int),
               "team_more_sot (sot_for>sot_against)", ["sot_for", "sot_against"])
    three_way(df, lambda d: (d["sot_for"] > d["sot_against"]).astype(int),
              lambda t: [R._p_a_gt_b(R.team_sot_mu(lf, 1.0), R.team_sot_mu(la, 1.0))
                         for lf, la in zip(t["lam_for"], t["lam_against"])],
              "team_more_sot")

    # ---- Target 3: both_teams_sot_1plus (match-level; home-row subset) ----
    home = df[df["is_home"] == 1].copy()
    sets3 = {"a: |gap|": ["fav_underdog_gap_abs"],
             "b: +total": ["fav_underdog_gap_abs", "total_line_prob"],
             "d: +lam_tot": ["fav_underdog_gap_abs", "total_line_prob", "lam_home", "lam_away"]}
    run_target(home, lambda d: ((d["sot_for"] >= 1) & (d["sot_against"] >= 1)).astype(int),
               "both_teams_sot_1plus (match)", ["sot_for", "sot_against"],
               sets=sets3, print_coefs_for=("d: +lam_tot",))
    three_way(home, lambda d: ((d["sot_for"] >= 1) & (d["sot_against"] >= 1)).astype(int),
              lambda t: [R._p_ge(R.team_sot_mu(lh, 1.0), 1) * R._p_ge(R.team_sot_mu(la, 1.0), 1)
                         for lh, la in zip(t["lam_home"], t["lam_away"])],
              "both_teams_sot_1plus",
              features_d=sets3["d: +lam_tot"])

    # ---- consolidated summary ----
    print("\n\n================  SUMMARY  ================")
    s = pd.DataFrame(rows)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
