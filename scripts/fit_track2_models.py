"""Fit + persist the two Track-2 gate-validated corpus models (cards-cmp, match-SOT).

Both passed the out-of-sample gate (scripts/validate_track2 run) with meaningful,
well-calibrated margins; team_more_fouls did NOT clear a worth-shipping margin and
is intentionally excluded (stays shadow). Validation is OOS; deployment fits on the
full totals-available corpus. Coefficients -> data/models/corpus_models.json so the
live path evaluates a plain logistic (no sklearn / no corpus read at lock).

    python scripts/fit_track2_models.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.linear_model import LogisticRegression

CORPUS = Path("data/historical/stat_lines.csv")
OUT = Path("data/models/corpus_models.json")
MATCH_SOT_THRESHOLDS = [6, 7, 8, 9, 10]   # "N total SOT": contest typical 8


def _fit(df, features, y):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(df[features].to_numpy(), y.to_numpy())
    return ({f: float(c) for f, c in zip(features, clf.coef_[0])},
            float(clf.intercept_[0]), float(y.mean()))


def main():
    df = pd.read_csv(CORPUS)
    df = df[df["total_line_prob"].notna() & df["favorite_gap"].notna()].copy()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {"_meta": {"fit_utc": now, "corpus": str(CORPUS),
                     "source": "club football (football-data.co.uk)",
                     "note": "OOS-gate-validated; deployed fit on full corpus"}}

    # team_more_cards: features favorite_gap + total_line_prob + is_home
    f = ["favorite_gap", "total_line_prob", "is_home"]
    y = ((df["yellows_for"] + df["reds_for"]) > (df["yellows_against"] + df["reds_against"])).astype(int)
    coefs, b0, base = _fit(df, f, y)
    out["team_more_cards"] = {"features": f, "coefs": coefs, "intercept": b0,
                              "base_rate": round(base, 4), "n": int(len(df))}
    print(f"team_more_cards: base={base:.3f} b0={b0:+.3f} {coefs}")

    # match_total_sot_over: features favorite_gap + total_line_prob, per threshold,
    # one row per match (home-subset)
    home = df[df["is_home"] == 1].copy()
    f2 = ["favorite_gap", "total_line_prob"]
    th = {}
    for k in MATCH_SOT_THRESHOLDS:
        yk = ((home["sot_for"] + home["sot_against"]) >= k).astype(int)
        if yk.nunique() < 2 or yk.mean() < 0.02 or yk.mean() > 0.98:
            print(f"  match_sot>={k}: skip (base {yk.mean():.3f})"); continue
        coefs, b0, base = _fit(home, f2, yk)
        th[str(k)] = {"coefs": coefs, "intercept": b0, "base_rate": round(base, 4)}
        print(f"  match_sot>={k}: base={base:.3f} b0={b0:+.3f} {coefs}")
    out["match_total_sot_over"] = {"features": f2, "thresholds": th, "n": int(len(home))}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
