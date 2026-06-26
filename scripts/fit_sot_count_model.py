"""Fit the deployable team_sot_over count-row model and persist coefficients.

The SOT revalidation (scripts/validate_sot.py) proved a simple logistic on
[favorite_gap, total_line_prob] beats both the flat shadow and the high-biased
live model (SOT~Poisson(1.01+3.03*lambda)) out-of-sample at every line, with
neutral calibration. This script fits that logistic PER THRESHOLD (k = ceil(line))
on the FULL corpus and writes coefficients to data/models/sot_count_model.json,
so the live pricing path can evaluate it with a plain logistic (no sklearn, no
168k-row read at lock time).

Validation is OOS (the harness); deployment fits on all data. Re-run to refit.

    python scripts/fit_sot_count_model.py
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
OUT = Path("data/models/sot_count_model.json")
FEATURES = ["favorite_gap", "total_line_prob"]
THRESHOLDS = [2, 3, 4, 5, 6]   # team_sot_over lines 1.5..5.5 -> ceil = 2..6


def main():
    df = pd.read_csv(CORPUS)
    df = df.dropna(subset=FEATURES + ["sot_for"])
    n_all = len(df)
    model = {"_meta": {
        "fit_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_rows": int(n_all), "features": FEATURES,
        "corpus": str(CORPUS), "source": "club football (football-data.co.uk)",
        "note": "OOS-validated via scripts/validate_sot.py; deployed fit on full corpus",
    }, "thresholds": {}}
    print(f"fitting on {n_all:,} rows with {FEATURES}")
    for k in THRESHOLDS:
        y = (df["sot_for"] >= k).astype(int)
        if y.nunique() < 2 or y.mean() < 0.02 or y.mean() > 0.98:
            print(f"  k>={k}: skip (base rate {y.mean():.3f})")
            continue
        clf = LogisticRegression(max_iter=2000)
        clf.fit(df[FEATURES].to_numpy(), y.to_numpy())
        model["thresholds"][str(k)] = {
            "intercept": float(clf.intercept_[0]),
            "favorite_gap": float(clf.coef_[0][0]),
            "total_line_prob": float(clf.coef_[0][1]),
            "base_rate": float(y.mean()),
        }
        print(f"  k>={k}: base={y.mean():.3f}  b0={clf.intercept_[0]:+.3f}  "
              f"fav_gap={clf.coef_[0][0]:+.3f}  total={clf.coef_[0][1]:+.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, indent=2))
    print(f"wrote {OUT} ({len(model['thresholds'])} thresholds)")


if __name__ == "__main__":
    main()
