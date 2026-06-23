"""Deliverable 1 — run the per-class/subtype edge table on resolved rows.

Fits k on the PRE-LOCK proxy (c_hat = the logged shadow/field estimate),
p_model = pipeline_submit, y = result. Realized crowd is NOT used to fit.

    python scripts/edge_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib.measurement import LOG_PATH, build_edge_frame
from odds_lib.edge import compute_edge_table


def main():
    df = pd.read_csv(LOG_PATH, dtype=str)
    # p_model = RAW model (p_hat column), c_hat = pre-lock proxy (shadow column).
    # NOT pipeline_submit, which is the edge-weighted output -> would fit k on a
    # quantity that already contains k (circular). build_edge_frame enforces this.
    edf = build_edge_frame(df)
    if edf.empty:
        print("no resolved rows yet."); return
    table = compute_edge_table(edf)

    n_matches = edf["match"].nunique()
    print(f"DELIVERABLE 1 — per-class edge table  ({len(edf)} resolved rows, {n_matches} match clusters)")
    print("=" * 110)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(table.to_string())
    print("\nk_hat = UNCLIPPED fit (may be <0 = anti-predictive five-alarm); k_deployed = clip(k_shrunk,[0,1]).")
    print("eff_n_k / n_active = the REAL fitting sample (rows where the model took a position) — governs trust, not n.")
    print(f"Fit on c_hat (pre-lock shadow), NOT realized crowd. m_prior={8} matches; lam in squared-deviation units.")
    print("\nREAD: at this cluster count every row is LOW-confidence / prior-dominated. The table is a")
    print("monitoring baseline; deployed k should track the structural prior until clusters + n_active grow.")


if __name__ == "__main__":
    main()
