"""Guards for the edge estimator (Deliverable 1).

    .venv/bin/python tests/test_edge.py

Key invariants: k_hat is UNCLIPPED (can be negative); deployed k is clipped
to [0,1]; lambda_prior is in squared-deviation units so the prior dominates
at small sum_d2; a no-position class sits at its prior.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib.edge import compute_edge_table, classify, K_PRIOR


def _df(rows):
    out = []
    for (cls, sub, match, p, c, y) in rows:
        out.append({"source_class": cls, "source_subtype": sub, "match": match,
                    "p_model": p, "c_hat": c, "y": y,
                    "rbp_final": 0.0, "rbp_model_cf": 0.0, "rbp_baseline_cf": 0.0})
    return pd.DataFrame(out)


def test_classify():
    assert classify("MARKET", "team_win") == ("MARKET", "market")
    assert classify("RATE_SOT", "total_sot_2h_over") == ("RATE_SOT", "total_2h")
    assert classify("RATE_SOT", "team_sot_over") == ("RATE_SOT", "single")
    assert classify("RATE_SOT_CMP", "team_more_sot_2h") == ("RATE_SOT", "comparison")
    assert classify("PENDING", "team_offsides_over") == ("SHADOW", "shadow")


def test_k_hat_unclipped_negative_but_deployed_clipped():
    # model deviates up (d>0) but outcomes come in below the proxy (resid<0)
    # -> anti-predictive -> k_hat < 0, but deployed must be >= 0.
    df = _df([("RATE_SOT", "single", "m1", 0.8, 0.5, 0),
              ("RATE_SOT", "single", "m2", 0.8, 0.5, 0)])
    row = compute_edge_table(df).loc[("RATE_SOT", "single")]
    assert row["k_hat"] < 0                 # diagnostic, unclipped
    assert 0.0 <= row["k_deployed"] <= 1.0   # deployed clipped


def test_no_position_class_sits_at_prior():
    # d == 0 everywhere -> no fitting signal -> k_shrunk == k_prior, deployed too.
    df = _df([("SHADOW", "shadow", "m1", 0.5, 0.5, 1),
              ("SHADOW", "shadow", "m2", 0.4, 0.4, 0)])
    row = compute_edge_table(df).loc[("SHADOW", "shadow")]
    assert row["n_active"] == 0
    assert abs(row["k_deployed"] - K_PRIOR[("SHADOW", "shadow")]) < 1e-9


def test_prior_dominates_at_small_sample():
    # one row of "perfect" data (k_hat=1) should NOT override an 8-match prior.
    df = _df([("MARKET", "market", "m1", 0.8, 0.5, 1)])   # d=0.3, resid=0.5, k_hat=1.67
    row = compute_edge_table(df).loc[("MARKET", "market")]
    assert row["k_shrunk"] < row["k_hat"]                 # pulled toward prior
    assert abs(row["k_shrunk"] - 0.9) < 0.2               # near the 0.90 prior


def test_thin_class_frozen_to_prior():
    # n_active < 4 -> deployed = k_prior EXACTLY, even if the shrunk k drifted
    # on a lucky row (this is the total_2h "0.25 -> 0.40 nudge" fix).
    df = _df([("RATE_SOT", "total_2h", "m1", 0.8, 0.5, 1)])  # 1 active row, would drift up
    row = compute_edge_table(df).loc[("RATE_SOT", "total_2h")]
    assert bool(row["frozen"]) is True
    assert abs(row["k_deployed"] - K_PRIOR[("RATE_SOT", "total_2h")]) < 1e-9   # == 0.90


def test_cluster_gate_freezes_even_with_many_active_rows():
    # 6 active rows but only 3 MATCH clusters (correlated): must stay FROZEN on
    # the prior, NOT deploy a fitted/clipped k driven by ~3 independent obs.
    rows = [("ENGINE", "engine", m, 0.9, 0.4, 1)
            for m in ("m1", "m1", "m2", "m2", "m3", "m3")]
    row = compute_edge_table(_df(rows)).loc[("ENGINE", "engine")]
    assert row["n_active"] == 6 and row["clusters"] == 3
    assert bool(row["frozen"]) is True
    assert abs(row["k_deployed"] - K_PRIOR[("ENGINE", "engine")]) < 1e-9


def test_engine_refreezes_at_five_clusters():
    # 5 match clusters, all active -> STILL frozen on the 0.90 prior (MIN_CLUSTERS=10).
    # Guards the re-freeze after ENGINE wrongly unfroze to a fitted k=1.0 at 5 clusters.
    rows = ([("ENGINE", "engine", f"m{i}", 0.9, 0.4, 1) for i in range(5)]
            + [("ENGINE", "engine", f"m{i}", 0.92, 0.45, 1) for i in range(5)])
    row = compute_edge_table(_df(rows)).loc[("ENGINE", "engine")]
    assert row["clusters"] == 5 and row["n_active"] >= 4
    assert bool(row["frozen"]) is True
    assert abs(row["k_deployed"] - K_PRIOR[("ENGINE", "engine")]) < 1e-9


def test_deployed_in_unit_interval():
    df = _df([("ENGINE", "engine", "m1", 0.95, 0.4, 1),
              ("ENGINE", "engine", "m2", 0.9, 0.45, 1)])
    row = compute_edge_table(df).loc[("ENGINE", "engine")]
    assert 0.0 <= row["k_deployed"] <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
