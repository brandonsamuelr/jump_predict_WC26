"""Audit live field-error performance from data/calibration_log.csv.

Splits rows by ``row_kind``:

- ``actual_submission`` rows: what Brandon actually submitted. Used for
  realized RBP analysis and field-error analysis if a ``p_field_est`` is
  present.
- ``engine_counterfactual`` rows: what the engine would have submitted.
  Used for field-error analysis ONLY (no RBP attribution — Brandon did
  not submit these numbers).

Both kinds contribute to field-error reporting; only actual_submission
rows contribute to RBP totals.

Reports
-------
- overall field MAE (each kind)
- MAE by question_type, target_role_bucket, decision_mode
- field bias by question_type
- worst 10 field misses (engine_counterfactual rows where engine had a chance)
- actual RBP total + per-question average (actual_submission rows only)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.calibration import CALIBRATION_LOG_PATH


def _field_error(df: pd.DataFrame) -> pd.Series:
    """Compute field_error = p_field_est - locked_crowd_pct/100, even for
    rows where the column wasn't written by the engine (recompute on the fly).
    """
    p_field = pd.to_numeric(df.get("p_field_est"), errors="coerce")
    crowd = pd.to_numeric(df.get("field_prob"), errors="coerce")
    return p_field - (crowd / 100.0)


def _stats(series: pd.Series, prefix: str = "") -> dict:
    s = series.dropna()
    if s.empty:
        return {f"{prefix}n": 0}
    return {
        f"{prefix}n": int(len(s)),
        f"{prefix}mae": float(s.abs().mean()),
        f"{prefix}bias": float(s.mean()),
        f"{prefix}max_abs": float(s.abs().max()),
    }


def _table(df: pd.DataFrame, group_cols: list[str], err_col: str) -> pd.DataFrame:
    sub = df.dropna(subset=[err_col]).copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in sub.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(g)
        row["mae"] = float(g[err_col].abs().mean())
        row["bias"] = float(g[err_col].mean())
        row["max_abs"] = float(g[err_col].abs().max())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


def main() -> None:
    if not CALIBRATION_LOG_PATH.exists():
        print(f"no calibration log at {CALIBRATION_LOG_PATH}")
        return
    df = pd.read_csv(CALIBRATION_LOG_PATH)
    if "row_kind" not in df.columns:
        df["row_kind"] = "actual_submission"  # legacy rows before split

    df["field_error_calc"] = _field_error(df)

    print(f"== calibration log audit ({len(df)} rows) ==")
    print(df["row_kind"].value_counts().to_string())

    for kind in ["actual_submission", "engine_counterfactual"]:
        sub = df[df["row_kind"] == kind].copy()
        if sub.empty:
            continue
        print(f"\n--- row_kind = {kind} ({len(sub)} rows) ---")
        s = _stats(sub["field_error_calc"], prefix="field_")
        if s.get("field_n", 0) > 0:
            print(
                f"  field_error: n={s['field_n']} mae={s['field_mae']:+.4f} "
                f"bias={s['field_bias']:+.4f} max_abs={s['field_max_abs']:+.4f}"
            )

            print("  MAE by question_type:")
            t = _table(sub, ["question_type"], "field_error_calc")
            for _, r in t.iterrows():
                print(f"    {r['question_type']:<30} n={int(r['n']):>3} mae={r['mae']:.4f} "
                      f"bias={r['bias']:+.4f} max={r['max_abs']:.4f}")

            if "target_role_bucket" in sub.columns and sub["target_role_bucket"].notna().any():
                print("  MAE by role bucket:")
                t = _table(sub, ["target_role_bucket"], "field_error_calc")
                for _, r in t.iterrows():
                    print(f"    {str(r['target_role_bucket']):<20} n={int(r['n']):>3} mae={r['mae']:.4f} "
                          f"bias={r['bias']:+.4f}")

            if "decision_mode" in sub.columns and sub["decision_mode"].notna().any():
                print("  MAE by decision_mode:")
                t = _table(sub, ["decision_mode"], "field_error_calc")
                for _, r in t.iterrows():
                    print(f"    {str(r['decision_mode']):<40} n={int(r['n']):>3} mae={r['mae']:.4f} "
                          f"bias={r['bias']:+.4f}")

            print("\n  worst 10 field misses:")
            w = sub.dropna(subset=["field_error_calc"]).copy()
            w["abs_err"] = w["field_error_calc"].abs()
            cols = ["match_norm", "question_type", "p_field_est", "field_prob", "field_error_calc"]
            cols = [c for c in cols if c in w.columns]
            for _, r in w.nlargest(10, "abs_err")[cols].iterrows():
                print(f"    {r['match_norm']:<25} {r['question_type']:<30} "
                      f"est={float(r['p_field_est']):.3f} crowd={float(r['field_prob']):.1f} "
                      f"err={float(r['field_error_calc']):+.3f}")
        else:
            print("  no rows with p_field_est on this kind.")

        # RBP attribution — only actual_submission rows are real money.
        if kind == "actual_submission":
            rbp = pd.to_numeric(sub.get("actual_rbp"), errors="coerce").dropna()
            if not rbp.empty:
                print(f"\n  REALIZED RBP (actual_submission only):")
                print(f"    total RBP = {rbp.sum():+.2f} across {len(rbp)} submissions")
                print(f"    mean RBP per submission = {rbp.mean():+.2f}")
                print(f"    win rate (RBP>0): {(rbp>0).mean()*100:.1f}%")
                top = sub.assign(rbp=rbp).nlargest(5, "rbp")[["match_norm", "question_type", "rbp"]]
                bot = sub.assign(rbp=rbp).nsmallest(5, "rbp")[["match_norm", "question_type", "rbp"]]
                print("    top 5:")
                for _, r in top.iterrows():
                    print(f"      {r['match_norm']:<25} {r['question_type']:<30} {float(r['rbp']):+.2f}")
                print("    bottom 5:")
                for _, r in bot.iterrows():
                    print(f"      {r['match_norm']:<25} {r['question_type']:<30} {float(r['rbp']):+.2f}")


if __name__ == "__main__":
    main()
