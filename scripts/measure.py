"""Per-tier measurement report from the resolved rows in the measurement log.

    python scripts/measure.py

Answers, per tier (and overall):
  - mean_rbp: did this tier beat the field? (>0 = yes)
  - beat_field_rate: fraction of rows with positive RBP
  - edge_vs_shadow: did leaning on our p beat just shadowing? (>0 = yes)
  - edge_vs_llm: did we beat your LLM estimates? (>0 = yes; needs llm_estimate)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib.measurement import LOG_PATH, score_rows, tier_report


def main():
    if not LOG_PATH.exists():
        print(f"no measurement log yet at {LOG_PATH}; run scripts/log_slate.py first")
        return
    df = pd.read_csv(LOG_PATH, dtype=str)
    scored = score_rows(df)
    total = len(df)
    if scored.empty:
        print(f"{total} rows logged, 0 resolved. Fill `result` + `actual_rbp` "
              f"in {LOG_PATH} after games resolve, then rerun.")
        return
    print(f"{len(scored)} of {total} logged rows resolved.\n")
    pd.set_option("display.width", 220)
    print(tier_report(scored).to_string())
    print("\nrbp_final = what you actually submitted; rbp_pipeline = trust-the-optimizer;")
    print("rbp_shadow = always-shadow; rbp_llm = your hand estimates.")
    print("pipeline_vs_shadow>0 => leaning beat shadowing.  pipeline_vs_llm>0 => model beat your LLM.")
    print("(compare rbp_final vs rbp_pipeline to see if your manual overrides helped.)")


if __name__ == "__main__":
    main()
