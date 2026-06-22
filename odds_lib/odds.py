"""Odds conversion, de-vigging, and cross-book consensus.

Strict-by-default: incomplete book-markets are dropped (with a warning) or
raised, never silently normalized away.

The de-vig grouping is ``(event_id, book, market_key, line)`` so totals at
different points (e.g. Over/Under 2.5 vs 3.0) are treated as independent
mini-markets that each de-vig to sum 1.0.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


REQUIRED_COLS = {
    "forecast_run_id",
    "event_id",
    "match_date",
    "match",
    "book",
    "market_key",
    "line",
    "outcome",
    "american_odds",
}

BOOK_MARKET_COLS = ["event_id", "book", "market_key", "line"]
MARKET_COLS = ["forecast_run_id", "event_id", "market_key", "line"]

# Default "sharp" basket. Pinnacle is the canonical reference; the three
# exchanges are low-margin and informative on liquid markets. Override via
# the ``sharp_books`` arg to ``process_match`` (or ``--sharp-books`` on the
# CLI). When a market/outcome has no sharp quote, the sharp_* columns are
# NaN and downstream code treats this as ``sharp_num_books = 0``.
DEFAULT_SHARP_BOOKS = ("Pinnacle", "Betfair", "Matchbook", "Smarkets")

OUTCOME_KEY_COLS = [
    "forecast_run_id",
    "event_id",
    "match_date",
    "match",
    "market_key",
    "line",
    "outcome",
]


def odds_to_prob(odds, odds_type: str = "american") -> np.ndarray:
    """Convert sportsbook odds to raw (vig-inclusive) implied probabilities."""
    odds = np.asarray(odds, dtype=float)
    if odds_type != "american":
        raise ValueError(f"unsupported odds_type: {odds_type!r}")
    if np.any(odds == 0):
        raise ValueError("american odds cannot be 0")
    pos = odds > 0
    result = np.empty_like(odds)
    result[pos] = 100.0 / (odds[pos] + 100.0)
    result[~pos] = -odds[~pos] / (-odds[~pos] + 100.0)
    return result


def remove_vig(prob_arr) -> np.ndarray:
    """Multiplicative (proportional) vig removal.

    Caller is responsible for passing a complete outcome set for a single
    book's market; this function does not validate completeness.
    """
    prob_arr = np.asarray(prob_arr, dtype=float)
    total = prob_arr.sum()
    if total <= 0:
        raise ValueError("probabilities must sum to a positive number")
    return prob_arr / total


def _validate_completeness(df: pd.DataFrame, on_incomplete: str) -> pd.DataFrame:
    """Drop or raise on book-markets that don't offer the full outcome set.

    Expected outcomes per ``(forecast_run_id, event_id, market_key, line)``
    are inferred as the union of outcomes seen across all books for that
    micro-market.
    """
    keep_mask = pd.Series(True, index=df.index)
    incomplete_report = []

    market_group_cols = ["forecast_run_id", "event_id", "market_key", "line"]
    for keys, market_df in df.groupby(market_group_cols, sort=False, dropna=False):
        expected = set(market_df["outcome"].unique())
        for book, book_df in market_df.groupby("book", sort=False):
            offered = set(book_df["outcome"])
            if offered != expected:
                missing = expected - offered
                extra = offered - expected
                incomplete_report.append(
                    {
                        "event_id": keys[1],
                        "market_key": keys[2],
                        "line": keys[3],
                        "book": book,
                        "missing": sorted(missing),
                        "extra": sorted(extra),
                    }
                )
                keep_mask.loc[book_df.index] = False

    if incomplete_report:
        msg = (
            f"Incomplete book-markets detected ({len(incomplete_report)}):\n"
            + "\n".join(str(r) for r in incomplete_report)
        )
        if on_incomplete == "raise":
            raise ValueError(msg)
        if on_incomplete == "drop":
            warnings.warn(msg, stacklevel=3)
            return df[keep_mask].copy()
        raise ValueError(f"unknown on_incomplete mode: {on_incomplete!r}")

    return df


def _aggregate_consensus(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate book-level novig probabilities per outcome, prefixing cols.

    Returns one row per ``OUTCOME_KEY_COLS`` tuple with:
    ``{prefix}num_books`` (int), ``{prefix}books_used`` (comma-joined str,
    sorted alphabetically for deterministic diffs), ``{prefix}std_prob``,
    ``{prefix}min_prob``, ``{prefix}max_prob``, ``{prefix}range_prob`` and
    a ``market_prob_{prefix.rstrip('_')}`` mean column.
    """
    if df.empty:
        return pd.DataFrame(
            columns=OUTCOME_KEY_COLS
            + [
                f"market_prob_{prefix.rstrip('_')}",
                f"{prefix}num_books",
                f"{prefix}books_used",
                f"{prefix}std_prob",
                f"{prefix}min_prob",
                f"{prefix}max_prob",
                f"{prefix}range_prob",
            ]
        )
    agg = (
        df.groupby(OUTCOME_KEY_COLS, as_index=False, sort=False, dropna=False)
        .agg(
            **{
                f"market_prob_{prefix.rstrip('_')}": ("novig_prob", "mean"),
                f"{prefix}num_books": ("book", "nunique"),
                f"{prefix}books_used": (
                    "book",
                    lambda s: ", ".join(sorted(set(s))),
                ),
                f"{prefix}std_prob": ("novig_prob", "std"),
                f"{prefix}min_prob": ("novig_prob", "min"),
                f"{prefix}max_prob": ("novig_prob", "max"),
            }
        )
    )
    agg[f"{prefix}range_prob"] = (
        agg[f"{prefix}max_prob"] - agg[f"{prefix}min_prob"]
    )
    return agg


def process_match(
    markets: pd.DataFrame,
    on_incomplete: str = "drop",
    sharp_books: tuple[str, ...] = DEFAULT_SHARP_BOOKS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Turn raw odds into consensus probabilities per micro-market.

    For every outcome we emit two parallel diagnostics: the consensus across
    *all* books and the consensus across the *sharp* basket (``sharp_books``,
    default :data:`DEFAULT_SHARP_BOOKS`). Sharp diagnostics are NaN for
    outcomes no sharp book quoted; downstream code reads
    ``sharp_num_books == 0`` for that case.

    Returns
    -------
    df : book-level table with raw_prob and novig_prob columns.
    consensus : per outcome row with all_* / sharp_* diagnostics,
        ``abs_diff_all_vs_sharp``, plus legacy column names (``market_prob``,
        ``submit_prob``, ``mean_prob``, ``median_prob``, ``min_prob``,
        ``max_prob``, ``std_prob``, ``num_books``) preserved for
        backwards-compat with market_sheet/daily_run.
    submit_sums : per-micro-market sums of ``submit_prob``, for sanity checks.
    """
    missing = REQUIRED_COLS - set(markets.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = markets.copy()
    df = _validate_completeness(df, on_incomplete)
    if df.empty:
        raise ValueError("no complete book-markets remain after validation")

    df["raw_prob"] = odds_to_prob(df["american_odds"].values, odds_type="american")
    df["novig_prob"] = (
        df.groupby(BOOK_MARKET_COLS, sort=False, dropna=False)["raw_prob"]
        .transform(remove_vig)
    )

    book_sums = (
        df.groupby(BOOK_MARKET_COLS, as_index=False, dropna=False)["novig_prob"]
        .sum()
        .rename(columns={"novig_prob": "book_novig_sum"})
    )
    bad = book_sums[~np.isclose(book_sums["book_novig_sum"], 1.0)]
    if len(bad) > 0:
        raise ValueError(f"book-level no-vig sums do not equal 1:\n{bad}")

    all_agg = _aggregate_consensus(df, prefix="all_")
    sharp_df = df[df["book"].isin(sharp_books)]
    sharp_agg = _aggregate_consensus(sharp_df, prefix="sharp_")

    consensus = all_agg.merge(sharp_agg, on=OUTCOME_KEY_COLS, how="left")

    # Median across all books (kept for backwards-compat with old reports).
    median = (
        df.groupby(OUTCOME_KEY_COLS, as_index=False, sort=False, dropna=False)
        .agg(median_prob=("novig_prob", "median"))
    )
    consensus = consensus.merge(median, on=OUTCOME_KEY_COLS, how="left")

    # Outcomes with no sharp quote get sharp_num_books = 0 (rather than NaN)
    # so the boolean gates downstream can compare cleanly.
    consensus["sharp_num_books"] = (
        consensus["sharp_num_books"].fillna(0).astype(int)
    )
    consensus["sharp_books_used"] = consensus["sharp_books_used"].fillna("")

    consensus["abs_diff_all_vs_sharp"] = (
        consensus["market_prob_all"] - consensus["market_prob_sharp"]
    ).abs()

    # market_prob / submit_prob remain pinned to the all-books mean for now.
    consensus["market_prob"] = consensus["market_prob_all"]
    consensus["submit_prob"] = consensus["market_prob"]

    # Backwards-compat aliases used by market_sheet.py / daily_run.py.
    consensus["mean_prob"] = consensus["market_prob_all"]
    consensus["min_prob"] = consensus["all_min_prob"]
    consensus["max_prob"] = consensus["all_max_prob"]
    consensus["std_prob"] = consensus["all_std_prob"]
    consensus["num_books"] = consensus["all_num_books"]

    submit_sums = (
        consensus.groupby(MARKET_COLS, as_index=False, dropna=False)["submit_prob"]
        .sum()
        .rename(columns={"submit_prob": "submit_prob_sum"})
    )
    bad = submit_sums[~np.isclose(submit_sums["submit_prob_sum"], 1.0)]
    if len(bad) > 0:
        raise ValueError(
            "consensus submit_prob does not sum to 1 per micro-market "
            "(likely a validation gap):\n"
            f"{bad}"
        )

    return df, consensus, submit_sums
