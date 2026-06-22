"""Append / upsert helpers for daily logs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Append rows to a CSV, creating the file with a header if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def upsert_csv(
    df: pd.DataFrame,
    path: str | Path,
    key_cols: list[str],
) -> None:
    """Write rows to CSV, replacing any existing rows matching ``key_cols``.

    Rationale: re-running a notebook cell on the same odds should not produce
    duplicate rows. The newest version of each ``key_cols`` tuple wins.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = set(key_cols) - set(df.columns)
    if missing:
        raise ValueError(f"key_cols not present in df: {sorted(missing)}")

    if not path.exists():
        df.to_csv(path, index=False)
        return

    existing = pd.read_csv(path)
    missing_in_existing = set(key_cols) - set(existing.columns)
    if missing_in_existing:
        raise ValueError(
            f"existing file is missing key_cols {sorted(missing_in_existing)}; "
            "schema mismatch with new rows"
        )

    keys_in_new = df[key_cols].apply(tuple, axis=1)
    keys_in_existing = existing[key_cols].apply(tuple, axis=1)
    kept = existing[~keys_in_existing.isin(set(keys_in_new))]

    combined = pd.concat([kept, df], ignore_index=True)
    combined.to_csv(path, index=False)
