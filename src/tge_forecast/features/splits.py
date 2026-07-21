"""Chronological train / validation / test splits for time series."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChronoSplit:
    """Frames for a time-ordered split (never shuffled)."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def _fraction_split(work: pd.DataFrame, timestamp_col: str) -> ChronoSplit:
    """Fallback when history is shorter than the requested month windows."""
    n = len(work)
    test_start = int(n * 0.70)
    val_start = int(n * 0.55)
    # Ensure non-empty train/test
    test_start = min(max(test_start, 2), n - 1)
    val_start = min(max(val_start, 1), test_start - 1) if test_start > 1 else 0

    train = work.iloc[:val_start].reset_index(drop=True)
    val = work.iloc[val_start:test_start].reset_index(drop=True)
    test = work.iloc[test_start:].reset_index(drop=True)

    logger.warning(
        "History too short for month-based split — using chronological "
        "fractions (~55%% / 15%% / 30%%): train=%s val=%s test=%s",
        len(train),
        len(val),
        len(test),
    )
    return ChronoSplit(
        train=train,
        val=val,
        test=test,
        train_end=pd.Timestamp(train[timestamp_col].max()),
        val_end=(
            pd.Timestamp(val[timestamp_col].max())
            if not val.empty
            else pd.Timestamp(train[timestamp_col].max())
        ),
        test_end=pd.Timestamp(test[timestamp_col].max()),
    )


def chronological_split(
    frame: pd.DataFrame,
    *,
    test_months: int = 12,
    val_months: int = 3,
    timestamp_col: str = "timestamp",
) -> ChronoSplit:
    """Split by time: train → val → test (never shuffled).

    ``test`` = last ``test_months`` of data.
    ``val`` = ``val_months`` immediately before the test window.
    ``train`` = everything before validation.

    If the series is shorter than the requested windows, falls back to
    chronological row fractions (still ordered — never random).
    """
    if test_months < 1:
        raise ValueError("test_months must be >= 1")
    if val_months < 0:
        raise ValueError("val_months must be >= 0")
    if timestamp_col not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    work = frame.sort_values(timestamp_col).reset_index(drop=True)
    ts = pd.to_datetime(work[timestamp_col])
    test_end = ts.max()
    test_start = test_end - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)

    if ts.min() > val_start:
        return _fraction_split(work, timestamp_col)

    test_mask = ts > test_start
    val_mask = (ts > val_start) & (ts <= test_start)
    train_mask = ts <= val_start

    train = work.loc[train_mask].reset_index(drop=True)
    val = work.loc[val_mask].reset_index(drop=True)
    test = work.loc[test_mask].reset_index(drop=True)

    if train.empty or test.empty:
        return _fraction_split(work, timestamp_col)

    if val_months > 0 and val.empty:
        logger.warning(
            "Validation set is empty (val_months=%s). "
            "Consider a longer history or smaller val_months.",
            val_months,
        )

    logger.info(
        "Chrono split: train=%s (%s → %s) | val=%s (%s → %s) | test=%s (%s → %s)",
        len(train),
        train[timestamp_col].min(),
        train[timestamp_col].max(),
        len(val),
        val[timestamp_col].min() if not val.empty else None,
        val[timestamp_col].max() if not val.empty else None,
        len(test),
        test[timestamp_col].min(),
        test[timestamp_col].max(),
    )

    return ChronoSplit(
        train=train,
        val=val,
        test=test,
        train_end=pd.Timestamp(train[timestamp_col].max()),
        val_end=(
            pd.Timestamp(val[timestamp_col].max()) if not val.empty else pd.Timestamp(test_start)
        ),
        test_end=pd.Timestamp(test_end),
    )


def assign_split_labels(
    frame: pd.DataFrame,
    split: ChronoSplit,
    *,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Attach a ``split`` column (train/val/test) using chrono boundaries."""
    out = frame.copy()
    ts = pd.to_datetime(out[timestamp_col])
    out["split"] = "train"
    out.loc[ts > split.train_end, "split"] = "val"
    out.loc[ts > split.val_end, "split"] = "test"
    return out
