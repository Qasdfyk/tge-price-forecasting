"""Feature engineering for RDN price forecasting."""

from __future__ import annotations

import logging
from typing import Any

import holidays
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "price_pln_mwh"


def _ensure_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with sorted, timezone-aware ``timestamp`` (Europe/Warsaw)."""
    work = frame.copy()
    if "timestamp" not in work.columns:
        raise ValueError("Dataset must contain a 'timestamp' column")
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True).dt.tz_convert("Europe/Warsaw")
    return work.sort_values("timestamp").reset_index(drop=True)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and Polish public-holiday features (cyclic encodings included)."""
    work = frame.copy()
    ts = work["timestamp"]

    work["hour"] = ts.dt.hour.astype("int16")
    work["day_of_week"] = ts.dt.dayofweek.astype("int16")
    work["month"] = ts.dt.month.astype("int16")
    work["day_of_year"] = ts.dt.dayofyear.astype("int16")
    work["is_weekend"] = (work["day_of_week"] >= 5).astype("int8")

    # Cyclical encodings (better for neural nets than raw integers)
    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24.0)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24.0)
    work["dow_sin"] = np.sin(2 * np.pi * work["day_of_week"] / 7.0)
    work["dow_cos"] = np.cos(2 * np.pi * work["day_of_week"] / 7.0)
    work["month_sin"] = np.sin(2 * np.pi * work["month"] / 12.0)
    work["month_cos"] = np.cos(2 * np.pi * work["month"] / 12.0)

    years = range(int(ts.dt.year.min()), int(ts.dt.year.max()) + 1)
    pl_holidays = holidays.country_holidays("PL", years=years)
    dates = ts.dt.date
    work["is_holiday"] = dates.map(lambda d: int(d in pl_holidays)).astype("int8")

    return work


def add_lag_features(
    frame: pd.DataFrame,
    *,
    price_lags: list[int],
    exog_lags: dict[str, list[int]] | None = None,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add autoregressive and exogenous lag columns (shift in hours)."""
    work = frame.copy()
    if target_column not in work.columns:
        raise ValueError(f"Missing target column: {target_column}")

    for lag in price_lags:
        work[f"{target_column}_lag_{lag}h"] = work[target_column].shift(lag)

    for column, lags in (exog_lags or {}).items():
        if column not in work.columns:
            logger.warning("Skipping lags for missing column: %s", column)
            continue
        for lag in lags:
            work[f"{column}_lag_{lag}h"] = work[column].shift(lag)

    return work


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    price_lags: list[int],
    exog_lags: dict[str, list[int]] | None = None,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Full feature pipeline: calendar + lags; drop rows with incomplete lag history."""
    work = _ensure_timestamp(frame)
    work = add_calendar_features(work)
    work = add_lag_features(
        work,
        price_lags=price_lags,
        exog_lags=exog_lags,
        target_column=target_column,
    )

    n_before = len(work)
    work = work.dropna().reset_index(drop=True)
    logger.info(
        "Feature frame: %s → %s rows after dropping incomplete lag windows (removed %s)",
        n_before,
        len(work),
        n_before - len(work),
    )
    return work


def resolve_feature_columns(
    frame: pd.DataFrame,
    *,
    base_feature_columns: list[str],
    price_lags: list[int],
    exog_lags: dict[str, list[int]] | None = None,
    target_column: str = TARGET_COLUMN,
) -> list[str]:
    """Build the ordered list of model input columns present in ``frame``."""
    calendar_cols = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
    ]
    lag_cols = [f"{target_column}_lag_{lag}h" for lag in price_lags]
    for column, lags in (exog_lags or {}).items():
        lag_cols.extend(f"{column}_lag_{lag}h" for lag in lags)

    candidates = [*base_feature_columns, *calendar_cols, *lag_cols]
    missing = [col for col in candidates if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    return candidates


def feature_config_summary(meta: dict[str, Any]) -> str:
    """Short human-readable summary for logs / CLI."""
    return (
        f"features={len(meta.get('feature_columns', []))} | "
        f"lookback={meta.get('lookback')} | horizon={meta.get('horizon')} | "
        f"train={meta.get('n_train')} val={meta.get('n_val')} test={meta.get('n_test')}"
    )
