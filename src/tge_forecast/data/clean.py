"""Clean and join sources into an hourly modeling dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)


def _to_warsaw_hourly(frame: pd.DataFrame, value_cols: list[str], how: str) -> pd.DataFrame:
    """Normalize to hourly slots in Europe/Warsaw.

    Args:
        frame: Data with a tz-aware ``timestamp`` column.
        value_cols: Columns to aggregate.
        how: ``mean`` or ``last``.
    """
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", *value_cols])

    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True).dt.tz_convert(
        ZoneInfo("Europe/Warsaw")
    )
    work = work.set_index("timestamp").sort_index()
    present = [col for col in value_cols if col in work.columns]
    numeric = work[present].apply(pd.to_numeric, errors="coerce")

    if how == "mean":
        hourly = numeric.resample("h").mean()
    elif how == "last":
        hourly = numeric.resample("h").last()
    else:
        raise ValueError(f"Unsupported aggregation: {how}")

    return hourly.reset_index()


def build_hourly_dataset(
    *,
    tge_path: Path,
    pse_load_path: Path,
    pse_generation_path: Path,
    weather_path: Path,
    output_path: Path,
    pse_agg: str = "mean",
) -> pd.DataFrame:
    """Join TGE + PSE + weather into a single hourly DataFrame.

    Target: ``price_pln_mwh`` (RDN Fixing I).
    Missing values after the join are logged; rows without the target price are dropped.
    """
    tge = pd.read_parquet(tge_path)
    load = pd.read_parquet(pse_load_path)
    gen = pd.read_parquet(pse_generation_path)
    weather = pd.read_parquet(weather_path)

    tge_h = _to_warsaw_hourly(tge, ["price_pln_mwh"], how="last")
    load_h = _to_warsaw_hourly(load, ["load_actual", "load_fcst"], how=pse_agg)
    gen_h = _to_warsaw_hourly(gen, ["gen_wi", "gen_fv", "kse_pow_dem"], how=pse_agg)
    weather_h = _to_warsaw_hourly(
        weather,
        [col for col in weather.columns if col.startswith("weather_")],
        how="last",
    )

    merged = tge_h.merge(load_h, on="timestamp", how="left")
    merged = merged.merge(gen_h, on="timestamp", how="left")
    merged = merged.merge(weather_h, on="timestamp", how="left")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    n_before = len(merged)
    merged = merged.dropna(subset=["price_pln_mwh"]).reset_index(drop=True)
    logger.info(
        "Dataset: %s → %s rows after dropping missing prices (removed %s)",
        n_before,
        len(merged),
        n_before - len(merged),
    )

    missing_ratio = merged.isna().mean().sort_values(ascending=False)
    nontrivial = missing_ratio[missing_ratio > 0]
    if not nontrivial.empty:
        logger.warning("Missing-value share after join:\n%s", nontrivial.to_string())

    # Calendar features — useful from Step 2 onward
    ts = merged["timestamp"]
    merged["hour"] = ts.dt.hour
    merged["day_of_week"] = ts.dt.dayofweek
    merged["month"] = ts.dt.month
    merged["is_weekend"] = (merged["day_of_week"] >= 5).astype("int8")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    merged.to_csv(output_path.with_suffix(".csv"), index=False)
    logger.info("Wrote dataset → %s (+ CSV)", output_path)
    return merged
