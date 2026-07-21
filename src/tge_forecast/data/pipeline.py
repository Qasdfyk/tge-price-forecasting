"""Orchestrate historical data download and cleaning (Step 1)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from tge_forecast.config import load_config, resolve_project_path
from tge_forecast.data.clean import build_hourly_dataset
from tge_forecast.data.fetch_pse import fetch_pse_range
from tge_forecast.data.fetch_tge import fetch_tge_range
from tge_forecast.data.fetch_weather import fetch_weather_range

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def run_download_pipeline(
    *,
    config_path: Path | str | None = None,
    skip_tge: bool = False,
    skip_pse: bool = False,
    skip_weather: bool = False,
    force_tge: bool = False,
) -> Path:
    """Download all sources and build the hourly dataset.

    Returns:
        Path to ``data/processed/hourly_dataset.parquet``.
    """
    config: dict[str, Any] = load_config(config_path)
    paths = config["paths"]
    date_range = config["date_range"]
    start = _parse_date(date_range["start_date"])
    end = _parse_date(date_range["end_date"])

    raw_root = resolve_project_path(paths["raw_dir"])
    processed_dir = resolve_project_path(paths["processed_dir"])
    tge_dir = raw_root / "tge"
    pse_dir = raw_root / "pse"
    weather_dir = raw_root / "weather"

    tge_cfg = config["tge"]
    pse_cfg = config["pse"]
    weather_cfg = config["weather"]
    cleaning_cfg = config["cleaning"]

    tge_path = tge_dir / "tge_rdn_fixing1.parquet"
    if not skip_tge:
        tge_path = fetch_tge_range(
            start,
            end,
            output_dir=tge_dir,
            base_url=tge_cfg["base_url"],
            request_delay_seconds=float(tge_cfg["request_delay_seconds"]),
            max_retries=int(tge_cfg["max_retries"]),
            force=force_tge,
        )
    elif not tge_path.exists():
        raise FileNotFoundError(f"skip_tge=True, but file is missing: {tge_path}")

    if not skip_pse:
        pse_paths = fetch_pse_range(
            start,
            end,
            output_dir=pse_dir,
            base_url=pse_cfg["base_url"],
            load_endpoint=pse_cfg["load_endpoint"],
            generation_endpoint=pse_cfg["generation_endpoint"],
            page_size=int(pse_cfg["page_size"]),
        )
    else:
        pse_paths = {
            "load": pse_dir / "pse_load.parquet",
            "generation": pse_dir / "pse_generation.parquet",
        }
        for path in pse_paths.values():
            if not path.exists():
                raise FileNotFoundError(f"skip_pse=True, but file is missing: {path}")

    weather_path = weather_dir / "weather_hourly.parquet"
    if not skip_weather:
        weather_path = fetch_weather_range(
            start,
            end,
            output_dir=weather_dir,
            archive_url=weather_cfg["archive_url"],
            latitude=float(weather_cfg["latitude"]),
            longitude=float(weather_cfg["longitude"]),
            timezone=weather_cfg["timezone"],
            hourly_variables=list(weather_cfg["hourly_variables"]),
        )
    elif not weather_path.exists():
        raise FileNotFoundError(f"skip_weather=True, but file is missing: {weather_path}")

    output_path = processed_dir / "hourly_dataset.parquet"
    build_hourly_dataset(
        tge_path=tge_path,
        pse_load_path=pse_paths["load"],
        pse_generation_path=pse_paths["generation"],
        weather_path=weather_path,
        output_path=output_path,
        pse_agg=cleaning_cfg.get("pse_agg", "mean"),
    )
    return output_path
