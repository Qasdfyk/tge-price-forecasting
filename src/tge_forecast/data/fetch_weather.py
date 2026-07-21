"""Historical weather from the Open-Meteo Archive API."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
    "relative_humidity_2m",
]


def fetch_weather_range(
    start_date: date,
    end_date: date,
    *,
    output_dir: Path,
    archive_url: str = "https://archive-api.open-meteo.com/v1/archive",
    latitude: float = 52.23,
    longitude: float = 21.01,
    timezone: str = "Europe/Warsaw",
    hourly_variables: list[str] | None = None,
    timeout: float = 60.0,
) -> Path:
    """Fetch hourly weather and write Parquet.

    Args:
        start_date: Inclusive start of the range.
        end_date: Inclusive end of the range.
        output_dir: Directory ``data/raw/weather``.
        archive_url: Open-Meteo Archive endpoint.
        latitude: Point latitude.
        longitude: Point longitude.
        timezone: Response timezone.
        hourly_variables: Open-Meteo hourly variable list.
        timeout: HTTP timeout in seconds.

    Returns:
        Path to ``weather_hourly.parquet``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variables = hourly_variables or DEFAULT_VARIABLES

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(variables),
        "timezone": timezone,
    }

    logger.info(
        "Fetching Open-Meteo (%s,%s) %s → %s",
        latitude,
        longitude,
        start_date,
        end_date,
    )
    response = requests.get(archive_url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Unexpected Open-Meteo response (missing hourly.time)")

    frame = pd.DataFrame(hourly)
    frame["timestamp"] = pd.to_datetime(frame["time"]).dt.tz_localize(
        ZoneInfo(timezone),
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    frame = frame.drop(columns=["time"]).sort_values("timestamp").reset_index(drop=True)

    # Prefix columns to avoid name collisions on join
    rename = {col: f"weather_{col}" for col in frame.columns if col != "timestamp"}
    frame = frame.rename(columns=rename)

    out_path = output_dir / "weather_hourly.parquet"
    frame.to_parquet(out_path, index=False)
    logger.info("Wrote %s weather rows → %s", len(frame), out_path)
    return out_path
