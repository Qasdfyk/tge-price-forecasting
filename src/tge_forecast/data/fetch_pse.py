"""PSE API client (api.raporty.pse.pl) — load demand and renewables generation."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

WARSAW_TZ = ZoneInfo("Europe/Warsaw")
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "tge-forecast/0.1 (research; contact: local-dev)",
}


def _daterange_filter(start: date, end: date) -> str:
    """Build an inclusive OData filter on ``business_date``."""
    return (
        f"business_date ge '{start.isoformat()}' and "
        f"business_date le '{end.isoformat()}'"
    )


def fetch_pse_endpoint(
    endpoint: str,
    start_date: date,
    end_date: date,
    *,
    base_url: str = "https://api.raporty.pse.pl/api",
    page_size: int = 500,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Fetch all records from a PSE endpoint, following ``nextLink`` pagination.

    Args:
        endpoint: Resource name, e.g. ``kse-load`` or ``pdgobpkd``.
        start_date: Start of the ``business_date`` range.
        end_date: End of the ``business_date`` range.
        base_url: PSE API base URL.
        page_size: Page size (``$first``).
        session: Optional HTTP session.
        timeout: Request timeout in seconds.

    Returns:
        List of JSON records from the ``value`` field.
    """
    http = session or requests.Session()
    url: str | None = f"{base_url.rstrip('/')}/{endpoint}"
    params: dict[str, str] | None = {
        "$filter": _daterange_filter(start_date, end_date),
        "$first": str(page_size),
    }

    records: list[dict[str, Any]] = []
    while url:
        response = http.get(url, headers=DEFAULT_HEADERS, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("value", [])
        if not isinstance(batch, list):
            raise ValueError(f"Unexpected PSE response ({endpoint})")
        records.extend(batch)
        # Later pages: full nextLink URL (do not re-send params)
        url = payload.get("nextLink")
        params = None
        logger.debug("%s: fetched %s records so far", endpoint, len(records))

    return records


def _records_to_frame(records: list[dict[str, Any]], value_cols: list[str]) -> pd.DataFrame:
    """Convert PSE records to a DataFrame with a timezone-aware ``timestamp``."""
    if not records:
        return pd.DataFrame(columns=["timestamp", *value_cols])

    frame = pd.DataFrame(records)
    if "dtime" not in frame.columns:
        raise ValueError("Missing dtime column in PSE response")

    # PSE publishes dtime as Warsaw local time without an explicit offset.
    frame["timestamp"] = pd.to_datetime(frame["dtime"]).dt.tz_localize(
        WARSAW_TZ,
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    keep = ["timestamp", *[col for col in value_cols if col in frame.columns]]
    return frame[keep].sort_values("timestamp").reset_index(drop=True)


def fetch_pse_range(
    start_date: date,
    end_date: date,
    *,
    output_dir: Path,
    base_url: str = "https://api.raporty.pse.pl/api",
    load_endpoint: str = "kse-load",
    generation_endpoint: str = "pdgobpkd",
    page_size: int = 500,
) -> dict[str, Path]:
    """Fetch load demand and renewables generation; write Parquet files.

    Returns:
        Mapping ``{"load": path, "generation": path}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    paths: dict[str, Path] = {}
    jobs: list[tuple[str, str, list[str]]] = [
        ("load", load_endpoint, ["load_actual", "load_fcst", "business_date"]),
        (
            "generation",
            generation_endpoint,
            ["gen_wi", "gen_fv", "kse_pow_dem", "business_date"],
        ),
    ]

    for name, endpoint, cols in tqdm(jobs, desc="PSE API"):
        logger.info("Fetching PSE/%s (%s → %s)", endpoint, start_date, end_date)
        records = fetch_pse_endpoint(
            endpoint,
            start_date,
            end_date,
            base_url=base_url,
            page_size=page_size,
            session=session,
        )
        frame = _records_to_frame(records, cols)
        out_path = output_dir / f"pse_{name}.parquet"
        frame.to_parquet(out_path, index=False)
        paths[name] = out_path
        logger.info("Wrote %s PSE/%s rows → %s", len(frame), name, out_path)

    return paths
