"""Fetch RDN Fixing I prices from the public TGE results page.

TGE does not provide a public API. Hourly results are published at:
https://tge.pl/energia-elektryczna-rdn?dateShow=DD-MM-YYYY
where ``dateShow`` is the publication day = delivery date − 1 day.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)

WARSAW_TZ = ZoneInfo("Europe/Warsaw")
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
}
HOUR_RE = re.compile(r"_H\d{2}$")
MIN_PRICE = -2000.0
MAX_PRICE = 10_000.0


def _parse_pl_number(text: str) -> float | None:
    """Parse a Polish-formatted number (e.g. ``1 234,56``) to float."""
    cleaned = text.strip().replace("\xa0", " ").replace("\u202f", " ")
    if cleaned in {"", "-", "—", "N/A", "n/a", "brak"}:
        return None

    cleaned = cleaned.replace(" ", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def _build_timestamps(delivery_date: date, n_hours: int) -> list[datetime]:
    """Build local hourly timestamps with correct DST handling."""
    local_midnight = datetime(
        delivery_date.year,
        delivery_date.month,
        delivery_date.day,
        tzinfo=WARSAW_TZ,
    )
    midnight_utc = local_midnight.astimezone(ZoneInfo("UTC"))
    return [
        (midnight_utc + timedelta(hours=offset)).astimezone(WARSAW_TZ)
        for offset in range(n_hours)
    ]


def fetch_tge_day(
    delivery_date: date,
    *,
    base_url: str = "https://tge.pl/energia-elektryczna-rdn",
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch Fixing I prices for a single delivery day.

    Args:
        delivery_date: Physical energy delivery date.
        base_url: Base URL of the RDN results page.
        session: Optional ``requests`` session (TCP reuse).
        timeout: HTTP timeout in seconds.

    Returns:
        DataFrame with columns ``timestamp``, ``price_pln_mwh``, ``delivery_date``.

    Raises:
        ValueError: If the table is empty or prices are out of a sane range.
        requests.HTTPError: On HTTP failure.
    """
    http = session or requests.Session()
    publish_date = delivery_date - timedelta(days=1)
    url = f"{base_url}?dateShow={publish_date.strftime('%d-%m-%Y')}"

    response = http.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    tbody = soup.select_one("#rdn > tbody")
    if tbody is None:
        raise ValueError(f"Table #rdn not found for {delivery_date.isoformat()}")

    raw_prices: list[float] = []
    for row in tbody.select("tr"):
        cells = row.select("td")
        if len(cells) < 3:
            continue
        instrument = cells[0].get_text(strip=True)
        if not HOUR_RE.search(instrument):
            continue
        price = _parse_pl_number(cells[2].get_text(strip=True))
        if price is None:
            raise ValueError(f"Missing Fixing I price for {instrument}")
        if not (MIN_PRICE <= price <= MAX_PRICE):
            raise ValueError(f"Price out of range ({price}) for {instrument}")
        raw_prices.append(price)

    if not (23 <= len(raw_prices) <= 25):
        raise ValueError(
            f"Unexpected hour count ({len(raw_prices)}) for {delivery_date.isoformat()}"
        )

    timestamps = _build_timestamps(delivery_date, len(raw_prices))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_pln_mwh": raw_prices,
            "delivery_date": delivery_date.isoformat(),
        }
    )


def fetch_tge_range(
    start_date: date,
    end_date: date,
    *,
    output_dir: Path,
    base_url: str = "https://tge.pl/energia-elektryczna-rdn",
    request_delay_seconds: float = 0.8,
    max_retries: int = 3,
    force: bool = False,
) -> Path:
    """Fetch a TGE date range and write Parquet (plus per-day CSV cache).

    Args:
        start_date: Inclusive start of the range.
        end_date: Inclusive end of the range.
        output_dir: Directory ``data/raw/tge``.
        base_url: TGE results page URL.
        request_delay_seconds: Delay between requests (be polite to TGE).
        max_retries: Retry count for transient failures.
        force: Overwrite existing daily cache files.

    Returns:
        Path to the aggregated ``tge_rdn_fixing1.parquet`` file.
    """
    output_dir = Path(output_dir)
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    frames: list[pd.DataFrame] = []
    days = pd.date_range(start_date, end_date, freq="D")

    for day in tqdm(days, desc="TGE RDN"):
        delivery = day.date()
        cache_path = daily_dir / f"{delivery.isoformat()}.csv"

        if cache_path.exists() and not force:
            frames.append(pd.read_csv(cache_path, parse_dates=["timestamp"]))
            continue

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                frame = fetch_tge_day(delivery, base_url=base_url, session=session)
                frame.to_csv(cache_path, index=False)
                frames.append(frame)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 — retry + aggregate per-day errors
                last_error = exc
                logger.warning(
                    "TGE %s attempt %s/%s failed: %s",
                    delivery,
                    attempt,
                    max_retries,
                    exc,
                )
                time.sleep(request_delay_seconds * attempt)

        if last_error is not None:
            logger.error("Skipped TGE day %s: %s", delivery, last_error)

        time.sleep(request_delay_seconds)

    if not frames:
        raise RuntimeError("No TGE days fetched — check date range / network.")

    result = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    out_path = output_dir / "tge_rdn_fixing1.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %s TGE rows → %s", len(result), out_path)
    return out_path


def load_tge_parquet(path: Path | str) -> pd.DataFrame:
    """Load persisted TGE prices from Parquet."""
    return pd.read_parquet(path)
