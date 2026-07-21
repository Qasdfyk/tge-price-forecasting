# TGE RDN Price Forecasting

Day-ahead (**RDN**) electricity price forecasting using data from the Polish Power Exchange
(**TGE**), **PSE**, and **Open-Meteo**.

CV project in two phases: Data Science (backtesting) → DevOps/MLOps (Kubernetes CronJob).

## Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project setup + download / clean data | **in progress** |
| 2 | Feature engineering + PyTorch Dataset | — |
| 3 | Model (Lightning) + MLflow + backtest | — |
| 4 | PostgreSQL + live inference script | — |
| 5 | Docker + Kubernetes (minikube CronJob) | — |

## Layout (Step 1)

```text
configs/data.yaml          # date range, endpoints, paths
data/raw/{tge,pse,weather} # raw Parquet/CSV
data/processed/            # hourly_dataset.parquet (+ .csv)
scripts/download_data.py   # CLI wrapper
src/tge_forecast/
  config.py
  data/
    fetch_tge.py           # Fixing I scrape (no public TGE API)
    fetch_pse.py           # PSE API: kse-load + pdgobpkd
    fetch_weather.py       # Open-Meteo Archive
    clean.py               # hourly join
    pipeline.py / cli.py
```

## Prerequisites

- Python **3.11+**
- Poetry (install below if missing)

### Install Poetry (Windows / PowerShell)

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

After install, **close and reopen the terminal**, then verify:

```powershell
poetry --version
```

## Environment setup (Step 1)

```powershell
cd C:\Users\adams\OneDrive\Pulpit\infa\tge-price-forecasting
poetry install
```

## Download data

Set the date range in `configs/data.yaml` (`date_range`).

**Smoke test (3 days)** — already configured by default:

```yaml
date_range:
  start_date: "2026-07-18"
  end_date: "2026-07-20"
```

Then:

```powershell
poetry run download-data
# or:
poetry run python scripts/download_data.py
```

A full ~1-year range takes several minutes because TGE is scraped day-by-day with an
~0.8 s delay (be polite to the server).

Options:

```powershell
poetry run download-data --skip-tge      # reuse TGE cache
poetry run download-data --force-tge    # overwrite daily TGE cache
poetry run download-data -v             # DEBUG logs
```

Output: `data/processed/hourly_dataset.parquet` (+ CSV alongside).

## Data sources

| Source | Method | Notes |
|--------|--------|-------|
| **TGE RDN Fixing I** | HTML scrape `tge.pl/energia-elektryczna-rdn` | **No public API.** No manual download needed. |
| **PSE** | `api.raporty.pse.pl` (`kse-load`, `pdgobpkd`) | Load + wind/PV generation (15-min → hourly mean). |
| **Weather** | Open-Meteo Archive | Point: Warsaw (national MVP proxy). No API key. |

### Do you need a manual CSV?

**Not for the first run.** The pipeline downloads everything.

Optional later (longer history / backup): a CSV loader from
[energy.instrat.pl](https://energy.instrat.pl/ceny/energia-rdn-godzinowe/) if scraping is too
slow for a multi-year backtest.

## Code quality

```powershell
poetry run ruff check src scripts
poetry run black --check src scripts
```

## Next

After you confirm this works, we move to **Step 2**: feature engineering (lags T-24h / T-168h)
and time-series `Dataset` / `DataLoader`.
