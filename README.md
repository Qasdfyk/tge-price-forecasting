# TGE RDN Price Forecasting

Day-ahead (**RDN**) electricity price forecasting using data from the Polish Power Exchange
(**TGE**), **PSE**, and **Open-Meteo**.

CV project in two phases: Data Science (backtesting) → DevOps/MLOps (Kubernetes CronJob).

## Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project setup + download / clean data | done (smoke) |
| 1.5 | Exploratory data visualization | **in progress** |
| 2 | Feature engineering + PyTorch Dataset | — |
| 3 | Model (Lightning) + MLflow + backtest | — |
| 4 | PostgreSQL + live inference script | — |
| 5 | Docker + Kubernetes (minikube CronJob) | — |

## Layout

```text
configs/data.yaml
data/raw/{tge,pse,weather}/
data/processed/hourly_dataset.parquet
reports/figures/               # Step 1.5 PNGs
scripts/download_data.py
scripts/visualize_data.py
src/tge_forecast/data/
  fetch_*.py / clean.py / pipeline.py / cli.py
  visualize.py / viz_cli.py
```

## Prerequisites

- Python **3.11+**
- Poetry

### Install Poetry

**PowerShell (Windows):**

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

If `poetry` is not on `PATH`, use `py -m poetry …` instead (or add
`%APPDATA%\Python\Python312\Scripts` to PATH and reopen the terminal).

**Linux / macOS:**

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc or ~/.zshrc
poetry --version
```

## Environment setup

**PowerShell:**

```powershell
cd C:\Users\adams\OneDrive\Pulpit\infa\tge-price-forecasting
py -m poetry install
```

**Linux / macOS:**

```bash
cd ~/path/to/tge-price-forecasting
poetry install
```

## Step 1 — Download data

Set the date range in `configs/data.yaml` (`date_range`).

Smoke test (3 days) is the default:

```yaml
date_range:
  start_date: "2026-07-18"
  end_date: "2026-07-20"
```

**PowerShell:**

```powershell
py -m poetry run download-data -v
# or:
py -m poetry run python scripts/download_data.py -v
```

**Linux / macOS:**

```bash
poetry run download-data -v
# or:
poetry run python scripts/download_data.py -v
```

Useful flags:

```text
--skip-tge / --skip-pse / --skip-weather   reuse cache
--force-tge                                overwrite daily TGE cache
-v                                         DEBUG logs
```

Output: `data/processed/hourly_dataset.parquet` (+ CSV).

A full ~1-year range takes several minutes (TGE scraped day-by-day, ~0.8 s delay).

## Step 1.5 — Visualize data

Requires Step 1 output. Writes `reports/figures/eda_overview.png`
(price, load, renewables, weather, hourly profile, correlation heatmap).

**PowerShell:**

```powershell
py -m poetry install
py -m poetry run visualize-data -v
# or:
py -m poetry run python scripts/visualize_data.py -v
```

**Linux / macOS:**

```bash
poetry install
poetry run visualize-data -v
# or:
poetry run python scripts/visualize_data.py -v
```

Optional flags:

```text
-d PATH / --dataset PATH     custom dataset path (parquet or csv)
-o DIR  / --output-dir DIR   figure output directory
--show                       also open an interactive matplotlib window
```

## Data sources

| Source | Method | Notes |
|--------|--------|-------|
| **TGE RDN Fixing I** | HTML scrape `tge.pl/energia-elektryczna-rdn` | No public API. No manual download needed. |
| **PSE** | `api.raporty.pse.pl` (`kse-load`, `pdgobpkd`) | Load + wind/PV (15-min → hourly mean). |
| **Weather** | Open-Meteo Archive | Warsaw point (national MVP proxy). No API key. |

## Code quality

**PowerShell:**

```powershell
py -m poetry run ruff check src scripts
py -m poetry run black --check src scripts
```

**Linux / macOS:**

```bash
poetry run ruff check src scripts
poetry run black --check src scripts
```

## Next

After you confirm Step 1.5 works, we move to **Step 2**: feature engineering
(lags T-24h / T-168h) and time-series `Dataset` / `DataLoader`.
