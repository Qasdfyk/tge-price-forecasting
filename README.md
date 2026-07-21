# TGE RDN Price Forecasting

End-to-end system for forecasting day-ahead (**RDN**) electricity prices on the Polish
Power Exchange (**TGE**), from historical modelling to a production-style daily inference job.

## Goals

- Build a reproducible **Data Science** pipeline: ingest market and exogenous data, engineer
  features, train a deep sequence model, and evaluate with chronological backtesting.
- Deliver an **MLOps** path: containerised inference, PostgreSQL storage, and a Kubernetes
  CronJob that fetches fresh inputs and writes next-day forecasts every day.

## Scope

| Phase | Deliverables |
|-------|----------------|
| Data Science | Historical ingest, EDA, feature engineering, PyTorch / Lightning training, MLflow tracking, backtest |
| DevOps / MLOps | Docker image, local PostgreSQL, Kubernetes manifests (minikube), daily CronJob |

## Data sources

| Source | Role |
|--------|------|
| **TGE** (RDN Fixing I) | Target: hourly day-ahead clearing prices (PLN/MWh) |
| **PSE** | System load and renewables generation (wind / PV) |
| **Open-Meteo** | Weather covariates (temperature, wind, radiation, etc.) |

## Stack

Poetry · Pandas · PyTorch / Lightning · scikit-learn · MLflow · Docker · Kubernetes · PostgreSQL · Ruff / Black

## Repository layout

```text
configs/           # Data and feature configuration
src/tge_forecast/  # Application package (data, features, models, …)
scripts/           # Thin CLI entrypoints
data/              # Raw / processed datasets (local)
k8s/               # Kubernetes manifests (later)
reports/           # Generated figures (local)
```

## Quick start

```bash
poetry install
poetry run download-data
poetry run visualize-data
poetry run build-features
```

On Windows, if `poetry` is not on `PATH`, use `py -m poetry …`. For GPU training, install a
CUDA build of PyTorch inside the Poetry environment (see local setup notes).

## Licence

Private / portfolio project — not for redistribution of TGE market data beyond personal use.
