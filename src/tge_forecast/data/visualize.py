"""Exploratory visualization of the hourly modeling dataset (Step 1.5)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data/processed/hourly_dataset.parquet")
DEFAULT_OUTPUT_DIR = Path("reports/figures")


def load_hourly_dataset(path: Path) -> pd.DataFrame:
    """Load the processed hourly dataset from Parquet or CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Run `download-data` first (Step 1).")

    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Europe/Warsaw")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _style_axes(ax: Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelrotation=30)


def build_overview_figure(frame: pd.DataFrame) -> Figure:
    """Build a multi-panel EDA overview figure."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    ts = frame["timestamp"]

    ax = axes[0, 0]
    ax.plot(ts, frame["price_pln_mwh"], color="#1f4e79", linewidth=1.2)
    _style_axes(ax, "RDN Fixing I price", "PLN/MWh")

    ax = axes[0, 1]
    ax.plot(ts, frame["load_actual"], label="actual", color="#c0392b", linewidth=1.1)
    if "load_fcst" in frame.columns:
        ax.plot(
            ts,
            frame["load_fcst"],
            label="forecast",
            color="#e67e22",
            linewidth=1.0,
            alpha=0.85,
        )
    ax.legend(loc="best", fontsize=8)
    _style_axes(ax, "PSE system load", "MW")

    ax = axes[1, 0]
    ax.plot(ts, frame["gen_wi"], label="wind", color="#2980b9", linewidth=1.1)
    ax.plot(ts, frame["gen_fv"], label="PV", color="#f1c40f", linewidth=1.1)
    ax.legend(loc="best", fontsize=8)
    _style_axes(ax, "PSE renewables generation", "MW")

    ax = axes[1, 1]
    ax.plot(
        ts,
        frame["weather_temperature_2m"],
        label="temp °C",
        color="#8e44ad",
        linewidth=1.1,
    )
    ax_twin = ax.twinx()
    ax_twin.plot(
        ts,
        frame["weather_shortwave_radiation"],
        label="radiation",
        color="#27ae60",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.set_title("Weather (Warsaw proxy)")
    ax.set_ylabel("Temperature (°C)")
    ax_twin.set_ylabel("Shortwave radiation (W/m²)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelrotation=30)
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax_twin.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc="best", fontsize=8)

    ax = axes[2, 0]
    hourly_profile = frame.groupby("hour", observed=True)["price_pln_mwh"].mean()
    ax.bar(hourly_profile.index, hourly_profile.values, color="#1f4e79", alpha=0.85)
    ax.set_xticks(range(0, 24, 2))
    _style_axes(ax, "Average price by hour of day", "PLN/MWh")
    ax.set_xlabel("Hour")

    ax = axes[2, 1]
    corr_cols = [
        "price_pln_mwh",
        "load_actual",
        "gen_wi",
        "gen_fv",
        "weather_temperature_2m",
        "weather_shortwave_radiation",
        "weather_wind_speed_10m",
    ]
    present = [col for col in corr_cols if col in frame.columns]
    corr = frame[present].corr(numeric_only=True)
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(present)))
    ax.set_yticks(np.arange(len(present)))
    short_labels = [col.replace("weather_", "w_") for col in present]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title("Feature correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("TGE RDN hourly dataset — exploratory overview", fontsize=14, y=1.01)
    return fig


def save_overview_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    filename: str = "eda_overview.png",
    dpi: int = 150,
    show: bool = False,
) -> Path:
    """Render and save the overview figure.

    Returns:
        Path to the written PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    fig = build_overview_figure(frame)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("Wrote figure → %s", out_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def run_visualization(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> Path:
    """Load the dataset and write EDA figures."""
    frame = load_hourly_dataset(dataset_path)
    logger.info(
        "Loaded %s rows (%s → %s)",
        len(frame),
        frame["timestamp"].min(),
        frame["timestamp"].max(),
    )
    return save_overview_figure(frame, output_dir, show=show)
