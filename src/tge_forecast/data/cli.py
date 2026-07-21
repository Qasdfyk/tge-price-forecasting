"""CLI: download and clean historical data."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from tge_forecast.data.pipeline import run_download_pipeline

console = Console()


def main(
    config: Path = typer.Option(
        Path("configs/data.yaml"),
        "--config",
        "-c",
        help="Path to configs/data.yaml",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    skip_tge: bool = typer.Option(False, help="Skip TGE scraping (use cache)."),
    skip_pse: bool = typer.Option(False, help="Skip PSE API (use cache)."),
    skip_weather: bool = typer.Option(False, help="Skip Open-Meteo (use cache)."),
    force_tge: bool = typer.Option(False, help="Overwrite daily TGE cache."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging."),
) -> None:
    """Download TGE / PSE / Open-Meteo data and build hourly_dataset."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    console.print("[bold]Step 1[/bold]: downloading and cleaning data…")
    output = run_download_pipeline(
        config_path=config,
        skip_tge=skip_tge,
        skip_pse=skip_pse,
        skip_weather=skip_weather,
        force_tge=force_tge,
    )
    console.print(f"[green]Done.[/green] Dataset: {output}")


def app() -> None:
    """Poetry entry-point (`download-data`)."""
    typer.run(main)


if __name__ == "__main__":
    app()
