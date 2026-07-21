"""CLI: exploratory visualization of the hourly dataset (Step 1.5)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from tge_forecast.data.visualize import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    run_visualization,
)

console = Console()


def main(
    dataset: Path = typer.Option(
        DEFAULT_DATASET,
        "--dataset",
        "-d",
        help="Path to hourly_dataset.parquet or .csv",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directory for PNG figures",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="Open an interactive matplotlib window (in addition to saving PNG).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging."),
) -> None:
    """Create EDA charts from the processed hourly dataset."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    console.print("[bold]Step 1.5[/bold]: visualizing hourly dataset…")
    out_path = run_visualization(
        dataset_path=dataset,
        output_dir=output_dir,
        show=show,
    )
    console.print(f"[green]Done.[/green] Figure: {out_path}")


def app() -> None:
    """Poetry entry-point (`visualize-data`)."""
    typer.run(main)


if __name__ == "__main__":
    app()
