"""CLI: build features, chronological splits, and GPU DataLoaders (Step 2)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from tge_forecast.features.pipeline import run_feature_pipeline

console = Console()


def main(
    config: Path = typer.Option(
        Path("configs/features.yaml"),
        "--config",
        "-c",
        help="Path to configs/features.yaml",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    skip_loaders: bool = typer.Option(
        False,
        "--skip-loaders",
        help="Only write features/meta/scaler (skip DataLoader smoke batch).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging."),
) -> None:
    """Run Step 2 feature engineering and verify one DataLoader batch."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    console.print("[bold]Step 2[/bold]: feature engineering + Dataset/DataLoader…")

    try:
        import torch

        cuda_msg = f"CUDA available: {torch.cuda.is_available()}"
        if torch.cuda.is_available():
            cuda_msg += f" ({torch.cuda.get_device_name(0)})"
        console.print(cuda_msg)
    except OSError:
        console.print(
            "[red]PyTorch failed to load (WinError 1114 / c10.dll).[/red]\n"
            "Fix (PowerShell):\n"
            "  py -m poetry run pip uninstall -y torch\n"
            "  py -m poetry run pip install torch "
            "--index-url https://download.pytorch.org/whl/cu124\n"
            "Also install: Microsoft Visual C++ Redistributable 2015-2022 (x64).\n"
            "Meanwhile you can still build tables with: "
            "[bold]py -m poetry run build-features --skip-loaders[/bold]"
        )
        if not skip_loaders:
            raise typer.Exit(code=1) from None

    _, meta, loaders = run_feature_pipeline(
        config_path=config,
        build_loaders=not skip_loaders,
    )

    console.print(
        f"[green]Done.[/green] features={meta['n_features']} "
        f"rows={meta['n_rows_featured']} "
        f"train/val/test={meta['n_train']}/{meta['n_val']}/{meta['n_test']}"
    )
    console.print(f"Features: {meta['features_path']}")
    console.print("Meta:     data/processed/feature_meta.json")
    console.print(f"Scaler:   {meta['scaler_path']}")

    if loaders is not None:
        batch_x, batch_y = next(iter(loaders.train))
        console.print(
            f"Train batch: x={tuple(batch_x.shape)} y={tuple(batch_y.shape)} "
            f"| pin_memory={loaders.train.pin_memory} "
            f"workers={loaders.train.num_workers}"
        )


def app() -> None:
    """Poetry entry-point (`build-features`)."""
    typer.run(main)


if __name__ == "__main__":
    app()
