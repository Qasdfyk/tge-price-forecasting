"""Load project YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the data configuration from YAML.

    Args:
        path: Path to the YAML file. Defaults to ``configs/data.yaml``.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return data


def resolve_project_path(relative: str | Path) -> Path:
    """Return an absolute path relative to the repository root."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
