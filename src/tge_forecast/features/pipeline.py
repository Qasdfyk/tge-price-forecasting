"""Step 2 pipeline: features → chrono split → scale → DataLoaders."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tge_forecast.config import load_config, resolve_project_path
from tge_forecast.features.engineering import (
    TARGET_COLUMN,
    build_feature_frame,
    feature_config_summary,
    resolve_feature_columns,
)
from tge_forecast.features.splits import assign_split_labels, chronological_split

logger = logging.getLogger(__name__)


def _arrays_from_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    targets = frame[target_column].to_numpy(dtype=np.float32)
    return features, targets


def run_feature_pipeline(
    *,
    config_path: Path | str | None = None,
    build_loaders: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    """Build features, chronological splits, scaler (fit on train only), optional loaders.

    Returns:
        Featured full frame, metadata dict, and DataLoader bundle (or None).
    """
    config_file = (
        Path(config_path)
        if config_path is not None
        else resolve_project_path("configs/features.yaml")
    )
    config: dict[str, Any] = load_config(config_file)
    paths = config["paths"]
    target_column = config.get("target_column", TARGET_COLUMN)
    price_lags = list(config["price_lags"])
    exog_lags = {str(k): list(v) for k, v in config.get("exog_lags", {}).items()}
    base_cols = list(config["base_feature_columns"])
    split_cfg = config["splits"]
    seq_cfg = config["sequence"]
    loader_cfg = config["dataloader"]

    input_path = resolve_project_path(paths["input_dataset"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing {input_path}. Run download-data (Step 1) first.")

    raw = pd.read_parquet(input_path)
    n_raw = len(raw)

    # Drop lag horizons that exceed available history (smoke datasets, etc.)
    usable_price_lags = [lag for lag in price_lags if lag < n_raw]
    if not usable_price_lags:
        usable_price_lags = [1]
    if usable_price_lags != price_lags:
        logger.warning(
            "Adjusted price_lags %s → %s (only %s input rows)",
            price_lags,
            usable_price_lags,
            n_raw,
        )
        price_lags = usable_price_lags

    usable_exog: dict[str, list[int]] = {}
    for column, lags in exog_lags.items():
        kept = [lag for lag in lags if lag < n_raw]
        if kept:
            usable_exog[column] = kept
    exog_lags = usable_exog

    featured = build_feature_frame(
        raw,
        price_lags=price_lags,
        exog_lags=exog_lags,
        target_column=target_column,
    )
    feature_columns = resolve_feature_columns(
        featured,
        base_feature_columns=base_cols,
        price_lags=price_lags,
        exog_lags=exog_lags,
        target_column=target_column,
    )

    split = chronological_split(
        featured,
        test_months=int(split_cfg["test_months"]),
        val_months=int(split_cfg["val_months"]),
    )

    # Fit scaler on TRAIN ONLY (no leakage into val/test)
    scaler = StandardScaler()
    scaler.fit(split.train[feature_columns].to_numpy(dtype=np.float64))

    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        out[feature_columns] = scaler.transform(
            frame[feature_columns].to_numpy(dtype=np.float64)
        ).astype(np.float32)
        return out

    train_s = _transform(split.train)
    val_s = _transform(split.val)
    test_s = _transform(split.test)

    features_path = resolve_project_path(paths["features_dataset"])
    meta_path = resolve_project_path(paths["meta_path"])
    scaler_path = resolve_project_path(paths["scaler_path"])
    features_path.parent.mkdir(parents=True, exist_ok=True)

    labeled = assign_split_labels(featured, split)
    labeled.to_parquet(features_path, index=False)
    joblib.dump(scaler, scaler_path)

    meta: dict[str, Any] = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "price_lags": price_lags,
        "exog_lags": exog_lags,
        "lookback": int(seq_cfg["lookback"]),
        "horizon": int(seq_cfg["horizon"]),
        "test_months": int(split_cfg["test_months"]),
        "val_months": int(split_cfg["val_months"]),
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "train_end": str(split.train_end),
        "val_end": str(split.val_end),
        "test_end": str(split.test_end),
        "features_path": str(features_path),
        "scaler_path": str(scaler_path),
        "input_path": str(input_path),
        "n_rows_featured": len(featured),
        "n_features": len(feature_columns),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote features → %s", features_path)
    logger.info("Wrote meta → %s (%s)", meta_path, feature_config_summary(meta))
    logger.info("Wrote scaler → %s", scaler_path)

    loaders = None
    if build_loaders:
        from tge_forecast.data.dataset import build_dataloaders

        lookback = int(seq_cfg["lookback"])
        horizon = int(seq_cfg["horizon"])
        min_rows = lookback + horizon

        # Auto-shrink windows for short smoke datasets so the pipeline is testable
        max_available = min(len(train_s), len(test_s))
        if max_available < min_rows:
            lookback = max(1, max_available // 3)
            horizon = max(1, max_available // 6)
            logger.warning(
                "Sequence too short for configured windows — "
                "using lookback=%s horizon=%s for this run",
                lookback,
                horizon,
            )
            meta["lookback"] = lookback
            meta["horizon"] = horizon
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        train_x_s, train_y = _arrays_from_frame(train_s, feature_columns, target_column)
        test_x_s, test_y = _arrays_from_frame(test_s, feature_columns, target_column)
        val_x_s: np.ndarray | None
        val_y: np.ndarray | None
        if not val_s.empty:
            val_x_s, val_y = _arrays_from_frame(val_s, feature_columns, target_column)
        else:
            val_x_s, val_y = None, None

        loaders = build_dataloaders(
            train_features=train_x_s,
            train_targets=train_y,
            val_features=val_x_s,
            val_targets=val_y,
            test_features=test_x_s,
            test_targets=test_y,
            lookback=lookback,
            horizon=horizon,
            batch_size=int(loader_cfg["batch_size"]),
            pin_memory=bool(loader_cfg.get("pin_memory", True)),
            num_workers=int(loader_cfg.get("num_workers", 4)),
            persistent_workers=bool(loader_cfg.get("persistent_workers", True)),
        )

    return featured, meta, loaders
