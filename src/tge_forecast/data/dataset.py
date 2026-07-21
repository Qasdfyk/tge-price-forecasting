"""GPU-oriented PyTorch Dataset and DataLoader for hourly sequences."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def default_num_workers(configured: int | None = None) -> int:
    """Pick a stable worker count (Windows-friendly default)."""
    if configured is not None:
        return max(0, int(configured))
    cpu = os.cpu_count() or 2
    if sys.platform == "win32":
        return min(4, max(1, cpu // 2))
    return min(8, max(1, cpu - 1))


def _torch() -> Any:
    """Import torch lazily (avoids hard fail when only building feature tables)."""
    import torch

    return torch


class PriceSequenceDataset:
    """Sliding-window dataset for sequence models (LSTM / Transformer).

    Each sample:
      - ``x``: shape ``(lookback, n_features)`` — history ending at t-1
      - ``y``: shape ``(horizon,)`` — target prices for t … t+horizon-1
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        lookback: int,
        horizon: int,
    ) -> None:
        if features.ndim != 2:
            raise ValueError("features must be 2-D (T, F)")
        if targets.ndim != 1:
            raise ValueError("targets must be 1-D (T,)")
        if len(features) != len(targets):
            raise ValueError("features and targets length mismatch")
        if lookback < 1 or horizon < 1:
            raise ValueError("lookback and horizon must be >= 1")

        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.lookback = lookback
        self.horizon = horizon

        self._n_samples = len(self.targets) - lookback - horizon + 1
        if self._n_samples < 1:
            raise ValueError(
                f"Not enough rows ({len(self.targets)}) for lookback={lookback}, "
                f"horizon={horizon}"
            )

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch = _torch()
        start = index
        mid = index + self.lookback
        end = mid + self.horizon
        x = torch.from_numpy(self.features[start:mid])
        y = torch.from_numpy(self.targets[mid:end])
        return x, y


@dataclass(frozen=True)
class DataLoaderBundle:
    """Train / val / test loaders ready for Lightning (Step 3)."""

    train: Any
    val: Any
    test: Any
    n_features: int
    lookback: int
    horizon: int


def create_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool = True,
    num_workers: int | None = None,
    persistent_workers: bool = True,
) -> Any:
    """Build a DataLoader with CUDA-friendly settings.

    Notes:
      - ``pin_memory=True`` speeds host→GPU copies when training with CUDA.
      - ``shuffle`` must be False for val/test; for train, shuffling sequence
        *windows* is OK (time order is preserved inside each window).
    """
    from torch.utils.data import DataLoader

    workers = default_num_workers(num_workers)
    use_persistent = bool(persistent_workers and workers > 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        multiprocessing_context="spawn" if workers > 0 and sys.platform == "win32" else None,
    )


def build_dataloaders(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray | None,
    val_targets: np.ndarray | None,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    lookback: int,
    horizon: int,
    batch_size: int = 64,
    pin_memory: bool = True,
    num_workers: int | None = 4,
    persistent_workers: bool = True,
) -> DataLoaderBundle:
    """Create train/val/test sequence DataLoaders."""
    _torch()  # fail fast with a clear import error if torch is broken

    train_ds = PriceSequenceDataset(
        train_features, train_targets, lookback=lookback, horizon=horizon
    )
    test_ds = PriceSequenceDataset(test_features, test_targets, lookback=lookback, horizon=horizon)

    val_loader = None
    if (
        val_features is not None
        and val_targets is not None
        and len(val_targets) >= lookback + horizon
    ):
        val_ds = PriceSequenceDataset(val_features, val_targets, lookback=lookback, horizon=horizon)
        val_loader = create_dataloader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
    else:
        logger.warning("Validation set too small for sequences — val loader disabled")

    bundle = DataLoaderBundle(
        train=create_dataloader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        ),
        val=val_loader,
        test=create_dataloader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        ),
        n_features=int(train_features.shape[1]),
        lookback=lookback,
        horizon=horizon,
    )
    logger.info(
        "DataLoaders ready: train=%s test=%s n_features=%s pin_memory=%s workers=%s",
        len(train_ds),
        len(test_ds),
        bundle.n_features,
        pin_memory,
        default_num_workers(num_workers),
    )
    return bundle
