"""Feature engineering package (Step 2)."""

from tge_forecast.features.engineering import build_feature_frame, resolve_feature_columns
from tge_forecast.features.splits import chronological_split

__all__ = [
    "build_feature_frame",
    "chronological_split",
    "resolve_feature_columns",
    "run_feature_pipeline",
]


def __getattr__(name: str):
    if name == "run_feature_pipeline":
        from tge_forecast.features.pipeline import run_feature_pipeline

        return run_feature_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
