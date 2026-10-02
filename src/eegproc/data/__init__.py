"""DataFrame schemas and trial-safe array preparation for EEG models."""

from .schema import EEGFrame, FeatureColumn, median_split, parse_feature_column
from .windowing import WindowedArrays, to_supervised_arrays

__all__ = [
    "EEGFrame", "FeatureColumn", "median_split", "parse_feature_column",
    "WindowedArrays", "to_supervised_arrays",
]
