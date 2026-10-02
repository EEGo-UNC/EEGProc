"""Column roles for signal samples and precomputed EEG features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..preprocessing import FREQUENCY_BANDS


@dataclass(frozen=True)
class FeatureColumn:
    """Channel, optional band, and optional statistic in a feature name."""

    channel: str
    band: str | None = None
    feature: str | None = None


def parse_feature_column(name: str) -> FeatureColumn:
    """Parse ``channel[_band][_statistic]`` without substring matching."""
    tokens = str(name).split("_")
    bands = {*FREQUENCY_BANDS, "beta"}
    if len(tokens) > 1 and tokens[1] in bands:
        return FeatureColumn(tokens[0], tokens[1], "_".join(tokens[2:]) or None)
    return FeatureColumn(tokens[0], None, "_".join(tokens[1:]) or None)


@dataclass
class EEGFrame:
    """An EEG table with explicit subject, trial, label, and feature columns.

    Rows within a trial are sorted by ``time_column`` when supplied. If it is
    omitted, ``sample_index`` is used when present; otherwise row order is kept.
    Label and identifier columns are always excluded from inferred features.
    """

    data: pd.DataFrame
    fs: float
    kind: Literal["signal", "features"] = "signal"
    subject_columns: tuple[str, ...] = ("subject",)
    trial_columns: tuple[str, ...] = ("trial",)
    time_column: str | None = None
    label_columns: tuple[str, ...] = ("label",)
    feature_columns: tuple[str, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            raise ValueError("data must be a nonempty pandas DataFrame.")
        if not self.data.columns.is_unique:
            raise ValueError("DataFrame column names must be unique.")
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("fs must be finite and positive.")
        if self.kind not in {"signal", "features"}:
            raise ValueError("kind must be 'signal' or 'features'.")
        self.subject_columns = tuple(self.subject_columns)
        self.trial_columns = tuple(self.trial_columns)
        self.label_columns = tuple(self.label_columns)
        if not self.subject_columns or not self.trial_columns:
            raise ValueError("Declare at least one subject and trial column.")
        if self.time_column is None and "sample_index" in self.data.columns:
            self.time_column = "sample_index"
        metadata = set((*self.subject_columns, *self.trial_columns, *self.label_columns))
        if self.time_column is not None:
            metadata.add(self.time_column)
        missing = metadata - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing declared columns: {sorted(missing)}.")
        if self.feature_columns is None:
            self.feature_columns = tuple(
                name for name in self.data.columns
                if name not in metadata and pd.api.types.is_numeric_dtype(self.data[name])
            )
        else:
            self.feature_columns = tuple(self.feature_columns)
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Declare at least one feature, without duplicate columns.")
        if set(self.feature_columns) & metadata:
            raise ValueError("Feature columns cannot include labels or identifiers.")
        if set(self.feature_columns) - set(self.data.columns):
            raise ValueError("Some feature columns are absent from the DataFrame.")
        if self.data[list(metadata)].isna().any().any():
            raise ValueError("Identifiers, labels, and sample ordering must not be missing.")
        values = self.data[list(self.feature_columns)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("EEG features must be finite numeric values.")


def median_split(label_column: str, *, threshold: float):
    """Create a fixed-threshold label transform; no threshold is fitted to data."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    def transform(values):
        if isinstance(values, pd.DataFrame):
            values = values[label_column]
        return (np.asarray(values) >= threshold).astype(np.int32)

    return transform
