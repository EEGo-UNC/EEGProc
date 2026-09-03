"""Shared loading helpers for counterfactual plotting scripts.

The counterfactual runner stores one trial with shape
``(1, windows, timesteps, channels)``.  Plotting flattens the chronological
window and timestep axes while keeping channels separate.  Each decoder branch
is handled independently because the runner deliberately does not fuse branch
reconstructions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


DREAMER_CHANNEL_NAMES = (
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
)

DEFAULT_BAND_NAMES = ("Theta", "Alpha", "Beta")


def _as_trial(array: np.ndarray, *, key: str) -> np.ndarray:
    """Return a finite ``(windows, timesteps, channels)`` trial."""
    trial = np.asarray(array, dtype=float)
    if trial.ndim == 4:
        if trial.shape[0] != 1:
            raise ValueError(f"{key!r} must contain exactly one trial.")
        trial = trial[0]
    elif trial.ndim == 2:
        trial = trial[None, ...]
    if trial.ndim != 3 or any(size < 1 for size in trial.shape):
        raise ValueError(
            f"{key!r} must have shape (1,W,T,C), (W,T,C), or (T,C); "
            f"received {trial.shape}."
        )
    if not np.isfinite(trial).all():
        raise ValueError(f"{key!r} contains non-finite values.")
    return trial


def _decode_names(values: np.ndarray) -> list[str]:
    names = []
    for value in np.asarray(values).reshape(-1).tolist():
        names.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return names


def load_counterfactual_trial(
    npz_path: Path,
    *,
    branch: str | None = None,
    reference: str = "input",
) -> tuple[np.ndarray, np.ndarray, str, list[str] | None]:
    """Load aligned reference and counterfactual trials from runner output.

    ``reference='input'`` computes the standard counterfactual difference by
    comparing ``x_prime_<branch>`` directly with saved input ``x``.
    ``reference='reconstruction'`` compares with ``x_reconstructed_<branch>``
    instead, isolating latent intervention change from reconstruction error.
    """
    npz_path = Path(npz_path)
    if reference not in {"reconstruction", "input"}:
        raise ValueError("reference must be 'reconstruction' or 'input'.")

    with np.load(npz_path, allow_pickle=False) as data:
        branches = sorted(
            key.removeprefix("x_prime_")
            for key in data.files
            if key.startswith("x_prime_")
        )
        if not branches:
            raise KeyError(
                f"{npz_path} has no x_prime_<branch> arrays. "
                "Pass a counterfactual.npz file produced by run_counterfactuals."
            )
        if branch is None:
            if len(branches) != 1:
                raise ValueError(
                    "Multiple decoder branches are available; choose one with "
                    f"--branch. Available branches: {branches}"
                )
            branch = branches[0]
        if branch not in branches:
            raise ValueError(
                f"Unknown branch {branch!r}. Available branches: {branches}"
            )

        counterfactual_key = f"x_prime_{branch}"
        reference_key = (
            f"x_reconstructed_{branch}" if reference == "reconstruction" else "x"
        )
        if reference_key not in data.files:
            raise KeyError(f"{npz_path} is missing required array {reference_key!r}.")
        reference_trial = _as_trial(data[reference_key], key=reference_key)
        counterfactual_trial = _as_trial(
            data[counterfactual_key], key=counterfactual_key
        )
        saved_names = (
            _decode_names(data["channel_names"])
            if "channel_names" in data.files
            else None
        )

    if reference_trial.shape != counterfactual_trial.shape:
        raise ValueError(
            "Reference and counterfactual shapes differ: "
            f"{reference_trial.shape} != {counterfactual_trial.shape}."
        )
    return reference_trial, counterfactual_trial, branch, saved_names


def flatten_trial(trial: np.ndarray) -> np.ndarray:
    """Flatten chronological window/time axes to ``(samples, channels)``."""
    return trial.reshape(-1, trial.shape[-1])


def split_channel_bands(
    signal: np.ndarray,
    *,
    n_channels: int,
    n_bands: int,
    feature_order: str,
) -> np.ndarray:
    """Return ``(samples, bands, channels)`` from flattened channel-band data.

    ``channel-major`` means the flattened feature axis is ordered as all bands
    for channel 1, then all bands for channel 2. ``band-major`` means all
    channels for band 1, then all channels for band 2.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or not signal.size or not np.isfinite(signal).all():
        raise ValueError("signal must be a finite, nonempty (samples, features) array.")
    if n_channels < 1 or n_bands < 1:
        raise ValueError("n_channels and n_bands must be positive.")
    expected_features = n_channels * n_bands
    if signal.shape[1] != expected_features:
        raise ValueError(
            f"Expected {n_channels} channels * {n_bands} bands = "
            f"{expected_features} features; received {signal.shape[1]}."
        )
    if feature_order == "channel-major":
        return signal.reshape(-1, n_channels, n_bands).transpose(0, 2, 1)
    if feature_order == "band-major":
        return signal.reshape(-1, n_bands, n_channels)
    raise ValueError("feature_order must be 'channel-major' or 'band-major'.")


def resolve_channel_names(
    n_channels: int,
    provided: Iterable[str] | None = None,
    saved: Iterable[str] | None = None,
) -> list[str]:
    """Resolve labels, using the standard 14-channel DREAMER order as fallback."""
    source = provided if provided is not None else saved
    if source is not None:
        names = [str(name) for name in source]
        if len(names) != n_channels:
            raise ValueError(
                f"Expected {n_channels} channel names, received {len(names)}."
            )
        if len(set(names)) != len(names):
            raise ValueError("Channel names must be unique.")
        return names
    if n_channels == len(DREAMER_CHANNEL_NAMES):
        return list(DREAMER_CHANNEL_NAMES)
    return [f"Ch {index + 1}" for index in range(n_channels)]


def resolve_feature_names(
    n_features: int,
    *,
    channel_names: Iterable[str] | None = None,
    saved_names: Iterable[str] | None = None,
    band_names: Iterable[str] = DEFAULT_BAND_NAMES,
    feature_order: str = "channel-major",
) -> list[str]:
    """Build heatmap labels for raw channels or flattened channel-band features."""
    if saved_names is not None:
        saved = [str(name) for name in saved_names]
        if len(saved) == n_features:
            return saved

    bands = [str(name) for name in band_names]
    provided_channels = None if channel_names is None else [str(name) for name in channel_names]
    if provided_channels is not None and len(provided_channels) == n_features:
        return provided_channels

    if provided_channels is None:
        if n_features == len(DREAMER_CHANNEL_NAMES):
            return list(DREAMER_CHANNEL_NAMES)
        if n_features == len(DREAMER_CHANNEL_NAMES) * len(bands):
            channels = list(DREAMER_CHANNEL_NAMES)
        else:
            return [f"Feature {index + 1}" for index in range(n_features)]
    else:
        channels = provided_channels

    if len(channels) * len(bands) != n_features:
        raise ValueError(
            f"Expected {len(channels)} channels * {len(bands)} bands = "
            f"{len(channels) * len(bands)} feature labels, but the trial has "
            f"{n_features} features."
        )
    if feature_order == "channel-major":
        return [f"{channel} · {band}" for channel in channels for band in bands]
    if feature_order == "band-major":
        return [f"{channel} · {band}" for band in bands for channel in channels]
    raise ValueError("feature_order must be 'channel-major' or 'band-major'.")
