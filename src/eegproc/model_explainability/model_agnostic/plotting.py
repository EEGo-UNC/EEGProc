"""Array loading and plotting helpers independent of any dataset or model."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle, Polygon

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
    """Load aligned reference and counterfactual trials from a saved archive.

    Save the archive from an optimization result with
    ``numpy.savez_compressed(path, **result["arrays"], channel_positions=...)``.

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
                "Pass an archive containing the optimizer result arrays."
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
    """Resolve explicit or saved labels, falling back to numbered channels."""
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
    return [f"Ch {index + 1}" for index in range(n_channels)]


def summarize_activity(signal: np.ndarray, *, measure: str) -> np.ndarray:
    """Reduce ``(samples, channels)`` to one whole-trial value per channel."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or not signal.size or not np.isfinite(signal).all():
        raise ValueError("signal must be a finite, nonempty (samples, channels) array.")
    if measure == "mean-absolute":
        return np.mean(np.abs(signal), axis=0)
    if measure == "rms":
        return np.sqrt(np.mean(np.square(signal), axis=0))
    if measure == "mean":
        return np.mean(signal, axis=0)
    raise ValueError("measure must be 'mean-absolute', 'rms', or 'mean'.")


def _draw_head(ax) -> None:
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="black", linewidth=2.0))
    ax.add_patch(
        Polygon(
            [(-0.11, 0.98), (0.0, 1.12), (0.11, 0.98)],
            closed=False,
            fill=False,
            color="black",
            linewidth=2.0,
        )
    )
    ax.plot([-1.0, -1.08, -1.0], [0.18, 0.0, -0.18], color="black", linewidth=2)
    ax.plot([1.0, 1.08, 1.0], [0.18, 0.0, -0.18], color="black", linewidth=2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-1.10, 1.16)
    ax.axis("off")


def plot_band_topographies(
    values: np.ndarray,
    *,
    channel_names: list[str],
    band_names: list[str],
    colorbar_label: str,
    channel_positions: np.ndarray | None = None,
    title: str | None = None,
    shared_scale: bool = False,
    signed: bool | None = None,
):
    """Plot ``(bands, channels)`` values with band-relative color scales."""
    values = np.asarray(values, dtype=float)
    expected_shape = (len(band_names), len(channel_names))
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(
            f"values must be finite and shaped {expected_shape}; received {values.shape}."
        )
    if len(channel_names) < 3:
        raise ValueError("At least three positioned channels are required.")

    if channel_positions is None:
        raise ValueError("Provide channel_positions in normalized scalp coordinates.")
    positions = np.asarray(channel_positions, dtype=float)
    if positions.shape != (len(channel_names), 2) or not np.isfinite(positions).all():
        raise ValueError("channel_positions must be finite and shaped (n_channels, 2).")
    x_positions, y_positions = positions[:, 0], positions[:, 1]
    triangulation = mtri.Triangulation(x_positions, y_positions)
    grid_axis = np.linspace(-1.0, 1.0, 250)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    outside_head = grid_x**2 + grid_y**2 > 1.0

    if signed is None:
        signed = bool(np.any(values < 0))
    shared_bounds = None
    if shared_scale:
        if signed:
            limit = float(np.max(np.abs(values))) or 1.0
            shared_bounds = (-limit, limit)
        else:
            shared_bounds = (0.0, float(np.max(values)) or 1.0)

    fig, axes = plt.subplots(
        1,
        len(band_names),
        figsize=(5.2 * len(band_names), 5.8),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]
    peak_channels = {}

    for ax, band_name, band_values in zip(axes, band_names, values):
        bounds = shared_bounds
        if bounds is None and signed:
            limit = float(np.max(np.abs(band_values))) or 1.0
            bounds = (-limit, limit)
        elif bounds is None:
            lower = float(np.min(band_values))
            upper = float(np.max(band_values))
            if np.isclose(lower, upper):
                padding = max(abs(upper) * 0.01, 1e-12)
                lower -= padding
                upper += padding
            bounds = (lower, upper)
        lower, upper = bounds
        if signed:
            levels = np.linspace(lower, upper, 61)
            cmap = "RdBu_r"
        else:
            levels = np.linspace(lower, upper, 61)
            cmap = "magma"

        interpolator = mtri.LinearTriInterpolator(triangulation, band_values)
        grid_values = interpolator(grid_x, grid_y)
        grid_values = np.ma.masked_where(outside_head, grid_values)
        contour = ax.contourf(
            grid_x,
            grid_y,
            grid_values,
            levels=levels,
            cmap=cmap,
            extend="both" if signed else "neither",
        )
        ax.scatter(
            x_positions,
            y_positions,
            c=band_values,
            cmap=cmap,
            edgecolors="black",
            linewidths=0.8,
            s=70,
            zorder=4,
            vmin=float(levels[0]),
            vmax=float(levels[-1]),
        )
        for channel_name, (x_position, y_position) in zip(channel_names, positions):
            ax.annotate(
                channel_name,
                (x_position, y_position),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                zorder=5,
            )
        _draw_head(ax)
        peak_index = int(np.argmax(np.abs(band_values) if signed else band_values))
        peak_channel = channel_names[peak_index]
        peak_channels[band_name] = peak_channel
        ax.set_title(f"{band_name}\nPeak: {peak_channel}")

        if not shared_scale:
            colorbar = fig.colorbar(contour, ax=ax, shrink=0.72, pad=0.02)
            colorbar.set_label(colorbar_label)

    if shared_scale:
        colorbar = fig.colorbar(contour, ax=axes.tolist(), shrink=0.78, pad=0.02)
        colorbar.set_label(colorbar_label)
    if title:
        fig.suptitle(title)
    return fig, peak_channels
