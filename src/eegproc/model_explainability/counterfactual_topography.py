"""Plot one whole-trial counterfactual scalp topography per EEG band.

SIC inputs contain flattened channel-band features. With the default DREAMER
configuration, 42 features are interpreted as 14 electrodes × 3 already
filtered bands. The script splits those features and displays Theta, Alpha,
and Beta maps side by side. Each band uses its own labeled color scale by
default so lower-amplitude bands retain their spatial contrast.

By default, each map contains the signed mean decoded counterfactual difference
at each electrode over the complete trial. A zero-centered diverging scale
shows whether the counterfactual is above or below its reference on average.

Example::

    PYTHONPATH=src python -m eegproc.model_explainability.counterfactual_topography \
        runs/.../subject_0_trial_0/counterfactual.npz \
        --branch gcn_gru
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Circle, Polygon

if __package__:
    from .counterfactual_plotting import (
        DEFAULT_BAND_NAMES,
        flatten_trial,
        load_counterfactual_trial,
        resolve_channel_names,
        split_channel_bands,
    )
else:
    from counterfactual_plotting import (  # type: ignore[no-redef]
        DEFAULT_BAND_NAMES,
        flatten_trial,
        load_counterfactual_trial,
        resolve_channel_names,
        split_channel_bands,
    )


# Approximate azimuthal positions for the DREAMER Emotiv EPOC montage.
DREAMER_POSITIONS = {
    "AF3": (-0.40, 0.84),
    "F7": (-0.78, 0.48),
    "F3": (-0.42, 0.48),
    "FC5": (-0.68, 0.16),
    "T7": (-0.92, 0.00),
    "P7": (-0.76, -0.45),
    "O1": (-0.34, -0.82),
    "O2": (0.34, -0.82),
    "P8": (0.76, -0.45),
    "T8": (0.92, 0.00),
    "FC6": (0.68, 0.16),
    "F4": (0.42, 0.48),
    "F8": (0.78, 0.48),
    "AF4": (0.40, 0.84),
}


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


def _channel_positions(channel_names: list[str]) -> np.ndarray:
    missing = [name for name in channel_names if name.upper() not in DREAMER_POSITIONS]
    if missing:
        raise ValueError(
            "Scalp positions are unavailable for channels "
            f"{missing}. This plot currently supports the DREAMER 14-channel montage."
        )
    return np.asarray([DREAMER_POSITIONS[name.upper()] for name in channel_names])


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
    title: str,
    colorbar_label: str,
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

    positions = _channel_positions(channel_names)
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

    fig.suptitle(title, fontsize=14)
    if shared_scale:
        colorbar = fig.colorbar(contour, ax=axes.tolist(), shrink=0.78, pad=0.02)
        colorbar.set_label(colorbar_label)
    return fig, peak_channels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one whole-trial counterfactual scalp map per EEG band."
    )
    parser.add_argument("npz_path", type=Path)
    parser.add_argument(
        "--branch",
        help=(
            "Reconstruction path, for example gcn_gru, bilstm, or joint; "
            "required when multiple paths exist."
        ),
    )
    parser.add_argument(
        "--reference",
        choices=("reconstruction", "input"),
        default="reconstruction",
    )
    parser.add_argument(
        "--quantity",
        choices=("difference", "counterfactual", "reference"),
        default="difference",
        help="Signal to average; default maps counterfactual change.",
    )
    parser.add_argument(
        "--measure",
        choices=("mean-absolute", "rms", "mean"),
        default="mean",
        help=(
            "Whole-trial reduction. The default mean retains the sign of the "
            "decoded counterfactual difference."
        ),
    )
    parser.add_argument(
        "--band-names",
        nargs="+",
        default=list(DEFAULT_BAND_NAMES),
        help="Band labels in saved feature order (default: Theta Alpha Beta).",
    )
    parser.add_argument(
        "--feature-order",
        choices=("channel-major", "band-major"),
        default="channel-major",
        help="How channel-band pairs are flattened on the feature axis.",
    )
    parser.add_argument(
        "--shared-scale",
        action="store_true",
        help=(
            "Use one color scale across all bands. By default each band is "
            "scaled independently to preserve within-band spatial contrast."
        ),
    )
    parser.add_argument(
        "--channel-names",
        nargs="+",
        help="Electrode names. The 14-channel DREAMER order is automatic.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference, counterfactual, branch, saved_names = load_counterfactual_trial(
        args.npz_path, branch=args.branch, reference=args.reference
    )
    band_names = [str(name) for name in args.band_names]
    n_features = reference.shape[-1]
    if n_features % len(band_names):
        raise ValueError(
            f"The trial has {n_features} features, which cannot be split into "
            f"{len(band_names)} bands. Pass the correct --band-names."
        )
    n_channels = n_features // len(band_names)
    saved_channel_names = (
        saved_names if saved_names is not None and len(saved_names) == n_channels else None
    )
    channel_names = resolve_channel_names(
        n_channels, provided=args.channel_names, saved=saved_channel_names
    )

    if args.quantity == "difference":
        signal = flatten_trial(counterfactual - reference)
    elif args.quantity == "counterfactual":
        signal = flatten_trial(counterfactual)
    else:
        signal = flatten_trial(reference)
    band_signal = split_channel_bands(
        signal,
        n_channels=n_channels,
        n_bands=len(band_names),
        feature_order=args.feature_order,
    )
    values = np.stack(
        [
            summarize_activity(band_signal[:, band_index, :], measure=args.measure)
            for band_index in range(len(band_names))
        ],
        axis=0,
    )
    fig, peak_channels = plot_band_topographies(
        values,
        channel_names=channel_names,
        band_names=band_names,
        title=(
            f"{branch}: whole-trial {args.quantity} "
            f"(reference={args.reference}, {args.measure}, band-relative scale)"
        ),
        colorbar_label=(
            "signed mean counterfactual − reference"
            if args.quantity == "difference" and args.measure == "mean"
            else f"{args.measure} {args.quantity}"
        ),
        shared_scale=args.shared_scale,
        signed=args.measure == "mean",
    )
    output = args.output or args.npz_path.with_name(
        f"{args.npz_path.stem}_{branch}_{args.quantity}_topography.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    for band_name, peak_channel in peak_channels.items():
        print(f"Peak channel ({band_name}): {peak_channel}")
    print(f"Saved plot to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
