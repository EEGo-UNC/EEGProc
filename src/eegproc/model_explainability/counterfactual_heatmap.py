"""Plot one channel-by-time counterfactual change heatmap per EEG band.

SIC's default 42-feature axis represents 14 electrodes × 3 already filtered
bands. This plot splits that axis and creates separate Theta, Alpha, and Beta
sections. Every section has the same 14 electrode rows, time axis, and color
semantics, so features are never displayed as 42 independent EEG channels.

Raw signed differences are not useful for band-filtered EEG: their sign flips
with the waveform phase and produces dense vertical striping at the sampling
rate. The default plot therefore shows RMS decoded change in one-second bins.
This exposes when and where the counterfactual changes band activity while
retaining the original electrode-by-time organization.

The default reference is ``x_reconstructed_<branch>``, which isolates the
latent intervention from ordinary decoder reconstruction error. Pass
``--reference input`` only when the intended quantity is ``x_prime - x``.

Example::

    PYTHONPATH=src python -m eegproc.model_explainability.counterfactual_heatmap \
        runs/.../subject_0_trial_0/counterfactual.npz \
        --branch gcn_gru --sampling-rate 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def counterfactual_difference(
    reference_trial: np.ndarray, counterfactual_trial: np.ndarray
) -> np.ndarray:
    """Return signed counterfactual change shaped ``(samples, features)``."""
    if reference_trial.shape != counterfactual_trial.shape:
        raise ValueError("Reference and counterfactual trials must have equal shapes.")
    return flatten_trial(counterfactual_trial - reference_trial)


def aggregate_temporal_change(
    difference: np.ndarray,
    *,
    sampling_rate_hz: float,
    time_bin_seconds: float,
    measure: str,
) -> np.ndarray:
    """Reduce sample-level differences to informative temporal change bins.

    Parameters
    ----------
    difference
        Array shaped ``(samples, bands, channels)``.
    sampling_rate_hz
        Sampling frequency of the saved model input.
    time_bin_seconds
        Width of each non-overlapping temporal bin.
    measure
        ``rms`` measures change energy, ``mean-absolute`` measures average
        magnitude, and ``signed-mean`` retains the direction of the change.
    """
    difference = np.asarray(difference, dtype=float)
    if difference.ndim != 3 or not difference.size:
        raise ValueError(
            "difference must be a nonempty (samples, bands, channels) array."
        )
    if not np.isfinite(difference).all():
        raise ValueError("difference must contain only finite values.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if time_bin_seconds <= 0:
        raise ValueError("time_bin_seconds must be positive.")
    if measure not in {"rms", "mean-absolute", "signed-mean"}:
        raise ValueError(
            "measure must be 'rms', 'mean-absolute', or 'signed-mean'."
        )

    bin_samples = max(1, int(round(sampling_rate_hz * time_bin_seconds)))
    bins = []
    for start in range(0, difference.shape[0], bin_samples):
        chunk = difference[start : start + bin_samples]
        if measure == "rms":
            summarized = np.sqrt(np.mean(np.square(chunk), axis=0))
        elif measure == "mean-absolute":
            summarized = np.mean(np.abs(chunk), axis=0)
        else:
            summarized = np.mean(chunk, axis=0)
        bins.append(summarized)
    return np.stack(bins, axis=0)


def plot_band_difference_heatmaps(
    difference: np.ndarray,
    *,
    channel_names: list[str],
    band_names: list[str],
    sampling_rate_hz: float = 128.0,
    time_bin_seconds: float = 1.0,
    measure: str = "rms",
    shared_scale: bool = False,
    title: str | None = None,
):
    """Plot time-binned ``(samples, bands, channels)`` change sections."""
    difference = np.asarray(difference, dtype=float)
    expected_tail = (len(band_names), len(channel_names))
    if difference.ndim != 3 or difference.shape[1:] != expected_tail:
        raise ValueError(
            "difference must be shaped (samples, bands, channels); "
            f"expected (*, {expected_tail[0]}, {expected_tail[1]}), "
            f"received {difference.shape}."
        )
    if not difference.size or not np.isfinite(difference).all():
        raise ValueError("difference must be finite and nonempty.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    n_samples, n_bands, n_channels = difference.shape
    binned = aggregate_temporal_change(
        difference,
        sampling_rate_hz=sampling_rate_hz,
        time_bin_seconds=time_bin_seconds,
        measure=measure,
    )
    x_end = n_samples / float(sampling_rate_hz)
    x_label = "Time across trial (s)"

    shared_limit = None
    if shared_scale:
        shared_limit = float(np.quantile(np.abs(binned), 0.99))
        if not np.isfinite(shared_limit) or shared_limit == 0:
            shared_limit = 1.0

    fig = plt.figure(
        figsize=(16, max(12.0, 4.4 * n_bands)), constrained_layout=True
    )
    grid = fig.add_gridspec(
        n_bands,
        3,
        width_ratios=(8.0, 2.0, 0.28),
        wspace=0.08,
        hspace=0.12,
    )
    heat_axes = []
    summary_axes = []

    for band_index, band_name in enumerate(band_names):
        ax_heat = fig.add_subplot(
            grid[band_index, 0],
            sharex=heat_axes[0] if heat_axes else None,
        )
        ax_summary = fig.add_subplot(grid[band_index, 1], sharey=ax_heat)
        heat_axes.append(ax_heat)
        summary_axes.append(ax_summary)

        band_difference = difference[:, band_index, :]
        plotted = binned[:, band_index, :].T
        scale = shared_limit
        if scale is None:
            scale = float(np.quantile(np.abs(plotted), 0.99))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
        color_options = (
            {"cmap": "RdBu_r", "vmin": -scale, "vmax": scale}
            if measure == "signed-mean"
            else {"cmap": "magma", "vmin": 0.0, "vmax": scale}
        )
        image = ax_heat.imshow(
            plotted,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(0.0, x_end, n_channels - 0.5, -0.5),
            **color_options,
        )
        ax_heat.set_yticks(np.arange(n_channels))
        ax_heat.set_yticklabels(channel_names)
        ax_heat.set_ylabel("EEG channel")
        ax_heat.set_title(f"{band_name} band")
        if band_index < n_bands - 1:
            ax_heat.tick_params(axis="x", labelbottom=False)
        else:
            ax_heat.set_xlabel(x_label)

        if measure == "rms":
            mean_by_channel = np.sqrt(np.mean(np.square(band_difference), axis=0))
            summary_label = "Trial RMS\nchange"
        else:
            mean_by_channel = np.mean(np.abs(band_difference), axis=0)
            summary_label = "Trial mean\n|change|"
        peak_index = int(np.argmax(mean_by_channel))
        bar_colors = [
            "tab:red" if index == peak_index else "tab:blue"
            for index in range(n_channels)
        ]
        ax_summary.barh(
            np.arange(n_channels), mean_by_channel, color=bar_colors, alpha=0.85
        )
        ax_summary.set_xlabel(summary_label)
        ax_summary.set_title(f"Peak: {channel_names[peak_index]}")
        ax_summary.tick_params(axis="y", labelleft=False)
        ax_summary.grid(axis="x", alpha=0.25)

        colorbar_axis = fig.add_subplot(grid[band_index, 2])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(
            f"{measure.replace('-', ' ')} change\n"
            f"({time_bin_seconds:g} s bins)"
        )

    fig.suptitle(
        title
        or f"Time-binned counterfactual change by EEG band ({measure})",
        fontsize=14,
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot separate channel-by-time counterfactual heatmaps per band."
    )
    parser.add_argument("npz_path", type=Path)
    parser.add_argument(
        "--branch",
        help="Decoder branch, for example gcn_gru or bilstm; required if both exist.",
    )
    parser.add_argument(
        "--reference",
        choices=("reconstruction", "input"),
        default="reconstruction",
        help=(
            "Baseline for the decoded change. The default reconstruction "
            "isolates the counterfactual intervention from decoder error."
        ),
    )
    parser.add_argument(
        "--channel-names",
        nargs="+",
        help="Electrode names. The 14-channel DREAMER order is automatic.",
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
        "--sampling-rate",
        type=float,
        default=128.0,
        help="Sampling rate in Hz (default: 128).",
    )
    parser.add_argument(
        "--time-bin-seconds",
        type=float,
        default=1.0,
        help="Width of non-overlapping temporal summaries (default: 1.0 s).",
    )
    parser.add_argument(
        "--measure",
        choices=("rms", "mean-absolute", "signed-mean"),
        default="rms",
        help="Change statistic inside each time bin (default: rms).",
    )
    parser.add_argument(
        "--absolute",
        action="store_const",
        const="mean-absolute",
        dest="measure",
        help="Compatibility alias for --measure mean-absolute.",
    )
    parser.add_argument(
        "--shared-scale",
        action="store_true",
        help=(
            "Use one color scale across bands. By default each band has its "
            "own labeled scale so lower-amplitude structure remains visible."
        ),
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
    difference = counterfactual_difference(reference, counterfactual)
    band_difference = split_channel_bands(
        difference,
        n_channels=n_channels,
        n_bands=len(band_names),
        feature_order=args.feature_order,
    )
    fig = plot_band_difference_heatmaps(
        band_difference,
        channel_names=channel_names,
        band_names=band_names,
        sampling_rate_hz=args.sampling_rate,
        time_bin_seconds=args.time_bin_seconds,
        measure=args.measure,
        shared_scale=args.shared_scale,
        title=(
            f"{branch}: decoded counterfactual change "
            f"(reference={args.reference}, {args.measure}, "
            f"{args.time_bin_seconds:g} s bins)"
        ),
    )
    output = args.output or args.npz_path.with_name(
        f"{args.npz_path.stem}_{branch}_counterfactual_difference_heatmap.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved plot to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
