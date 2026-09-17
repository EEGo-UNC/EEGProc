"""Metadata-driven counterfactual topographies with optional unit restoration."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..counterfactuals.plotting import (
    DEFAULT_BAND_NAMES,
    flatten_trial,
    load_counterfactual_trial,
    resolve_channel_names,
    split_channel_bands,
)
from ..counterfactuals.topography import (
    plot_band_topographies,
    summarize_activity,
)


def restore_source_units(signal, offset, scale):
    """Invert an affine per-window normalization with broadcasting."""

    signal = np.asarray(signal, dtype=float)
    scale = np.asarray(scale, dtype=float)
    offset = np.zeros_like(scale) if offset is None else np.asarray(offset, dtype=float)
    try:
        restored = signal * scale + offset
    except ValueError as error:
        raise ValueError(
            f"Normalization transform shapes {offset.shape}/{scale.shape} cannot "
            f"broadcast to trial shape {signal.shape}."
        ) from error
    if not np.isfinite(restored).all():
        raise ValueError("Restored signal contains non-finite values.")
    return restored


def _strings(data, key):
    if key not in data.files:
        return None
    return [str(value) for value in np.asarray(data[key]).reshape(-1)]


def _scalar(data, key):
    values = _strings(data, key)
    return None if not values else values[0]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot adapter-generated counterfactuals using saved signal metadata."
    )
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--branch")
    parser.add_argument(
        "--reference", choices=("reconstruction", "input"), default="reconstruction"
    )
    parser.add_argument(
        "--quantity",
        choices=("difference", "counterfactual", "reference"),
        default="difference",
    )
    parser.add_argument(
        "--measure", choices=("mean", "mean-absolute", "rms"), default="mean"
    )
    parser.add_argument(
        "--physical-units",
        action="store_true",
        help="Invert the normalization transform stored in the NPZ before reducing.",
    )
    parser.add_argument(
        "--signal-unit",
        help="Override the saved source unit label, for example 'µV'.",
    )
    parser.add_argument("--band-names", nargs="+")
    parser.add_argument(
        "--feature-order", choices=("channel-major", "band-major")
    )
    parser.add_argument("--channel-names", nargs="+")
    parser.add_argument("--shared-scale", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    reference, counterfactual, branch, loader_names = load_counterfactual_trial(
        args.npz_path,
        branch=args.branch,
        reference=args.reference,
    )
    with np.load(args.npz_path, allow_pickle=False) as data:
        saved_channel_names = _strings(data, "channel_names")
        saved_band_names = _strings(data, "band_names")
        saved_feature_order = _scalar(data, "feature_order")
        saved_signal_unit = _scalar(data, "signal_unit")
        positions = (
            np.asarray(data["channel_positions"], dtype=float)
            if "channel_positions" in data.files
            else None
        )
        if args.physical_units:
            if "normalization_scale" not in data.files:
                raise ValueError(
                    "Physical-unit plotting requires normalization_scale in the NPZ. "
                    "Rerun through the model-agnostic runner with a metadata-aware loader."
                )
            scale = np.asarray(data["normalization_scale"], dtype=float)
            offset = (
                np.asarray(data["normalization_offset"], dtype=float)
                if "normalization_offset" in data.files
                else None
            )
            reference = restore_source_units(reference, offset, scale)
            counterfactual = restore_source_units(counterfactual, offset, scale)

    band_names = args.band_names or saved_band_names or list(DEFAULT_BAND_NAMES)
    feature_order = args.feature_order or saved_feature_order or "channel-major"
    n_features = reference.shape[-1]
    if n_features % len(band_names):
        raise ValueError(
            f"{n_features} features cannot be split into {len(band_names)} bands."
        )
    n_channels = n_features // len(band_names)
    inherited_names = saved_channel_names or loader_names
    channel_names = resolve_channel_names(
        n_channels,
        provided=args.channel_names,
        saved=(
            inherited_names
            if inherited_names is not None and len(inherited_names) == n_channels
            else None
        ),
    )
    if positions is not None and positions.shape != (n_channels, 2):
        raise ValueError(
            f"Saved channel_positions has shape {positions.shape}, expected {(n_channels, 2)}."
        )

    if args.quantity == "difference":
        signal = counterfactual - reference
    elif args.quantity == "counterfactual":
        signal = counterfactual
    else:
        signal = reference
    split = split_channel_bands(
        flatten_trial(signal),
        n_channels=n_channels,
        n_bands=len(band_names),
        feature_order=feature_order,
    )
    values = np.stack(
        [
            summarize_activity(split[:, band_index, :], measure=args.measure)
            for band_index in range(len(band_names))
        ]
    )
    unit = args.signal_unit or saved_signal_unit
    if not args.physical_units:
        unit = "normalized input units"
    elif unit is None:
        unit = "source signal units"
    statistic = {
        "mean": "signed mean",
        "mean-absolute": "mean absolute",
        "rms": "RMS",
    }[args.measure]
    label = f"{statistic} {args.quantity} ({unit})"
    fig, peaks = plot_band_topographies(
        values,
        channel_names=channel_names,
        band_names=[str(value) for value in band_names],
        title=(
            f"{branch}: whole-trial {args.quantity} "
            f"(reference={args.reference}, {statistic})"
        ),
        colorbar_label=label,
        shared_scale=args.shared_scale,
        signed=args.measure == "mean",
        channel_positions=positions,
    )
    scale_name = "physical" if args.physical_units else "normalized"
    output = args.output or args.npz_path.with_name(
        f"{args.npz_path.stem}_{branch}_{args.quantity}_{args.measure}_{scale_name}_topography.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    for band_name, channel_name in peaks.items():
        print(f"Peak channel ({band_name}): {channel_name}")
    print(f"Saved plot to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
