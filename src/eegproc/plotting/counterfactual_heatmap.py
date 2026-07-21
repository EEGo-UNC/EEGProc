from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt


def decode_channel_names(channel_names: np.ndarray) -> list[str]:
    names: list[str] = []
    for item in channel_names.tolist():
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


def bandpass_filter(
    signal_1d: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    sos = butter(
        N=order,
        Wn=[low_hz, high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, signal_1d)


def build_theta_matrix(
    counterfactual_features: np.ndarray,
    channel_index: int,
    fs: float,
    low_hz: float = 4.0,
    high_hz: float = 8.0,
) -> np.ndarray:
    """
    counterfactual_features shape:
        (n_logged_steps, batch, time, channels)

    Returns:
        theta_matrix with shape (n_logged_steps, time)
    """
    n_logged_steps = counterfactual_features.shape[0]
    theta_rows = []

    for i in range(n_logged_steps):
        # single-window run -> batch dimension should be 1
        signal_1d = counterfactual_features[i, 0, :, channel_index]
        theta_signal = bandpass_filter(
            signal_1d=signal_1d,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        theta_rows.append(theta_signal)

    return np.stack(theta_rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    parser.add_argument(
        "--channel",
        type=str,
        default="F7",
        help="EEG channel to plot (default: F7).",
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        help=(
            "Plot theta difference relative to step 0 "
            "(the original reconstruction) instead of raw theta amplitude."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path.",
    )
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=False)

    required_keys = [
        "result__history__counterfactual_features",
        "result__history__feature_step",
        "result__history__target_probability",
        "result__history__step",
        "time_seconds",
        "channel_names",
        "sampling_rate_hz",
    ]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(
            "The NPZ file is missing required keys for heatmap plotting:\n"
            + "\n".join(missing)
            + "\n\nRe-run the counterfactual generation with feature logging enabled."
        )

    counterfactual_features = data["result__history__counterfactual_features"]
    feature_steps = data["result__history__feature_step"]
    all_steps = data["result__history__step"]
    target_probability = data["result__history__target_probability"]
    time_seconds = data["time_seconds"]
    sampling_rate_hz = float(data["sampling_rate_hz"])
    channel_names = decode_channel_names(data["channel_names"])

    if args.channel not in channel_names:
        raise ValueError(
            f"Channel {args.channel!r} not found. Available channels:\n{channel_names}"
        )

    channel_index = channel_names.index(args.channel)

    theta_matrix = build_theta_matrix(
        counterfactual_features=counterfactual_features,
        channel_index=channel_index,
        fs=sampling_rate_hz,
        low_hz=4.0,
        high_hz=8.0,
    )

    # Optional difference-from-step-0 view
    if args.delta:
        theta_matrix_to_plot = theta_matrix - theta_matrix[0:1, :]
        colorbar_label = f"{args.channel} theta (4–8 Hz) Δ from step 0"
        title = f"{args.channel} theta heatmap across optimization steps (delta)"
    else:
        theta_matrix_to_plot = theta_matrix
        colorbar_label = f"{args.channel} theta (4–8 Hz) amplitude"
        title = f"{args.channel} theta heatmap across optimization steps"

    # Build figure
    fig, (ax_heat, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Heatmap
    im = ax_heat.imshow(
        theta_matrix_to_plot,
        aspect="auto",
        origin="lower",
        extent=[
            float(time_seconds[0]),
            float(time_seconds[-1]),
            float(feature_steps[0]),
            float(feature_steps[-1]),
        ],
    )
    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label(colorbar_label)

    ax_heat.set_title(title)
    ax_heat.set_xlabel("Time (s)")
    ax_heat.set_ylabel("Optimization step")

    # Probability trajectory
    ax_prob.plot(all_steps, target_probability, linewidth=2)
    ax_prob.axhline(0.5, linestyle="--", linewidth=1, label="Class flip threshold")
    ax_prob.axhline(0.8, linestyle=":", linewidth=1, label="Success threshold")
    ax_prob.set_title("Target-class probability during optimization")
    ax_prob.set_xlabel("Optimization step")
    ax_prob.set_ylabel("P(target class)")
    ax_prob.set_ylim(0.0, 1.0)
    ax_prob.legend()

    if args.output is None:
        suffix = "_delta" if args.delta else ""
        output_path = args.npz_path.with_name(
            args.npz_path.stem + f"_{args.channel}_theta_heatmap{suffix}.png"
        )
    else:
        output_path = args.output

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()