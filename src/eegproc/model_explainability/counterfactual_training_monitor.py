"""Plot counterfactual optimization history as a 3-D trajectory.

Axes are optimization epoch, target-class probability, and decoded
counterfactual difference. Existing histories provide the last quantity in the
``decoded`` column: decoded counterfactual MSE relative to the original input.

Example::

    PYTHONPATH=src python -m eegproc.model_explainability.counterfactual_training_monitor \
        runs/.../subject_0_trial_0/history.csv --no-show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_training_history(
    history_csv: Path, *, difference_column: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load finite epoch, target probability, and difference columns."""
    history_csv = Path(history_csv)
    with history_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        epoch_column = "epoch" if "epoch" in fieldnames else "step"
        if epoch_column not in fieldnames:
            raise KeyError("history CSV requires an 'epoch' or 'step' column.")
        if "target_probability" not in fieldnames:
            raise KeyError("history CSV requires 'target_probability'.")
        if difference_column is None:
            if "decoded" in fieldnames:
                difference_column = "decoded"
            elif "counterfactual_difference" in fieldnames:
                difference_column = "counterfactual_difference"
            else:
                raise KeyError(
                    "history CSV requires 'decoded' or "
                    "'counterfactual_difference'."
                )
        if difference_column not in fieldnames:
            raise KeyError(
                f"Difference column {difference_column!r} is absent. "
                f"Available columns: {fieldnames}"
            )

        rows = []
        for row in reader:
            try:
                values = tuple(
                    float(row[column])
                    for column in (
                        epoch_column,
                        "target_probability",
                        difference_column,
                    )
                )
            except (KeyError, TypeError, ValueError):
                # A live reader can encounter the final row while it is being written.
                continue
            if all(np.isfinite(value) for value in values):
                rows.append(values)
    if not rows:
        raise ValueError(f"No complete finite history rows found in {history_csv}.")
    matrix = np.asarray(rows, dtype=float)
    return matrix[:, 0], matrix[:, 1], matrix[:, 2], difference_column


def plot_training_trajectory(
    epochs: np.ndarray,
    target_probability: np.ndarray,
    difference: np.ndarray,
    *,
    difference_label: str,
    full_probability_range: bool = False,
    ax=None,
):
    """Draw epoch × target probability × counterfactual difference."""
    arrays = [
        np.asarray(values, dtype=float)
        for values in (epochs, target_probability, difference)
    ]
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("Training trajectory inputs must be one-dimensional.")
    if len({len(values) for values in arrays}) != 1 or not len(arrays[0]):
        raise ValueError("Training trajectory inputs must have equal nonzero lengths.")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Training trajectory inputs must be finite.")
    epochs, target_probability, difference = arrays

    if ax is None:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        ax.clear()

    ax.plot(
        epochs,
        difference,
        target_probability,
        color="tab:blue",
        linewidth=2,
        alpha=0.85,
    )
    # TODO: Color trajectory points by counterfactual validity to add the
    # requested validity heatmap once that metric is available.
    points = ax.scatter(
        epochs,
        difference,
        target_probability,
        color="tab:blue",
        s=28,
        depthshade=True,
    )
    ax.scatter(
        [epochs[-1]],
        [difference[-1]],
        [target_probability[-1]],
        color="tab:red",
        s=65,
        label="Latest epoch",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Counterfactual difference ({difference_label})")
    ax.set_zlabel("target_p")
    if full_probability_range:
        ax.set_zlim(0.0, 1.0)
    else:
        probability_min = float(np.min(target_probability))
        probability_max = float(np.max(target_probability))
        probability_span = probability_max - probability_min
        probability_padding = (
            0.10 * probability_span if probability_span > 0 else 0.01
        )
        lower_limit = max(0.0, probability_min - probability_padding)
        upper_limit = min(1.0, probability_max + probability_padding)
        if lower_limit == upper_limit:
            lower_limit = max(0.0, lower_limit - 0.01)
            upper_limit = min(1.0, upper_limit + 0.01)
        ax.set_zlim(lower_limit, upper_limit)
    ax.set_title(
        "Counterfactual optimization trajectory\n"
        f"z source: {difference_label}"
    )
    ax.legend(loc="upper left")
    return fig, ax, points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot epoch, target probability, and counterfactual difference."
    )
    parser.add_argument("history_csv", type=Path)
    parser.add_argument(
        "--difference-column",
        help="Default: decoded (MSE from decoded counterfactual to original input).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Re-read the CSV whenever another process updates it.",
    )
    parser.add_argument("--refresh-seconds", type=float, default=1.0)
    parser.add_argument(
        "--full-probability-range",
        action="store_true",
        help="Use a fixed target_p range of 0–1 instead of zooming to its variation.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser


def _render(args, ax=None):
    epochs, target_p, difference, difference_label = read_training_history(
        args.history_csv, difference_column=args.difference_column
    )
    fig, ax, _ = plot_training_trajectory(
        epochs,
        target_p,
        difference,
        difference_label=difference_label,
        full_probability_range=args.full_probability_range,
        ax=ax,
    )
    output = args.output or args.history_csv.with_name(
        f"{args.history_csv.stem}_counterfactual_training_trajectory.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig, ax, output, len(epochs)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.refresh_seconds <= 0:
        parser.error("--refresh-seconds must be positive")
    if args.watch and args.no_show:
        parser.error("--watch requires an interactive window; remove --no-show")

    if not args.watch:
        fig, _, output, _ = _render(args)
        if not args.no_show:
            plt.show()
        else:
            plt.close(fig)
        print(f"Saved plot to: {output}")
        return 0

    plt.ion()
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Waiting for history: {args.history_csv}")
    plt.show(block=False)
    last_signature = None
    output = args.output or args.history_csv.with_name(
        f"{args.history_csv.stem}_counterfactual_training_trajectory.png"
    )
    try:
        while plt.fignum_exists(fig.number):
            if args.history_csv.exists():
                signature = (
                    args.history_csv.stat().st_mtime_ns,
                    args.history_csv.stat().st_size,
                )
                if signature != last_signature:
                    try:
                        fig, ax, output, n_rows = _render(args, ax=ax)
                    except (KeyError, OSError, ValueError):
                        pass
                    else:
                        last_signature = signature
                        fig.canvas.draw_idle()
                        print(f"Updated from {n_rows} epochs: {output}")
            plt.pause(args.refresh_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
