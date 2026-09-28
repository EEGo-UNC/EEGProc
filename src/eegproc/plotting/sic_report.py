"""Comparison figures for a set of SIC sweep runs.

Built for the baseline-plus-two-arms sensitivity study: one run establishes a
reference, two more push a single hyperparameter in opposite directions, and
these panels show whether that move helped, hurt, or did nothing.

Usage::

    python -m eegproc.plotting.sic_report \\
        runs/sweep/valence/baseline \\
        runs/sweep/valence/vc_beta_lo \\
        runs/sweep/valence/vc_beta_hi \\
        --out-dir reports/sic_sweep_valence

Each path may point at the sweep output root, the timestamped run directory, or
a ``configuration_XXXX`` directory; :mod:`eegproc.plotting.sic_results_io`
resolves it. Run labels come from ``run_commit.txt`` when the sweep script wrote
one, otherwise from the directory name; ``--labels`` overrides both.

Figures produced:

1. ``brier_vs_shots``      -- headline: Brier against calibration shots
2. ``source_diagnostics``  -- source-split class balance and confidence per epoch
3. ``per_subject_brier``   -- per-subject calibrated Brier at the selection level
4. ``calibration_gain``    -- calibrated minus zero-shot, per shot level
5. ``ece_vs_shots``        -- probability honesty against calibration shots

Brier score and ECE are minimized, so on every panel here lower is better.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .result_figures import _resolve_ax
from .sic_results_io import (
    SICRun,
    load_sic_runs,
    shot_level_frame,
    subject_frame,
    summary_table,
)


# Colour-blind-safe and consistent across every panel, so a run keeps its
# identity from figure to figure.
_RUN_COLOURS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860")

_SUMMARY_METRICS = (
    "brier_score",
    "ece",
    "balanced_accuracy",
    "accuracy",
    "roc_auc",
)


def _colour_for(index: int) -> str:
    return _RUN_COLOURS[index % len(_RUN_COLOURS)]


def _run_order(runs: list[SICRun]) -> list[str]:
    return [run.label for run in runs]


def plot_metric_vs_shots(
    runs: list[SICRun],
    metric: str = "brier_score",
    ax=None,
):
    """Zero-shot and calibrated ``metric`` against calibration shots.

    The ``shots=0`` point is the untouched population model on all target
    trials. Calibrated points are solid, the paired zero-shot baseline at each
    level is dashed -- the vertical gap between them is what calibration bought.
    """
    levels = shot_level_frame(runs)
    if levels.empty:
        raise ValueError("No shot-level metrics found in the supplied runs.")

    subset = levels[levels["metric"] == metric]
    if subset.empty:
        available = sorted(levels["metric"].unique())
        raise ValueError(f"Metric {metric!r} not found. Available: {available}")

    fig, ax = _resolve_ax(ax)

    for index, label in enumerate(_run_order(runs)):
        colour = _colour_for(index)
        run_rows = subset[subset["run"] == label]

        calibrated = run_rows[run_rows["phase"] == "calibrated"].sort_values("shots")
        anchor = run_rows[(run_rows["shots"] == 0) & (run_rows["phase"] == "zero_shot")]
        if not anchor.empty:
            calibrated = pd.concat([anchor, calibrated], ignore_index=True)

        ax.plot(
            calibrated["shots"],
            calibrated["value"],
            marker="o",
            color=colour,
            linewidth=1.8,
            label=f"{label} (calibrated)",
        )

        paired = run_rows[
            (run_rows["phase"] == "zero_shot") & (run_rows["shots"] > 0)
        ].sort_values("shots")
        if not paired.empty:
            ax.plot(
                paired["shots"],
                paired["value"],
                marker="s",
                markersize=4,
                linestyle="--",
                color=colour,
                linewidth=1.0,
                alpha=0.55,
                label=f"{label} (zero-shot)",
            )

    ax.set_xlabel("Calibration shots (whole target trials)")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    direction = "lower is better" if metric in {"brier_score", "ece"} else "higher is better"
    ax.set_title(f"{metric.replace('_', ' ').capitalize()} vs. calibration shots ({direction})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    return fig


def plot_source_diagnostics(
    runs: list[SICRun],
    ax=None,
):
    """Source-split prediction behaviour per epoch, for every run.

    Reads ``sic_prediction_diagnostics.csv``, which is built from
    ``X_train=X_source`` only -- the held-out target subject never enters it. That
    is what makes this panel usable as evidence for choosing a hyperparameter,
    where per-epoch held-out-subject metrics would not be: selecting against the
    target would turn the study into an oracle estimate rather than a LOSOCV one.

    Top axis is the predicted class-1 rate against the subset's true rate. The
    diagnostic subset is stratified to be class-balanced, so the reference line
    sits near 0.5 and any sustained departure is the classifier collapsing toward
    one class. Bottom axis is mean confidence with a +/-1 SD band; high and flat
    means confidently uniform, which is a different failure from collapse.
    """
    frames = []
    for run in runs:
        if run.diagnostics.empty:
            continue
        frame = run.diagnostics.copy()
        frame["run"] = run.label
        frames.append(frame)
    if not frames:
        raise ValueError(
            "No sic_prediction_diagnostics.csv found. Re-run with "
            "--prediction-diagnostics to produce this panel."
        )

    diagnostics = pd.concat(frames, ignore_index=True)
    train_rows = diagnostics[diagnostics["split"] == "train"]
    if train_rows.empty:
        train_rows = diagnostics

    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    else:
        fig, axes = ax.figure, ax

    collapse_ax, confidence_ax = axes

    reference = float(train_rows["true_class_1_fraction"].mean())
    collapse_ax.axhline(
        reference,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"true class-1 rate = {reference:.2f}",
    )

    for index, label in enumerate(_run_order(runs)):
        run_rows = train_rows[train_rows["run"] == label]
        if run_rows.empty:
            continue
        colour = _colour_for(index)

        predicted = run_rows.groupby("epoch")["predicted_class_1_fraction"].mean()
        collapse_ax.plot(
            predicted.index, predicted.to_numpy(), color=colour, linewidth=1.6, label=label
        )

        confidence = run_rows.groupby("epoch")["confidence_mean"].mean()
        spread = run_rows.groupby("epoch")["confidence_std"].mean()
        confidence_ax.plot(
            confidence.index, confidence.to_numpy(), color=colour, linewidth=1.6, label=label
        )
        confidence_ax.fill_between(
            confidence.index,
            (confidence - spread).to_numpy(),
            (confidence + spread).to_numpy(),
            color=colour,
            alpha=0.15,
            linewidth=0,
        )

    collapse_ax.set_ylim(0, 1)
    collapse_ax.set_ylabel("Predicted class-1 rate")
    collapse_ax.set_title(
        "Source-split prediction behaviour\n"
        "(source subjects only; the target subject is never evaluated here)",
        fontsize=10,
    )
    collapse_ax.legend(fontsize=8, loc="best")
    collapse_ax.grid(True, linewidth=0.5, alpha=0.4)

    confidence_ax.set_xlabel("Source training epoch")
    confidence_ax.set_ylabel("Mean confidence (±1 SD)")
    confidence_ax.grid(True, linewidth=0.5, alpha=0.4)

    return fig


def plot_per_subject(
    runs: list[SICRun],
    shots: int,
    metric: str = "brier_score",
    ax=None,
):
    """Per-subject calibrated ``metric``, subjects ordered by the first run.

    Ordering every run by the baseline's ranking makes it visible whether an arm
    improved the cohort uniformly or only rescued the subjects that were already
    worst.
    """
    per_subject = subject_frame(runs, shots=shots, phase="calibrated")
    subset = per_subject[per_subject["metric"] == metric]
    if subset.empty:
        raise ValueError(f"No per-subject {metric!r} at {shots} shots.")

    reference = subset[subset["run"] == runs[0].label].sort_values("value")
    order = reference["target_subject"].tolist()

    fig, ax = _resolve_ax(ax)
    positions = np.arange(len(order))

    for index, label in enumerate(_run_order(runs)):
        run_rows = subset[subset["run"] == label].set_index("target_subject")
        values = [run_rows["value"].get(subject, np.nan) for subject in order]
        ax.plot(
            positions,
            values,
            marker="o",
            markersize=4,
            color=_colour_for(index),
            linewidth=1.4,
            label=f"{label} (mean {np.nanmean(values):.4f})",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(subject) for subject in order], rotation=45, ha="right")
    ax.set_xlabel(f"Target subject (ordered by {runs[0].label})")
    ax.set_ylabel(f"{metric.replace('_', ' ').capitalize()} at {shots} shots")
    ax.set_title(f"Per-subject {metric.replace('_', ' ')} after {shots}-shot calibration")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    return fig


def plot_calibration_gain(
    runs: list[SICRun],
    metric: str = "brier_score",
    ax=None,
):
    """Calibrated minus paired zero-shot, per shot level, as grouped bars.

    The pipeline defines delta as ``post_calibration_minus_paired_zero_shot``,
    so for Brier a *negative* bar is an improvement. Error bars are the
    across-subject standard deviation of that difference.
    """
    levels = shot_level_frame(runs)
    subset = levels[(levels["metric"] == metric) & (levels["phase"] == "delta")]
    if subset.empty:
        raise ValueError(f"No delta values for metric {metric!r}.")

    shot_values = sorted(subset["shots"].unique())
    labels = _run_order(runs)
    fig, ax = _resolve_ax(ax)

    width = 0.8 / max(len(labels), 1)
    base = np.arange(len(shot_values))

    for index, label in enumerate(labels):
        run_rows = subset[subset["run"] == label].set_index("shots")
        means = [run_rows["value"].get(shots, np.nan) for shots in shot_values]
        errors = [run_rows["std"].get(shots, np.nan) for shots in shot_values]
        ax.bar(
            base + index * width - 0.4 + width / 2,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=_colour_for(index),
            label=label,
            error_kw={"linewidth": 0.9, "alpha": 0.7},
        )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(base)
    ax.set_xticklabels([str(shots) for shots in shot_values])
    ax.set_xlabel("Calibration shots")
    ax.set_ylabel(f"Δ {metric.replace('_', ' ')} (calibrated − zero-shot)")
    improvement = "negative = improvement" if metric in {"brier_score", "ece"} else "positive = improvement"
    ax.set_title(f"What calibration bought ({improvement})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    return fig


def _save(fig, out_dir: Path, name: str, dpi: int = 150) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def build_report(
    runs: list[SICRun],
    out_dir: Path,
    selection_shots: int | None = None,
) -> list[Path]:
    """Write every panel plus the summary table, returning the paths written."""
    out_dir = Path(out_dir)
    if selection_shots is None:
        recorded = [run.selection_shots for run in runs if run.selection_shots]
        available = runs[0].shot_levels
        selection_shots = recorded[0] if recorded else (available[-1] if available else 0)

    written: list[Path] = []

    written.append(_save(plot_metric_vs_shots(runs, "brier_score"), out_dir, "brier_vs_shots"))
    written.append(_save(plot_metric_vs_shots(runs, "ece"), out_dir, "ece_vs_shots"))
    written.append(_save(plot_calibration_gain(runs, "brier_score"), out_dir, "calibration_gain"))
    written.append(
        _save(
            plot_per_subject(runs, shots=selection_shots, metric="brier_score"),
            out_dir,
            "per_subject_brier",
        )
    )

    # Diagnostics are optional at run time (--prediction-diagnostics), so a run
    # without them is still usable for everything else.
    try:
        written.append(
            _save(plot_source_diagnostics(runs), out_dir, "source_diagnostics")
        )
    except ValueError as error:
        print(f"Skipped source_diagnostics: {error}")

    table = summary_table(runs, _SUMMARY_METRICS)
    table_path = out_dir / "summary.csv"
    table.to_csv(table_path, index=False)
    written.append(table_path)

    print(f"\nSummary ({selection_shots}-shot selection level)")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    for run in runs:
        commit = run.commit.get("commit", "unknown")
        print(f"\n{run.label}: commit={commit} subjects={run.n_subjects} dir={run.config_dir}")

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Comparison figures for SIC sweep runs. Pass the baseline first; "
            "per-subject ordering follows it."
        )
    )
    parser.add_argument("run_dirs", nargs="+", help="One or more SIC run directories.")
    parser.add_argument("--out-dir", default="reports/sic_sweep", type=Path)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Explicit run labels; must align with run_dirs.",
    )
    parser.add_argument(
        "--selection-shots",
        type=int,
        default=None,
        help="Shot level for the per-subject panel. Defaults to the level the runs ranked on.",
    )
    args = parser.parse_args(argv)

    runs = load_sic_runs(args.run_dirs, args.labels)
    written = build_report(runs, args.out_dir, args.selection_shots)

    print(f"\nWrote {len(written)} files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
