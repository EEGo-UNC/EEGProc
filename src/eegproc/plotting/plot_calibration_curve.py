#!/usr/bin/env python3
"""Plot SIC calibration shots versus balanced accuracy for each user and the mean.

Standalone usage (no imports from EEGProc and no model loading):
    python plot_calibration_curve.py --results path/to/sic_calibration_results.json

Also works from src/eegproc/plotting/ with:
    PYTHONPATH=src python -m eegproc.plotting.plot_calibration_curve --results ...

Defaults: thin blue-gray user lines with alpha=0.18 (82% transparent), and a
red mean line with alpha=1.0 (fully opaque). Shots=0 uses strict zero-shot on
all trials; shots>0 uses saved post-calibration subject means. The average gives
each user equal weight, regardless of trial count or number of calibration folds.
No paired-zero-shot predictions or pooled trial metrics are used.

Missing user/shot values remain gaps. The mean uses available users at each
shot count and warns if the cohort varies. CSV outputs include subject counts.
Each user is labeled at its last finite point. Nearby endpoint labels are
separated vertically, with subtle connector lines back to the actual points.
Dependencies: numpy and matplotlib. Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


def extract_curves(raw: dict, include_zero_shot: bool = True):
    """Return (user IDs, sorted shots, user-by-shot matrix) from saved means.

    The flat subject summaries are preferred. Nested summaries fill missing
    entries only, so the same result never gets counted twice.
    """
    if not isinstance(raw, dict):
        raise ValueError("Expected a SIC results JSON object.")
    values: dict[tuple[str, int], float] = {}

    def add(user, shots, value):
        if user is None or value is None:
            return
        try:
            shot_number = float(shots)
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid calibration value for user {user}, shots={shots}: {value!r}") from exc
        if not math.isfinite(shot_number) or shot_number < 0 or not shot_number.is_integer():
            raise ValueError(f"Invalid calibration shot count: {shots!r}")
        shots = int(shot_number)
        if shots == 0 and not include_zero_shot:
            return
        if math.isnan(score):
            return
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Balanced accuracy must be in [0, 1]: user={user}, shots={shots}, value={value!r}")
        values.setdefault((str(user), shots), score)

    for row in raw.get("subject_summary_rows", []) or []:
        user = row.get("target_subject", row.get("subject_id"))
        add(user, 0, row.get("zero_shot_all_balanced_accuracy"))
        for key, value in row.items():
            match = re.fullmatch(r"([1-9]\d*)_shot_calibrated_balanced_accuracy", key)
            if match:
                add(user, match.group(1), value)

    for subject in raw.get("subject_results", []) or []:
        user = subject.get("target_subject", subject.get("subject_id"))
        summary = subject.get("subject_summary") or {}
        add(user, 0, (summary.get("zero_shot_all_trials_scores") or {}).get("balanced_accuracy"))
        zero = subject.get("zero_shot_all_trials") or {}
        zero_metrics = zero.get("trial_fold_metrics") or zero.get("fold_metrics") or {}
        if isinstance(zero_metrics, dict):
            add(user, 0, zero_metrics.get("balanced_accuracy"))
        for shots, level in (summary.get("calibration_levels") or {}).items():
            add(user, shots, (level.get("calibrated_mean_scores") or {}).get("balanced_accuracy"))
        for level in subject.get("calibration_levels", []) or []:
            shots = level.get("calibration_shots")
            scores = (level.get("summary") or {}).get("calibrated_mean_scores") or {}
            add(user, shots, scores.get("balanced_accuracy"))

    if not values:
        raise ValueError(
            "No subject-level balanced accuracies found. Use sic_calibration_results.json "
            "with subject_summary_rows or subject_results, not sic_overall_metrics.json."
        )
    users = sorted({user for user, _ in values},
                   key=lambda user: (0, int(user)) if re.fullmatch(r"-?\d+", user) else (1, user))
    shots = sorted({shot for _, shot in values})
    if not any(shot > 0 for shot in shots):
        raise ValueError("No post-calibration shot levels found; this file contains only zero-shot scores.")
    matrix = np.array([[values.get((user, shot), np.nan) for shot in shots] for user in users], dtype=float)
    return users, shots, matrix


def _label_curve_ends(ax, users, shots, scores):
    """Label last valid points, packing labels separately at each ending shot.

    Compute spacing after layout, in display units, so label separation stays
    consistent when the figure is saved at a different DPI. Connector lines
    identify the true endpoint whenever a label needs vertical displacement.
    """
    fig = ax.figure
    fig.canvas.draw()
    groups = {}
    for user, row in zip(users, scores):
        finite = np.flatnonzero(np.isfinite(row))
        if not len(finite):
            continue
        last = int(finite[-1])
        x, y = shots[last], float(row[last])
        groups.setdefault(x, []).append((user, y, ax.transData.transform((x, y))[1]))

    fontsize = 8.5
    point_pixels = fig.dpi / 72.0
    requested_gap = (fontsize + 5) * point_pixels
    low, high = ax.bbox.y0 + requested_gap / 2, ax.bbox.y1 - requested_gap / 2
    annotations = []
    for x, group in groups.items():
        group.sort(key=lambda item: item[2])
        gap = min(requested_gap, (high - low) / max(len(group) - 1, 1))
        positions = np.clip([item[2] for item in group], low, high)
        for i in range(1, len(positions)):
            positions[i] = max(positions[i], positions[i - 1] + gap)
        if positions[-1] > high:
            positions[-1] = high
            for i in range(len(positions) - 2, -1, -1):
                positions[i] = min(positions[i], positions[i + 1] - gap)
        for (user, y, original_y), label_y in zip(group, positions):
            offset = (label_y - original_y) / point_pixels
            connector = ({"arrowstyle": "-", "color": "#6F8091", "alpha": 0.35,
                          "linewidth": 0.65, "shrinkA": 2, "shrinkB": 2}
                         if abs(offset) > 1 else None)
            annotations.append(ax.annotate(
                f"User {user}", xy=(x, y), xytext=(8, offset), textcoords="offset points",
                ha="left", va="center", fontsize=fontsize, color="#6F8091", alpha=0.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
                arrowprops=connector, annotation_clip=False, zorder=6,
            ))
    return annotations


def plot_curves(users, shots, scores, *, subject_alpha=0.18, ylim=None,
                title="Subject calibration as shots increase"):
    """Draw one faint line per user and an opaque red, equally weighted mean."""
    if not 0 <= subject_alpha <= 1:
        raise ValueError("--subject-alpha must be in [0, 1].")
    if ylim is not None and not (0 <= ylim[0] < ylim[1] <= 1):
        raise ValueError("--ylim needs two increasing values between 0 and 1.")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(users), len(shots)) or not len(users) or not len(shots):
        raise ValueError("Score matrix must contain one row per user and one column per shot count.")
    counts = np.isfinite(scores).sum(axis=0)
    if (counts == 0).any():
        raise ValueError("Each displayed shot count needs at least one finite score.")
    means = np.nanmean(scores, axis=0)
    complete = bool(np.isfinite(scores).all())
    if not complete:
        warnings.warn(
            "Some users have missing shot levels. Their lines contain gaps; "
            "the mean uses available users at each shot count. Counts are in the mean CSV.",
            stacklevel=2,
        )

    last_indices = [int(np.flatnonzero(np.isfinite(row))[-1]) for row in scores if np.isfinite(row).any()]
    max_endpoint_labels = max(np.bincount(last_indices))
    fig, ax = plt.subplots(figsize=(9.5, max(5.8, 2.0 + 0.19 * max_endpoint_labels)))
    for user, row in zip(users, scores):
        # Matplotlib alpha=0 is invisible, alpha=1 is zero transparency.
        ax.plot(shots, row, color="#6F8091", alpha=subject_alpha,
                linewidth=1.2, marker="o", markersize=3, zorder=2,
                label=f"User {user}")
    mean_line, = ax.plot(shots, means, color="#D62728", alpha=1.0,
                         linewidth=2.8, marker="o", markersize=6, zorder=5)
    user_handle = Line2D([], [], color="#6F8091", alpha=0.35, linewidth=1.2)
    mean_label = "Average across all users" if complete else "Average across available users"
    ax.legend([user_handle, mean_line], [f"Individual users (n={len(users)})", mean_label],
              frameon=False, loc="best", fontsize=10)
    ax.set_title(title, fontsize=15, loc="left", pad=18)
    ax.set_xlabel("Calibration shots", fontsize=11, labelpad=10)
    ax.set_ylabel("Balanced accuracy", fontsize=11, labelpad=10)
    ax.set_xticks(shots)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if ylim is None:
        low, high = float(np.nanmin(scores)), float(np.nanmax(scores))
        padding = max(0.04, (high - low) * 0.15)
        ylim = (max(0, low - padding), min(1, high + padding))
    ax.set_ylim(*ylim)
    ax.margins(x=0.04)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#CBD0D6")
    ax.tick_params(colors="#374151", length=3)
    note = ("0 shots: strict zero-shot on all trials. Higher shots: post-calibration on held-out trials."
            if 0 in shots else "Post-calibration balanced accuracy on held-out trials.")
    note += "\nEach user's calibrated score is averaged across its calibration folds; users have equal weight."
    if not complete:
        note += "\nAvailable users by shot count: " + ", ".join(f"{shot}: {n}" for shot, n in zip(shots, counts)) + "."
    fig.text(0.11, 0.025, note, fontsize=8, color="#6B7280", va="bottom")
    # Reserve space on the right for endpoint labels; this is not another shot.
    fig.tight_layout(rect=(0, 0.115 if not complete else 0.10, 0.91, 1))
    _label_curve_ends(ax, users, shots, scores)
    return fig, means, counts


def build_plot(results_path, out_dir=None, *, include_zero_shot=True, subject_alpha=0.18,
               formats=("png", "pdf"), dpi=300, ylim=None,
               title="Subject calibration as shots increase"):
    """Load one SIC configuration and save its figure plus underlying CSVs."""
    results_path = Path(results_path)
    with results_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    users, shots, scores = extract_curves(raw, include_zero_shot=include_zero_shot)
    if include_zero_shot and 0 not in shots:
        warnings.warn("No strict zero-shot subject scores found; 0 shots is omitted.", stacklevel=2)
    if not formats or any(fmt not in {"png", "pdf", "svg"} for fmt in formats):
        raise ValueError("--formats must contain png, pdf and/or svg.")
    if dpi <= 0:
        raise ValueError("--dpi must be positive.")
    fig, means, counts = plot_curves(users, shots, scores, subject_alpha=subject_alpha,
                                    ylim=ylim, title=title)
    out_dir = Path(out_dir) if out_dir is not None else results_path.parent / "figures" / "calibration_curve"
    stem = "calibration_balanced_accuracy"
    paths = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in dict.fromkeys(formats):
            path = out_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            paths.append(path)
    finally:
        plt.close(fig)
    subject_csv, mean_csv = out_dir / f"{stem}_subjects.csv", out_dir / f"{stem}_mean.csv"
    with subject_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "calibration_shots", "balanced_accuracy", "stage"])
        for user, row in zip(users, scores):
            for shot, score in zip(shots, row):
                writer.writerow([user, shot, score if np.isfinite(score) else "",
                                 "zero_shot_all_trials" if shot == 0 else "post_calibration"])
    with mean_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["calibration_shots", "mean_balanced_accuracy", "n_subjects"])
        writer.writerows(zip(shots, means, counts))
    for shot, mean, count in zip(shots, means, counts):
        print(f"{shot:>3} shots | mean balanced_accuracy={mean:.6f} ({mean:.2%}) | users={count}")
    print(f"Saved {len(paths)} figure files and 2 CSVs to {out_dir}")
    return paths + [subject_csv, mean_csv]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="Path to one configuration's sic_calibration_results.json.")
    parser.add_argument("--out", default=None, help="Output directory. Default: <results-dir>/figures/calibration_curve.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated png,pdf,svg (default: png,pdf).")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--subject-alpha", type=float, default=0.18,
                        help="User-line opacity, 0=invisible and 1=opaque (default: 0.18). The red mean stays opaque.")
    parser.add_argument("--no-zero-shot", action="store_true", help="Exclude the strict zero-shot point at 0 shots.")
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Optional y-axis limits, e.g. --ylim 0 1. Default: fit observed scores with padding.")
    parser.add_argument("--title", default="Subject calibration as shots increase")
    args = parser.parse_args()
    try:
        build_plot(args.results, args.out, include_zero_shot=not args.no_zero_shot,
                   subject_alpha=args.subject_alpha,
                   formats=tuple(fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()),
                   dpi=args.dpi, ylim=args.ylim, title=args.title)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
