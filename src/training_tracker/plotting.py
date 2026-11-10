"""Headless report generation using the repository's existing result panels."""
from __future__ import annotations

import hashlib
import math
import textwrap
from pathlib import Path

from .core import load_campaign, load_json, scan_runs, write_json
from .results import subject_metrics


def generate_plots(runs_dir: Path, campaign: dict, output: Path, *,
                   run_ids: list[str] | None = None, image_format: str = "png",
                   chance: float | None = None,
                   compare_campaigns: list[Path] | None = None) -> list[Path]:
    """Write figures, exact source tables, and a provenance manifest.

    Campaigns must share the objective metric. Targets and smoke/full stages
    are kept in separate comparisons. Summary-only runs still get a figure.
    """
    if image_format not in {"png", "svg", "pdf"}:
        raise ValueError("image_format must be png, svg, or pdf.")
    if chance is not None and not math.isfinite(chance):
        raise ValueError("chance must be finite.")
    metric = campaign["objective"]["metric"]
    sources = [(campaign, runs_dir)]
    for path in compare_campaigns or []:
        other = load_campaign(path)
        if other["objective"]["metric"] != metric:
            raise ValueError("Comparison campaigns must use the same objective metric.")
        sources.append((other, path.resolve().parent / "runs"))
    summaries, subjects, skipped = [], [], []
    found = set()
    for current, directory in sources:
        for row in scan_runs(directory, current):
            if run_ids and row["run_id"] not in run_ids:
                continue
            found.add(row["run_id"])
            if row["status"] != "completed":
                skipped.append({"run_id": row["run_id"], "reason": row.get("error", row["status"])})
                continue
            label = f"{current['campaign']} / {row['run_id']}"
            common = {"model": label, "campaign": current["campaign"],
                      "run_id": row["run_id"], "target": row["target"], "stage": row["stage"]}
            result = load_json(row["run_dir"] / "result.json")
            unit = result.get("aggregation_unit", "subject" if "config_results" in result or result.get("subject_metrics") else "evaluations")
            if not isinstance(unit, str):
                raise ValueError("aggregation_unit must be a string.")
            summaries.append({**common, "mean": row["mean"], "std": row["std"],
                              "aggregation_unit": unit,
                              "result_path": str((row["run_dir"] / "result.json").resolve())})
            try:
                values = subject_metrics(result, metric)
            except (ValueError, KeyError, TypeError) as exc:
                values = []
                skipped.append({"run_id": row["run_id"], "reason": f"Subject panel: {exc}"})
            if not values:
                skipped.append({"run_id": row["run_id"], "reason": "No usable per-subject scores; summary retained."})
            subjects.extend({**common, **item} for item in values)
    missing = set(run_ids or []) - found
    if missing:
        raise ValueError(f"Unknown run IDs: {sorted(missing)}.")
    if not summaries:
        raise ValueError("No completed runs with valid objective metrics to plot. Sync results first.")
    # Plotting stays optional for the tracking core and never imports TensorFlow.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from eegproc.plotting.result_figures import (
        plot_metric_summary, plot_model_comparison, plot_per_subject_accuracy,
    )

    output.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    subject_frame = pd.DataFrame(subjects, columns=["model", "campaign", "run_id", "target", "stage", "subject_id", metric])
    summary_frame.to_csv(output / "runs.csv", index=False)
    subject_frame.to_csv(output / "subjects.csv", index=False)
    paths = []
    comparisons = []

    def save(fig, key, title):
        path = output / f"{key}.{image_format}"
        try:
            fig.suptitle(title, fontsize=10, wrap=True)
            if chance is not None:
                ax = fig.axes[0]
                ax.axhline(chance, color="#666666", linestyle=":", label=f"Reference = {chance:g}")
                low, high = ax.get_ylim()
                ax.set_ylim(min(low, chance - 0.02), max(high, chance + 0.02))
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(path, dpi=160, bbox_inches="tight")
        finally:
            plt.close(fig)
        paths.append(path)

    for row in summaries:
        key = hashlib.sha256(row["model"].encode()).hexdigest()[:12]
        selected = subject_frame[subject_frame["model"] == row["model"]]
        saved = pd.DataFrame([{"metric": metric, "mean": row["mean"], "std": row["std"]}])
        fig = plot_metric_summary(selected, metrics=(metric,), summary=saved, aggregation_unit=row["aggregation_unit"])
        title = f"{row['campaign']} · {row['run_id']}\n{row['target']} / {row['stage']} · {metric}"
        save(fig, key + "-summary", title)
        if not selected.empty:
            fig = plot_per_subject_accuracy(selected, metric=metric)
            save(fig, key + "-subjects", title)
    if not subject_frame.empty:
        for (target, stage), group in subject_frame.groupby(["target", "stage"]):
            if group["model"].nunique() < 2:
                continue
            # Compare paired subjects only; do not hide differing fold coverage.
            sets = [set(frame["subject_id"]) for _, frame in group.groupby("model")]
            shared = set.intersection(*sets)
            if not shared:
                skipped.append({"target": target, "stage": stage, "reason": "No shared subjects for comparison."})
                continue
            paired = group[group["subject_id"].isin(shared)].copy()
            paired["aggregation_unit"] = "subject"
            comparisons.append({"target": target, "stage": stage, "shared_subject_ids": sorted(shared),
                                "subjects_per_run": {name: len(frame) for name, frame in group.groupby("model")}})
            fig = plot_model_comparison(paired, metrics=metric, maximize=campaign["objective"].get("maximize", True))
            fig.set_size_inches(10, 6)
            # Shorten long IDs on the canvas; full identities remain in CSV/JSON.
            ax = fig.axes[0]
            labels = [tick.get_text() for tick in ax.get_xticklabels()]
            ax.set_xticklabels(["\n".join(textwrap.shorten(part, width=46, placeholder="...")
                                         for part in label.split(" / ")) for label in labels],
                               fontsize=8, rotation=15, ha="right")
            key = hashlib.sha256(f"{target}/{stage}".encode()).hexdigest()[:12]
            save(fig, key + "-comparison", f"{target} / {stage} · {len(shared)} shared subjects · {metric}")
    write_json(output / "plots.json", {
        "schema_version": 1, "metric": metric, "chance_reference": chance,
        "figures": [path.name for path in paths], "runs": summaries, "skipped": skipped,
        "comparisons": comparisons,
        "interpretation": "Descriptive results only. SD across the recorded aggregation unit is not a confidence interval. "
                          "A reference line does not establish non-randomness. "
                          "Comparison bars use only shared subjects; test-fold-based selection can bias results.",
    })
    (output / "README.md").write_text(
        "# Training result figures\n\n" +
        "\n".join(f"![{path.stem}]({path.name})\n" for path in paths) +
        "\nExact inputs: runs.csv and subjects.csv. Provenance and skipped panels: plots.json.\n\n"
        "Error bars show standard deviation across the recorded aggregation unit, not confidence intervals. "
        "Comparisons use the intersection of subjects and separate targets and smoke/full stages. "
        "Patterns and reference lines alone do not establish statistical significance.\n",
        encoding="utf-8",
    )
    return paths
