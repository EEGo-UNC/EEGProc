"""Offline trajectories, population distributions, and band-power scalp maps.

Only archived CSV/JSON/NPZ files are read. No TensorFlow import or checkpoint
inference is needed. PNG and PDF outputs share the saved numerical inputs.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..counterfactuals.topography import plot_band_topographies
from .artifacts import write_npz, write_json


def _csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig, path):
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_report(report_directory, output):
    report_directory, output = Path(report_directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((report_directory / "results.json").read_text())
    round_trip = report.get("round_trip_evaluation", "latent_only") != "latent_only"
    is_full_trial = (report.get("typicality_definition") or {}).get("representation") == "vc_trial_embedding"
    displacement_label = "VC full-trial embedding displacement" if is_full_trial else "VC sequence displacement"
    distributions = _csv(report_directory / "discrepancy_distributions.csv")
    subjects = _csv(report_directory / "subject_counterfactuals.csv")
    trials = _csv(report_directory / "trial_metrics.csv")
    tasks = sorted({r["task"] for r in distributions})
    groups = (("observed_class_0", "base_reencoded", "typicality_reencoded", "observed_class_1")
              if round_trip else ("observed_class_0", "base", "typicality", "observed_class_1"))
    typical_field = "decoded_typical_percent" if round_trip else "typical_percent"
    for task in tasks:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        values, positions = [], []
        for index, group in enumerate(groups, 1):
            selected = [float(r["discrepancy"]) - float(r["threshold"]) for r in distributions
                        if r["task"] == task and r["representation"] == group]
            if selected:
                values.append(selected)
                positions.append(index)
        if values:
            ax.boxplot(values, positions=positions, widths=0.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xticks=range(1, 5), xticklabels=["Observed class 0", "Base CFO", "Typicality CFO", "Observed class 1"],
               ylabel=r"$D_{C_1} - \tau_{C_1}$ (fold threshold subtracted)",
               title=f"{task.title()}: {'re-encoded' if round_trip else 'latent'} trial discrepancy")
        _save(fig, output / f"{task}_discrepancy")

        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        for objective, color in (("base", "#536878"), ("typicality", "#087f8c")):
            selected = sorted([r for r in subjects if r["task"] == task and r["objective"] == objective
                               and r[typical_field]], key=lambda r: int(r["subject_id"]))
            ax.plot([int(r["subject_id"]) for r in selected], [float(r[typical_field]) for r in selected],
                    "o-", color=color, label=objective)
        ax.set(xlabel="Held-out subject", ylabel="Typical trials (%)", ylim=(-3, 103),
               title=f"{task.title()}: per-subject {'re-encoded ' if round_trip else ''}typicality")
        ax.legend()
        _save(fig, output / f"{task}_subject_typicality")

        for objective in ("base", "typicality"):
            selected = [r for r in trials if r["task"] == task and r["objective"] == objective and r["status"] == "completed"]
            _power_maps(selected, output / f"{task}_{objective}_population_power", f"{task.title()}, {objective}: mean decoded power change")

    for example in report["examples"]:
        if "artifact_directory" not in example:
            continue
        directory = Path(example["artifact_directory"])
        summary = json.loads((directory / "result.json").read_text())
        history = _csv(directory / "history.csv")
        steps = [int(r["step"]) for r in history]
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout="constrained")
        axes[0].plot(steps, [float(r["target_probability"]) for r in history], label="Latent classifier")
        axes[0].axhline(summary["required_target_probability"], linestyle="--", color="grey", label="Target probability")
        axes[0].set(ylabel="Class-1 probability", ylim=(0, 1))
        axes[1].plot(steps, [float(r["discrepancy"]) for r in history], label="Latent candidate")
        axes[1].axhline(example["threshold"], linestyle="--", color="grey")
        axes[1].set(ylabel=r"$D_{C_1}$")
        if round_trip:
            decoded = summary["decoded_trials"][summary["report_output"]]
            for name, step, label, marker in (
                ("original_reconstruction", 0, "Re-encoded reconstruction", "s"),
                ("counterfactual", summary["selected_step"], "Re-encoded counterfactual", "D"),
            ):
                axes[0].scatter([step], [decoded[name]["target_probability"]], label=label, marker=marker, zorder=3)
                axes[1].scatter([step], [decoded[name]["discrepancy"]], label=label, marker=marker, zorder=3)
            axes[1].legend()
        axes[0].legend()
        axes[2].plot(steps, [float(r["d_z"]) for r in history], label=displacement_label)
        axes[2].plot(steps, [float(r[f"delta_dec_{summary['report_output']}"]) for r in history], label="Decoder displacement")
        axes[2].set(xlabel="Optimization step", ylabel="RMSE (respective units)")
        axes[2].legend()
        for ax in axes:
            ax.axvline(summary["selected_step"], color="#b65b33", linestyle=":", linewidth=1)
        fig.suptitle(f"{example['task'].title()}: subject {example['subject_id']}, trial {example['trial_id']}")
        _save(fig, output / f"{example['task']}_single_trial_trajectory")
        _power_maps([example], output / f"{example['task']}_single_trial_power", "Selected example: decoded power change")
    write_json(output / "figure_metadata.json", {
        "report": str(report_directory.resolve()), "report_complete": report["complete"],
        "population_maps_cohort": "all finite completed endpoints, including unsuccessful counterfactuals",
        "power_reference": "decoded original reconstruction; power differences retain each run's documented units",
        "discrepancy_plot": "D minus each trial's fold threshold; raw values and thresholds remain in report CSV",
        "typicality_definition": report.get("typicality_definition"),
        "round_trip_evaluation": report.get("round_trip_evaluation", "latent_only"),
        "example_rule": f"{'decoded-target-and-reencoded-typical' if round_trip else 'latent-target-and-typical'} arm nearest task median {displacement_label}",
    })


def _power_maps(rows, path, title):
    if not rows:
        return
    changes, keys = [], []
    names, bands = None, None
    for row in rows:
        directory = Path(row["artifact_directory"])
        with np.load(directory / "physiology_counterfactual.npz", allow_pickle=False) as data:
            after = data["spectral_power"]
        with np.load(directory / "physiology_reconstruction.npz", allow_pickle=False) as data:
            before = data["spectral_power"]
        with np.load(directory / "counterfactual.npz", allow_pickle=False) as data:
            current_names, current_bands = data["channel_names"].tolist(), data["band_names"].tolist()
        if names is not None and (names != current_names or bands != current_bands):
            raise ValueError("Cannot combine topographies with different channel/band orders")
        names, bands = current_names, current_bands
        changes.append(after - before)
        keys.append([int(row["subject_id"]), int(row["trial_id"])])
    changes = np.stack(changes)
    write_npz(path.with_suffix(".npz"), trial_keys=np.asarray(keys), power_changes=changes,
              mean_power_change=changes.mean(axis=0), channel_names=np.asarray(names), band_names=np.asarray(bands))
    fig, _ = plot_band_topographies(changes.mean(axis=0).T, channel_names=names, band_names=bands,
                                title=f"{title} (n={len(rows)})", colorbar_label="Power difference (signal unit squared)",
                                signed=True, shared_scale=True)
    _save(fig, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    plot_report(args.report_directory, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
