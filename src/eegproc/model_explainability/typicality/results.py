"""Rebuild manuscript tables, distributions, and examples from saved files.

This module does not import TensorFlow or load a checkpoint. Percentages use
all eligible trials, including unsuccessful and crashed optimizations. Pending
trials are counted and results are explicitly provisional until all finish.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .artifacts import write_json, write_csv, write_npz


def recognition_metrics(labels, probabilities, *, ece_bins=15):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), 2) or not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("Need nonempty binary trial predictions")
    if not np.isin(labels, [0, 1]).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid labels or probabilities")
    if not isinstance(ece_bins, int) or ece_bins < 1:
        raise ValueError("ece_bins must be positive")
    predicted = probabilities.argmax(axis=1)
    recalls = [float((predicted[labels == c] == c).mean()) if np.any(labels == c) else None for c in (0, 1)]
    confidence = probabilities.max(axis=1)
    bins = np.minimum((confidence * ece_bins).astype(int), ece_bins - 1)
    calibration = []
    ece = 0.0
    for index in range(ece_bins):
        mask = bins == index
        if mask.any():
            acc = float((predicted[mask] == labels[mask]).mean())
            conf = float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(acc - conf)
            calibration.append({"bin": index, "n": int(mask.sum()), "accuracy": acc, "confidence": conf})
    return {"n_trials": len(labels), "balanced_accuracy": float(np.mean(recalls)) if None not in recalls else None,
            "recall_0": recalls[0], "recall_1": recalls[1],
            "auroc": float(roc_auc_score(labels, probabilities[:, 1])) if len(np.unique(labels)) == 2 else None,
            "ece": float(ece), "ece_bins": ece_bins, "calibration_bins": calibration}


def _quantiles(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    low, middle, high = np.quantile(values, [0.25, 0.5, 0.75])
    return {"median": float(middle), "q25": float(low), "q75": float(high), "n": len(values)}


def population_summary(rows):
    """One task/objective group. Missing metrics are not zero-valued distances."""
    n = len(rows)
    result = {"n_eligible": n,
              "n_completed": sum(r["status"] == "completed" for r in rows),
              "n_error": sum(r["status"] == "error" for r in rows),
              "n_pending": sum(r["status"] == "pending" for r in rows)}
    for field in ("typical", "latent_target_success", "typicality_success"):
        result[f"{field}_percent"] = 100 * sum(bool(r.get(field, False)) for r in rows) / n if n else None
    for field in ("d_z", "delta_dec", "e_rec"):
        for key, value in _quantiles([r.get(field) for r in rows]).items():
            result[f"{field}_{key}"] = value
    assessed = sum(r.get("physiological_passed") is not None for r in rows)
    result["n_physiologically_assessed"] = assessed
    # A known optimization error is a failure; an unavailable scientific check
    # is unknown, not a failure or a pass. Do not print a misleading Phys.%.
    unknown = any(r["status"] == "completed" and r.get("physiological_passed") is None for r in rows)
    result["physiological_pass_percent"] = (100 * sum(r.get("physiological_passed") is True for r in rows) / n
                                             if n and not unknown and not result["n_pending"] else None)
    result["available_physiological_checks_pass_percent"] = (
        100 * sum(r.get("available_physiological_checks_passed") is True for r in rows) / n if n else None)
    result["provisional"] = result["n_pending"] > 0
    return result


def _latest_result(trial_directory):
    complete = sorted(Path(trial_directory).glob("attempt_*/complete.json"))
    if complete:
        return complete[-1].parent
    attempts = sorted(Path(trial_directory).glob("attempt_*/result.json"))
    return attempts[-1].parent if attempts else None


def collect_study(root):
    root = Path(root)
    study = json.loads((root / "study.json").read_text())
    task = study["task"]
    rows, observed, recognition, folds = [], [], [], []
    for fold_entry in study["folds"]:
        subject = fold_entry["subject_id"]
        fold_dir = root / f"subject_{subject}"
        if not (fold_dir / "fold.json").exists():
            folds.append({"task": task, "subject_id": subject, "status": "pending"})
            continue
        fold = json.loads((fold_dir / "fold.json").read_text())
        folds.append({"task": task, "subject_id": subject, "status": fold["status"],
                      "n_eligible": len(fold["eligible_trial_ids"]), "threshold": fold["threshold"]})
        recognition.append({"task": task, "subject_id": subject, **fold["recognition"]})
        with np.load(fold_dir / "observations.npz", allow_pickle=False) as data:
            for i, trial in enumerate(data["trial_ids"]):
                observed.append({"task": task, "subject_id": subject, "trial_id": int(trial),
                                 "representation": f"observed_class_{data['labels'][i]}",
                                 "discrepancy": float(data["discrepancy"][i]),
                                 "threshold": fold["threshold"],
                                 "correct": bool(data["probabilities"][i].argmax() == data["labels"][i])})
        for trial in fold["eligible_trial_ids"]:
            for objective in ("base", "typicality"):
                record = {"task": task, "subject_id": subject, "trial_id": trial,
                          "objective": objective, "status": "pending", "threshold": fold["threshold"],
                          "typical": False, "latent_target_success": False,
                          "typicality_success": False}
                attempt = _latest_result(fold_dir / f"trial_{trial}" / objective)
                if attempt:
                    summary = json.loads((attempt / "result.json").read_text())
                    record.update(status=summary["status"], artifact_directory=str(attempt.resolve()))
                    if summary["status"] == "completed":
                        typ = summary["typicality"]
                        decoded = summary["decoded_trials"][summary["report_output"]]
                        physiology = summary["physiology"]
                        latent_target_success = summary["latent_counterfactual"]["success"]
                        record.update(
                            typical=typ["typical"],
                            latent_target_success=latent_target_success,
                            typicality_success=bool(typ["typical"] and latent_target_success),
                            original_discrepancy=typ["original_discrepancy"],
                            discrepancy=typ["counterfactual_discrepancy"],
                            d_z=summary["d_z"], delta_dec=float(np.sqrt(decoded["decoded_change_mse"])),
                            e_rec=float(np.sqrt(decoded["original_reconstruction_mse"])),
                            original_probability=summary["original"]["target_probability"],
                            latent_probability=summary["latent_counterfactual"]["target_probability"],
                            vcsc_original=summary["vcsc_original_input"],
                            vcsc_original_reconstruction=decoded["vcsc_original_reconstruction"],
                            vcsc_counterfactual=decoded["vcsc_counterfactual"],
                            selected_step=summary["selected_step"], steps_completed=summary["steps_completed"],
                            stop_reason=summary["stop_reason"],
                            physiological_passed=physiology["all_required_passed"],
                            physiological_passed_count=physiology["passed_count"],
                            physiological_available_count=physiology["available_count"],
                            physiological_required_count=physiology["required_count"],
                            available_physiological_checks_passed=physiology["available_checks_passed"],
                        )
                        observed.append({"task": task, "subject_id": subject, "trial_id": trial,
                                         "representation": objective, "discrepancy": record["discrepancy"],
                                         "threshold": fold["threshold"], "correct": None})
                    else:
                        record["error"] = summary.get("error")
                rows.append(record)
    return rows, observed, recognition, folds


def build_report(roots, output, *, probe_results=()):
    rows, observed, recognition, folds = [], [], [], []
    seen = set()
    for root in roots:
        collected = collect_study(root)
        for fold in collected[3]:
            key = (fold["task"], fold["subject_id"])
            if key in seen:
                raise ValueError(f"Duplicate task/subject across study inputs: {key}")
            seen.add(key)
        for destination, values in zip((rows, observed, recognition, folds), collected):
            destination.extend(values)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = sorted({fold["task"] for fold in folds})
    populations, subject_rows, recognition_rows, subject_changes, examples = [], [], [], [], []
    for task in tasks:
        task_folds = [f for f in folds if f["task"] == task]
        for objective in ("base", "typicality"):
            group = [r for r in rows if r["task"] == task and r["objective"] == objective]
            pending_folds = sum(f["status"] == "pending" for f in task_folds)
            group_summary = population_summary(group)
            group_summary["provisional"] |= pending_folds > 0
            populations.append({"task": task, "objective": objective, **group_summary,
                                "n_expected_subjects": len(task_folds),
                                "n_pending_folds": pending_folds})
            for fold in task_folds:
                subject = fold["subject_id"]
                subject_rows.append({"task": task, "subject_id": subject, "objective": objective,
                                     **population_summary([r for r in group if r["subject_id"] == subject]),
                                     "fold_status": fold["status"]})
        metrics = [r for r in recognition if r["task"] == task]
        rec = {"task": task, "n_expected_folds": len(task_folds)}
        for name in ("balanced_accuracy", "recall_0", "recall_1", "auroc", "ece"):
            values = [r[name] for r in metrics if r[name] is not None]
            rec[f"{name}_mean"] = float(np.mean(values)) if values else None
            rec[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) >= 2 else None
            rec[f"{name}_n"] = len(values)
        recognition_rows.append(rec)
        subjects = sorted({r["subject_id"] for r in subject_rows if r["task"] == task})
        paired_rates = []
        for subject in subjects:
            pair = [r for r in subject_rows if r["task"] == task and r["subject_id"] == subject]
            if len(pair) == 2 and all(r["n_eligible"] and not r["provisional"] and r["fold_status"] == "completed" for r in pair):
                paired_rates.append([r["typical_percent"] for r in pair])
        changes = {"task": task, "n_expected_subjects": len(subjects), "n_evaluable_subjects": len(paired_rates),
                   "n_improved": sum(p[1] > p[0] for p in paired_rates)}
        for index, objective in enumerate(("base", "typicality")):
            changes.update({f"{objective}_{key}": val for key, val in _quantiles([p[index] for p in paired_rates]).items()})
        pop = [p for p in populations if p["task"] == task]
        for metric in ("latent_target_success", "typical", "typicality_success"):
            a, b = [p[f"{metric}_percent"] for p in pop]
            changes[f"{metric}_change_percentage_points"] = b - a if a is not None and b is not None else None
        subject_changes.append(changes)
        successful = [r for r in rows if r["task"] == task and r["objective"] == "typicality" and r["typicality_success"]]
        if successful:
            median = np.median([r["d_z"] for r in successful])
            selected = min(successful, key=lambda r: (abs(r["d_z"] - median), r["subject_id"], r["trial_id"]))
            examples.append({**selected, "selection_rule": "latent-target-and-typical typicality arm nearest its task median d_z; ties by subject/trial",
                             "successful_median_d_z": float(median)})
        else:
            examples.append({"task": task, "status": "no_typicality_success_example"})
    for name, values in (("trial_metrics", rows), ("discrepancy_distributions", observed),
                         ("population_counterfactuals", populations), ("subject_counterfactuals", subject_rows),
                         ("emotion_recognition_folds", [{k: v for k, v in r.items() if k != "calibration_bins"} for r in recognition]),
                         ("emotion_recognition", recognition_rows), ("subject_typicality_changes", subject_changes), ("fold_status", folds)):
        write_csv(output / f"{name}.csv", values)
    write_json(output / "calibration_bins.json", recognition)
    write_json(output / "single_trial_examples.json", examples)
    probes = []
    for path in probe_results:
        probe = json.loads(Path(path).read_text())
        if probe.get("task") not in tasks or any(p["task"] == probe["task"] for p in probes):
            raise ValueError("Probe results must identify one unique task present in this report")
        probes.append(probe)
    payload = {"schema_version": 1, "population": populations, "emotion_recognition": recognition_rows,
               "subject_typicality": subject_changes, "examples": examples,
               "subject_identifiability": probes,
               "complete": all(f["status"] == "completed" for f in folds) and all(r["status"] != "pending" for r in rows),
               "notes": ["Distances are RMSE in latent/input coordinates; IQR is Q25,Q75 over all finite selected endpoints, including failures.",
                         "Unknown physiological checks are NA, never silently passed.",
                         "ECE uses equal-width top-label confidence bins; fold SD uses ddof=1.",
                         "Subject probe must be run explicitly with a declared coordinate policy."]}
    write_json(output / "results.json", payload)
    write_json(output / "subject_probe_status.json", {"status": "provided" if probes else "not_evaluated", "tasks": [p["task"] for p in probes], "reason": "Independent LOSO latent coordinate systems can confound subject identification; inspect each probe's coordinate policy."})
    if probes:
        write_csv(output / "subject_identifiability.csv", [{"task": p["task"], **row} for p in probes for row in p["rows"]])
    _write_latex(output / "tables.tex", populations, recognition_rows, probes)
    return payload


def _write_latex(path, populations, recognition, probes=()):
    def fmt(value, percent=False):
        return "--" if value is None else f"{value * (100 if percent else 1):.2f}"
    def distance(row, name):
        return f"{fmt(row[name + '_median'])} [{fmt(row[name + '_q25'])}, {fmt(row[name + '_q75'])}]"
    lines = [r"% Generated from archived trials. -- means unavailable; inspect results.json for completeness.",
             r"\begin{tabular}{lccccc}", r"Task & Bal. acc. (\%) & Recall 0 (\%) & Recall 1 (\%) & AUROC & ECE \\ \hline"]
    for row in recognition:
        entries = [f"{fmt(row[name + '_mean'], percent)} $\\pm$ {fmt(row[name + '_std'], percent)}"
                   for name, percent in (("balanced_accuracy", True), ("recall_0", True), ("recall_1", True), ("auroc", False), ("ece", False))]
        lines.append(" & ".join([row["task"].title(), *entries]) + r" \\")
    lines.extend([r"\end{tabular}", "", r"\begin{tabular}{llcccccc}",
                  r"Task & Objective & Latent (\%) & Typ. (\%) & Both (\%) & $d_z$ & $\Delta_{dec}$ & Phys. (\%) \\ \hline"])
    for row in populations:
        entries = [row["task"].title(), "Base CFO" if row["objective"] == "base" else r"$+\mathcal{L}_{typ}$",
                   fmt(row["latent_target_success_percent"]), fmt(row["typical_percent"]), fmt(row["typicality_success_percent"]),
                   distance(row, "d_z"), distance(row, "delta_dec"), fmt(row["physiological_pass_percent"])]
        lines.append(" & ".join(entries) + r" \\")
    lines.append(r"\end{tabular}")
    if probes:
        lines.extend(["", r"% Check subject_identifiability.json coordinate policy before interpreting these values.",
                      r"\begin{tabular}{lcc}", r"Representation & Valence probe BA (\%) & Arousal probe BA (\%) \\ \hline"])
        for representation, label in (("original", r"Original $Z$"), ("base", r"Base $Z^{cf}$"), ("typicality", r"Typicality-constrained $Z^{cf}$")):
            values = []
            for task in ("valence", "arousal"):
                matches = [r["balanced_accuracy_percent"] for p in probes if p["task"] == task for r in p["rows"] if r["representation"] == representation]
                values.append(fmt(matches[0] if matches else None))
            lines.append(" & ".join([label, *values]) + r" \\")
        lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--probe-results", nargs="*", type=Path, default=[])
    args = parser.parse_args(argv)
    build_report(args.studies, args.out_dir, probe_results=args.probe_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
