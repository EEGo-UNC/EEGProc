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

from .artifacts import write_json, write_csv, read_summary, completed_attempt
from .coverage import DIRECTIONS, eligibility_records


def compatible_typicality_definition(roots):
    """Keep old reports readable, but never pool distinct score definitions."""
    definition = None
    for root in roots:
        manifest = Path(root) / "study.json"
        if not manifest.is_file():
            raise ValueError("Expected a typicality.runner archive with study.json")
        study = json.loads(manifest.read_text())
        current = {
            "score": study.get("typicality_definition", "legacy_window_moment_KL"),
            "representation": study.get("typicality_representation",
                                        study.get("arguments", {}).get("typicality_sequence", "legacy_unspecified")),
        }
        if definition is not None and current != definition:
            raise ValueError("Cannot combine incompatible typicality definitions; report legacy KL and full-trial Mahalanobis studies separately")
        definition = current
    if definition is None:
        raise ValueError("At least one typicality study is required")
    return definition


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
    for field in ("latent_flip", "confidence_met", "typical", "latent_target_success", "typicality_success"):
        result[f"{field}_n"] = sum(bool(r.get(field, False)) for r in rows)
        result[f"{field}_percent"] = 100 * sum(bool(r.get(field, False)) for r in rows) / n if n else None
    result["n_round_trip_evaluated"] = sum(bool(r.get("round_trip_evaluated")) for r in rows)
    expected = any(r.get("round_trip_expected") or r.get("round_trip_evaluated") for r in rows)
    for field in ("decoded_target_success", "decoded_typical", "decoded_typicality_success",
                  "reconstruction_preserves_prediction"):
        unknown = any(r["status"] == "completed" and r.get(field) is None for r in rows)
        result[f"{field}_percent"] = (100 * sum(r.get(field) is True for r in rows) / n
                                      if n and expected and not unknown else None)
    for prefix in ("reencoded", "reencoded_reconstruction", "reencoded_repair"):
        # Percent of ELIGIBLE trials (failed/pending count as not successful),
        # matching the latent rates above; only when this run scored them.
        if any(f"{prefix}_target_success" in r for r in rows):
            for field in ("target_success", "typical", "success_and_typical"):
                result[f"{prefix}_{field}_percent"] = (
                    100 * sum(r.get(f"{prefix}_{field}") is True for r in rows) / n if n else None)
    if any("delta_dec_explanation" in r for r in rows):
        for field in ("repair_feasible", "repair_preserves_prediction"):
            known = [r[field] for r in rows if field in r]
            result[f"{field}_percent"] = 100 * sum(known) / len(known) if known else None
        for field in ("delta_dec_explanation", "delta_dec_repair"):
            for key, value in _quantiles([r.get(field) for r in rows]).items():
                result[f"{field}_{key}"] = value
    for field in ("d_z", "delta_dec", "e_rec"):
        for key, value in _quantiles([r.get(field) for r in rows]).items():
            result[f"{field}_{key}"] = value
    # Phase-invariant distances and latent-edit structure exist only in runs
    # made with the temporal/phase code; older archives keep their columns.
    for field in ("physiological_worst_exceedance", "physiological_mean_exceedance", "delta_env", "e_rec_env", "delta_env_explanation", "delta_env_repair",
                  "latent_edit_l1", "latent_edit_tv", "latent_edit_active_fraction", "latent_edit_episodes"):
        if any(field in r for r in rows):
            for key, value in _quantiles([r.get(field) for r in rows]).items():
                result[f"{field}_{key}"] = value
    for field in ("reconstruction_confidence_drop", "counterfactual_target_probability_drop",
                  "reconstruction_embedding_cycle_rmse", "counterfactual_embedding_cycle_rmse"):
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
    result["n_unsuccessful"] = sum(r["status"] == "completed" and not r.get("latent_target_success", False) for r in rows)
    result["provisional"] = result["n_pending"] > 0
    return result


def _latest_result(trial_directory):
    complete = completed_attempt(trial_directory)
    if complete:
        return complete
    attempts = sorted(Path(trial_directory).glob("attempt_*/result.json"))
    return attempts[-1].parent if attempts else None


def collect_study(root):
    root = Path(root)
    study = json.loads((root / "study.json").read_text())
    task = study["task"]
    target = study.get("target_class", 1)
    source = study.get("source_class", 1 - target)
    direction = study.get("direction", f"{source}_to_{target}")
    if DIRECTIONS.get(direction) != (source, target):
        raise ValueError("Study direction disagrees with source and target classes")
    identity = dict(direction=direction, source_class=source, target_class=target)
    rows, observed, recognition, folds = [], [], [], []
    for fold_entry in study["folds"]:
        subject = fold_entry["subject_id"]
        fold_dir = root / f"subject_{subject}"
        if not (fold_dir / "fold.json").exists():
            folds.append({"task": task, **identity, "subject_id": subject, "status": "pending"})
            continue
        fold = json.loads((fold_dir / "fold.json").read_text())
        if len(set(fold["eligible_trial_ids"])) != len(fold["eligible_trial_ids"]):
            raise ValueError("Duplicate eligible trial IDs")
        if fold.get("direction", direction) != direction or fold.get("target_class", target) != target:
            raise ValueError("Fold direction disagrees with study")
        folds.append({"task": task, **identity, "subject_id": subject, "status": fold["status"],
                      "n_eligible": len(fold["eligible_trial_ids"]), "threshold": fold["threshold"]})
        recognition.append({"task": task, **identity, "subject_id": subject, **fold["recognition"]})
        data = read_summary(fold_dir / "observations.json")
        for i, trial in enumerate(data["trial_ids"]):
            observed.append({"task": task, **identity, "subject_id": subject, "trial_id": int(trial),
                             "representation": f"observed_class_{data['labels'][i]}",
                             "discrepancy": float(data["discrepancy"][i]),
                             "threshold": fold["threshold"],
                             "correct": bool(data["probabilities"][i].argmax() == data["labels"][i])})
        expected_audit = eligibility_records(task=task, direction=direction, subject=subject,
            trial_ids=data["trial_ids"], labels=data["labels"], probabilities=data["probabilities"],
            trial_filter=study.get("arguments", {}).get("trial_ids"))
        if {r["trial_id"] for r in expected_audit if r["eligible"]} != set(fold["eligible_trial_ids"]):
            raise ValueError("Eligibility manifest disagrees with original labels and predictions")
        audit_path = fold_dir / "eligibility.json"
        if "direction" in study and not audit_path.is_file():
            raise ValueError("Direction-aware studies require an eligibility audit")
        if audit_path.is_file() and json.loads(audit_path.read_text()) != expected_audit:
            raise ValueError("Saved eligibility audit disagrees with original observations")
        folds[-1]["eligibility"] = expected_audit
        for trial in fold["eligible_trial_ids"]:
            for objective in study.get("objectives", ["base", "typicality"]):
                record = {"task": task, **identity, "subject_id": subject, "trial_id": trial,
                          "objective": objective, "status": "pending", "threshold": fold["threshold"],
                          "typical": False, "latent_target_success": False,
                          "typicality_success": False,
                          "round_trip_expected": study.get("round_trip_evaluation", "latent_only") != "latent_only",
                          "round_trip_evaluated": False}
                attempt = _latest_result(fold_dir / f"trial_{trial}" / objective)
                if attempt:
                    summary = json.loads((attempt / "result.json").read_text())
                    record.update(status=summary["status"], artifact_directory=str(attempt.resolve()))
                    if (summary.get("target_class", target) != target
                            or summary.get("direction", direction) != direction
                            or summary.get("subject_id", subject) != subject
                            or summary.get("trial_id", trial) != trial
                            or summary.get("objective", objective) != objective):
                        raise ValueError("Result identity disagrees with study")
                    if summary["status"] == "completed" and not (attempt / "complete.json").is_file():
                        record.update(status="pending", error="Uncommitted attempt")
                    elif summary["status"] == "completed":
                        typ = summary["typicality"]
                        decoded = summary["decoded_trials"][summary["report_output"]]
                        physiology = summary["physiology"]
                        latent_target_success = summary["latent_counterfactual"]["success"]
                        latent = summary["latent_counterfactual"]
                        required = summary["required_target_probability"]
                        flip = latent["predicted_class"] == target
                        confidence = latent["target_probability"] >= required
                        if bool(latent_target_success) != bool(flip and confidence):
                            raise ValueError("Saved success disagrees with target argmax and confidence")
                        if float(typ["threshold"]) != float(fold["threshold"]):
                            raise ValueError("Result typicality threshold disagrees with fold")
                        record.update(
                            latent_flip=flip, confidence_met=confidence,
                            required_target_probability=required,
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
                        if physiology.get("graded_count"):
                            # Graded deviation (half-interval widths past the
                            # source interval); absent in older archives.
                            record.update(
                                physiological_worst_exceedance=physiology["worst_family_exceedance"],
                                physiological_mean_exceedance=physiology["mean_family_exceedance"],
                                **{f"physiological_exceedance_{family}": value
                                   for family, value in physiology["exceedance_by_family"].items()})
                        # Re-encoded delivered EEG (all arms in newer runs; only the
                        # typicality arm in older ones). Kept apart from the
                        # round_trip_* protocol columns, which describe selection.
                        for prefix, key in (("reencoded", "decoded_counterfactual"),
                                            ("reencoded_reconstruction", "decoded_reconstruction"),
                                            ("reencoded_repair", "decoded_repair")):
                            scored = summary.get(key)
                            if not scored:
                                continue
                            record[f"{prefix}_probability"] = scored["probabilities"][target]
                            record[f"{prefix}_discrepancy"] = scored["discrepancy"]
                            if "success" in scored:
                                record[f"{prefix}_target_success"] = scored["success"]
                                record[f"{prefix}_typical"] = scored["typical"]
                                record[f"{prefix}_success_and_typical"] = bool(scored["success"] and scored["typical"])
                        if "decoded_change_env" in decoded:
                            # RMS difference of log band-power envelopes:
                            # insensitive to phase shifts that dominate MSE.
                            record.update(delta_env=float(np.sqrt(decoded["decoded_change_env"])),
                                          e_rec_env=float(np.sqrt(decoded["original_reconstruction_env"])))
                        edit = summary.get("latent_edit")
                        if edit:
                            record.update(latent_edit_l1=edit["l1_mean"], latent_edit_tv=edit["tv_mean"],
                                          latent_edit_active_fraction=edit["active_fraction"],
                                          latent_edit_episodes=edit["n_episodes"])
                        repair = summary.get("repair_baseline")
                        if repair and repair.get("available"):
                            # Change measured against the repaired decoder output
                            # instead of the raw reconstruction (see runner).
                            record.update(
                                delta_dec_explanation=repair["delta_dec_explanation"],
                                delta_dec_repair=repair["delta_dec_repair"],
                                repair_feasible=repair["feasible"],
                                repair_preserves_prediction=repair["preserves_prediction"],
                                vcsc_repair=repair["vcsc_repair"],
                            )
                            if "explanation_change_env" in repair:
                                record.update(
                                    delta_env_explanation=float(np.sqrt(repair["explanation_change_env"])),
                                    delta_env_repair=float(np.sqrt(repair["repair_change_env"])))
                        if "counterfactual" in decoded and "original_reconstruction" in decoded:
                            cf, rec = decoded["counterfactual"], decoded["original_reconstruction"]
                            record.update(
                                round_trip_evaluated=True,
                                decoded_target_success=cf["success"],
                                decoded_typical=cf["typical"],
                                decoded_typicality_success=bool(cf["success"] and cf["typical"]),
                                decoded_probability=cf["target_probability"],
                                decoded_discrepancy=cf["discrepancy"],
                                reconstruction_probability=rec["target_probability"],
                                reconstruction_discrepancy=rec["discrepancy"],
                                reconstruction_preserves_prediction=decoded["reconstruction_preserves_prediction"],
                                reconstruction_confidence_drop=decoded["reconstruction_confidence_drop"],
                                counterfactual_target_probability_drop=decoded["counterfactual_target_probability_drop"],
                                reconstruction_embedding_cycle_rmse=rec["embedding_cycle_rmse"],
                                counterfactual_embedding_cycle_rmse=cf["embedding_cycle_rmse"],
                            )
                            observed.append({"task": task, **identity, "subject_id": subject, "trial_id": trial,
                                             "representation": f"{objective}_reencoded", "discrepancy": cf["discrepancy"],
                                             "threshold": fold["threshold"], "correct": None})
                        observed.append({"task": task, **identity, "subject_id": subject, "trial_id": trial,
                                         "representation": objective, "discrepancy": record["discrepancy"],
                                         "threshold": fold["threshold"], "correct": None})
                    else:
                        record["error"] = summary.get("error")
                attempts = list((fold_dir / f"trial_{trial}" / objective).glob("attempt_*"))
                record["n_attempts"] = len(attempts)
                record["n_error_attempts"] = sum(
                    (a / "result.json").is_file() and json.loads((a / "result.json").read_text()).get("status") == "error"
                    for a in attempts)
                rows.append(record)
    return rows, observed, recognition, folds


def study_roots(roots):
    """Accept individual studies or a task/suite containing direction shards."""
    studies = []
    for root in map(Path, roots):
        found = [root] if (root / "study.json").is_file() else sorted(p.parent for p in root.rglob("study.json"))
        if not found:
            raise ValueError(f"No study.json found under {root}")
        studies.extend(found)
    return studies


def build_report(roots, output, *, expected_subjects=None, expected_tasks=None,
                 expected_directions=None, require_complete=False):
    roots = study_roots(roots)
    definition = compatible_typicality_definition(roots)
    manifests = [json.loads((root / "study.json").read_text()) for root in roots]
    objective_sets = {tuple(m.get("objectives", ["base", "typicality"])) for m in manifests}
    if len(objective_sets) != 1:
        raise ValueError("Cannot combine different objective sets")
    objectives = objective_sets.pop()
    protocols = {m.get("round_trip_evaluation", "latent_only") for m in manifests}
    if protocols != {"latent_only"}:
        raise ValueError("This report requires the latent-only validity protocol")
    for task in {m["task"] for m in manifests}:
        task_manifests = [m for m in manifests if m["task"] == task]
        for key in ("typicality_quantile", "target_probability"):
            values = {m.get(key, m.get("arguments", {}).get(key)) for m in task_manifests}
            if len(values) != 1:
                raise ValueError(f"Cannot combine different {key} settings within {task}")
    rows, observed, recognition, folds = [], [], [], []
    seen = set()
    for root in roots:
        collected = collect_study(root)
        for fold in collected[3]:
            key = (fold["task"], fold["direction"], fold["subject_id"])
            if key in seen:
                raise ValueError(f"Duplicate task/direction/subject across study shards: {key}")
            seen.add(key)
        for destination, values in zip((rows, observed, recognition, folds), collected):
            destination.extend(values)
    if expected_subjects is not None:
        if expected_subjects < 1:
            raise ValueError("expected_subjects must be positive")
        tasks = expected_tasks or sorted({m["task"] for m in manifests})
        directions = expected_directions or sorted({f["direction"] for f in folds})
        expected = {(task, direction, subject) for task in tasks for direction in directions
                    for subject in range(expected_subjects)}
        if seen - expected:
            raise ValueError("Study shards fall outside the expected task/direction/subject coverage")
        for task, direction, subject in sorted(expected - seen):
            source, target = DIRECTIONS[direction]
            folds.append(dict(task=task, direction=direction, source_class=source,
                              target_class=target, subject_id=subject, status="pending"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    populations, subject_rows, coverage, examples = [], [], [], []
    audit = [row for fold in folds for row in fold.pop("eligibility", [])]
    groups = sorted({(fold["task"], fold["direction"]) for fold in folds})
    for task, direction in groups:
        source, target = DIRECTIONS[direction]
        identity = dict(task=task, direction=direction, source_class=source, target_class=target)
        task_folds = [f for f in folds if (f["task"], f["direction"]) == (task, direction)]
        decisions = [r for r in audit if (r["task"], r["direction"]) == (task, direction)]
        def counts(decisions):
            return dict(n_observed=len(decisions), n_eligible=sum(r["eligible"] for r in decisions),
                        n_excluded=sum(not r["eligible"] for r in decisions),
                        **{f"n_{reason}": sum(r["exclusion_reason"] == reason for r in decisions)
                           for reason in ("outside_source_class", "originally_misclassified", "trial_filter")})
        coverage.append({**identity, "scope": "population", **counts(decisions),
                         "n_expected_subjects": len(task_folds),
                         "n_pending_folds": sum(f["status"] != "completed" for f in task_folds)})
        for fold in task_folds:
            subject_counts = counts([r for r in decisions if r["subject_id"] == fold["subject_id"]])
            if fold["status"] == "pending":
                subject_counts = dict.fromkeys(subject_counts, None)
            coverage.append({**identity, "scope": "subject", "subject_id": fold["subject_id"],
                             **subject_counts,
                             "fold_status": fold["status"]})
        for objective in objectives:
            group = [r for r in rows if (r["task"], r["direction"], r["objective"]) == (task, direction, objective)]
            summary = population_summary(group)
            summary["provisional"] |= any(f["status"] != "completed" for f in task_folds)
            populations.append({**identity, "objective": objective, **summary,
                                "n_expected_subjects": len(task_folds)})
            for fold in task_folds:
                summary = population_summary([r for r in group if r["subject_id"] == fold["subject_id"]])
                summary["provisional"] |= fold["status"] != "completed"
                subject_rows.append({**identity, "subject_id": fold["subject_id"], "objective": objective,
                                     **summary, "fold_status": fold["status"]})
            # Deterministic examples show successful and unsuccessful finite endpoints.
            for success in (True, False):
                candidates = [r for r in group if r["status"] == "completed"
                              and r["latent_target_success"] == success
                              and r.get("d_z") is not None and np.isfinite(r["d_z"])]
                if candidates:
                    median = float(np.median([r["d_z"] for r in candidates]))
                    selected = min(candidates, key=lambda r: (abs(r["d_z"] - median), r["subject_id"], r["trial_id"]))
                    examples.append({**selected, "example_type": "success" if success else "unsuccessful",
                                     "selection_rule": "nearest within-group median d_z; ties by subject/trial"})
    # Normalize each score within its fold before making cross-fold distributions.
    for row in observed:
        threshold = row["threshold"]
        row["normalized_discrepancy"] = row["discrepancy"] / threshold if threshold > 0 else None
    for name, values in (("trial_metrics", rows), ("population_counterfactuals", populations),
                         ("subject_counterfactuals", subject_rows), ("fold_status", folds),
                         ("eligibility", audit), ("exclusions", [r for r in audit if not r["eligible"]]),
                         ("coverage", coverage), ("discrepancy_distributions", observed),
                         ("emotion_recognition_folds", [{k: v for k, v in r.items() if k != "calibration_bins"} for r in recognition])):
        write_csv(output / f"{name}.csv", values)
    payload = dict(schema_version=4, typicality_definition=definition,
                   studies=[str(root.resolve()) for root in roots],
                   round_trip_evaluation="latent_only", population=populations, coverage=coverage,
                   complete=all(f["status"] == "completed" for f in folds) and all(r["status"] != "pending" for r in rows),
                   notes=["Rates use all eligible trials per task, direction, and objective. Errors count as failures; pending work makes rates provisional.",
                          "Retries retain their attempt records and each eligible trial contributes once to each objective denominator.",
                          "Distances summarize all finite selected endpoints, including unsuccessful cases, with sample counts and quartiles.",
                          "Typicality uses the saved target-class Gaussian and source-only 10th-percentile cutoff in new studies.",
                          "Latent success requires target argmax and confidence; decoded classification validity is unmeasured.",
                          "Unknown physiological checks remain unavailable. Zero-eligible subjects have undefined rates."])
    write_json(output / "results.json", payload)
    write_json(output / "single_trial_examples.json", examples)
    write_json(output / "calibration_bins.json", recognition)
    _write_latex(output / "tables.tex", populations)
    if require_complete and not payload["complete"]:
        raise ValueError("Counterfactual report is incomplete; inspect fold_status.csv and trial_metrics.csv")
    return payload


def _write_latex(path, populations):
    lines = [r"% All eligible trials; finite endpoints include failures. -- means unavailable or provisional.",
             r"\begin{tabular}{lllrrrrrr}",
             r"Task & Direction & Objective & $N$ & Flip (\%) & Conf. (\%) & Typ. (\%) & Both (\%) & $d_z$ [Q1,Q3] \\ \hline"]
    def fmt(value):
        return "--" if value is None else f"{value:.2f}"
    for row in populations:
        values = [row["task"], row["direction"].replace("_to_", r"$\to$"), row["objective"].replace("_", r"\_"), str(row["n_eligible"])]
        rates = ("latent_flip_percent", "confidence_met_percent", "typical_percent", "typicality_success_percent")
        values.extend("--" if row["provisional"] else fmt(row[key]) for key in rates)
        values.append("--" if row["provisional"] else f'{fmt(row["d_z_median"])} [{fmt(row["d_z_q25"])}, {fmt(row["d_z_q75"])}]')
        lines.append(" & ".join(values) + r" \\")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int,
                        help="Require IDs 0..N-1 in each requested task/direction; missing shards remain pending.")
    parser.add_argument("--expected-tasks", nargs="+", choices=("valence", "arousal"))
    parser.add_argument("--expected-directions", nargs="+", choices=tuple(DIRECTIONS))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    if (args.expected_tasks or args.expected_directions) and args.expected_subjects is None:
        parser.error("Expected task/direction coverage requires --expected-subjects")
    build_report(args.studies, args.out_dir, expected_subjects=args.expected_subjects,
                 expected_tasks=args.expected_tasks, expected_directions=args.expected_directions,
                 require_complete=args.require_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
