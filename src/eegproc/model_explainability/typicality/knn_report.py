"""Rebuild independent encoded-neighbour evaluations without loading a model.

Usage: python -m eegproc.model_explainability.typicality.knn_report
    --roots RUN/0_to_1 RUN/1_to_0 --out-dir RUN/knn_report
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import completed_attempt, file_sha256, read_summary, summary_path, write_csv, write_json
from .neighbors import EncodedKNNReference, KNN_DEFINITION, KNN_REPRESENTATION


def fit_archived_source(source, *, held_out_subject, target_class, k=5, quantile=.10,
                        calibration_fraction=.25, seed=42):
    if "embeddings" not in source:
        raise ValueError("Full source vc_trial_embedding vectors are missing. Scalar Gaussian discrepancies cannot recover neighbour distances; rerun/export with embedding archival enabled.")
    return EncodedKNNReference.fit_source(
        source["embeddings"], labels=source["labels"], subject_ids=source["subject_ids"],
        trial_ids=source["trial_ids"], held_out_subject=held_out_subject, target_class=target_class,
        k=k, quantile=quantile, calibration_fraction=calibration_fraction, seed=seed)


def assess_endpoints(reference, original, optimized):
    before = reference.assess(original)
    after = reference.assess(optimized)
    return {"original": before, "optimized": after,
            "distance_change": after["distance"] - before["distance"],
            "distance_over_threshold_change": (
                after["distance_over_threshold"] - before["distance_over_threshold"]
                if reference.threshold > 0 else None)}


def _score_row(reference, embedding, **metadata):
    return {**metadata, "evaluation_status": "scored", **reference.assess(embedding)}


def _roots(paths):
    roots = []
    for path in map(Path, paths):
        candidates = [path] if (path / "study.json").is_file() else [path / "0_to_1", path / "1_to_0"]
        found = [candidate for candidate in candidates if (candidate / "study.json").is_file()]
        if not found:
            raise ValueError(f"No study.json or direction-specific studies found in {path}")
        roots.extend(found)
    resolved = [path.resolve() for path in roots]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Duplicate study roots would repeat neighbour-evaluation trials")
    return sorted(resolved)


def _attempt(arm):
    committed = completed_attempt(arm)
    if committed is not None:
        return json.loads((committed / "result.json").read_text())
    latest = sorted(Path(arm).glob("attempt_*/result.json"))
    if latest:
        summary = json.loads(latest[-1].read_text())
        if summary.get("status") == "error":
            return summary
    return {"status": "pending"}


def _population(rows):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ("task", "direction", "target_class", "role", "objective"))
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        scored = [row for row in group if row["evaluation_status"] == "scored"]
        inside = sum(row["inside"] for row in scored)
        normalized = [row["distance_over_threshold"] for row in scored if row["distance_over_threshold"] is not None]
        percentiles = [row["calibration_percentile"] for row in scored]
        result.append(dict(zip(("task", "direction", "target_class", "role", "objective"), key),
            n_total=len(group), n_scored=len(scored), n_unscored=len(group) - len(scored),
            n_attempt_errors=sum(row["attempt_status"] == "error" for row in group),
            n_attempt_pending=sum(row["attempt_status"] == "pending" for row in group),
            inside_n=inside,
            inside_percent_of_scored=100 * inside / len(scored) if scored else None,
            confirmed_inside_percent_of_all=100 * inside / len(group),
            median_distance_over_threshold=float(np.median(normalized)) if normalized else None,
            median_calibration_percentile=float(np.median(percentiles)) if percentiles else None))
    return result


def build_knn_report(roots, output, *, k=5, quantile=.10, calibration_fraction=.25, seed=42):
    output = Path(output)
    roots = _roots(roots)
    rows, folds, seen = [], [], set()
    for root in roots:
        study = json.loads((root / "study.json").read_text())
        if study.get("typicality_representation") != KNN_REPRESENTATION:
            raise ValueError("Independent neighbours require the frozen full-trial vc_trial_embedding space")
        target = int(study.get("target_class", 1))
        direction = study.get("direction", f"{1-target}_to_{target}")
        task = study["task"]
        for entry in study["folds"]:
            subject = int(entry["subject_id"])
            identity = (task, direction, subject)
            if identity in seen:
                raise ValueError("Duplicate task/direction/subject shards would repeat neighbour-evaluation trials")
            seen.add(identity)
            directory = root / f"subject_{subject}"
            common = dict(task=task, direction=direction, target_class=target, subject_id=subject)
            if not (directory / "fold.json").is_file():
                folds.append({**common, "status": "pending", "n_eligible": None})
                continue
            fold = json.loads((directory / "fold.json").read_text())
            if int(fold["subject_id"]) != subject or int(fold.get("target_class", target)) != target:
                raise ValueError(f"Fold identity or target class disagrees with study: {directory}")
            source = read_summary(directory / "calibration" / "source_trials")
            reference = fit_archived_source(source, held_out_subject=subject, target_class=target,
                k=k, quantile=quantile, calibration_fraction=calibration_fraction, seed=seed)
            reference.save(output / "references" / task / direction / f"subject_{subject}.json")
            folds.append({**common, "status": fold["status"], "n_eligible": len(fold["eligible_trial_ids"]),
                          "threshold": reference.threshold, "n_bank_trials": len(reference.bank),
                          "n_calibration_trials": len(reference.calibration_embeddings),
                          "study_manifest_sha256": file_sha256(root / "study.json"),
                          "fold_metadata_sha256": file_sha256(directory / "fold.json"),
                          "source_archive_sha256": file_sha256(summary_path(directory / "calibration" / "source_trials")),
                          "observations_archive_sha256": file_sha256(summary_path(directory / "observations")),
                          "bank_subject_ids": reference.metadata["bank_subject_ids"],
                          "calibration_subject_ids": reference.metadata["calibration_subject_ids"]})
            observed = read_summary(directory / "observations")
            if "embeddings" not in observed:
                raise ValueError(f"Full held-out classification embeddings are missing in {directory}; re-export observations with the updated runner")
            embeddings = np.asarray(observed["embeddings"], dtype=float)
            if embeddings.shape != (len(observed["trial_ids"]), reference.bank.shape[1]):
                raise ValueError(f"Observation embedding dimensions disagree in {directory}")
            by_trial = {}
            for trial, label, embedding in zip(observed["trial_ids"], observed["labels"], embeddings):
                trial = int(trial)
                if trial in by_trial:
                    raise ValueError("Duplicate held-out trial IDs")
                by_trial[trial] = embedding
                rows.append(_score_row(reference, embedding, **common, trial_id=trial,
                    role="real_target_class" if int(label) == target else "real_other_class",
                    objective="observed", attempt_status="observed"))
            for trial in fold["eligible_trial_ids"]:
                trial = int(trial)
                if trial not in by_trial:
                    raise ValueError("Eligible trial is missing from encoded observations")
                for objective in study.get("objectives", ["base", "typicality"]):
                    summary = _attempt(directory / f"trial_{trial}" / objective)
                    expected = dict(common, trial_id=trial, objective=objective)
                    if any(key in summary and summary[key] != value for key, value in expected.items()):
                        raise ValueError("Counterfactual identity disagrees with its source fold")
                    metadata = dict(common, trial_id=trial, objective=objective,
                                    attempt_status=summary["status"])
                    rows.append(_score_row(reference, by_trial[trial], **metadata, role="original_before_optimization"))
                    endpoints = summary.get("classification_embeddings", {})
                    if "original" in endpoints:
                        original = np.asarray(endpoints["original"])
                        if original.shape != by_trial[trial].shape or not np.allclose(
                                original, by_trial[trial], rtol=1e-5, atol=1e-7):
                            raise ValueError("Counterfactual original embedding disagrees with encoded observations")
                    for key in ("optimized", "decoded_counterfactual"):
                        if key == "decoded_counterfactual" and key not in endpoints:
                            continue
                        if summary["status"] == "completed" and key in endpoints:
                            rows.append(_score_row(reference, endpoints[key], **metadata,
                                role="optimized_counterfactual" if key == "optimized" else "decoded_then_reencoded_counterfactual"))
                        else:
                            rows.append({**metadata, "role": "optimized_counterfactual",
                                "evaluation_status": ("missing_embedding" if summary["status"] == "completed" else summary["status"])})
    population = _population(rows)
    provisional = any(fold["status"] != "completed" for fold in folds)
    for group in population:
        group["provisional"] = provisional or group["n_attempt_pending"] > 0 or group["n_unscored"] > 0
    report = {
        "schema_version": 1, "definition": KNN_DEFINITION, "representation": KNN_REPRESENTATION,
        "roots": [str(path) for path in roots], "k": k, "quantile": quantile,
        "calibration_fraction": calibration_fraction, "seed": seed,
        "folds": folds, "population": population,
        "interpretation": [
            "The score is distance to the kth real true-target source embedding, not the mean neighbour distance.",
            "Bank and cutoff-calibration subjects are disjoint source participants. Neither uses the actual held-out participant.",
            "This evaluates a new score in the same frozen encoder; it is not an independently trained representation or a physiological test.",
            "The metric must remain outside optimization, iterate selection and early stopping.",
            "Raw distances are retained per trial. Cross-fold summaries use distance/cutoff and source-calibration percentiles.",
            "All eligible counterfactual attempts remain in the denominator. Missing vectors, errors and pending attempts are explicit; confirmed_inside_percent_of_all is only a confirmed lower bound when any are unscored.",
        ],
    }
    write_csv(output / "knn_trial_metrics.csv", rows)
    write_csv(output / "knn_population.csv", population)
    write_csv(output / "knn_folds.csv", folds)
    write_json(output / "knn_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=.10)
    parser.add_argument("--calibration-fraction", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        build_knn_report(args.roots, args.out_dir, k=args.k, quantile=args.quantile,
                         calibration_fraction=args.calibration_fraction, seed=args.seed)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
