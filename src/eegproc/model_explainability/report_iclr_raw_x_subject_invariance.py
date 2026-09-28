"""Compare original class-1 EEG X across subjects in each frozen LOSO fold.

The source reference is fitted to original, prepared X from source subjects'
correctly predicted, embedding-typical class-1 trials. Held-out original X is
scored in exactly the same waveform space. No decoder or model is run.

The report also makes a source-subject-out comparison: each eligible source
subject and the actual held-out subject are scored against a reference fitted
without that source subject. The paired ranking is the less biased comparison;
the full-source 95th-percentile region is descriptive because its source scores
are in-sample.

Run from an EEGProc checkout that contains the SIC raw trial loader::

    PYTHONPATH=src python -m eegproc.model_explainability.report_iclr_raw_x_subject_invariance \
        runs/counterfactuals/final-ICLR-report \
        --raw-eeg datasets/dreamer_eeg.npy \
        --raw-labels datasets/dreamer_labels.npy \
        --out-dir runs/counterfactuals/final-ICLR-raw-x-invariance
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from .report_iclr_subject_invariance import (
    _load_summary, _median, _probability_higher, _source_percentiles,
    analyze_fold, file_sha256, stratified_sample_indices, write_csv,
    write_json,
)


def _roots(inputs):
    roots = set()
    for item in map(Path, inputs):
        if not item.exists():
            raise ValueError(f"Study input does not exist: {item}")
        report_path = item if item.is_file() and item.name == "report.json" else item / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            studies = report.get("studies")
            if not isinstance(studies, list) or not studies:
                raise ValueError(f"{report_path}: missing nonempty study list")
            for study in studies:
                root = Path(study)
                if not root.is_absolute():
                    root = report_path.parent / root
                if not (root / "study.json").is_file():
                    raise ValueError(f"{report_path}: linked study is unavailable: {root}")
                roots.add(root.resolve())
        elif (item / "study.json").is_file():
            roots.add(item.resolve())
        else:
            roots.update(path.parent.resolve() for path in item.rglob("study.json"))
    if not roots:
        raise ValueError("No study.json found beneath the supplied inputs")
    return sorted(roots)


def _dataset_for_task(study, *, raw_eeg, raw_labels):
    """Replay the archived preprocessing and check its recorded fingerprints."""
    from .typicality.artifacts import array_sha256

    arguments = study["arguments"]
    spec = arguments.get("data_loader")
    if not spec or ":" not in spec:
        raise ValueError("Study needs its archived package.module:loader data_loader")
    module_name, function_name = spec.rsplit(":", 1)
    loader = getattr(importlib.import_module(module_name), function_name)
    config = dict(arguments["data_config"])
    config["raw_eeg_npy"] = str(raw_eeg)
    config["raw_labels_npy"] = str(raw_labels)
    dataset = loader(config)
    if dataset.features.ndim != 4 or not np.isfinite(dataset.features).all():
        raise ValueError("Expected finite prepared, full-trial EEG (trial, window, sample, feature)")
    for name in ("subject_ids", "trial_ids", "labels", "normalization_offset", "normalization_scale"):
        values = getattr(dataset, name)
        if values is None or array_sha256(values) != study["dataset_sha256"][name]:
            raise ValueError(f"Local EEG or preprocessing differs from study: {name}")
    keys = list(zip(dataset.subject_ids.astype(int), dataset.trial_ids.astype(int)))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate prepared subject/trial IDs")
    return dataset, dict(zip(keys, range(len(keys))))


def fit_region(trials, *, variance_floor=1e-6):
    """Fit the same pointwise diagonal squared-Mahalanobis form as Eq. 7."""
    values = np.asarray(trials, dtype=np.float64)
    if (values.ndim != 4 or len(values) < 2 or not np.isfinite(values).all()
            or not np.isfinite(variance_floor) or variance_floor <= 0):
        raise ValueError("Need at least two finite, equally shaped raw EEG trials and a positive floor")
    mean = values.mean(axis=0)
    variance = np.maximum(values.var(axis=0), variance_floor)
    return mean, variance


def score_region(trials, mean, variance):
    values = np.asarray(trials, dtype=np.float64)
    if values.ndim == mean.ndim:
        values = values[None]
    if values.shape[1:] != mean.shape or not np.isfinite(values).all():
        raise ValueError("Raw EEG query shape or values differ from the source reference")
    return np.mean(np.square(values - mean) / variance, axis=tuple(range(1, values.ndim)))


def _trial_keys(subjects, trials, selected, index, dataset, *, held_subject, role):
    keys = [(int(subjects[i]), int(trials[i])) for i in selected]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate {role} trial IDs")
    for key in keys:
        if ((key[0] == held_subject) != (role == "heldout")
                or key not in index or int(dataset.labels[index[key]]) != 1):
            raise ValueError(f"Invalid {role} original class-1 trial: {key}")
    return keys


def _fold(root, study, entry, *, dataset, index, samples_per_subject, seed,
          variance_floor, quantile):
    task, held_subject = study["task"], int(entry["subject_id"])
    fold_dir = root / f"subject_{held_subject}"
    # This checks archive identity, source membership, and the frozen region.
    _, _, _, archive_hashes = analyze_fold(
        root, task, entry, samples_per_subject=0, seed=seed,
    )
    fold = json.loads((fold_dir / "fold.json").read_text())
    _, source = _load_summary(fold_dir / "calibration/source_trials", (
        "subject_ids", "trial_ids", "labels", "probabilities", "discrepancy",
    ))
    _, heldout = _load_summary(fold_dir / "observations", (
        "trial_ids", "labels", "probabilities", "discrepancy",
    ))
    source_subjects = np.asarray(source["subject_ids"], dtype=int)
    source_trials = np.asarray(source["trial_ids"], dtype=int)
    source_labels = np.asarray(source["labels"], dtype=int)
    source_probs = np.asarray(source["probabilities"], dtype=float)
    source_embedding_scores = np.asarray(source["discrepancy"], dtype=float)
    typical = (source_labels == 1) & (source_probs.argmax(axis=1) == 1) & (
        source_embedding_scores <= float(fold["threshold"])
    )
    source_selected, source_sampling = stratified_sample_indices(
        source_subjects, source_trials, typical,
        samples_per_subject=samples_per_subject, seed=seed,
        # Keep the role seed identical to the existing decoded-waveform report
        # so the same source trials are used when sampling is enabled.
        fold_subject=held_subject, role="source_correct_typical_class1_decoded_reference",
    )
    for item in source_sampling:
        item["seed_role"] = item["role"]
        item["role"] = "source_typical_class1_X"
    held_trials = np.asarray(heldout["trial_ids"], dtype=int)
    held_subjects = np.full(len(held_trials), held_subject, dtype=int)
    held_selected, held_sampling = stratified_sample_indices(
        held_subjects, held_trials, np.asarray(heldout["labels"], dtype=int) == 1,
        samples_per_subject=samples_per_subject, seed=seed,
        fold_subject=held_subject, role="heldout_true_class_1",
    )
    for item in held_sampling:
        item["seed_role"] = item["role"]
        item["role"] = "heldout_true_class1_X"
    source_keys = _trial_keys(source_subjects, source_trials, source_selected, index, dataset,
                              held_subject=held_subject, role="source")
    held_keys = _trial_keys(held_subjects, held_trials, held_selected, index, dataset,
                            held_subject=held_subject, role="heldout")
    archive_hashes["raw_source_selection"] = "true class 1; predicted class 1; archived embedding discrepancy <= archived threshold"
    if len(source_keys) < 2:
        return ({
            "task": task, "fold_subject": held_subject,
            "status": "insufficient_typical_source_class1_X",
            "n_source_typical_class1": len(source_keys),
            "n_source_subjects_with_typical_class1": len({key[0] for key in source_keys}),
            "n_heldout_true_class1": len(held_keys), "n_heldout_scored": 0,
            "n_heldout_inside_source_raw_x": 0, "n_cross_subject_comparisons": 0,
            "source_raw_x_threshold": None, "source_raw_x_median": None,
            "heldout_raw_x_median": None, "median_gap_over_raw_x_threshold": None,
            "heldout_inside_source_raw_x_percent": None,
            "heldout_median_source_raw_x_percentile": None,
            "cross_subject_probability_heldout_discrepancy_higher": None,
        }, [], [], source_sampling + held_sampling, archive_hashes)
    source_x = np.asarray(dataset.features[[index[key] for key in source_keys]], dtype=np.float32)
    held_x = np.asarray(dataset.features[[index[key] for key in held_keys]], dtype=np.float32)
    mean, variance = fit_region(source_x, variance_floor=variance_floor)
    source_scores = score_region(source_x, mean, variance)
    held_scores = score_region(held_x, mean, variance)
    threshold = float(np.quantile(source_scores, quantile, method="higher"))
    source_subject_array = np.asarray([key[0] for key in source_keys])
    source_rows = [{
        "task": task, "fold_subject": held_subject, "role": "source_typical_class1_X",
        "subject_id": key[0], "trial_id": key[1], "raw_x_discrepancy": float(value),
        "raw_x_threshold": threshold,
        "inside_raw_x_region": bool(value <= threshold),
    } for key, value in zip(source_keys, source_scores)]
    held_percentiles = _source_percentiles(held_scores, source_scores)
    held_rows = [{
        "task": task, "fold_subject": held_subject, "role": "heldout_true_class1_X",
        "subject_id": key[0], "trial_id": key[1], "raw_x_discrepancy": float(value),
        "raw_x_threshold": threshold,
        "inside_raw_x_region": bool(value <= threshold),
        "source_raw_x_percentile": float(percentile),
    } for key, value, percentile in zip(held_keys, held_scores, held_percentiles)]

    # Compare two subjects only against a reference fitted without either one.
    # This avoids treating the fitting scores as a held-out source baseline.
    crossfit_rows = []
    pair_probabilities = []
    for source_subject in sorted(set(source_subject_array)):
        fit_mask = source_subject_array != source_subject
        if np.count_nonzero(fit_mask) < 2 or not held_keys:
            continue
        inner_mean, inner_variance = fit_region(source_x[fit_mask], variance_floor=variance_floor)
        source_cv = score_region(source_x[~fit_mask], inner_mean, inner_variance)
        held_cv = score_region(held_x, inner_mean, inner_variance)
        pair_probabilities.append(_probability_higher(held_cv, source_cv))
        for key, value in zip(np.asarray(source_keys, dtype=int)[~fit_mask], source_cv):
            crossfit_rows.append({
                "task": task, "fold_subject": held_subject,
                "reference_excluded_source_subject": int(source_subject),
                "role": "source_typical_class1_X", "subject_id": int(key[0]),
                "trial_id": int(key[1]), "raw_x_discrepancy": float(value),
            })
        for key, value in zip(held_keys, held_cv):
            crossfit_rows.append({
                "task": task, "fold_subject": held_subject,
                "reference_excluded_source_subject": int(source_subject),
                "role": "heldout_true_class1_X", "subject_id": key[0],
                "trial_id": key[1], "raw_x_discrepancy": float(value),
            })
    row = {
        "task": task, "fold_subject": held_subject,
        "status": "scored",
        "n_source_typical_class1": len(source_keys),
        "n_source_subjects_with_typical_class1": len(set(source_subject_array)),
        "n_heldout_true_class1": len(held_keys),
        "n_heldout_scored": len(held_keys),
        "n_heldout_inside_source_raw_x": int(np.count_nonzero(held_scores <= threshold)),
        "n_cross_subject_comparisons": len(pair_probabilities),
        "source_raw_x_threshold": threshold,
        "source_raw_x_median": _median(source_scores),
        "heldout_raw_x_median": _median(held_scores),
        "median_gap_over_raw_x_threshold": (
            (_median(held_scores) - _median(source_scores)) / threshold
            if held_keys and threshold > 0 else None
        ),
        "heldout_inside_source_raw_x_percent": (
            float(100 * np.mean(held_scores <= threshold)) if held_keys else None
        ),
        "heldout_median_source_raw_x_percentile": _median(held_percentiles),
        "cross_subject_probability_heldout_discrepancy_higher": _median(pair_probabilities),
    }
    return row, source_rows + held_rows, crossfit_rows, source_sampling + held_sampling, archive_hashes


def build_report(roots, output, *, raw_eeg, raw_labels, samples_per_subject=3,
                 seed=42, variance_floor=1e-6, quantile=0.95):
    if (isinstance(samples_per_subject, bool) or int(samples_per_subject) != samples_per_subject
            or samples_per_subject < 0 or not 0 < quantile < 1):
        raise ValueError("Invalid sampling count or source quantile")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {output}")
    folds, trials, crossfit, sampling, hashes = [], [], [], [], []
    seen, datasets = set(), {}
    for root in _roots(roots):
        study_path = root / "study.json"
        study = json.loads(study_path.read_text())
        task = study["task"]
        if task not in datasets:
            dataset, index = _dataset_for_task(study, raw_eeg=raw_eeg, raw_labels=raw_labels)
            datasets[task] = (dataset, index, study["dataset_sha256"])
        dataset, index, fingerprints = datasets[task]
        if study["dataset_sha256"] != fingerprints:
            raise ValueError(f"Different prepared EEG datasets in {task} study shards")
        for entry in study["folds"]:
            key = task, int(entry["subject_id"])
            if key in seen:
                raise ValueError(f"Duplicate task/held-out subject: {key}")
            seen.add(key)
            row, scores, cv_scores, selected, inputs = _fold(
                root, study, entry, dataset=dataset, index=index,
                samples_per_subject=samples_per_subject, seed=seed,
                variance_floor=variance_floor, quantile=quantile,
            )
            folds.append(row)
            trials.extend(scores)
            crossfit.extend(cv_scores)
            sampling.extend(selected)
            hashes.append({**inputs, "study_json_sha256": file_sha256(study_path)})
    if not folds:
        raise ValueError("No LOSO folds in supplied studies")
    aggregates = []
    for task in sorted({row["task"] for row in folds}):
        group = [row for row in folds if row["task"] == task]
        n_heldout = sum(row["n_heldout_true_class1"] for row in group)
        n_scored = sum(row["n_heldout_scored"] for row in group)
        n_heldout_inside = sum(row["n_heldout_inside_source_raw_x"] for row in group)
        def med(field):
            return _median([row[field] for row in group if row[field] is not None])
        aggregates.append({
            "task": task, "n_folds": len(group),
            "n_scored_folds": sum(row["status"] == "scored" for row in group),
            "unscored_subject_ids": [row["fold_subject"] for row in group if row["status"] != "scored"],
            "n_folds_with_heldout_class1": sum(row["n_heldout_true_class1"] > 0 for row in group),
            "n_source_typical_class1": sum(row["n_source_typical_class1"] for row in group),
            "n_heldout_true_class1": n_heldout,
            "n_heldout_scored": n_scored,
            "n_heldout_inside_source_raw_x": n_heldout_inside,
            "pooled_heldout_inside_source_raw_x_percent": (
                100 * n_heldout_inside / n_scored if n_scored else None
            ),
            "median_fold_heldout_inside_source_raw_x_percent": med("heldout_inside_source_raw_x_percent"),
            "median_fold_raw_x_discrepancy_gap_over_threshold": med("median_gap_over_raw_x_threshold"),
            "median_fold_cross_subject_probability_heldout_higher": med(
                "cross_subject_probability_heldout_discrepancy_higher"),
        })
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "raw_x_trials.csv", trials)
    write_csv(output / "raw_x_cross_subject_scores.csv", crossfit)
    write_csv(output / "raw_x_folds.csv", folds)
    write_csv(output / "raw_x_aggregate.csv", aggregates)
    write_json(output / "raw_x_sampling.json", sampling)
    report = {
        "schema_version": 1,
        "waveform_space": "original prepared model-input EEG X; no decoder or re-encoding",
        "formula": "mean((X - source_mean)^2 / max(source_variance, variance_floor))",
        "source_reference": "source subjects' real, true class-1, correctly predicted, embedding-typical X",
        "heldout_selection": "all true class-1 X, regardless of prediction or embedding discrepancy",
        "threshold": "source raw-X discrepancy quantile; source scores used for the full-reference threshold are in-sample",
        "cross_subject_comparison": "for each source subject j, score its typical X and held-out X against the same raw-X reference fitted without j; report the median across j of P(D_heldout > D_source_j), ties 0.5",
        "quantile": float(quantile), "variance_floor": float(variance_floor),
        "samples_per_subject": int(samples_per_subject), "seed": int(seed),
        "raw_eeg_sha256": file_sha256(raw_eeg),
        "raw_labels_sha256": file_sha256(raw_labels),
        "aggregate": aggregates, "input_hashes": hashes,
        "limitations": [
            "The full-source 95th-percentile threshold is descriptive because the source EEG used to fit the raw-X reference also determines it; use cross-subject comparison as the less biased baseline.",
            "Pointwise waveform discrepancy is sensitive to temporal phase and alignment.",
            "Raw-X discrepancy and latent-Z or decoded-waveform discrepancy have different reference spaces and cannot be compared numerically without separate calibration.",
            "Archival source-typical selection depends on the frozen source-trained classifier; held-out selection depends only on true class.",
        ],
    }
    write_json(output / "raw_x_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+",
                        help="Study roots, a parent directory, or a report directory with report.json")
    parser.add_argument("--raw-eeg", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-subject", type=int, default=3,
                        help="3 (default) matches the decoded-waveform report; 0 uses every eligible trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance-floor", type=float, default=1e-6)
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args(argv)
    try:
        build_report(
            args.roots, args.out_dir, raw_eeg=args.raw_eeg, raw_labels=args.raw_labels,
            samples_per_subject=args.samples_per_subject, seed=args.seed,
            variance_floor=args.variance_floor, quantile=args.quantile,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Raw-X subject-invariance report: {error}\n")
    print(f"Wrote raw-X subject-invariance report to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
