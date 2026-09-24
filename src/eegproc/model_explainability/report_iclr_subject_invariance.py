"""Compare real class-1 trials with source-trained class-1 regions in LOSO folds.

Each held-out subject is scored only in its own fold's frozen classifier space.
The source subjects supplied that fold's learned class-1 Gaussian; held-out
trials did not. This analysis reads saved real-trial embeddings and never loads
a checkpoint or runs counterfactual optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SCORE_DEFINITION = "full_trial_diagonal_squared_mahalanobis_per_dimension"
REPRESENTATION = "vc_trial_embedding"


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
            field for row in rows for field in row
        )))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path, required):
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            f"Missing real-trial archive {path}; this analysis needs "
            "calibration/source_trials.npz and observations.npz from typicality.runner."
        )
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def _validate_predictions(labels, probabilities, discrepancies, *, name):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    discrepancies = np.asarray(discrepancies, dtype=float)
    if labels.ndim != 1 or discrepancies.shape != labels.shape or probabilities.shape != (len(labels), 2):
        raise ValueError(f"{name} labels, predictions, and scores must align")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(probabilities).all():
        raise ValueError(f"{name} labels or probabilities are invalid")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError(f"{name} probabilities must be nonnegative and sum to one")
    if not np.isfinite(discrepancies).all() or np.any(discrepancies < 0):
        raise ValueError(f"{name} discrepancies must be finite and nonnegative")
    return labels, probabilities, discrepancies


def _stable_seed(seed, *parts):
    value = "|".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def stratified_sample_indices(subject_ids, trial_ids, eligible, *, samples_per_subject, seed,
                              fold_subject, role):
    subjects = np.asarray(subject_ids, dtype=int)
    trials = np.asarray(trial_ids, dtype=int)
    eligible = np.asarray(eligible, dtype=bool)
    if subjects.shape != trials.shape or subjects.shape != eligible.shape or subjects.ndim != 1:
        raise ValueError("Subject IDs, trial IDs, and eligibility must align")
    if len(set(zip(subjects.tolist(), trials.tolist()))) != len(subjects):
        raise ValueError("Duplicate subject/trial keys")
    selected, manifest = [], []
    for subject in sorted(np.unique(subjects).tolist()):
        available = np.flatnonzero(eligible & (subjects == subject))
        available = available[np.argsort(trials[available], kind="stable")]
        subject_seed = _stable_seed(seed, fold_subject, role, subject)
        ordered = available[np.random.default_rng(subject_seed).permutation(len(available))]
        count = len(ordered) if samples_per_subject == 0 else min(samples_per_subject, len(ordered))
        chosen = ordered[:count]
        selected.extend(chosen.tolist())
        manifest.append({
            "fold_subject": int(fold_subject), "role": role, "subject_id": int(subject),
            "seed": subject_seed, "n_available_true_class_1": int(len(available)),
            "n_selected": int(count),
            "shortfall": max(0, samples_per_subject - count) if samples_per_subject else 0,
            "selected_trial_ids": sorted(trials[chosen].astype(int).tolist()),
        })
    selected = np.asarray(selected, dtype=int)
    if len(selected):
        selected = selected[np.lexsort((trials[selected], subjects[selected]))]
    return selected, manifest


class TypicalityRegion:
    """Read the frozen Gaussian and independently verify archived scores."""

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        if not (directory / "region.json").is_file() or not (directory / "region.npz").is_file():
            raise ValueError(f"Missing learned class-1 region in {directory}; this analysis "
                             "needs the real-trial calibration archives from typicality.runner")
        metadata = json.loads((directory / "region.json").read_text())
        if (metadata.get("schema_version") != 2 or metadata.get("definition") != SCORE_DEFINITION
                or metadata.get("representation") != REPRESENTATION):
            raise ValueError(f"Incompatible typicality region in {directory}")
        with np.load(directory / "region.npz", allow_pickle=False) as archive:
            region = cls()
            region.prior_mean = np.asarray(archive["prior_mean"], dtype=float)
            region.prior_variance = np.asarray(archive["prior_variance"], dtype=float)
            region.tau = float(archive["tau"])
            region.variance_floor = float(archive["variance_floor"])
        if (region.prior_mean.ndim != 1 or region.prior_variance.shape != region.prior_mean.shape
                or not np.isfinite(region.prior_mean).all()
                or not np.isfinite(region.prior_variance).all()
                or np.any(region.prior_variance < 0)
                or not np.isfinite(region.tau) or region.tau < 0
                or not np.isfinite(region.variance_floor) or region.variance_floor <= 0):
            raise ValueError(f"Invalid learned class-1 region in {directory}")
        region.metadata = metadata
        return region

    def score(self, embeddings):
        values = np.asarray(embeddings, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.prior_mean):
            raise ValueError("Embedding shape does not match learned class-1 Gaussian")
        return np.mean(np.square(values - self.prior_mean) /
                       np.maximum(self.prior_variance, self.variance_floor), axis=1)


def compatible_typicality_definition(roots):
    definitions = set()
    for root in roots:
        study = json.loads((Path(root) / "study.json").read_text())
        definitions.add((study.get("typicality_definition"), study.get("typicality_representation")))
    if len(definitions) != 1:
        raise ValueError("Cannot combine different typicality definitions or representations")
    score, representation = definitions.pop()
    return {"score": score, "representation": representation}


def _median(values):
    return float(np.median(values)) if len(values) else None


def _percent_inside(values, tau):
    return float(100 * np.mean(np.asarray(values) <= tau)) if len(values) else None


def _probability_higher(held, source):
    """P(D_held > D_source), counting ties as one half."""
    if not len(held) or not len(source):
        return None
    comparisons = np.subtract.outer(held, source)
    return float((np.count_nonzero(comparisons > 0) +
                  0.5 * np.count_nonzero(comparisons == 0)) / comparisons.size)


def _source_percentiles(held, source):
    """Midrank empirical source-CDF percentile for every held-out score."""
    if not len(source):
        return None
    ordered = np.sort(source)
    return 100 * (np.searchsorted(ordered, held, side="left") +
                  0.5 * (np.searchsorted(ordered, held, side="right") -
                         np.searchsorted(ordered, held, side="left"))) / len(ordered)


def _bootstrap_median_interval(values, *, seed, replicates):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    samples = np.random.default_rng(seed).choice(values, size=(replicates, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(float(value) for value in np.quantile(medians, [0.025, 0.975]))


def _check_scores(archive, region, *, name):
    labels, probabilities, scores = _validate_predictions(
        archive["labels"], archive["probabilities"], archive["discrepancy"], name=name,
    )
    embeddings = np.asarray(archive["embeddings"], dtype=float)
    if embeddings.shape != (len(labels), len(region.prior_mean)):
        raise ValueError(f"{name} embeddings do not match the learned class-1 distribution")
    if not np.allclose(region.score(embeddings), scores, rtol=1e-5, atol=1e-7):
        raise ValueError(f"{name} discrepancies do not match the learned class-1 distribution")
    return labels, probabilities, scores


def _trial_rows(task, fold_subject, role, indices, subjects, trials, labels, probabilities, scores, tau,
                percentiles=None):
    return [
        {
            "task": task,
            "fold_subject": fold_subject,
            "role": role,
            "subject_id": int(subjects[index]),
            "trial_id": int(trials[index]),
            "true_class": int(labels[index]),
            "predicted_class": int(probabilities[index].argmax()),
            "discrepancy": float(scores[index]),
            "source_threshold": tau,
            "discrepancy_over_threshold": float(scores[index] / tau) if tau > 0 else None,
            "inside_source_region": bool(scores[index] <= tau),
            "source_empirical_percentile": float(percentiles[position]) if percentiles is not None else None,
        }
        for position, index in enumerate(indices)
    ]


def analyze_fold(root, task, fold_entry, *, samples_per_subject=3, seed=42):
    """Sample all true class-1 trials independently of prediction and score."""
    root = Path(root)
    held_subject = int(fold_entry["subject_id"])
    fold_dir = root / f"subject_{held_subject}"
    fold_path = fold_dir / "fold.json"
    if not fold_path.is_file():
        raise ValueError(f"Missing LOSO fold metadata: {fold_path}")
    fold = json.loads(fold_path.read_text())
    if int(fold["subject_id"]) != held_subject:
        raise ValueError(f"Fold subject ID mismatch in {fold_path}")
    checkpoint = fold.get("checkpoint", {})
    if checkpoint.get("stage") != "zero_shot_source_model":
        raise ValueError(f"Fold {held_subject} does not declare a zero-shot source-trained checkpoint")
    region_path = fold_dir / "calibration"
    region = TypicalityRegion.load(region_path)
    if int(region.metadata["held_out_subject"]) != held_subject:
        raise ValueError(f"Region does not belong to held-out subject {held_subject}")
    tau = region.tau
    if not np.isclose(float(fold["threshold"]), tau):
        raise ValueError(f"Fold and region thresholds differ for subject {held_subject}")

    source_path = region_path / "source_trials.npz"
    held_path = fold_dir / "observations.npz"
    source = _load_npz(source_path, (
        "subject_ids", "trial_ids", "labels", "probabilities", "discrepancy", "embeddings",
    ))
    held = _load_npz(held_path, (
        "trial_ids", "labels", "probabilities", "discrepancy", "embeddings",
    ))
    slabels, sprobs, sscores = _check_scores(source, region, name="source")
    hlabels, hprobs, hscores = _check_scores(held, region, name="held-out")
    ssubjects = np.asarray(source["subject_ids"], dtype=int)
    strials = np.asarray(source["trial_ids"], dtype=int)
    htrials = np.asarray(held["trial_ids"], dtype=int)
    if ssubjects.shape != slabels.shape or strials.shape != slabels.shape or htrials.shape != hlabels.shape:
        raise ValueError("Trial IDs and subject IDs must align with scores")
    if np.any(ssubjects == held_subject):
        raise ValueError("Held-out subject leaked into the source reference")
    if set(np.unique(ssubjects).tolist()) != set(region.metadata["source_subject_ids"]):
        raise ValueError("Source subjects differ from the region's calibration subjects")
    if set(ssubjects.tolist()) != set(checkpoint.get("source_subject_ids", [])):
        raise ValueError("Source subjects differ from the checkpoint manifest")
    hsubjects = np.full(len(hlabels), held_subject, dtype=int)

    source_indices, source_sampling = stratified_sample_indices(
        ssubjects, strials, slabels == 1, samples_per_subject=samples_per_subject,
        seed=seed, fold_subject=held_subject, role="source_true_class_1",
    )
    held_indices, held_sampling = stratified_sample_indices(
        hsubjects, htrials, hlabels == 1, samples_per_subject=samples_per_subject,
        seed=seed, fold_subject=held_subject, role="heldout_true_class_1",
    )
    control_indices = np.flatnonzero(hlabels == 0)
    control_indices = control_indices[np.argsort(htrials[control_indices], kind="stable")]
    source_scores, held_scores = sscores[source_indices], hscores[held_indices]
    control_scores = hscores[control_indices]
    source_cdf_scores = sscores[slabels == 1]
    held_percentiles = _source_percentiles(held_scores, source_cdf_scores)
    control_percentiles = _source_percentiles(control_scores, source_cdf_scores)
    source_median, held_median = _median(source_scores), _median(held_scores)
    source_coverage, held_coverage = (_percent_inside(scores, tau)
                                      for scores in (source_scores, held_scores))
    row = {
        "task": task,
        "fold_subject": held_subject,
        "source_threshold": tau,
        "n_source_subjects": len(source_sampling),
        "n_source_subjects_with_true1": sum(item["n_available_true_class_1"] > 0
                                            for item in source_sampling),
        "n_source_true1_available": int(np.count_nonzero(slabels == 1)),
        "n_source_cdf_trials": len(source_cdf_scores),
        "n_source_true1_sampled": len(source_indices),
        "n_heldout_true1_available": int(np.count_nonzero(hlabels == 1)),
        "n_heldout_true1_sampled": len(held_indices),
        "n_heldout_class0_negative_control": len(control_indices),
        "source_median_discrepancy": source_median,
        "heldout_median_discrepancy": held_median,
        "median_discrepancy_gap": held_median - source_median
        if source_median is not None and held_median is not None else None,
        "source_median_over_threshold": source_median / tau if source_median is not None and tau > 0 else None,
        "heldout_median_over_threshold": held_median / tau if held_median is not None and tau > 0 else None,
        "median_gap_over_threshold": (held_median - source_median) / tau
        if source_median is not None and held_median is not None and tau > 0 else None,
        "source_inside_percent": source_coverage,
        "heldout_inside_percent": held_coverage,
        "heldout_minus_source_inside_pp": held_coverage - source_coverage
        if source_coverage is not None and held_coverage is not None else None,
        "probability_heldout_discrepancy_higher": _probability_higher(held_scores, source_scores),
        "heldout_median_source_percentile": _median(held_percentiles)
        if held_percentiles is not None else None,
        "heldout_class0_inside_percent": _percent_inside(control_scores, tau),
        "heldout_class0_median_source_percentile": _median(control_percentiles)
        if control_percentiles is not None else None,
    }
    trial_rows = _trial_rows(
        task, held_subject, "source", source_indices, ssubjects, strials,
        slabels, sprobs, sscores, tau,
    ) + _trial_rows(
        task, held_subject, "heldout", held_indices, hsubjects, htrials,
        hlabels, hprobs, hscores, tau, percentiles=held_percentiles,
    ) + _trial_rows(
        task, held_subject, "heldout_class0_negative_control", control_indices, hsubjects, htrials,
        hlabels, hprobs, hscores, tau, percentiles=control_percentiles,
    )
    return row, trial_rows, source_sampling + held_sampling, {
        "study_root": str(root.resolve()),
        "task": task,
        "fold_subject": held_subject,
        "fold_json_sha256": file_sha256(fold_path),
        "region_json_sha256": file_sha256(region_path / "region.json"),
        "region_npz_sha256": file_sha256(region_path / "region.npz"),
        "source_trials_npz_sha256": file_sha256(source_path),
        "observations_npz_sha256": file_sha256(held_path),
    }


def _aggregate(rows, *, seed, bootstrap_replicates):
    def mean(field):
        values = [row[field] for row in rows if row[field] is not None]
        return float(np.mean(values)) if values else None

    gaps = [row["median_gap_over_threshold"] for row in rows
            if row["median_gap_over_threshold"] is not None]
    coverage = [row["heldout_inside_percent"] for row in rows
                if row["heldout_inside_percent"] is not None]
    coverage_gaps = [row["heldout_minus_source_inside_pp"] for row in rows
                     if row["heldout_minus_source_inside_pp"] is not None]
    percentiles = [row["heldout_median_source_percentile"] for row in rows
                   if row["heldout_median_source_percentile"] is not None]
    coverage_low, coverage_high = _bootstrap_median_interval(
        coverage, seed=_stable_seed(seed, rows[0]["task"], "coverage"), replicates=bootstrap_replicates,
    )
    percentile_low, percentile_high = _bootstrap_median_interval(
        percentiles, seed=_stable_seed(seed, rows[0]["task"], "percentile"), replicates=bootstrap_replicates,
    )
    return {
        "task": rows[0]["task"],
        "n_folds": len(rows),
        "n_comparable_folds": len(gaps),
        "n_heldout_scored_folds": len(coverage),
        "median_fold_gap_over_threshold": _median(gaps),
        "q25_fold_gap_over_threshold": float(np.quantile(gaps, 0.25)) if gaps else None,
        "q75_fold_gap_over_threshold": float(np.quantile(gaps, 0.75)) if gaps else None,
        "n_folds_heldout_median_higher": sum(gap > 0 for gap in gaps),
        "source_inside_macro_percent": mean("source_inside_percent"),
        "heldout_inside_macro_percent": mean("heldout_inside_percent"),
        "heldout_minus_source_inside_macro_pp": mean("heldout_minus_source_inside_pp"),
        "heldout_minus_source_inside_subject_median_pp": _median(coverage_gaps),
        "probability_heldout_discrepancy_higher_macro": mean("probability_heldout_discrepancy_higher"),
        "heldout_class0_inside_macro_percent": mean("heldout_class0_inside_percent"),
        "heldout_coverage_subject_median_percent": _median(coverage),
        "heldout_coverage_subject_q25_percent": float(np.quantile(coverage, 0.25)) if coverage else None,
        "heldout_coverage_subject_q75_percent": float(np.quantile(coverage, 0.75)) if coverage else None,
        "heldout_coverage_subject_bootstrap95_low_percent": coverage_low,
        "heldout_coverage_subject_bootstrap95_high_percent": coverage_high,
        "heldout_source_percentile_subject_median": _median(percentiles),
        "heldout_source_percentile_subject_q25": float(np.quantile(percentiles, 0.25)) if percentiles else None,
        "heldout_source_percentile_subject_q75": float(np.quantile(percentiles, 0.75)) if percentiles else None,
        "heldout_source_percentile_subject_bootstrap95_low": percentile_low,
        "heldout_source_percentile_subject_bootstrap95_high": percentile_high,
    }


def _paper_paragraph(aggregates):
    lines = [r"\paragraph{Subject-invariance.}"]
    for row in aggregates:
        n = row["n_heldout_scored_folds"]
        task = row["task"].capitalize()
        if not n:
            lines.append(f"For {task}, no held-out real class-1 trials were available for this analysis.")
            continue
        coverage = row["heldout_coverage_subject_median_percent"]
        low = row["heldout_coverage_subject_bootstrap95_low_percent"]
        high = row["heldout_coverage_subject_bootstrap95_high_percent"]
        percentile = row["heldout_source_percentile_subject_median"]
        percentile_q25 = row["heldout_source_percentile_subject_q25"]
        percentile_q75 = row["heldout_source_percentile_subject_q75"]
        lines.append(
            f"For {task} ({n} held-out subjects), the subject-median fraction of sampled real "
            f"class-1 trials inside the source-trained class-1 region was {coverage:.1f}\\% "
            f"(subject-bootstrap 95\\% CI [{low:.1f}, {high:.1f}]\\%). "
            f"The subject-median empirical source-discrepancy percentile was {percentile:.1f} "
            f"(IQR [{percentile_q25:.1f}, {percentile_q75:.1f}])."
        )
    lines.append("These within-fold comparisons describe cross-subject class-1 compatibility; "
                 "they do not establish full subject-invariant representations.")
    return "\n".join(lines) + "\n"


def build_subject_invariance_report(roots, output, *, samples_per_subject=3, seed=42,
                                    bootstrap_replicates=10000):
    """Write reproducible trial, fold, and task summaries from LOSO archives."""
    study_roots = []
    for root in map(Path, roots):
        if (root / "study.json").is_file():
            study_roots.append(root)
        else:
            study_roots.extend(sorted({path.parent for path in root.rglob("study.json")}))
    if not study_roots:
        raise ValueError("No study.json found under the supplied input directories")
    definition = compatible_typicality_definition(study_roots)
    if definition != {"score": SCORE_DEFINITION, "representation": REPRESENTATION}:
        raise ValueError("Subject-invariance analysis requires full-trial class-1 Mahalanobis archives")
    if isinstance(samples_per_subject, bool) or int(samples_per_subject) != samples_per_subject or samples_per_subject < 0:
        raise ValueError("samples_per_subject must be a nonnegative integer")
    if (isinstance(bootstrap_replicates, bool) or int(bootstrap_replicates) != bootstrap_replicates
            or bootstrap_replicates < 1):
        raise ValueError("bootstrap_replicates must be a positive integer")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {output}")
    folds, trials, sampling, hashes = [], [], [], []
    seen = set()
    for root in study_roots:
        study_path = root / "study.json"
        study = json.loads(study_path.read_text())
        task = study["task"]
        for entry in study["folds"]:
            key = (task, int(entry["subject_id"]))
            if key in seen:
                raise ValueError(f"Duplicate task/fold across study roots: {key}")
            seen.add(key)
            fold, real, selected, input_hashes = analyze_fold(
                root, task, entry, samples_per_subject=samples_per_subject, seed=seed,
            )
            folds.append(fold)
            trials.extend(real)
            sampling.extend(selected)
            hashes.append({**input_hashes, "study_json_sha256": file_sha256(study_path)})
    if not folds:
        raise ValueError("No LOSO folds found in the supplied studies")
    aggregates = [_aggregate([row for row in folds if row["task"] == task],
                             seed=seed, bootstrap_replicates=bootstrap_replicates)
                  for task in sorted({row["task"] for row in folds})]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "subject_invariance_trials.csv", trials)
    write_csv(output / "subject_invariance_folds.csv", folds)
    write_csv(output / "subject_invariance_aggregate.csv", aggregates)
    write_json(output / "subject_invariance_sampling.json", sampling)
    (output / "subject_invariance_paragraph.tex").write_text(_paper_paragraph(aggregates))
    report = {
        "schema_version": 1,
        "typicality_definition": definition,
        "selection": "true class 1, irrespective of predicted class or discrepancy",
        "comparison": "within each LOSO fold: held-out real trials versus sampled source real trials scored against that fold's frozen learned class-1 Gaussian",
        "negative_control": "all held-out true class-0 trials are reported separately and never enter the class-1 estimates",
        "source_empirical_percentile_reference": "all available source true class-1 trials in the matching fold",
        "normalization": "discrepancy divided by that fold's source-calibrated threshold; raw scores are not pooled across models",
        "samples_per_subject": int(samples_per_subject),
        "seed": int(seed),
        "bootstrap_unit": "held-out subject (one LOSO fold)",
        "bootstrap_replicates": int(bootstrap_replicates),
        "aggregate": aggregates,
        "input_hashes": hashes,
        "limitations": [
            "Source trials were used to train the model; held-out trials were not.",
            "Independent LOSO models have different latent coordinate systems; only within-fold contrasts are aggregated.",
            "Bootstrap intervals resample held-out subjects; overlapping LOSO training sets make them descriptive rather than independent-model guarantees.",
            "This descriptive compatibility test alone does not prove subject-invariant representations.",
        ],
    }
    write_json(output / "subject_invariance.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+",
                        help="typicality.runner study roots or a parent containing task/fold shards")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-subject", type=int, default=3,
                        help="Real class-1 trials per subject; 0 uses all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        build_subject_invariance_report(
            args.roots, args.out_dir, samples_per_subject=args.samples_per_subject, seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Subject-invariance report: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
