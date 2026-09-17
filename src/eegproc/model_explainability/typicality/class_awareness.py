"""Offline class-1 typicality and subject-invariance audit.

The audit consumes completed :mod:`typicality.runner` archives. It never
imports TensorFlow or reloads a checkpoint. The generation threshold is
retained, while an independent audit threshold is computed from a reproducible
subject-stratified sample of source trials satisfying ``y == 1`` and
``argmax(probabilities) == 1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import file_sha256, write_csv, write_json
from .results import _latest_result


def _validate_predictions(labels, probabilities, discrepancies, *, name):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    discrepancies = np.asarray(discrepancies, dtype=float)
    if labels.ndim != 1 or discrepancies.shape != labels.shape:
        raise ValueError(f"{name} labels and discrepancies must be aligned vectors")
    if probabilities.shape != (len(labels), 2):
        raise ValueError(f"{name} probabilities must be shaped (trials, 2)")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{name} labels must be binary")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError(f"{name} probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{name} probabilities must sum to one")
    if not np.isfinite(discrepancies).all() or np.any(discrepancies < 0):
        raise ValueError(f"{name} discrepancies must be finite and nonnegative")
    return labels, probabilities, discrepancies


def correct_class_one_mask(labels, probabilities):
    """Select trials whose true and argmax-predicted classes are both one."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), 2):
        raise ValueError("probabilities must be shaped (trials, 2)")
    return (labels == 1) & (probabilities.argmax(axis=1) == 1)


def _stable_seed(global_seed, *parts):
    payload = "|".join((str(int(global_seed)), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def stratified_sample_indices(
    subject_ids,
    trial_ids,
    eligible,
    *,
    samples_per_subject,
    seed,
    fold_subject,
    role,
):
    """Sample eligible trials per subject, independently of discrepancy."""
    subjects = np.asarray(subject_ids, dtype=int)
    trials = np.asarray(trial_ids, dtype=int)
    eligible = np.asarray(eligible, dtype=bool)
    if subjects.shape != trials.shape or subjects.shape != eligible.shape or subjects.ndim != 1:
        raise ValueError("subject_ids, trial_ids, and eligible must be aligned vectors")
    if len(set(zip(subjects.tolist(), trials.tolist()))) != len(subjects):
        raise ValueError("Trial keys must be unique")
    if isinstance(samples_per_subject, bool) or int(samples_per_subject) != samples_per_subject:
        raise ValueError("samples_per_subject must be a nonnegative integer")
    samples_per_subject = int(samples_per_subject)
    if samples_per_subject < 0:
        raise ValueError("samples_per_subject must be a nonnegative integer")

    selected = []
    manifest = []
    # Include subjects with no eligible trials so shortfalls are explicit.
    for subject in sorted(np.unique(subjects).tolist()):
        available = np.flatnonzero(eligible & (subjects == subject))
        available = available[np.argsort(trials[available], kind="stable")]
        subject_seed = _stable_seed(seed, fold_subject, role, subject)
        order = np.random.default_rng(subject_seed).permutation(len(available))
        take = len(available) if samples_per_subject == 0 else min(samples_per_subject, len(available))
        chosen = available[order[:take]]
        selected.extend(chosen.tolist())
        manifest.append(
            {
                "fold_subject": int(fold_subject),
                "role": role,
                "subject_id": int(subject),
                "seed": subject_seed,
                "n_available_correct_class_1": int(len(available)),
                "n_selected": int(len(chosen)),
                "shortfall": int(max(0, samples_per_subject - len(chosen)))
                if samples_per_subject
                else 0,
                "selected_trial_ids": sorted(trials[chosen].astype(int).tolist()),
            }
        )
    selected = np.asarray(selected, dtype=int)
    if len(selected):
        selected = selected[np.lexsort((trials[selected], subjects[selected]))]
    return selected, manifest


def _higher_quantile(values, quantile):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite source correct-class-1 scores are required")
    if not math.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    return float(np.quantile(values, quantile, method="higher"))


def _coverage(values, threshold):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or (len(values) and not np.isfinite(values).all()):
        raise ValueError("Coverage values must be a finite vector")
    n = int(len(values))
    if not n:
        return {
            "n": 0,
            "inside": 0,
            "percent": None,
            "wilson95_low_percent": None,
            "wilson95_high_percent": None,
        }
    inside = int(np.count_nonzero(values <= threshold))
    proportion = inside / n
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return {
        "n": n,
        "inside": inside,
        "percent": 100.0 * proportion,
        "wilson95_low_percent": 100.0 * max(0.0, center - half),
        "wilson95_high_percent": 100.0 * min(1.0, center + half),
    }


def _prefixed(prefix, values):
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _real_rows(
    *, task, fold_subject, role, indices, subject_ids, trial_ids, labels,
    probabilities, discrepancies, audit_tau, generation_tau,
):
    rows = []
    for index in indices:
        discrepancy = float(discrepancies[index])
        rows.append(
            {
                "task": task,
                "fold_subject": int(fold_subject),
                "role": role,
                "subject_id": int(subject_ids[index]),
                "trial_id": int(trial_ids[index]),
                "true_class": int(labels[index]),
                "predicted_class": int(probabilities[index].argmax()),
                "p_class_1": float(probabilities[index, 1]),
                "discrepancy": discrepancy,
                "audit_tau_correct1": audit_tau,
                "generation_tau_all_true1": generation_tau,
                "inside_audit_region": bool(discrepancy <= audit_tau),
                "inside_generation_region": bool(discrepancy <= generation_tau),
                "audit_margin": discrepancy - audit_tau,
                "audit_ratio": discrepancy / audit_tau if audit_tau > 0 else None,
            }
        )
    return rows


def _counterfactual_rows(*, root, task, fold, audit_tau):
    fold_subject = int(fold["subject_id"])
    generation_tau = float(fold["threshold"])
    fold_dir = Path(root) / f"subject_{fold_subject}"
    rows = []
    for trial in fold.get("eligible_trial_ids", []):
        for objective in ("base", "typicality"):
            row = {
                "task": task,
                "fold_subject": fold_subject,
                "trial_id": int(trial),
                "objective": objective,
                "status": "pending",
                "generation_tau_all_true1": generation_tau,
                "audit_tau_correct1": audit_tau,
            }
            attempt = _latest_result(fold_dir / f"trial_{trial}" / objective)
            if attempt is None:
                rows.append(row)
                continue
            summary = json.loads((attempt / "result.json").read_text())
            row["status"] = summary.get("status", "error")
            row["artifact_directory"] = str(attempt.resolve())
            if row["status"] != "completed":
                row["error"] = summary.get("error")
                rows.append(row)
                continue
            typicality = summary["typicality"]
            original_d = float(typicality["original_discrepancy"])
            counterfactual_d = float(typicality["counterfactual_discrepancy"])
            if not np.isfinite([original_d, counterfactual_d]).all():
                raise ValueError(f"Non-finite counterfactual discrepancy in {attempt}")
            original_inside = original_d <= audit_tau
            counterfactual_inside = counterfactual_d <= audit_tau
            transition = (
                "preserved_inside" if original_inside and counterfactual_inside
                else "exited" if original_inside
                else "entered" if counterfactual_inside
                else "stayed_outside"
            )
            latent_target_success = bool(summary["latent_counterfactual"]["success"])
            row.update(
                original_discrepancy=original_d,
                counterfactual_discrepancy=counterfactual_d,
                discrepancy_change=counterfactual_d - original_d,
                original_inside_audit_region=bool(original_inside),
                counterfactual_inside_audit_region=bool(counterfactual_inside),
                transition=transition,
                entered_audit_region=transition == "entered",
                preserved_inside_audit_region=transition == "preserved_inside",
                exited_audit_region=transition == "exited",
                counterfactual_audit_margin=counterfactual_d - audit_tau,
                counterfactual_audit_ratio=counterfactual_d / audit_tau if audit_tau > 0 else None,
                latent_target_success=latent_target_success,
                audit_typicality_success=bool(latent_target_success and counterfactual_inside),
                generation_typical=bool(typicality["typical"]),
            )
            rows.append(row)
    return rows


def _counterfactual_summary(rows):
    completed = [row for row in rows if row["status"] == "completed"]
    n_attempted, n_scored = len(rows), len(completed)

    def percent(field, denominator):
        return (100.0 * sum(bool(row.get(field, False)) for row in completed) / denominator
                if denominator else None)

    latent_successes = [row for row in completed if row.get("latent_target_success")]
    return {
        "n_attempted": n_attempted,
        "n_completed": n_scored,
        "n_error": sum(row["status"] == "error" for row in rows),
        "n_pending": sum(row["status"] == "pending" for row in rows),
        "typical_percent_all_attempts": percent("counterfactual_inside_audit_region", n_attempted),
        "typical_percent_scored": percent("counterfactual_inside_audit_region", n_scored),
        "entered_percent_all_attempts": percent("entered_audit_region", n_attempted),
        "preserved_inside_percent_all_attempts": percent("preserved_inside_audit_region", n_attempted),
        "latent_target_success_percent_all_attempts": percent("latent_target_success", n_attempted),
        "typicality_success_percent_all_attempts": percent("audit_typicality_success", n_attempted),
        "typical_among_latent_successes_percent": (
            100.0 * sum(row["counterfactual_inside_audit_region"] for row in latent_successes) / len(latent_successes)
            if latent_successes else None
        ),
        "n_latent_target_successes": len(latent_successes),
    }


def _load_npz(path, required):
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            "Class-awareness audit requires a completed typicality.runner archive; "
            f"missing {path}. Plain counterfactual-only runs need checkpoint-backed enrichment."
        )
    with np.load(path, allow_pickle=False) as data:
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
        return {name: np.asarray(data[name]) for name in required}


def audit_fold(
    root, task, fold_entry, *, samples_per_source_subject=3,
    samples_per_target_subject=3, seed=42, quantile=0.95,
):
    """Audit one saved LOSO fold without loading its model."""
    root = Path(root)
    fold_subject = int(fold_entry["subject_id"])
    fold_dir = root / f"subject_{fold_subject}"
    fold_path = fold_dir / "fold.json"
    if not fold_path.is_file():
        raise ValueError(f"Fold {fold_subject} is incomplete: missing {fold_path}")
    fold = json.loads(fold_path.read_text())
    if fold.get("status") != "completed":
        raise ValueError(f"Fold {fold_subject} is not completed")
    generation_tau = float(fold["threshold"])

    source_path = fold_dir / "calibration" / "source_trials.npz"
    source = _load_npz(source_path, ("subject_ids", "trial_ids", "labels", "probabilities", "discrepancy"))
    slabels, sprobs, sscores = _validate_predictions(
        source["labels"], source["probabilities"], source["discrepancy"], name="source"
    )
    ssubjects = np.asarray(source["subject_ids"], dtype=int)
    strials = np.asarray(source["trial_ids"], dtype=int)
    if ssubjects.shape != slabels.shape or strials.shape != slabels.shape:
        raise ValueError("Source IDs must align with source trials")
    if np.any(ssubjects == fold_subject):
        raise ValueError("Held-out subject leaked into source typicality calibration")
    source_correct = correct_class_one_mask(slabels, sprobs)
    source_selected, source_manifest = stratified_sample_indices(
        ssubjects, strials, source_correct, samples_per_subject=samples_per_source_subject,
        seed=seed, fold_subject=fold_subject, role="source_correct_class_1",
    )
    audit_tau = _higher_quantile(sscores[source_selected], quantile)
    all_source_tau = _higher_quantile(sscores[source_correct], quantile)

    observations_path = fold_dir / "observations.npz"
    held = _load_npz(observations_path, ("trial_ids", "labels", "probabilities", "discrepancy"))
    hlabels, hprobs, hscores = _validate_predictions(
        held["labels"], held["probabilities"], held["discrepancy"], name="held-out"
    )
    htrials = np.asarray(held["trial_ids"], dtype=int)
    if htrials.shape != hlabels.shape:
        raise ValueError("Held-out trial IDs must align with observations")
    hsubjects = np.full(len(hlabels), fold_subject, dtype=int)
    held_correct = correct_class_one_mask(hlabels, hprobs)
    held_selected, held_manifest = stratified_sample_indices(
        hsubjects, htrials, held_correct, samples_per_subject=samples_per_target_subject,
        seed=seed, fold_subject=fold_subject, role="heldout_correct_class_1",
    )
    pooled_tau = (_higher_quantile(np.concatenate((sscores[source_selected], hscores[held_selected])), quantile)
                  if len(held_selected) else None)

    real_rows = _real_rows(
        task=task, fold_subject=fold_subject, role="source_reference", indices=source_selected,
        subject_ids=ssubjects, trial_ids=strials, labels=slabels, probabilities=sprobs,
        discrepancies=sscores, audit_tau=audit_tau, generation_tau=generation_tau,
    )
    real_rows.extend(_real_rows(
        task=task, fold_subject=fold_subject, role="heldout_subject_invariance", indices=held_selected,
        subject_ids=hsubjects, trial_ids=htrials, labels=hlabels, probabilities=hprobs,
        discrepancies=hscores, audit_tau=audit_tau, generation_tau=generation_tau,
    ))
    counterfactual_rows = _counterfactual_rows(root=root, task=task, fold=fold, audit_tau=audit_tau)
    base = {
        "task": task,
        "fold_subject": fold_subject,
        "generation_tau_all_true1": generation_tau,
        "audit_tau_correct1_sampled": audit_tau,
        "audit_tau_correct1_all_available": all_source_tau,
        "pooled_tau_correct1_sampled_descriptive": pooled_tau,
        "quantile": quantile,
        "quantile_method": "higher",
        "n_source_subjects_available": len(source_manifest),
        "n_source_subjects_with_correct1": sum(
            row["n_available_correct_class_1"] > 0 for row in source_manifest
        ),
        "n_source_subjects_sampled": sum(row["n_selected"] > 0 for row in source_manifest),
        "n_source_correct1_available": int(source_correct.sum()),
        "n_source_correct1_sampled": int(len(source_selected)),
        "n_heldout_correct1_available": int(held_correct.sum()),
        "n_heldout_correct1_sampled": int(len(held_selected)),
        **_prefixed("source_sample", _coverage(sscores[source_selected], audit_tau)),
        **_prefixed("heldout_correct1_sample", _coverage(hscores[held_selected], audit_tau)),
        **_prefixed("heldout_all_true1", _coverage(hscores[hlabels == 1], audit_tau)),
        **_prefixed("heldout_misclassified_true1", _coverage(hscores[(hlabels == 1) & ~held_correct], audit_tau)),
    }
    fold_rows = []
    for objective in ("base", "typicality"):
        objective_rows = [row for row in counterfactual_rows if row["objective"] == objective]
        fold_rows.append({**base, "objective": objective, **_counterfactual_summary(objective_rows)})
    return {
        "fold_rows": fold_rows,
        "real_rows": real_rows,
        "counterfactual_rows": counterfactual_rows,
        "sampling": source_manifest + held_manifest,
        "hashes": {
            "source_trials_npz": file_sha256(source_path),
            "observations_npz": file_sha256(observations_path),
            "fold_json": file_sha256(fold_path),
        },
    }


def _mean(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def build_class_typicality_audit(
    roots, output, *, samples_per_source_subject=3, samples_per_target_subject=3,
    seed=42, quantile=0.95,
):
    """Build an offline class-awareness report from one or more study roots."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {output}")
    fold_rows, real_rows, counterfactual_rows, sampling, audited = [], [], [], [], []
    seen = set()
    for root in map(Path, roots):
        study_path = root / "study.json"
        if not study_path.is_file():
            raise ValueError(
                "Class-awareness audit requires a completed typicality.runner archive with study.json. "
                "Plain counterfactual-only runs need one checkpoint-backed enrichment pass."
            )
        study = json.loads(study_path.read_text())
        task = study["task"]
        for fold_entry in study["folds"]:
            key = (task, int(fold_entry["subject_id"]))
            if key in seen:
                raise ValueError(f"Duplicate task/fold across study roots: {key}")
            seen.add(key)
            result = audit_fold(
                root, task, fold_entry, samples_per_source_subject=samples_per_source_subject,
                samples_per_target_subject=samples_per_target_subject, seed=seed, quantile=quantile,
            )
            fold_rows.extend(result["fold_rows"])
            real_rows.extend(result["real_rows"])
            counterfactual_rows.extend(result["counterfactual_rows"])
            sampling.extend(result["sampling"])
            audited.append({
                "task": task,
                "fold_subject": int(fold_entry["subject_id"]),
                "study_root": str(root.resolve()),
                "study_sha256": file_sha256(study_path),
                **result["hashes"],
            })

    aggregates = []
    for task in sorted({row["task"] for row in fold_rows}):
        for objective in ("base", "typicality"):
            group = [row for row in fold_rows if row["task"] == task and row["objective"] == objective]
            aggregates.append({
                "task": task,
                "objective": objective,
                "n_folds": len(group),
                "heldout_correct1_coverage_macro_percent": _mean(group, "heldout_correct1_sample_percent"),
                "heldout_all_true1_coverage_macro_percent": _mean(group, "heldout_all_true1_percent"),
                "counterfactual_typical_macro_percent": _mean(group, "typical_percent_all_attempts"),
                "counterfactual_entry_macro_percent": _mean(group, "entered_percent_all_attempts"),
                "counterfactual_typicality_success_macro_percent": _mean(group, "typicality_success_percent_all_attempts"),
                "n_counterfactual_attempts": int(sum(row["n_attempted"] for row in group)),
            })

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "class_typicality_folds.csv", fold_rows)
    write_csv(output / "class_typicality_real_trials.csv", real_rows)
    write_csv(output / "class_typicality_counterfactuals.csv", counterfactual_rows)
    write_csv(output / "class_typicality_aggregate.csv", aggregates)
    write_json(output / "class_typicality_sampling.json", sampling)
    payload = {
        "schema_version": 1,
        "definition": "true_class == 1 and argmax(probabilities) == 1",
        "primary_threshold": "source-only sampled correct-class-1 discrepancy quantile",
        "heldout_role": "evaluation only; never contributes to primary threshold",
        "samples_per_source_subject": int(samples_per_source_subject),
        "samples_per_target_subject": int(samples_per_target_subject),
        "seed": int(seed),
        "quantile": float(quantile),
        "quantile_method": "higher",
        "audited_archives": audited,
        "aggregate": aggregates,
        "notes": [
            "Generation and audit thresholds are retained separately.",
            "The pooled source+target threshold is descriptive and is never used for the invariance claim.",
            "Correct-class-1 coverage is conditional on recognition; all-true-class-1 coverage is a sensitivity analysis.",
            "Independent fold latent coordinates are never pooled; only fold-level memberships and percentages are aggregated.",
        ],
    }
    write_json(output / "class_typicality_audit.json", payload)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+", help="Completed typicality.runner study roots")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-source-subject", type=int, default=3,
                        help="Correct class-1 trials per source subject; 0 uses all")
    parser.add_argument("--samples-per-target-subject", type=int, default=3,
                        help="Correct class-1 held-out trials per fold; 0 uses all")
    parser.add_argument("--typicality-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.samples_per_source_subject < 0 or args.samples_per_target_subject < 0:
        raise SystemExit("Sample counts must be nonnegative")
    build_class_typicality_audit(
        args.roots, args.out_dir,
        samples_per_source_subject=args.samples_per_source_subject,
        samples_per_target_subject=args.samples_per_target_subject,
        seed=args.seed, quantile=args.typicality_quantile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
