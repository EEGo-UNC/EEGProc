"""Compare held-out real X and typicality-arm R(Zcf) with source class-1 R(Z).

The benchmark consists of decoded latent states from other subjects' real,
correctly predicted, typical class-1 trials. The discrepancy uses the same
diagonal squared Mahalanobis formula as typicality, fitted to those decoded
waveforms. It is a distinct score from the learned embedding-space typicality
D. Generated counterfactuals are never re-encoded for this report.

Run from the EEGProc root with::

    PYTHONPATH=src python -m eegproc.model_explainability.report_iclr_waveform_subject_invariance \
        /path/to/incomplete-ICLR \
        --raw-eeg datasets/dreamer_eeg.npy --raw-labels datasets/dreamer_labels.npy \
        --checkpoint-root runs/full --samples-per-subject 3 --seed 42 \
        --out-dir runs/counterfactuals/incomplete-ICLR-waveform-subject-invariance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .report_iclr_subject_invariance import (
    _load_summary, _median, _percent_inside, _source_percentiles, analyze_fold,
    file_sha256, stratified_sample_indices, write_csv, write_json,
)
from .report_iclr_typicality_subject_invariance import _latest_committed_result, _roots


def _fit_waveform_region(source_trials, *, variance_floor=1e-6):
    """Fit a diagonal Gaussian in decoded EEG coordinates."""
    source_trials = iter(source_trials)
    try:
        first = np.asarray(next(source_trials), dtype=np.float64)
    except StopIteration as error:
        raise ValueError("No decoded source class-1 EEG trials for waveform calibration") from error
    if first.ndim != 3 or not np.isfinite(first).all():
        raise ValueError("Expected finite full-trial EEG shaped (windows, samples, features)")
    mean = first.copy()
    second_moment = np.zeros_like(mean)
    n = 1
    for trial in source_trials:
        value = np.asarray(trial, dtype=np.float64)
        if value.shape != mean.shape or not np.isfinite(value).all():
            raise ValueError("Decoded source EEG trials must have the same finite waveform shape")
        n += 1
        delta = value - mean
        mean += delta / n
        second_moment += delta * (value - mean)
    if n < 2 or not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("Need at least two decoded source class-1 EEG trials and a positive variance floor")
    variance = np.maximum(second_moment / n, variance_floor)
    return mean, variance, n


def _waveform_discrepancy(signal, mean, variance):
    value = np.asarray(signal, dtype=np.float64)
    if value.shape != mean.shape or not np.isfinite(value).all():
        raise ValueError("Query EEG waveform shape or values differ from the source reference")
    return float(np.mean(np.square(value - mean) / variance))


def _archived_decoded_eeg(result_path, *, original_x, report_output):
    """Read only X and R(Zcf); never substitute the baseline reconstruction."""
    archive_path = result_path.with_name("counterfactual.npz")
    if not archive_path.is_file():
        raise ValueError(f"{result_path}: missing decoded counterfactual EEG archive")
    with np.load(archive_path, allow_pickle=False) as archive:
        decoded_key = f"x_prime_{report_output}"
        if "x" not in archive or decoded_key not in archive:
            raise ValueError(f"{archive_path}: missing original X or decoded R(Zcf)")
        archived_x = np.asarray(archive["x"], dtype=np.float32)
        decoded = np.asarray(archive[decoded_key], dtype=np.float32)
    if (archived_x.shape != (1, *original_x.shape)
            or decoded.shape != archived_x.shape
            or not np.isfinite(archived_x).all() or not np.isfinite(decoded).all()
            or not np.allclose(archived_x[0], original_x, rtol=1e-5, atol=1e-6)):
        raise ValueError(f"{archive_path}: waveform shape, values, or original trial identity disagree")
    return decoded[0], archive_path


def _dataset_for_task(study, *, raw_eeg, raw_labels):
    from .model_agnostic.sic_adapter import load_sic_raw_trials
    from .typicality.artifacts import array_sha256

    config = dict(study["arguments"]["data_config"])
    config["raw_eeg_npy"] = str(raw_eeg)
    config["raw_labels_npy"] = str(raw_labels)
    dataset = load_sic_raw_trials(config)
    if (dataset.features.ndim != 4 or dataset.features.shape[-1] != 42
            or dataset.normalization_scale is None):
        raise ValueError("Expected original normalized full-trial DREAMER EEG")
    for name in ("subject_ids", "trial_ids", "labels", "normalization_offset", "normalization_scale"):
        if array_sha256(getattr(dataset, name)) != study["dataset_sha256"][name]:
            raise ValueError(f"Local DREAMER data or preprocessing differs from the study: {name}")
    keys = list(zip(dataset.subject_ids.astype(int), dataset.trial_ids.astype(int)))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate subject/trial identifiers in the original EEG dataset")
    return dataset, dict(zip(keys, range(len(keys))))


def _checkpoint_for_fold(entry, checkpoint_root, cache):
    expected = entry["sha256"]
    if expected in cache:
        return cache[expected]
    archived = Path(entry["path"])
    candidates = [archived] if archived.is_file() else []
    candidates.extend(path for path in checkpoint_root.rglob(archived.name) if path != archived)
    for path in candidates:
        if file_sha256(path) == expected:
            cache[expected] = path
            return path
    raise FileNotFoundError(
        f"Matching frozen checkpoint {archived.name} ({expected}) not found under {checkpoint_root}"
    )


def _decoded_source_reference(study, checkpoint, dataset, source_indices, *, batch_size=16):
    """Decode selected source X through its frozen Z once; do not encode R(Zcf)."""
    import tensorflow as tf
    from .model_agnostic.sic_adapter import create_sic_adapter

    args = study["arguments"]
    adapter = create_sic_adapter(
        model_path=checkpoint,
        config={"model_module": args["model_module"],
                "decoder_mode": args["decoder_mode"],
                "fixed_joint_alpha": args.get("fixed_joint_alpha")},
        sample_input=dataset.features[source_indices[:1]],
    )
    report_output = args["report_output"]
    signature = tf.TensorSpec((None, *dataset.features.shape[1:]), dtype=tf.float32)

    @tf.function(input_signature=[signature])
    def decode_source(raw):
        state = adapter.initial_state(raw)
        return adapter.reconstruct(state, raw)[report_output]

    decoded = np.empty((len(source_indices), *dataset.features.shape[1:]), dtype=np.float32)
    for start in range(0, len(source_indices), batch_size):
        raw = np.asarray(dataset.features[source_indices[start:start + batch_size]], dtype=np.float32)
        values = np.asarray(decode_source(raw), dtype=np.float32)
        if values.shape != raw.shape or not np.isfinite(values).all():
            raise ValueError("Decoded source class-1 EEG has invalid shape or values")
        decoded[start:start + len(raw)] = values
    del adapter
    return decoded


def _fold(root, study, entry, *, dataset, index, checkpoint_root, checkpoint_cache,
          samples_per_subject, seed, variance_floor):
    task, subject = study["task"], int(entry["subject_id"])
    fold_dir = root / f"subject_{subject}"
    _, archived_real, sampling, hashes = analyze_fold(
        root, task, entry, samples_per_subject=samples_per_subject, seed=seed,
    )
    fold = json.loads((fold_dir / "fold.json").read_text())
    _, source = _load_summary(fold_dir / "calibration/source_trials", (
        "subject_ids", "trial_ids", "labels", "probabilities", "discrepancy",
    ))
    slabels = np.asarray(source["labels"], dtype=int)
    sprobs = np.asarray(source["probabilities"], dtype=float)
    sscores = np.asarray(source["discrepancy"], dtype=float)
    selected, source_reference_sampling = stratified_sample_indices(
        source["subject_ids"], source["trial_ids"],
        (slabels == 1) & (sprobs.argmax(axis=1) == 1)
        & (sscores <= float(fold["threshold"])),
        samples_per_subject=samples_per_subject, seed=seed,
        fold_subject=subject, role="source_correct_typical_class1_decoded_reference",
    )
    keys = [(int(source["subject_ids"][i]), int(source["trial_ids"][i])) for i in selected]
    if len(keys) < 2 or any(key[0] == subject or key not in index
                           or dataset.labels[index[key]] != 1 for key in keys):
        raise ValueError(f"{fold_dir}: source typical real class-1 EEG membership is invalid")
    checkpoint = _checkpoint_for_fold(entry, checkpoint_root, checkpoint_cache)
    print(f"{task} subject {subject}: decoding {len(keys)} typical source class-1 trials", flush=True)
    source_decoded = _decoded_source_reference(study, checkpoint, dataset, [index[key] for key in keys])
    mean, variance, n_source = _fit_waveform_region(source_decoded, variance_floor=variance_floor)
    source_wave_scores = np.asarray([
        _waveform_discrepancy(value, mean, variance) for value in source_decoded
    ])
    wave_tau = float(np.quantile(source_wave_scores, 0.95, method="higher"))
    source_rows = [{
        "task": task, "fold_subject": subject,
        "role": "decoded_source_typical_class1_R(Z)",
        "subject_id": key[0], "trial_id": key[1], "true_class": 1,
        "waveform_discrepancy": float(score),
        "waveform_source_threshold": wave_tau,
        "inside_source_waveform_region": bool(score <= wave_tau),
        "source_waveform_percentile": float(percentile),
    } for key, score, percentile in zip(
        keys, source_wave_scores, _source_percentiles(source_wave_scores, source_wave_scores)
    )]
    real_rows = []
    for archived in archived_real:
        if archived["role"] != "heldout":
            continue
        trial = int(archived["trial_id"])
        key = (subject, trial)
        if key not in index or dataset.labels[index[key]] != 1:
            raise ValueError(f"{fold_dir}: sampled original real class-1 trial is absent")
        score = _waveform_discrepancy(dataset.features[index[key]], mean, variance)
        real_rows.append({
            "task": task, "fold_subject": subject, "role": "original_real_class1_X",
            "subject_id": subject, "trial_id": trial, "true_class": 1,
            "waveform_discrepancy": score, "waveform_source_threshold": wave_tau,
            "inside_source_waveform_region": score <= wave_tau,
            "source_waveform_percentile": float(_source_percentiles(np.asarray([score]), source_wave_scores)[0]),
        })
    _, observed = _load_summary(fold_dir / "observations", ("trial_ids", "labels", "probabilities"))
    obs_ids = np.asarray(observed["trial_ids"], dtype=int)
    obs_labels = np.asarray(observed["labels"], dtype=int)
    obs_preds = np.asarray(observed["probabilities"], dtype=float).argmax(axis=1)
    expected = {int(obs_ids[i]) for i in range(len(obs_ids)) if obs_labels[i] == 0 and obs_preds[i] == 0}
    eligible = list(map(int, fold["eligible_trial_ids"]))
    if len(set(eligible)) != len(eligible) or not set(eligible).issubset(expected):
        raise ValueError(f"{fold_dir}: eligible class-0 trials conflict with observations")
    cf_rows = []
    n_completed = n_pending = n_error = n_latent_nonflip = n_confident_flip = 0
    for trial in sorted(eligible):
        arm = fold_dir / f"trial_{trial}" / "typicality"
        path = _latest_committed_result(arm)
        if path is None:
            previous = sorted(arm.glob("attempt_*/result.json"))
            if previous and json.loads(previous[-1].read_text()).get("status") == "error":
                n_error += 1
            else:
                n_pending += 1
            continue
        summary = json.loads(path.read_text())
        for name, expected_value in (("task", task), ("subject_id", subject), ("trial_id", trial),
                                     ("true_class", 0), ("objective", "typicality"),
                                     ("checkpoint_sha256", fold["checkpoint"]["sha256"]),
                                     ("status", "completed")):
            if summary.get(name) != expected_value:
                raise ValueError(f"{path}: {name} differs from the fold archive")
        latent = summary["latent_counterfactual"]
        probabilities = np.asarray(latent["probabilities"], dtype=float)
        if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5)
                or latent.get("predicted_class") != int(probabilities.argmax())):
            raise ValueError(f"{path}: invalid optimized class prediction")
        required_probability = float(summary["required_target_probability"])
        confident_flip = bool(probabilities.argmax() == 1
                              and probabilities[1] >= required_probability)
        if (not 0 < required_probability <= 1
                or not np.isfinite(required_probability)
                or latent.get("success") != confident_flip):
            raise ValueError(f"{path}: saved counterfactual success conflicts with the study criterion")
        key = (subject, trial)
        if key not in index or dataset.labels[index[key]] != 0:
            raise ValueError(f"{path}: original class-0 EEG trial is absent")
        decoded, archive = _archived_decoded_eeg(
            path, original_x=dataset.features[index[key]], report_output=summary["report_output"],
        )
        score = _waveform_discrepancy(decoded, mean, variance)
        n_completed += 1
        if probabilities.argmax() != 1:
            n_latent_nonflip += 1
        if confident_flip:
            n_confident_flip += 1
        cf_rows.append({
            "task": task, "fold_subject": subject,
            "role": "decoded_typicality_counterfactual_R(Zcf)",
            "subject_id": subject, "trial_id": trial, "original_true_class": 0,
            "optimized_predicted_class": int(probabilities.argmax()),
            "optimized_class1_flip": bool(probabilities.argmax() == 1),
            "optimized_target_probability": float(probabilities[1]),
            "required_target_probability": required_probability,
            "optimized_confident_class1_flip": confident_flip,
            "waveform_discrepancy": score, "waveform_source_threshold": wave_tau,
            "inside_source_waveform_region": score <= wave_tau,
            "source_waveform_percentile": float(_source_percentiles(np.asarray([score]), source_wave_scores)[0]),
            "result_path": str(path.resolve()),
            "decoded_waveform_archive": str(archive.resolve()),
        })
        committed = json.loads(path.with_name("complete.json").read_text())["sha256"]
        hashes.setdefault("counterfactual_results", []).append({
            "result_path": str(path.resolve()), "result_sha256": file_sha256(path),
            "waveform_archive_path": str(archive.resolve()),
            "waveform_archive_declared_sha256": committed.get("counterfactual.npz"),
        })
    real_scores = np.asarray([row["waveform_discrepancy"] for row in real_rows])
    cf_scores = np.asarray([row["waveform_discrepancy"] for row in cf_rows
                            if row["optimized_class1_flip"]])
    source_median, real_median, cf_median = map(_median, (source_wave_scores, real_scores, cf_scores))
    row = {
        "task": task, "fold_subject": subject, "fold_status": fold["status"],
        "source_checkpoint": str(checkpoint.resolve()),
        "n_source_subjects": len(set(key[0] for key in keys)),
        "n_source_typical_real_class1": n_source,
        "n_heldout_original_real_class1_sampled": len(real_scores),
        "n_eligible_class0": len(eligible), "n_typicality_completed": n_completed,
        "n_typicality_error": n_error, "n_typicality_pending": n_pending,
        "n_optimized_class1_flip": len(cf_scores), "n_optimized_nonflip": n_latent_nonflip,
        "n_optimized_confident_class1_flip": n_confident_flip,
        "waveform_source_threshold": wave_tau,
        "source_waveform_median_discrepancy": source_median,
        "real_x_median_waveform_discrepancy": real_median,
        "decoded_cf_median_waveform_discrepancy": cf_median,
        "real_x_minus_source_median_over_waveform_threshold":
            (real_median - source_median) / wave_tau if real_median is not None and wave_tau > 0 else None,
        "decoded_cf_minus_source_median_over_waveform_threshold":
            (cf_median - source_median) / wave_tau if cf_median is not None and wave_tau > 0 else None,
        "decoded_cf_minus_real_x_median_over_waveform_threshold":
            (cf_median - real_median) / wave_tau if cf_median is not None and real_median is not None and wave_tau > 0 else None,
        "n_real_x_inside_source": int(np.count_nonzero(real_scores <= wave_tau)),
        "n_decoded_cf_inside_source": int(np.count_nonzero(cf_scores <= wave_tau)),
        "real_x_median_source_waveform_percentile": _median(_source_percentiles(real_scores, source_wave_scores)),
        "decoded_cf_median_source_waveform_percentile": _median(_source_percentiles(cf_scores, source_wave_scores)),
    }
    hashes["source_checkpoint"] = str(checkpoint.resolve())
    hashes["source_checkpoint_sha256"] = entry["sha256"]
    return row, source_rows + real_rows + cf_rows, sampling + source_reference_sampling, hashes


def _aggregate(rows, *, expected_subjects):
    comparable = [row for row in rows if row["n_heldout_original_real_class1_sampled"]
                  and row["n_optimized_class1_flip"]]

    def med(field):
        return _median([row[field] for row in comparable if row[field] is not None])

    ids = {row["fold_subject"] for row in rows}
    return {
        "task": rows[0]["task"], "n_observed_folds": len(rows),
        "n_expected_folds": expected_subjects,
        "missing_subject_ids": sorted(set(range(expected_subjects)) - ids),
        "n_comparable_folds": len(comparable),
        "n_folds_decoded_cf_lower_discrepancy_than_real_x": sum(
            row["decoded_cf_minus_real_x_median_over_waveform_threshold"] < 0
            for row in comparable
        ),
        "min_source_decoded_reference_trials": min(
            row["n_source_typical_real_class1"] for row in rows
        ),
        "n_running_folds": sum(row["fold_status"] != "completed" for row in rows),
        "n_eligible_class0": sum(row["n_eligible_class0"] for row in rows),
        "n_typicality_completed": sum(row["n_typicality_completed"] for row in rows),
        "n_typicality_error": sum(row["n_typicality_error"] for row in rows),
        "n_typicality_pending": sum(row["n_typicality_pending"] for row in rows),
        "n_optimized_class1_flip": sum(row["n_optimized_class1_flip"] for row in rows),
        "n_optimized_confident_class1_flip": sum(
            row["n_optimized_confident_class1_flip"] for row in rows
        ),
        "n_optimized_nonflip": sum(row["n_optimized_nonflip"] for row in rows),
        "n_real_x_sampled": sum(row["n_heldout_original_real_class1_sampled"] for row in rows),
        "n_real_x_inside_source": sum(row["n_real_x_inside_source"] for row in rows),
        "n_decoded_cf_inside_source": sum(row["n_decoded_cf_inside_source"] for row in rows),
        "fold_median_real_x_source_waveform_percentile": med("real_x_median_source_waveform_percentile"),
        "fold_median_decoded_cf_source_waveform_percentile": med("decoded_cf_median_source_waveform_percentile"),
        "fold_median_real_x_minus_source_over_waveform_threshold": med("real_x_minus_source_median_over_waveform_threshold"),
        "fold_median_decoded_cf_minus_source_over_waveform_threshold": med("decoded_cf_minus_source_median_over_waveform_threshold"),
        "fold_median_decoded_cf_minus_real_x_over_waveform_threshold": med("decoded_cf_minus_real_x_median_over_waveform_threshold"),
    }


def _paragraph(aggregates):
    lines = [
        r"\paragraph{Subject-invariance.}",
        "We compare held-out subjects' original class-1 EEG $X$ and "
        "typicality-generated class-0-to-1 waveforms $R(Z^{\\mathrm{cf}})$ "
        "with decoded $R(Z)$ from other subjects' correctly predicted, "
        "typical class-1 trials. The fold-specific EEG-space discrepancy "
        "uses the same diagonal squared Mahalanobis formula as typicality, "
        "fitted to the source $R(Z)$ waveforms.",
    ]
    for row in aggregates:
        task = row["task"].capitalize()
        if not row["n_comparable_folds"]:
            lines.append(
                f"For {task}, no observed fold has both sampled original "
                "class-1 EEG and a completed class-1 counterfactual waveform."
            )
            continue
        lines.append(
            f"For {task}, {row['n_decoded_cf_inside_source']} of "
            f"{row['n_optimized_class1_flip']} class-1 argmax counterfactuals "
            f"fall within the source $R(Z)$ 95th-percentile region, compared "
            f"with {row['n_real_x_inside_source']} of {row['n_real_x_sampled']} "
            f"sampled original class-1 $X$. The "
            f"counterfactual fold-median discrepancy is lower than that of "
            f"real $X$ in {row['n_folds_decoded_cf_lower_discrepancy_than_real_x']} "
            f"of {row['n_comparable_folds']} comparable folds. "
            f"{row['n_optimized_confident_class1_flip']} of "
            f"{row['n_typicality_completed']} completed attempts meet the "
            f"study's target-probability criterion."
        )
    if all(row["n_real_x_inside_source"] == 0 for row in aggregates):
        lines.append(
            "All sampled original class-1 $X$ fall outside the decoded "
            "reference region. Since $R(Z^{\\mathrm{cf}})$ and the reference "
            "$R(Z)$ share a decoder while $X$ does not, these counts do not "
            "establish subject invariance; they show overlap within the "
            "decoder's output space."
        )
    coverage = "; ".join(
        f"{row['n_observed_folds']}/{row['n_expected_folds']} {row['task']} folds"
        for row in aggregates
    )
    lines.append(
        f"The supplied archives cover {coverage}. This empirical EEG-space "
        "reference is distinct from the learned embedding-space typicality region."
    )
    return "\n".join(lines) + "\n"


def build_waveform_subject_invariance_report(roots, output, *, raw_eeg, raw_labels, checkpoint_root,
                                             samples_per_subject=3, seed=42,
                                             expected_subjects=23, variance_floor=1e-6):
    if (isinstance(samples_per_subject, bool) or int(samples_per_subject) != samples_per_subject
            or samples_per_subject < 0 or expected_subjects < 1):
        raise ValueError("Invalid subject sampling or expected-subject count")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {output}")
    roots = _roots(roots)
    folds, trials, samples, hashes, seen, datasets, checkpoint_cache = [], [], [], [], set(), {}, {}
    for root in roots:
        study_path = root / "study.json"
        study = json.loads(study_path.read_text())
        task = study["task"]
        if "typicality" not in study.get("objectives", ["typicality"]):
            raise ValueError(f"{study_path}: typicality arm is absent")
        if task not in datasets:
            datasets[task] = _dataset_for_task(study, raw_eeg=raw_eeg, raw_labels=raw_labels)
        dataset, index = datasets[task]
        for name in ("subject_ids", "trial_ids", "labels", "normalization_offset", "normalization_scale"):
            from .typicality.artifacts import array_sha256
            if array_sha256(getattr(dataset, name)) != study["dataset_sha256"][name]:
                raise ValueError(f"{study_path}: task dataset hash differs for {name}")
        for entry in study["folds"]:
            key = (task, int(entry["subject_id"]))
            if key in seen:
                raise ValueError(f"Duplicate task/fold across studies: {key}")
            seen.add(key)
            fold, trial_rows, sampling, inputs = _fold(
                root, study, entry, dataset=dataset, index=index,
                checkpoint_root=Path(checkpoint_root), checkpoint_cache=checkpoint_cache,
                samples_per_subject=samples_per_subject, seed=seed,
                variance_floor=variance_floor,
            )
            folds.append(fold)
            trials.extend(trial_rows)
            samples.extend(sampling)
            hashes.append({**inputs, "study_json_sha256": file_sha256(study_path)})
    if not folds:
        raise ValueError("No LOSO folds found in the input studies")
    aggregates = [_aggregate([row for row in folds if row["task"] == task],
                             expected_subjects=expected_subjects)
                  for task in sorted({row["task"] for row in folds})]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "waveform_subject_invariance_trials.csv", trials)
    write_csv(output / "waveform_subject_invariance_folds.csv", folds)
    write_csv(output / "waveform_subject_invariance_aggregate.csv", aggregates)
    write_json(output / "waveform_subject_invariance_sampling.json", samples)
    (output / "waveform_subject_invariance_paragraph.tex").write_text(_paragraph(aggregates))
    report = {
        "schema_version": 1,
        "discrepancy_definition": "full_trial_diagonal_squared_mahalanobis_per_waveform_coordinate",
        "formula": "mean((waveform - source_mean)^2 / max(source_variance, variance_floor))",
        "waveform_space": "normalized SIC model-input EEG, shape (windows, samples, channel-bands)",
        "real_signal": "original prepared X from DREAMER; never R(E(X))",
        "counterfactual_signal": "x_prime_<report_output> = decoded R(Zcf) from typicality-arm counterfactual.npz",
        "source_reference": "R(Z) from other subjects' real true-class-1 EEG with correct class-1 prediction and saved embedding typicality D <= source threshold",
        "source_reference_generation": "one frozen-checkpoint encoding of source X to Z followed by decoding Z to R(Z); no counterfactual re-encoding",
        "source_threshold": "95th percentile of source decoded R(Z) waveform discrepancies in each fold",
        "subject_invariance_conclusion": "inconclusive: shared decoder confounds the decoded counterfactual-versus-original EEG comparison",
        "selection": "sampled held-out real true class 1 regardless of prediction; all eligible typicality outputs counted, optimized latent class-1 argmax flips in waveform comparison, with confidence-qualified flips reported separately",
        "variance_floor": variance_floor,
        "samples_per_subject": int(samples_per_subject), "seed": int(seed),
        "expected_subjects": int(expected_subjects),
        "raw_eeg_sha256": file_sha256(raw_eeg),
        "raw_labels_sha256": file_sha256(raw_labels),
        "checkpoint_root": str(Path(checkpoint_root).resolve()),
        "aggregate": aggregates, "input_hashes": hashes,
        "limitations": [
            "The diagonal decoded-EEG Gaussian is empirical and is distinct from the checkpoint's learned embedding-space class-1 Gaussian.",
            "A decoded class-1 flip is not independently verified without re-encoding; class-1 selection uses the optimizer's saved latent prediction.",
            "Original X and decoded R(Z) occupy different waveform distributions in the supplied archives; decoder-induced alignment can explain the observed gap, so this report cannot establish subject invariance.",
            "Pointwise waveform discrepancy is sensitive to temporal phase and alignment.",
            "The supplied study archives are incomplete and task summaries are provisional.",
        ],
    }
    write_json(output / "waveform_subject_invariance.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+", help="typicality.runner studies or parent directory")
    parser.add_argument("--raw-eeg", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="Directory containing matching frozen LOSO .keras checkpoints")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-subject", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-subjects", type=int, default=23)
    parser.add_argument("--variance-floor", type=float, default=1e-6)
    args = parser.parse_args(argv)
    try:
        build_waveform_subject_invariance_report(
            args.roots, args.out_dir, raw_eeg=args.raw_eeg, raw_labels=args.raw_labels,
            checkpoint_root=args.checkpoint_root,
            samples_per_subject=args.samples_per_subject, seed=args.seed,
            expected_subjects=args.expected_subjects, variance_floor=args.variance_floor,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Waveform subject-invariance report: {error}\n")
    print(f"Wrote waveform subject-invariance report to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
