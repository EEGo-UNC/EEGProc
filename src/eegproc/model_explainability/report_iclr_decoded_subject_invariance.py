"""Compare held-out real class-1 X and class-0-to-1 decoded R(Z) with source class-1 trials.

Each comparison stays within a LOSO fold. Source real class-1 trials calibrate
the frozen classifier's class-1 region; held-out real X and typicality-arm
decoded counterfactual EEG are independently encoded and scored against it.
Only decoded outputs that the frozen classifier labels class 1 enter the
primary counterfactual comparison. All eligible trials remain in the counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .report_iclr_subject_invariance import (
    REPRESENTATION, SCORE_DEFINITION, TypicalityRegion, _load_summary,
    _median, _percent_inside, _source_percentiles, analyze_fold,
    compatible_typicality_definition, file_sha256, write_csv, write_json,
)
from .typicality.artifacts import completed_attempt


def _study_roots(roots):
    found = set()
    for root in map(Path, roots):
        if not root.exists():
            raise ValueError(f"Input study directory does not exist: {root}. Copy the run output here first, or use an existing study directory.")
        if (root / "study.json").is_file():
            found.add(root.resolve())
        else:
            found.update(path.parent.resolve() for path in root.rglob("study.json"))
    if not found:
        raise ValueError("No typicality.runner study.json found under the input directories")
    return sorted(found)


def _decoded_score(summary, region, *, path, task, subject, trial, checkpoint_sha256):
    for key, expected in (("task", task), ("subject_id", subject), ("trial_id", trial),
                          ("true_class", 0), ("objective", "typicality")):
        if summary.get(key) != expected:
            raise ValueError(f"{path}: {key} does not match the requested typicality trial")
    if checkpoint_sha256 and summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{path}: checkpoint differs from the fold's frozen classifier")
    if (summary.get("typicality_definition", SCORE_DEFINITION) != SCORE_DEFINITION
            or summary.get("typicality_representation", REPRESENTATION) != REPRESENTATION):
        raise ValueError(f"{path}: result uses a different typicality definition")
    decoded = summary.get("decoded_counterfactual")
    score_archive = None
    if decoded is None:
        # Earlier full round-trip studies retained the generated waveform and
        # its independent re-encoding, though compact latent-only runs did not.
        output = summary.get("report_output")
        round_trip = summary.get("decoded_trials", {}).get(output, {}).get("counterfactual")
        archive_path = path.with_name("counterfactual.npz")
        if summary.get("round_trip_evaluation") != "full_trial_decoder_encoder_v1" or round_trip is None:
            raise ValueError(
                f"{path}: missing decoded_counterfactual. This archive only saved latent D(Zcf); "
                "rerun typicality.runner with the updated code in a NEW output directory "
                "to measure D(E(R(Zcf)))."
            )
        if not archive_path.is_file():
            raise ValueError(f"{path}: round-trip score has no archived decoded EEG and re-encoded embedding")
        with np.load(archive_path, allow_pickle=False) as archive:
            signal_key = f"x_prime_{output}"
            embedding_key = f"classification_embedding_reencoded_{output}"
            if signal_key not in archive or embedding_key not in archive:
                raise ValueError(f"{archive_path}: missing decoded EEG or its re-encoded embedding")
            signal = np.asarray(archive[signal_key])
            embedding = np.asarray(archive[embedding_key])
        if signal.ndim != 4 or signal.shape[0] != 1 or not np.isfinite(signal).all():
            raise ValueError(f"{archive_path}: invalid decoded EEG waveform")
        if embedding.shape != (1, len(region.prior_mean)):
            raise ValueError(f"{archive_path}: invalid re-encoded decoded EEG shape")
        if not np.isclose(float(summary["typicality"]["threshold"]), region.tau):
            raise ValueError(f"{path}: source threshold differs from round-trip calibration")
        decoded = {
            "input": "decoded_counterfactual_eeg_reencoded_by_frozen_classifier",
            "report_output": output, "target_class": 1,
            "classification_embedding": embedding[0],
            "probabilities": round_trip.get("probabilities"),
            "predicted_class": round_trip.get("predicted_class"),
            "discrepancy": round_trip.get("discrepancy"),
            "source_threshold": region.tau,
        }
        score_archive = archive_path
    if (decoded.get("input") != "decoded_counterfactual_eeg_reencoded_by_frozen_classifier"
            or decoded.get("report_output") != summary.get("report_output")
            or decoded.get("target_class") != 1):
        raise ValueError(f"{path}: decoded EEG scoring metadata is incompatible")
    embedding = np.asarray(decoded.get("classification_embedding"), dtype=float)
    probabilities = np.asarray(decoded.get("probabilities"), dtype=float)
    if embedding.shape != (len(region.prior_mean),) or not np.isfinite(embedding).all():
        raise ValueError(f"{path}: invalid re-encoded decoded EEG embedding")
    if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
        raise ValueError(f"{path}: invalid decoded EEG class probabilities")
    score = float(decoded.get("discrepancy"))
    if not np.isfinite(score) or score < 0 or not np.isclose(
            score, region.score(embedding[None, :])[0], rtol=1e-5, atol=1e-7):
        raise ValueError(f"{path}: decoded EEG discrepancy does not match the source class-1 region")
    if not np.isclose(float(decoded.get("source_threshold")), region.tau):
        raise ValueError(f"{path}: decoded EEG threshold differs from source calibration")
    predicted = int(probabilities.argmax())
    if decoded.get("predicted_class") != predicted:
        raise ValueError(f"{path}: decoded EEG predicted class disagrees with probabilities")
    return score, predicted, probabilities, score_archive


def _fold(root, task, entry, *, samples_per_subject, seed):
    subject = int(entry["subject_id"])
    fold_dir = root / f"subject_{subject}"
    real_summary, real_trials, sampling, hashes = analyze_fold(
        root, task, entry, samples_per_subject=samples_per_subject, seed=seed,
    )
    fold = json.loads((fold_dir / "fold.json").read_text())
    region = TypicalityRegion.load(fold_dir / "calibration")
    _, source = _load_summary(fold_dir / "calibration/source_trials", ("labels", "discrepancy"))
    source_scores = np.asarray(source["discrepancy"], dtype=float)[np.asarray(source["labels"], dtype=int) == 1]
    _, observed = _load_summary(fold_dir / "observations", ("trial_ids", "labels", "probabilities"))
    trial_ids = np.asarray(observed["trial_ids"], dtype=int)
    labels = np.asarray(observed["labels"], dtype=int)
    probabilities = np.asarray(observed["probabilities"], dtype=float)
    if len(set(trial_ids.tolist())) != len(trial_ids):
        raise ValueError(f"Duplicate held-out trial IDs in {fold_dir}")
    lookup = {int(trial): i for i, trial in enumerate(trial_ids)}
    eligible = list(map(int, fold.get("eligible_trial_ids", [])))
    if len(set(eligible)) != len(eligible):
        raise ValueError(f"Duplicate eligible trial IDs in {fold_dir}")
    expected = {int(trial_ids[i]) for i in range(len(trial_ids))
                if labels[i] == 0 and probabilities[i].argmax() == 0}
    if set(eligible) != expected:
        raise ValueError(f"Eligible class-0 trials differ from observations in {fold_dir}")

    rows = [row for row in real_trials if row["role"] in ("source", "heldout")]
    real_scores = np.asarray([row["discrepancy"] for row in rows if row["role"] == "heldout"], dtype=float)
    cf_scores, cf_rows = [], []
    n_completed = n_error = n_pending = n_nonflip = 0
    for trial in sorted(eligible):
        arm = fold_dir / f"trial_{trial}" / "typicality"
        attempt = completed_attempt(arm)
        if attempt is None:
            attempts = sorted(arm.glob("attempt_*/result.json"))
            if attempts and json.loads(attempts[-1].read_text()).get("status") == "error":
                n_error += 1
            else:
                n_pending += 1
            continue
        path = attempt / "result.json"
        summary = json.loads(path.read_text())
        if summary.get("status") != "completed":
            raise ValueError(f"{path}: committed attempt is not completed")
        score, predicted, probs, score_archive = _decoded_score(
            summary, region, path=path, task=task, subject=subject, trial=trial,
            checkpoint_sha256=fold.get("checkpoint", {}).get("sha256"),
        )
        n_completed += 1
        if predicted != 1:
            n_nonflip += 1
        else:
            cf_scores.append(score)
        percentile = float(_source_percentiles(np.asarray([score]), source_scores)[0])
        cf_rows.append({
            "task": task, "fold_subject": subject, "role": "decoded_counterfactual",
            "subject_id": subject, "trial_id": trial, "true_class": 0,
            "predicted_class": predicted, "decoded_class1_flip": predicted == 1,
            "target_probability": float(probs[1]), "discrepancy": score,
            "source_threshold": region.tau,
            "discrepancy_over_threshold": score / region.tau if region.tau > 0 else None,
            "inside_source_region": score <= region.tau,
            "source_empirical_percentile": percentile,
            "result_path": str(path.resolve()),
            "score_provenance": "historical_roundtrip_npz" if score_archive else "current_result_json",
        })
        inputs = {"result_path": str(path.resolve()), "result_sha256": file_sha256(path)}
        if score_archive:
            inputs.update(waveform_archive_path=str(score_archive.resolve()),
                          waveform_archive_sha256=file_sha256(score_archive))
        hashes.setdefault("counterfactual_results", []).append(inputs)
    cf_scores = np.asarray(cf_scores, dtype=float)
    cf_percentiles = _source_percentiles(cf_scores, source_scores)
    real_percentiles = _source_percentiles(real_scores, source_scores)
    source_median = _median(source_scores)
    real_median = _median(real_scores)
    cf_median = _median(cf_scores)
    tau = region.tau
    row = {
        "task": task, "fold_subject": subject,
        "n_source_subjects": real_summary["n_source_subjects"],
        "n_source_class1": len(source_scores),
        "n_heldout_real_class1_sampled": len(real_scores),
        "n_eligible_class0": len(eligible), "n_typicality_completed": n_completed,
        "n_typicality_error": n_error, "n_typicality_pending": n_pending,
        "n_decoded_class1_flip": len(cf_scores), "n_decoded_nonflip": n_nonflip,
        "source_threshold": tau, "source_median_discrepancy": source_median,
        "real_x_median_discrepancy": real_median,
        "decoded_cf_median_discrepancy": cf_median,
        "real_x_minus_source_median_over_threshold":
            (real_median - source_median) / tau if real_median is not None and tau > 0 else None,
        "decoded_cf_minus_source_median_over_threshold":
            (cf_median - source_median) / tau if cf_median is not None and tau > 0 else None,
        "decoded_cf_minus_real_x_median_over_threshold":
            (cf_median - real_median) / tau if cf_median is not None and real_median is not None and tau > 0 else None,
        "real_x_inside_source_percent": _percent_inside(real_scores, tau),
        "decoded_cf_inside_source_percent": _percent_inside(cf_scores, tau),
        "real_x_median_source_percentile": _median(real_percentiles),
        "decoded_cf_median_source_percentile": _median(cf_percentiles),
    }
    return row, rows + cf_rows, sampling, hashes


def _aggregate(rows):
    comparable = [row for row in rows if row["n_heldout_real_class1_sampled"] > 0
                  and row["n_decoded_class1_flip"] > 0]

    def med(field):
        return _median([row[field] for row in comparable if row[field] is not None])

    return {
        "task": rows[0]["task"], "n_folds": len(rows),
        "n_comparable_folds": len(comparable),
        "n_eligible_class0": sum(row["n_eligible_class0"] for row in rows),
        "n_typicality_completed": sum(row["n_typicality_completed"] for row in rows),
        "n_typicality_error": sum(row["n_typicality_error"] for row in rows),
        "n_typicality_pending": sum(row["n_typicality_pending"] for row in rows),
        "n_decoded_class1_flip": sum(row["n_decoded_class1_flip"] for row in rows),
        "n_decoded_nonflip": sum(row["n_decoded_nonflip"] for row in rows),
        "subject_median_real_x_inside_source_percent": med("real_x_inside_source_percent"),
        "subject_median_decoded_cf_inside_source_percent": med("decoded_cf_inside_source_percent"),
        "subject_median_real_x_source_percentile": med("real_x_median_source_percentile"),
        "subject_median_decoded_cf_source_percentile": med("decoded_cf_median_source_percentile"),
        "subject_median_real_x_minus_source_over_threshold": med("real_x_minus_source_median_over_threshold"),
        "subject_median_decoded_cf_minus_source_over_threshold": med("decoded_cf_minus_source_median_over_threshold"),
        "subject_median_decoded_cf_minus_real_x_over_threshold": med("decoded_cf_minus_real_x_median_over_threshold"),
    }


def _paragraph(aggregates):
    lines = [r"\paragraph{Subject-invariance.}"]
    lines.append("We compare held-out subjects' real class-1 EEG $X$ and their class-0-to-1 "
                 r"typicality counterfactual EEG $R(Z^{\mathrm{cf}})$ with the other subjects' "
                 "real class-1 trials. Within each leave-one-subject-out fold, the frozen "
                 "classifier re-encodes both trial types and scores them against the same "
                 "source-trained class-1 region; counterfactual results include only decoded "
                 "signals classified as class 1.")
    for row in aggregates:
        task = row["task"].capitalize()
        if not row["n_comparable_folds"]:
            lines.append(f"For {task}, no fold has both sampled real class-1 trials and a "
                         "decoded class-1 counterfactual, so this comparison is unavailable.")
            continue
        real = row["subject_median_real_x_inside_source_percent"]
        cf = row["subject_median_decoded_cf_inside_source_percent"]
        real_pct = row["subject_median_real_x_source_percentile"]
        cf_pct = row["subject_median_decoded_cf_source_percentile"]
        real_gap = row["subject_median_real_x_minus_source_over_threshold"]
        cf_gap = row["subject_median_decoded_cf_minus_source_over_threshold"]
        fold_word = "fold" if row["n_comparable_folds"] == 1 else "folds"
        lines.append(f"For {task}, {row['n_decoded_class1_flip']} of "
                     f"{row['n_eligible_class0']} eligible counterfactuals were decoded "
                     f"class-1 flips. Across {row['n_comparable_folds']} comparable {fold_word}, "
                     f"the subject-median share inside the source class-1 region was "
                     f"{real:.1f}\\% for real $X$ and {cf:.1f}\\% for decoded $R(Z^{{\\mathrm{{cf}}}})$; "
                     f"their subject-median source-discrepancy percentiles were "
                     f"{real_pct:.1f} and {cf_pct:.1f}, respectively. "
                     f"The within-fold median discrepancy gaps from source class-1 trials, "
                     f"scaled by the fold threshold, were {real_gap:+.3f} for real $X$ "
                     f"and {cf_gap:+.3f} for decoded $R(Z^{{\\mathrm{{cf}}}})$.")
        if row["n_typicality_pending"] or row["n_typicality_error"]:
            lines.append(f"The {task} estimate is provisional: "
                         f"{row['n_typicality_pending']} attempts are pending and "
                         f"{row['n_typicality_error']} failed.")
    return "\n".join(lines) + "\n"


def build_decoded_subject_invariance_report(roots, output, *, samples_per_subject=3, seed=42):
    if (isinstance(samples_per_subject, bool) or int(samples_per_subject) != samples_per_subject
            or samples_per_subject < 0):
        raise ValueError("samples_per_subject must be a nonnegative integer")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {output}")
    roots = _study_roots(roots)
    definition = compatible_typicality_definition(roots)
    if definition != {"score": SCORE_DEFINITION, "representation": REPRESENTATION}:
        raise ValueError("Incompatible class-1 typicality definition or representation")
    folds, trials, samples, hashes, seen = [], [], [], [], set()
    for root in roots:
        study_path = root / "study.json"
        study = json.loads(study_path.read_text())
        if study.get("target_class", 1) != 1 or "typicality" not in study.get("objectives", ["typicality"]):
            raise ValueError(f"{study_path}: expected class-1 typicality optimization")
        for entry in study["folds"]:
            key = (study["task"], int(entry["subject_id"]))
            if key in seen:
                raise ValueError(f"Duplicate task/fold across studies: {key}")
            seen.add(key)
            fold, trial_rows, sampling, inputs = _fold(
                root, study["task"], entry, samples_per_subject=samples_per_subject, seed=seed,
            )
            folds.append(fold)
            trials.extend(trial_rows)
            samples.extend(sampling)
            hashes.append({**inputs, "study_json_sha256": file_sha256(study_path)})
    if not folds:
        raise ValueError("No LOSO folds found in the supplied studies")
    aggregates = [_aggregate([row for row in folds if row["task"] == task])
                  for task in sorted({row["task"] for row in folds})]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "decoded_subject_invariance_trials.csv", trials)
    write_csv(output / "decoded_subject_invariance_folds.csv", folds)
    write_csv(output / "decoded_subject_invariance_aggregate.csv", aggregates)
    write_json(output / "decoded_subject_invariance_sampling.json", samples)
    (output / "decoded_subject_invariance_paragraph.tex").write_text(_paragraph(aggregates))
    report = {
        "schema_version": 1,
        "comparison": "Within each LOSO fold: held-out real class-1 X and decoded typicality-arm class-0-to-1 R(Zcf), both re-encoded and scored against source real class-1 trials' frozen Gaussian",
        "selection": "Real X: true class 1, sampled regardless of prediction; decoded R(Zcf): all eligible class-0 trials counted, predicted-class-1 decoded flips in primary score comparison",
        "source_percentile_reference": "all available source real true class-1 trials in the same fold",
        "normalization": "within-fold discrepancy contrasts divided by that fold's source-calibrated threshold; raw discrepancies never pooled across models",
        "samples_per_subject": samples_per_subject, "seed": seed,
        "aggregate": aggregates, "input_hashes": hashes,
        "limitations": [
            "Decoded nonflips and failed or pending optimizations are counted but omitted from the class-1 discrepancy comparison.",
            "A typicality score measures compatibility in the frozen classifier embedding, not waveform similarity or complete subject invariance.",
            "Source subjects trained the classifier; held-out subjects did not.",
            "LOSO fold scores use distinct learned latent coordinate systems, so only within-fold contrasts are aggregated.",
        ],
    }
    write_json(output / "decoded_subject_invariance.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+",
                        help="typicality.runner study roots or a parent containing fold shards")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-subject", type=int, default=3,
                        help="Real class-1 trials per subject; 0 uses all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        build_decoded_subject_invariance_report(
            args.roots, args.out_dir, samples_per_subject=args.samples_per_subject, seed=args.seed,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Decoded subject-invariance report: {error}\n")
    print(f"Wrote decoded subject-invariance report to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
