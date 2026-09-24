"""Compare archived typicality scores of held-out real trials and optimized counterfactuals.

The frozen source-trained class-1 Gaussian supplies the same D in each LOSO
fold. Real EEG X is encoded to E(X); a typicality counterfactual is scored at
its optimized classification embedding Zcf. The generated waveform R(Zcf)
is not re-encoded or assigned this score. This distinction is recorded in the
outputs. Missing folds and unfinished attempts remain visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .report_iclr_subject_invariance import (
    REPRESENTATION, SCORE_DEFINITION, TypicalityRegion, _load_summary, _median,
    _percent_inside, _source_percentiles, analyze_fold,
    compatible_typicality_definition, file_sha256, write_csv, write_json,
)


def _roots(inputs):
    found = set()
    for root in map(Path, inputs):
        if not root.exists():
            raise ValueError(f"Input study directory does not exist: {root}")
        if (root / "study.json").is_file():
            found.add(root.resolve())
        else:
            found.update(path.parent.resolve() for path in root.rglob("study.json"))
    if not found:
        raise ValueError("No typicality.runner study.json found under the supplied inputs")
    return sorted(found)


def _latest_committed_result(arm):
    for marker in sorted(Path(arm).glob("attempt_*/complete.json"), reverse=True):
        committed = json.loads(marker.read_text()).get("sha256", {})
        result = marker.with_name("result.json")
        if "result.json" not in committed or not result.is_file():
            raise ValueError(f"Incomplete committed typicality attempt: {marker}")
        if file_sha256(result) != committed["result.json"]:
            raise ValueError(f"Committed typicality result changed: {result}")
        return result
    return None


def _counterfactual_score(path, *, task, subject, trial, checkpoint_hash, region):
    result = json.loads(path.read_text())
    for key, expected in (("task", task), ("subject_id", subject), ("trial_id", trial),
                          ("true_class", 0), ("objective", "typicality"),
                          ("checkpoint_sha256", checkpoint_hash),
                          ("typicality_definition", SCORE_DEFINITION),
                          ("typicality_representation", REPRESENTATION),
                          ("status", "completed")):
        if result.get(key) != expected:
            raise ValueError(f"{path}: {key} does not match the fold's typicality trial")
    typicality = result["typicality"]
    if (typicality.get("definition") != SCORE_DEFINITION
            or typicality.get("representation") != REPRESENTATION
            or not np.isclose(float(typicality["threshold"]), region.tau)):
        raise ValueError(f"{path}: incompatible typicality score or source threshold")
    score = float(typicality["counterfactual_discrepancy"])
    if not np.isfinite(score) or score < 0:
        raise ValueError(f"{path}: invalid optimized-embedding discrepancy")
    archive = path.with_name("counterfactual.npz")
    verified = False
    if archive.is_file():
        with np.load(archive, allow_pickle=False) as arrays:
            if "classification_embedding_prime" in arrays:
                embedding = np.asarray(arrays["classification_embedding_prime"], dtype=float)
                if not np.isclose(region.score(embedding)[0], score, rtol=1e-5, atol=1e-7):
                    raise ValueError(f"{path}: saved D(Zcf) disagrees with optimized embedding")
                verified = True
    latent = result["latent_counterfactual"]
    probabilities = np.asarray(latent["probabilities"], dtype=float)
    if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
        raise ValueError(f"{path}: invalid optimized-embedding class probabilities")
    predicted = int(probabilities.argmax())
    if latent.get("predicted_class") != predicted:
        raise ValueError(f"{path}: latent predicted class differs from probabilities")
    return score, predicted, probabilities, verified, archive if verified else None


def _fold(root, study, entry, *, samples_per_subject, seed):
    task, subject = study["task"], int(entry["subject_id"])
    fold_dir = root / f"subject_{subject}"
    _, real_rows, sampling, hashes = analyze_fold(
        root, task, entry, samples_per_subject=samples_per_subject, seed=seed,
    )
    fold = json.loads((fold_dir / "fold.json").read_text())
    region = TypicalityRegion.load(fold_dir / "calibration")
    _, source = _load_summary(fold_dir / "calibration/source_trials", (
        "labels", "probabilities", "discrepancy", "subject_ids", "trial_ids",
    ))
    source_labels = np.asarray(source["labels"], dtype=int)
    source_predictions = np.asarray(source["probabilities"], dtype=float).argmax(axis=1)
    source_scores = np.asarray(source["discrepancy"], dtype=float)
    reference = source_scores[(source_labels == 1) & (source_predictions == 1)
                              & (source_scores <= region.tau)]
    if len(reference) < 2:
        raise ValueError(f"{fold_dir}: need at least two source real class-1 trials that are predicted and typical")
    real = [row for row in real_rows if row["role"] == "heldout"]
    real_scores = np.asarray([row["discrepancy"] for row in real], dtype=float)
    for row, percentile in zip(real, _source_percentiles(real_scores, reference)):
        row.update(role="heldout_real_x", score_space="encoded_real_E(X)",
                   source_typical_percentile=float(percentile))
    _, observed = _load_summary(fold_dir / "observations", (
        "trial_ids", "labels", "probabilities",
    ))
    observed_ids = np.asarray(observed["trial_ids"], dtype=int)
    observed_labels = np.asarray(observed["labels"], dtype=int)
    observed_predictions = np.asarray(observed["probabilities"], dtype=float).argmax(axis=1)
    expected = {int(observed_ids[i]) for i in range(len(observed_ids))
                if observed_labels[i] == 0 and observed_predictions[i] == 0}
    eligible = list(map(int, fold["eligible_trial_ids"]))
    # A user may have intentionally restricted trial_ids in the study; never
    # let a declared eligible trial have the wrong original label/prediction.
    if len(set(eligible)) != len(eligible) or not set(eligible).issubset(expected):
        raise ValueError(f"{fold_dir}: eligible counterfactual trial IDs conflict with observations")
    cf_rows, flipped_scores = [], []
    n_completed = n_error = n_pending = n_nonflip = 0
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
        score, predicted, probabilities, verified, archive = _counterfactual_score(
            path, task=task, subject=subject, trial=trial,
            checkpoint_hash=fold["checkpoint"]["sha256"], region=region,
        )
        n_completed += 1
        if predicted == 1:
            flipped_scores.append(score)
        else:
            n_nonflip += 1
        cf_rows.append({
            "task": task, "fold_subject": subject,
            "role": "typicality_counterfactual", "subject_id": subject,
            "trial_id": trial, "true_class": 0,
            "score_space": "optimized_classification_embedding_Zcf",
            "predicted_class": predicted, "latent_class1_flip": predicted == 1,
            "target_probability": float(probabilities[1]),
            "discrepancy": score, "source_threshold": region.tau,
            "inside_source_region": bool(score <= region.tau),
            "source_typical_percentile": float(_source_percentiles(np.asarray([score]), reference)[0]),
            "score_recomputed_from_archived_Zcf": verified,
            "result_path": str(path.resolve()),
        })
        input_hash = {"result_path": str(path.resolve()), "result_sha256": file_sha256(path)}
        if archive is not None:
            committed = json.loads(path.with_name("complete.json").read_text())["sha256"]
            input_hash.update(optimized_embedding_archive_path=str(archive.resolve()),
                              optimized_embedding_archive_declared_sha256=committed.get("counterfactual.npz"))
        hashes.setdefault("counterfactual_results", []).append(input_hash)
    flipped_scores = np.asarray(flipped_scores, dtype=float)
    real_median, cf_median, source_median = map(_median, (real_scores, flipped_scores, reference))
    tau = region.tau
    row = {
        "task": task, "fold_subject": subject, "fold_status": fold["status"],
        "n_source_subjects": len(set(np.asarray(source["subject_ids"], dtype=int).tolist())),
        "n_source_typical_class1": len(reference),
        "n_heldout_real_class1_sampled": len(real_scores),
        "n_eligible_class0": len(eligible), "n_typicality_completed": n_completed,
        "n_typicality_error": n_error, "n_typicality_pending": n_pending,
        "n_latent_class1_flip": len(flipped_scores), "n_latent_nonflip": n_nonflip,
        "n_real_x_inside_source": int(np.count_nonzero(real_scores <= tau)),
        "n_optimized_cf_inside_source": int(np.count_nonzero(flipped_scores <= tau)),
        "source_threshold": tau,
        "source_typical_median_discrepancy": source_median,
        "real_x_median_discrepancy": real_median,
        "optimized_cf_median_discrepancy": cf_median,
        "real_x_minus_source_median_over_threshold":
            (real_median - source_median) / tau if real_median is not None and tau > 0 else None,
        "optimized_cf_minus_source_median_over_threshold":
            (cf_median - source_median) / tau if cf_median is not None and tau > 0 else None,
        "optimized_cf_minus_real_x_median_over_threshold":
            (cf_median - real_median) / tau if cf_median is not None and real_median is not None and tau > 0 else None,
        "real_x_inside_source_percent": _percent_inside(real_scores, tau),
        "optimized_cf_inside_source_percent": _percent_inside(flipped_scores, tau),
        "real_x_median_source_typical_percentile": _median(_source_percentiles(real_scores, reference)),
        "optimized_cf_median_source_typical_percentile": _median(_source_percentiles(flipped_scores, reference)),
    }
    return row, real + cf_rows, sampling, hashes


def _aggregate(rows, *, expected_subjects):
    comparable = [row for row in rows if row["n_heldout_real_class1_sampled"]
                  and row["n_latent_class1_flip"]]

    def median(field):
        return _median([row[field] for row in comparable if row[field] is not None])

    ids = {row["fold_subject"] for row in rows}
    return {
        "task": rows[0]["task"], "n_observed_folds": len(rows),
        "n_expected_folds": expected_subjects,
        "missing_subject_ids": sorted(set(range(expected_subjects)) - ids),
        "n_comparable_folds": len(comparable),
        "n_running_folds": sum(row["fold_status"] != "completed" for row in rows),
        "n_eligible_class0": sum(row["n_eligible_class0"] for row in rows),
        "n_typicality_completed": sum(row["n_typicality_completed"] for row in rows),
        "n_typicality_error": sum(row["n_typicality_error"] for row in rows),
        "n_typicality_pending": sum(row["n_typicality_pending"] for row in rows),
        "n_latent_class1_flip": sum(row["n_latent_class1_flip"] for row in rows),
        "n_latent_nonflip": sum(row["n_latent_nonflip"] for row in rows),
        "n_real_x_sampled": sum(row["n_heldout_real_class1_sampled"] for row in rows),
        "n_real_x_inside_source": sum(row["n_real_x_inside_source"] for row in rows),
        "n_optimized_cf_inside_source": sum(row["n_optimized_cf_inside_source"] for row in rows),
        "fold_median_real_x_inside_source_percent": median("real_x_inside_source_percent"),
        "fold_median_optimized_cf_inside_source_percent": median("optimized_cf_inside_source_percent"),
        "fold_median_real_x_source_typical_percentile": median("real_x_median_source_typical_percentile"),
        "fold_median_optimized_cf_source_typical_percentile": median("optimized_cf_median_source_typical_percentile"),
        "fold_median_real_x_minus_source_over_threshold": median("real_x_minus_source_median_over_threshold"),
        "fold_median_optimized_cf_minus_source_over_threshold": median("optimized_cf_minus_source_median_over_threshold"),
        "fold_median_optimized_cf_minus_real_x_over_threshold": median("optimized_cf_minus_real_x_median_over_threshold"),
    }


def _paragraph(aggregates):
    lines = [r"\paragraph{Subject-invariance.}",
             "Within each leave-one-subject-out fold, we compare the frozen class-1 typicality "
             "discrepancy of sampled held-out real class-1 EEG $X$ with that of typicality-arm "
             "class-0-to-1 counterfactuals. Both use the same frozen source-trained class-1 "
             "Gaussian; we benchmark their scores against other subjects' correctly "
             "classified, typical real class-1 trials. The real-trial score is $D(E(X))$; the "
             "counterfactual score is $D(Z^{\\mathrm{cf}})$ on the optimized classification "
             "embedding that produces $R(Z^{\\mathrm{cf}})$."]
    for row in aggregates:
        task = row["task"].capitalize()
        if not row["n_comparable_folds"]:
            lines.append(f"For {task}, no observed fold has both sampled real class-1 trials "
                         "and a class-1 optimized counterfactual.")
            continue
        lines.append(
            f"For {task}, {row['n_latent_class1_flip']} of {row['n_eligible_class0']} eligible "
            f"typicality attempts yielded a class-1 optimized embedding. Across "
            f"the observed folds, {row['n_optimized_cf_inside_source']} of "
            f"{row['n_latent_class1_flip']} scored counterfactuals and "
            f"{row['n_real_x_inside_source']} of {row['n_real_x_sampled']} sampled real "
            f"class-1 trials were inside the source-calibrated region. Across "
            f"{row['n_comparable_folds']} comparable folds, the fold-median source-typical "
            f"discrepancy percentile was {row['fold_median_real_x_source_typical_percentile']:.1f} "
            f"for real $X$ and {row['fold_median_optimized_cf_source_typical_percentile']:.1f} "
            f"for optimized counterfactuals; their median discrepancy gaps from the source "
            f"reference, divided by the fold threshold, were "
            f"{row['fold_median_real_x_minus_source_over_threshold']:+.3f} and "
            f"{row['fold_median_optimized_cf_minus_source_over_threshold']:+.3f}, respectively."
        )
        if (row["missing_subject_ids"] or row["n_running_folds"]
                or row["n_typicality_pending"] or row["n_typicality_error"]):
            lines.append(f"The {task} archive is incomplete ({row['n_observed_folds']} of "
                         f"{row['n_expected_folds']} expected folds; "
                         f"{row['n_typicality_pending']} pending and {row['n_typicality_error']} "
                         "failed typicality attempts).")
    lines.append("These scores assess the optimized class-1 representation; they do not "
                 "measure the discrepancy of the decoded waveform itself. Membership inside "
                 "the upper typicality threshold does not by itself show that counterfactual "
                 "scores follow the empirical real-trial discrepancy distribution.")
    return "\n".join(lines) + "\n"


def build_typicality_subject_invariance_report(roots, output, *, samples_per_subject=3,
                                               seed=42, expected_subjects=23):
    if (isinstance(samples_per_subject, bool) or int(samples_per_subject) != samples_per_subject
            or samples_per_subject < 0 or expected_subjects < 1):
        raise ValueError("Invalid subject sampling or expected-subject count")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {output}")
    study_roots = _roots(roots)
    definition = compatible_typicality_definition(study_roots)
    if definition != {"score": SCORE_DEFINITION, "representation": REPRESENTATION}:
        raise ValueError("Incompatible class-1 typicality definition or representation")
    folds, trials, samples, hashes, seen = [], [], [], [], set()
    for root in study_roots:
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
                root, study, entry, samples_per_subject=samples_per_subject, seed=seed,
            )
            folds.append(fold)
            trials.extend(trial_rows)
            samples.extend(sampling)
            hashes.append({**inputs, "study_json_sha256": file_sha256(study_path)})
    if not folds:
        raise ValueError("No LOSO folds found in the supplied studies")
    aggregates = [_aggregate([row for row in folds if row["task"] == task],
                             expected_subjects=expected_subjects)
                  for task in sorted({row["task"] for row in folds})]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "typicality_subject_invariance_trials.csv", trials)
    write_csv(output / "typicality_subject_invariance_folds.csv", folds)
    write_csv(output / "typicality_subject_invariance_aggregate.csv", aggregates)
    write_json(output / "typicality_subject_invariance_sampling.json", samples)
    (output / "typicality_subject_invariance_paragraph.tex").write_text(_paragraph(aggregates))
    report = {
        "schema_version": 1, "typicality_definition": definition,
        "score_space_real": "frozen classifier embedding E(X) of original held-out real EEG",
        "score_space_counterfactual": "optimized classification embedding Zcf, before decoding to R(Zcf)",
        "source_reference": "other subjects' real true-class-1 trials, predicted class 1 and within the source-calibrated typicality threshold",
        "selection": "sampled held-out true class 1 irrespective of prediction; all eligible class-0 typicality attempts counted, optimized class-1 flips in primary score comparison",
        "normalization": "within-fold gaps divided by the fold's source-calibrated threshold; raw scores never pooled across classifiers",
        "samples_per_subject": int(samples_per_subject), "seed": int(seed),
        "expected_subjects": int(expected_subjects), "aggregate": aggregates,
        "input_hashes": hashes,
        "limitations": [
            "This is not a score of decoded R(Zcf); no decoder-encoder round trip is used.",
            "Source subjects trained the classifier; the held-out subject did not.",
            "Only optimized class-1 flips enter the primary counterfactual discrepancy comparison; nonflips, errors, and pending attempts remain in the counts.",
            "Incomplete fold coverage and unfinished attempts make task aggregates provisional.",
        ],
    }
    write_json(output / "typicality_subject_invariance.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+",
                        help="typicality.runner study roots or a parent of task/fold shards")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-subject", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-subjects", type=int, default=23)
    args = parser.parse_args(argv)
    try:
        build_typicality_subject_invariance_report(
            args.roots, args.out_dir, samples_per_subject=args.samples_per_subject,
            seed=args.seed, expected_subjects=args.expected_subjects,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Typicality subject-invariance report: {error}\n")
    print(f"Wrote typicality subject-invariance report to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
