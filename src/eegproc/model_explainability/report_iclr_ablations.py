"""Summarize four-arm ICLR counterfactual studies from archived JSON files.

Run from the EEGProc root with::

    PYTHONPATH=src python -m eegproc.model_explainability.report_iclr_ablations \
        runs/counterfactuals/final-ICLR --out-dir runs/counterfactuals/final-ICLR-report

The input can contain task directories and/or fold_* study shards. No model,
checkpoint, TensorFlow, or EEG arrays are loaded. Flip, confidence, and their
conjunctions refer to latent predictions in latent-only studies; decoded
validity is unavailable there.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


OBJECTIVES = ("target_latent", "base", "typicality", "typicality_no_physiology")
LABELS = {
    "target_latent": r"$\mathcal{L}_{\mathrm{base}}$",
    "base": r"$\mathcal{L}_{\mathrm{cfo}}$",
    "typicality": r"$\mathcal{L}_{\mathrm{cfo}}^{\mathcal{T}_{C_1}}$",
    "typicality_no_physiology":
        r"$\mathcal{L}_{\mathrm{cfo}}^{\mathcal{T}_{C_1},\,\lambda_{\mathrm{phys}}=0}$",
}
TASKS = ("valence", "arousal")


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _study_paths(root: Path) -> list[Path]:
    if (root / "study.json").is_file():
        return [root]
    return sorted({path.parent for path in root.rglob("study.json")})


def _latest_attempt(arm: Path) -> Path | None:
    completed = sorted(arm.glob("attempt_*/complete.json"))
    if completed:
        return completed[-1].parent
    attempted = sorted(arm.glob("attempt_*/result.json"))
    return attempted[-1].parent if attempted else None


def _finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _endpoint(summary: dict, protocol: str, study_threshold) -> dict:
    row = {"status": summary["status"]}
    if row["status"] != "completed":
        row["error"] = summary.get("error")
        return row
    typical = summary["typicality"]["typical"]
    decoded = summary["decoded_trials"][summary["report_output"]]
    if protocol == "latent_only":
        prediction = summary["latent_counterfactual"]
    else:
        prediction = decoded.get("counterfactual")
        typical = prediction.get("typical") if prediction is not None else None
    threshold = _finite(summary.get("required_target_probability", study_threshold))
    if (threshold is not None and study_threshold is not None
            and not math.isclose(threshold, float(study_threshold), abs_tol=1e-12)):
        raise ValueError("Result and study disagree on required target probability")
    target_probability = _finite(prediction.get("target_probability")) if prediction else None
    target_class = summary.get("target_class", 1)
    predicted_class = prediction.get("predicted_class") if prediction else None
    if predicted_class is None and prediction and prediction.get("probabilities"):
        probabilities = prediction["probabilities"]
        predicted_class = max(range(len(probabilities)), key=lambda index: probabilities[index])
    flip = predicted_class == target_class if predicted_class is not None else None
    confidence = (target_probability >= threshold
                  if target_probability is not None and threshold is not None else None)
    if prediction is not None and flip is not None and confidence is not None:
        if prediction.get("success") is not None and bool(prediction["success"]) != (flip and confidence):
            raise ValueError("Saved success disagrees with flip and confidence criterion")
    physiology = summary["physiology"]
    change_mse = _finite(decoded.get("decoded_change_mse"))
    vcsc_penalty = _finite(decoded.get("vcsc_counterfactual"))
    vcsc_tolerance = _finite(summary.get("physiological_tolerance"))
    row.update(
        flip=flip, confidence_acquired=confidence, typical=typical,
        flip_typical=bool(flip and typical) if flip is not None and typical is not None else None,
        confident_flip_typical=(bool(flip and confidence and typical)
                                if flip is not None and confidence is not None and typical is not None else None),
        target_probability=target_probability, required_target_probability=threshold,
        d_z=_finite(summary.get("d_z")),
        delta_dec=math.sqrt(change_mse) if change_mse is not None and change_mse >= 0 else None,
        vcsc_penalty=vcsc_penalty, vcsc_tolerance=vcsc_tolerance,
        vcsc_passed=(vcsc_penalty <= vcsc_tolerance
                     if vcsc_penalty is not None and vcsc_tolerance is not None else None),
        physiological_passed=physiology.get("all_required_passed"),
        available_checks_passed=physiology.get("available_checks_passed"),
        physiological_available_count=physiology.get("available_count"),
        physiological_required_count=physiology.get("required_count"),
        selected_step=summary.get("selected_step"),
        steps_completed=summary.get("steps_completed"),
        stop_reason=summary.get("stop_reason"),
    )
    return row


def collect(root: Path):
    """Return all eligible trial/arm records and declared fold coverage."""
    studies = _study_paths(root)
    if not studies:
        raise ValueError(f"No study.json found under {root}")
    rows, folds, seen = [], [], set()
    definitions, protocols = set(), set()
    thresholds = {}
    for study_dir in studies:
        study = _read_json(study_dir / "study.json")
        task = study["task"].lower()
        if task not in TASKS:
            raise ValueError(f"Unexpected task {task!r} in {study_dir}")
        objectives = tuple(study.get("objectives", ()))
        if set(objectives) != set(OBJECTIVES) or len(objectives) != 4:
            raise ValueError(f"Expected all four ablations in {study_dir}; got {objectives}")
        protocol = study.get("round_trip_evaluation", "latent_only")
        protocols.add(protocol)
        study_threshold = study.get("arguments", {}).get("target_probability")
        if study_threshold is not None:
            study_threshold = _finite(study_threshold)
            if study_threshold is None or not 0 < study_threshold <= 1:
                raise ValueError(f"Invalid target-probability threshold in {study_dir}")
            if task in thresholds and thresholds[task] != study_threshold:
                raise ValueError(f"Cannot pool different {task} confidence thresholds")
            thresholds[task] = study_threshold
        definitions.add((study.get("typicality_definition"),
                         study.get("typicality_representation")))
        for entry in study["folds"]:
            subject = int(entry["subject_id"])
            key = (task, subject)
            if key in seen:
                raise ValueError(f"Duplicate task/subject {key} across study shards")
            seen.add(key)
            fold_dir = study_dir / f"subject_{subject}"
            fold_file = fold_dir / "fold.json"
            if not fold_file.is_file():
                folds.append({"task": task, "subject_id": subject,
                              "status": "pending", "n_eligible": None})
                continue
            fold = _read_json(fold_file)
            if int(fold["subject_id"]) != subject:
                raise ValueError(f"Subject mismatch in {fold_file}")
            trial_ids = fold["eligible_trial_ids"]
            if len(trial_ids) != len(set(trial_ids)):
                raise ValueError(f"Duplicate eligible trial ID in {fold_file}")
            folds.append({"task": task, "subject_id": subject,
                          "status": fold["status"], "n_eligible": len(trial_ids)})
            for trial in trial_ids:
                for objective in OBJECTIVES:
                    row = {"task": task, "subject_id": subject, "trial_id": trial,
                           "objective": objective, "status": "pending"}
                    attempt = _latest_attempt(fold_dir / f"trial_{trial}" / objective)
                    if attempt is not None:
                        summary = _read_json(attempt / "result.json")
                        for name, expected in (("task", task), ("subject_id", subject),
                                               ("trial_id", trial), ("objective", objective)):
                            if name in summary and summary[name] != expected:
                                raise ValueError(f"{name} mismatch in {attempt / 'result.json'}")
                        if summary["status"] == "completed" and not (attempt / "complete.json").is_file():
                            row["error"] = "Uncommitted completed attempt"
                        else:
                            row.update(_endpoint(summary, protocol, study_threshold))
                            row["artifact_directory"] = str(attempt.resolve())
                    rows.append(row)
    if len(protocols) != 1:
        raise ValueError(f"Cannot mix validity protocols: {sorted(protocols)}")
    if len(definitions) != 1:
        raise ValueError("Cannot pool different typicality definitions or representations")
    return rows, folds, studies, protocols.pop(), definitions.pop(), thresholds


def _quantiles(values):
    values = sorted(value for item in values if (value := _finite(item)) is not None)
    if not values:
        return None, None, None, 0

    def percentile(p):
        index = (len(values) - 1) * p
        low = math.floor(index)
        high = math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)

    return percentile(.5), percentile(.25), percentile(.75), len(values)


def summarize(rows: list[dict], *, pending_folds=0) -> dict:
    """Rates use every eligible trial; failed attempts count as failures."""
    n = len(rows)
    completed = [row for row in rows if row["status"] == "completed"]
    out = {"n_eligible": n, "n_completed": len(completed),
           "n_error": sum(row["status"] == "error" for row in rows),
           "n_pending": sum(row["status"] == "pending" for row in rows),
           "n_pending_folds": pending_folds}
    for field in ("flip", "confidence_acquired", "typical", "flip_typical",
                  "confident_flip_typical", "vcsc_passed", "physiological_passed",
                  "available_checks_passed"):
        unknown = any(row.get(field) is None for row in completed)
        if field == "physiological_passed" and out["n_pending"]:
            unknown = True
        out[field + "_percent"] = (100 * sum(row.get(field) is True for row in rows) / n
                                   if n and not unknown else None)
        out[field + "_n_assessed"] = sum(row.get(field) is not None for row in completed)
    for field in ("d_z", "delta_dec"):
        q50, q25, q75, count = _quantiles(row.get(field) for row in rows)
        out.update({field + "_median": q50, field + "_q25": q25,
                    field + "_q75": q75, field + "_n": count})
    out["provisional"] = bool(out["n_pending"] or pending_folds)
    return out


def _csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def _distance(row, field):
    digits = 4 if field == "delta_dec" else 3
    return (f"{_fmt(row[field + '_median'], digits)} "
            f"[{_fmt(row[field + '_q25'], digits)}, {_fmt(row[field + '_q75'], digits)}]"
            if row[field + "_median"] is not None else "--")


def _table(population: list[dict], *, include_provisional=False) -> str:
    caption = ("Counterfactual displacement and physiological diagnostics across eligible class-0 trials. "
               "Distances are median [Q1, Q3] over finite endpoints. "
               "VCSC is the fraction with raw VCSC penalty at or below the archived tolerance, "
               "including arms without a VCSC optimization penalty. "
               "Phys. (4/4) requires passage of all four assessable physiological checks. "
               "The fifth check, aperiodic exponent, is unavailable for the band-filtered decoder; "
               "-- means task results are incomplete or unavailable.")
    if include_provisional:
        caption += (" A dagger marks a provisional task: its values summarize only the "
                    "archived user folds and must not be interpreted as population results.")
    lines = [r"\begin{table}[H]", f"\\caption{{{caption}}}",
             r"\label{tab:counterfactual_results}", r"\begin{center}",
             r"\begin{tabular}{llcccc}",
             r"\textbf{Task} & \textbf{Objective} & $\mathbf{d_Z}$ & $\Delta_{\mathrm{dec}}$ & \textbf{VCSC (\%)} & \textbf{Phys. 4/4 (\%)} \\ \hline"]
    for task in TASKS:
        for objective in OBJECTIVES:
            row = next((item for item in population if item["task"] == task and
                        item["objective"] == objective), None)
            if row is None or (row["provisional"] and (not include_provisional or not row["n_eligible"])):
                entries = [task.title(), LABELS[objective], *("--",) * 4]
            else:
                name = task.title() + (r"$^{\dagger}$" if row["provisional"] else "")
                entries = [name, LABELS[objective],
                           _distance(row, "d_z"), _distance(row, "delta_dec"),
                           _fmt(row["vcsc_passed_percent"]),
                           _fmt(row["available_checks_passed_percent"])]
            lines.append(" & ".join(entries) + r" \\")
    lines += [r"\end{tabular}", r"\end{center}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_report(input_root: Path, out_dir: Path, *, expected_subjects: int = 23) -> dict:
    if expected_subjects < 1:
        raise ValueError("expected_subjects must be positive")
    rows, folds, studies, protocol, definition, thresholds = collect(input_root)
    check_counts = {(row.get("physiological_available_count"),
                     row.get("physiological_required_count"))
                    for row in rows if row["status"] == "completed"}
    if check_counts and check_counts != {(4, 5)}:
        raise ValueError(f"Phys. 4/4 table requires four of five available checks; found {check_counts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects, population = [], []
    missing_subjects = {}
    for task in TASKS:
        task_folds = [fold for fold in folds if fold["task"] == task]
        observed_ids = {fold["subject_id"] for fold in task_folds}
        missing_subjects[task] = sorted(set(range(expected_subjects)) - observed_ids)
        unexpected_ids = sorted(observed_ids - set(range(expected_subjects)))
        if unexpected_ids:
            raise ValueError(f"Unexpected {task} subject IDs: {unexpected_ids}")
        for objective in OBJECTIVES:
            group = [row for row in rows if row["task"] == task and row["objective"] == objective]
            population.append({"task": task, "objective": objective,
                               **summarize(group, pending_folds=len(missing_subjects[task]) + sum(
                                   fold["status"] != "completed" for fold in task_folds)),
                               "n_expected_subjects": expected_subjects})
            for fold in task_folds:
                subject_rows = [row for row in group if row["subject_id"] == fold["subject_id"]]
                subjects.append({"task": task, "subject_id": fold["subject_id"],
                                 "objective": objective, "fold_status": fold["status"],
                                 **summarize(subject_rows, pending_folds=int(fold["status"] != "completed"))})
    _csv(out_dir / "trial_optimizations.csv", rows)
    _csv(out_dir / "user_optimizations.csv", subjects)
    _csv(out_dir / "population_ablations.csv", population)
    user_dir = out_dir / "users"
    user_dir.mkdir(exist_ok=True)
    for fold in folds:
        selected = [row for row in subjects if row["task"] == fold["task"]
                    and row["subject_id"] == fold["subject_id"]]
        lines = [f"# {fold['task'].title()} user {fold['subject_id']}", "",
                 f"Fold status: {fold['status']}; eligible trials: {fold['n_eligible'] if fold['n_eligible'] is not None else 'unknown'}.",
                 "", "| Objective | Completed / eligible | Flip % | Conf. % | Typ. % | Flip+Typ. % | Conf.+Typ. % | d_Z median [Q1, Q3] | Delta_dec median [Q1, Q3] | Phys. % |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in selected:
            lines.append("| " + " | ".join((row["objective"],
                         f"{row['n_completed']} / {row['n_eligible']}",
                         *(_fmt(row[field + "_percent"]) for field in (
                             "flip", "confidence_acquired", "typical", "flip_typical",
                             "confident_flip_typical")),
                         _distance(row, "d_z"), _distance(row, "delta_dec"),
                         _fmt(row["physiological_passed_percent"]))) + " |")
        lines += ["", "Flip is target-class argmax; Conf. is target probability at or above the archived threshold.",
                  "Flip+Typ. and Conf.+Typ. are separate conjunctions; the latter also requires a flip.",
                  "Rates use all eligible trials. Distances use finite selected endpoints.",
                  "Phys. is unavailable when a required check was not assessed.", ""]
        (user_dir / f"{fold['task']}_user_{fold['subject_id']}.md").write_text(
            "\n".join(lines), encoding="utf-8")
    (out_dir / "counterfactual_results.tex").write_text(_table(population), encoding="utf-8")
    (out_dir / "counterfactual_results_with_provisional.tex").write_text(
        _table(population, include_provisional=True), encoding="utf-8")
    complete = (not any(missing_subjects.values())
                and all(fold["status"] == "completed" for fold in folds)
                and all(row["status"] != "pending" for row in rows))
    payload = {"input": str(input_root.resolve()), "studies": [str(path.resolve()) for path in studies],
               "validity_protocol": protocol, "confidence_thresholds": thresholds,
               "typicality_definition": definition[0],
               "typicality_representation": definition[1], "complete": complete,
               "expected_subjects_per_task": expected_subjects,
               "missing_subjects": missing_subjects, "folds": folds, "population": population}
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-subjects", type=int, default=23,
                        help="Expected consecutive user IDs per task (default: 23 for DREAMER)")
    args = parser.parse_args(argv)
    report = build_report(args.input_root, args.out_dir,
                          expected_subjects=args.expected_subjects)
    print(f"Wrote {args.out_dir}; complete={report['complete']}; protocol={report['validity_protocol']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
