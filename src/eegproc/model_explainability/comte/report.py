"""Rebuild a paired, failure-inclusive signal comparison without TensorFlow."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..typicality.artifacts import completed_attempt, file_sha256, write_csv, write_json

RATES = ("prediction_validity", "independent_typicality", "vc_typical",
         "physiological_acceptance", "available_physiological_acceptance",
         "joint_success", "available_checks_joint_success")
DISTANCES = ("signal_rmse", "relative_signal_l2", "changed_channels", "channel_sparsity",
             "generation_seconds", "evaluation_seconds", "total_seconds",
             "generation_signal_queries", "generation_classifier_queries", "gradient_evaluations",
             "evaluation_signal_queries", "total_classifier_queries")


def summarize(rows):
    n = len(rows)
    result = dict(n_eligible=n, n_completed=sum(r["status"] == "completed" for r in rows),
                  n_error=sum(r["status"] == "error" for r in rows),
                  n_pending=sum(r["status"] == "pending" for r in rows))
    result["provisional"] = bool(result["n_pending"])
    for key in RATES:
        successes = sum(r.get(key) is True for r in rows)
        unknown = sum(r["status"] in ("completed", "pending") and r.get(key) is None for r in rows)
        result[key + "_n"] = successes
        result[key + "_unknown_n"] = unknown
        result[key + "_percent"] = 100 * successes / n if n and not unknown else None
    for key in DISTANCES:
        values = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        result[key + "_n"] = len(values)
        for label, q in (("q25", .25), ("median", .5), ("q75", .75)):
            result[key + "_" + label] = float(np.quantile(values, q)) if values else None
    return result


def build_report(roots, output):
    output = Path(output)
    trials, subjects, attempts, inputs = [], [], [], []
    seen = set()
    signature = None
    for root in map(Path, roots):
        study = json.loads((root / "study.json").read_text())
        if study.get("comparison_schema") != 1 or study.get("methods") != ["cfo", "comte"]:
            raise ValueError("Expected a paired CoMTE signal comparison study")
        current = study["comparison_settings"]
        if signature is not None and current != signature:
            raise ValueError("Cannot pool comparisons with different thresholds or settings")
        signature = current
        study_hash = file_sha256(root / "study.json")
        inputs.append(dict(path=str(root.resolve()), sha256=study_hash))
        for entry in study["folds"]:
            key = (study["task"], study["direction"], entry["subject_id"])
            if key in seen:
                raise ValueError("Duplicate task/direction/held-out subject across comparison shards")
            seen.add(key)
            fold_dir = root / f"subject_{entry['subject_id']}"
            identity = dict(task=key[0], direction=key[1], subject_id=key[2])
            if not (fold_dir / "fold.json").is_file():
                for method in study["methods"]:
                    subjects.append({**identity, "method": method, "fold_status": "pending",
                                     "n_eligible": None, "provisional": True})
                continue
            fold = json.loads((fold_dir / "fold.json").read_text())
            if fold["checkpoint"] != entry or fold["direction"] != study["direction"]:
                raise ValueError("Comparison fold checkpoint or direction changed")
            for name, digest in fold["calibration_sha256"].items():
                path = fold_dir / "calibration" / name
                if not path.is_file() or file_sha256(path) != digest:
                    raise ValueError(f"Comparison calibration changed: {path}")
            eligible = fold["eligible_trial_ids"]
            if len(set(eligible)) != len(eligible):
                raise ValueError("Duplicate eligible trial IDs")
            audit = json.loads((fold_dir / "eligibility.json").read_text())
            if {r["trial_id"] for r in audit if r["eligible"]} != set(eligible):
                raise ValueError("Comparison eligibility disagrees with fold denominators")
            fold_rows = []
            for trial in eligible:
                for method in study["methods"]:
                    row = {**identity, "trial_id": trial, "method": method, "status": "pending"}
                    arm = fold_dir / f"trial_{trial}" / method
                    complete = completed_attempt(arm)
                    files = sorted(arm.glob("attempt_*/result.json"))
                    chosen = complete / "result.json" if complete else (files[-1] if files else None)
                    for result_path in files:
                        attempt = json.loads(result_path.read_text())
                        if (attempt.get("study_sha256") != study_hash
                                or attempt.get("checkpoint_sha256") != entry["sha256"]
                                or attempt.get("calibration_sha256") != fold["calibration_sha256"]
                                or any(attempt.get(k) != v for k, v in {**identity, "trial_id": trial, "method": method}.items())):
                            raise ValueError(f"Attempt identity/protocol mismatch: {result_path}")
                        attempts.append({**identity, "trial_id": trial, "method": method,
                                         "attempt": result_path.parent.name,
                                         "status": attempt["status"], "error": attempt.get("error"),
                                         "generation_seconds": attempt.get("generation_seconds"),
                                         "generation_signal_queries": attempt.get("generation_signal_queries"),
                                         "generation_classifier_queries": attempt.get("generation_classifier_queries"),
                                         "evaluation_seconds": attempt.get("evaluation_seconds"),
                                         "total_seconds": attempt.get("total_seconds"),
                                         "total_classifier_queries": attempt.get("total_classifier_queries"),
                                         "selected_for_trial": result_path == chosen and (complete is not None or attempt["status"] == "error")})
                    if chosen:
                        result = json.loads(chosen.read_text())
                        # A completed result without its atomic commit is pending.
                        if complete or result["status"] == "error":
                            row.update({k: v for k, v in result.items() if not isinstance(v, (dict, list))})
                            if complete and any(k not in result for k in (*RATES, *DISTANCES)):
                                raise ValueError(f"Missing shared signal evaluation: {chosen}")
                    trials.append(row)
                    fold_rows.append(row)
            for method in study["methods"]:
                subjects.append({**identity, "method": method, "fold_status": fold["status"],
                                 **summarize([r for r in fold_rows if r["method"] == method])})
    population, differences = [], []
    groups = sorted({(r["task"], r["direction"], r["method"]) for r in subjects})
    for task, direction, method in groups:
        selected = [r for r in trials if (r["task"], r["direction"], r["method"]) == (task, direction, method)]
        folds = [r for r in subjects if (r["task"], r["direction"], r["method"]) == (task, direction, method)]
        item = dict(task=task, direction=direction, method=method, n_subjects=len(folds), **summarize(selected))
        item["provisional"] |= any(f["provisional"] for f in folds)
        for field in RATES:
            values = [f.get(field + "_percent") for f in folds if f.get("n_eligible")]
            item[field + "_subject_macro_percent"] = (float(np.mean(values))
                if values and all(v is not None for v in values) and not item["provisional"] else None)
            if item["provisional"]:
                item[field + "_percent"] = None
        population.append(item)
    for task, direction, subject in sorted(seen):
        pair = {r["method"]: r for r in subjects if (r["task"], r["direction"], r["subject_id"]) == (task, direction, subject)}
        row = dict(task=task, direction=direction, subject_id=subject, difference="CFO minus CoMTE")
        for field in (*[k + "_percent" for k in RATES], *[k + "_median" for k in DISTANCES]):
            a, b = pair["cfo"].get(field), pair["comte"].get(field)
            row[field] = a - b if a is not None and b is not None else None
        differences.append(row)
    write_csv(output / "trials.csv", trials)
    write_csv(output / "attempts.csv", attempts)
    write_csv(output / "subjects.csv", subjects)
    write_csv(output / "population.csv", population)
    write_csv(output / "paired_subject_differences.csv", differences)
    payload = dict(comparison_schema=1, inputs=inputs, settings=signature,
                   subjects=subjects, population=population,
                   complete=not any(r["provisional"] for r in subjects))
    write_json(output / "comparison.json", payload)
    def fmt(v):
        return "NA" if v is None else f"{v:.3g}"
    lines = ["# CFO / CoMTE signal comparison", "",
             "Both endpoints pass through the same frozen encoder–classifier. Rates include unsuccessful trials and errors. Pending outcomes remain provisional. Distances include finite unsuccessful endpoints; each CSV reports its available count.", "",
             "Physiology requires every configured check plus VCSC. An unavailable check stays unknown. Available-check results are separate and do not certify full physiological acceptance.", "",
             "| Task | Direction | Subject | Method | N | Valid % | Independent typical % | Physiology % | Joint % | Signal RMSE | Changed channels | Seconds | Classifier queries |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in subjects:
        fields = [r["task"], r["direction"], str(r["subject_id"]), r["method"], str(r.get("n_eligible")),
                  *[fmt(r.get(k)) for k in ("prediction_validity_percent", "independent_typicality_percent",
                    "physiological_acceptance_percent", "joint_success_percent", "signal_rmse_median",
                    "changed_channels_median", "total_seconds_median", "total_classifier_queries_median")]]
        lines.append("| " + " | ".join(fields) + " |")
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    build_report(args.roots, args.out_dir)


if __name__ == "__main__":
    main()
