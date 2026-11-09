#!/usr/bin/env python3
"""How much work does the typicality constraint actually do?

Reads every ``trial_metrics.csv`` under one or more suites (a suite's own
``report/`` and each ``fold_NN/report/``) and answers three questions per task,
all in the fold's own frozen classifier space and against the fold's
source-calibrated threshold tau:

  1. original_inside: fraction of completed source-class trials whose ORIGINAL
     embedding is already inside the direction's target-class typicality region
     (original_discrepancy <= tau). If this is high, "every counterfactual is
     typical" says little, because most starting points already were.
  2. inside_per_arm: fraction of counterfactuals inside the region for each arm.
     The ``base`` arm stops when the target is reached and never sees the
     typicality penalty, so its fraction is the fraction that were already
     typical at the moment the ``typicality`` arm switches its penalty on.
  3. moved_by_typicality: among trials that were NOT inside after ``base``,
     the fraction the ``typicality`` arm brought inside. This is the only part
     of the typicality pass rate with a paired region-entry difference. It also
     counts trials whose ``typicality`` result is identical to ``base``: the
     penalty switches on only once the target is reached, so identical
     endpoints do not establish whether the penalty ran.

Only completed finite attempts are counted; each (direction, subject, trial)
is counted once per arm. These are occupancy rates among completed endpoints,
not all-eligible success rates. Directions are never pooled. Output: JSON and
CSV next to --out, and a short table on stdout.

    python -m eegproc.model_explainability.report_iclr_typicality_vacuity \\
        --suite runs/counterfactuals/arousal_suite_..._2111899 \\
        --suite runs/counterfactuals/valence_suite_..._2112037 \\
        --out runs/counterfactuals/typicality_vacuity
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def load_rows(suites):
    """Completed rows, one per (task, subject, trial, objective)."""
    rows, seen, files = [], {}, []
    for suite in suites:
        suite = Path(suite)
        if suite.is_file():
            candidates = [suite]
        else:
            candidates = sorted(suite.rglob("trial_metrics.csv"))
        for path in candidates:
            files.append(str(path))
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("status") != "completed":
                        continue
                    direction = row.get("direction") or "0_to_1"
                    if direction not in ("0_to_1", "1_to_0"):
                        raise ValueError("Unknown counterfactual direction")
                    key = (row["task"], direction, row["subject_id"], row["trial_id"], row["objective"])
                    threshold, original, discrepancy = (_float(row.get(k)) for k in
                                                        ("threshold", "original_discrepancy", "discrepancy"))
                    if threshold is None or threshold < 0 or original is None or discrepancy is None:
                        continue
                    values = (threshold, original, discrepancy)
                    if key in seen:
                        if seen[key] != values:
                            raise ValueError("Conflicting occupancy rows; keep protocols in separate reports")
                        continue
                    seen[key] = values
                    rows.append({"task": row["task"], "direction": direction,
                                 "subject": row["subject_id"], "trial": row["trial_id"],
                                 "objective": row["objective"], "threshold": threshold,
                                 "original": original, "discrepancy": discrepancy})
    return rows, files


def summarize(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[(row["task"], row.get("direction", "0_to_1"))].append(row)
    summary = {}
    for (task, direction), task_rows in sorted(by_task.items()):
        trials = {}
        arms = defaultdict(dict)
        for row in task_rows:
            trial = (row["subject"], row["trial"])
            trials[trial] = row
            arms[row["objective"]][trial] = row
        original_inside = [r["original"] <= r["threshold"] for r in trials.values()]
        per_subject = defaultdict(list)
        for (subject, _), row in trials.items():
            per_subject[subject].append(row["original"] <= row["threshold"])
        result = {
            "task": task, "direction": direction,
            "source_class": int(direction[0]), "target_class": int(direction[-1]),
            "n_trials": len(trials),
            "n_subjects": len(per_subject),
            "original_inside": float(np.mean(original_inside)),
            "original_inside_count": int(np.sum(original_inside)),
            "subjects_with_all_originals_inside": int(sum(all(v) for v in per_subject.values())),
            "original_margin_median": _median_ratio(trials.values(), "original"),
            "inside_per_arm": {}, "discrepancy_ratio_median_per_arm": {},
        }
        for arm, arm_rows in sorted(arms.items()):
            inside = [r["discrepancy"] <= r["threshold"] for r in arm_rows.values()]
            result["inside_per_arm"][arm] = {"fraction": float(np.mean(inside)), "count": int(np.sum(inside)),
                                             "n": len(inside)}
            result["discrepancy_ratio_median_per_arm"][arm] = _median_ratio(arm_rows.values(), "discrepancy")
        if "base" in arms and "typicality" in arms:
            common = sorted(set(arms["base"]) & set(arms["typicality"]))
            outside_after_base = [t for t in common if arms["base"][t]["discrepancy"] > arms["base"][t]["threshold"]]
            moved = [t for t in outside_after_base
                     if arms["typicality"][t]["discrepancy"] <= arms["typicality"][t]["threshold"]]
            identical = [t for t in common
                         if arms["typicality"][t]["discrepancy"] == arms["base"][t]["discrepancy"]]
            result["moved_by_typicality"] = {
                "n_paired": len(common), "typicality_identical_to_base": len(identical), "outside_after_base": len(outside_after_base),
                "brought_inside_by_typicality": len(moved),
                "fraction_of_typicality_passes_owed_to_penalty": (
                    len(moved) / max(1, sum(arms["typicality"][t]["discrepancy"] <= arms["typicality"][t]["threshold"]
                                            for t in common)))}
        summary[task if direction == "0_to_1" else f"{task}/{direction}"] = result
    return summary


def _median_ratio(rows, field):
    values = [r[field] / r["threshold"] for r in rows if r["threshold"] > 0]
    return float(np.median(values)) if values else None


def write_outputs(out, summary, files):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "typicality_vacuity.json").write_text(json.dumps(
        {"definition": __doc__.split("\n\n")[1].strip(), "inputs": files, "tasks": summary},
        indent=2, allow_nan=False) + "\n")
    with (out / "typicality_vacuity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "quantity", "arm", "value", "count", "n", "direction"])
        for task, result in summary.items():
            writer.writerow([result["task"], "original_inside", "", result["original_inside"],
                             result["original_inside_count"], result["n_trials"], result["direction"]])
            for arm, values in result["inside_per_arm"].items():
                writer.writerow([result["task"], "inside_after_optimization", arm, values["fraction"], values["count"],
                                 values["n"], result["direction"]])
            moved = result.get("moved_by_typicality")
            if moved:
                writer.writerow([result["task"], "brought_inside_by_typicality", "typicality",
                                 moved["fraction_of_typicality_passes_owed_to_penalty"],
                                 moved["brought_inside_by_typicality"], moved["n_paired"], result["direction"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", action="append", required=True,
                        help="Suite directory (or a single trial_metrics.csv); repeat for several")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows, files = load_rows(args.suite)
    if not rows:
        raise SystemExit("No completed rows with threshold/original_discrepancy/discrepancy found")
    summary = summarize(rows)
    write_outputs(args.out, summary, files)
    for task, result in summary.items():
        print(f"\n{task}: {result['n_trials']} completed class-{result['source_class']} trials "
              f"from {result['n_subjects']} held-out subjects ({result['direction']})")
        print(f"  original already inside region: {result['original_inside_count']}/{result['n_trials']}"
              f" ({100 * result['original_inside']:.1f}%), median original/tau {result['original_margin_median']}")
        for arm, values in result["inside_per_arm"].items():
            print(f"  {arm:<26} inside after optimization: {values['count']}/{values['n']}"
                  f" ({100 * values['fraction']:.1f}%), median discrepancy/tau "
                  f"{result['discrepancy_ratio_median_per_arm'][arm]}")
        moved = result.get("moved_by_typicality")
        if moved:
            print(f"  outside after base: {moved['outside_after_base']}/{moved['n_paired']};"
                  f" brought inside by the typicality penalty: {moved['brought_inside_by_typicality']};"
                  f" typicality identical to base: "
                  f"{moved['typicality_identical_to_base']}/{moved['n_paired']}")


if __name__ == "__main__":
    main()
