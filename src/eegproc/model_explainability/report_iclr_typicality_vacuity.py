#!/usr/bin/env python3
"""How much work does the typicality constraint actually do?

Reads every ``trial_metrics.csv`` under one or more suites (a suite's own
``report/`` and each ``fold_NN/report/``) and answers three questions per task,
all in the fold's own frozen classifier space and against the fold's
source-calibrated threshold tau. New bidirectional archives are reported
separately for each direction; legacy archives retain class-0-to-1 semantics:

  1. original_inside: fraction of eligible class-0 trials whose ORIGINAL
     embedding is already inside the class-1 typicality region
     (original_discrepancy <= tau). If this is high, "every counterfactual is
     typical" says little, because most starting points already were.
  2. inside_per_arm: fraction of counterfactuals inside the region for each arm.
     The ``base`` arm stops when the target is reached and never sees the
     typicality penalty, so its fraction is the fraction that were already
     typical at the moment the ``typicality`` arm switches its penalty on.
  3. moved_by_typicality: among trials that were NOT inside after ``base``,
     the fraction the ``typicality`` arm brought inside. This is the only part
     of the typicality pass rate the penalty can take credit for. It also
     counts trials whose ``typicality`` result is identical to ``base``: the
     penalty switches on only once the target is reached, so identical
     results mean it never ran.

Only completed attempts are counted; each (subject, trial) is counted once per
arm. Output: JSON and CSV next to --out, and a short table on stdout.

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


def _ratio_median(rows, value):
    # A zero source cutoff is valid. Membership remains well-defined, while
    # dividing a discrepancy by that cutoff does not produce a finite ratio.
    rows = list(rows)
    if any(row["threshold"] == 0 for row in rows):
        return None
    return float(np.median([row[value] / row["threshold"] for row in rows]))


def load_rows(suites):
    """Completed rows, one per (task, direction, subject, trial, objective)."""
    rows, seen, files = [], set(), []
    for suite in suites:
        suite = Path(suite)
        if suite.is_file():
            candidates = [suite]
        else:
            candidates = sorted(suite.rglob("report/trial_metrics.csv"))
        for path in candidates:
            files.append(str(path))
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("status") != "completed":
                        continue
                    direction = row.get("direction") or "legacy_0_to_1"
                    key = (row["task"], direction, row["subject_id"], row["trial_id"], row["objective"])
                    if key in seen:
                        continue
                    threshold, original, discrepancy = (_float(row.get(k)) for k in
                                                        ("threshold", "original_discrepancy", "discrepancy"))
                    if threshold is None or threshold < 0 or original is None or discrepancy is None:
                        continue
                    seen.add(key)
                    rows.append({"task": row["task"], "direction": direction,
                                 "subject": row["subject_id"], "trial": row["trial_id"],
                                 "objective": row["objective"], "threshold": threshold,
                                 "original": original, "discrepancy": discrepancy})
    return rows, files


def summarize(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[(row["task"], row["direction"])].append(row)
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
            "n_trials": len(trials),
            "n_subjects": len(per_subject),
            "original_inside": float(np.mean(original_inside)),
            "original_inside_count": int(np.sum(original_inside)),
            "subjects_with_all_originals_inside": int(sum(all(v) for v in per_subject.values())),
            "original_margin_median": _ratio_median(trials.values(), "original"),
            "inside_per_arm": {}, "discrepancy_ratio_median_per_arm": {},
        }
        for arm, arm_rows in sorted(arms.items()):
            inside = [r["discrepancy"] <= r["threshold"] for r in arm_rows.values()]
            result["inside_per_arm"][arm] = {"fraction": float(np.mean(inside)), "count": int(np.sum(inside)),
                                             "n": len(inside)}
            result["discrepancy_ratio_median_per_arm"][arm] = _ratio_median(arm_rows.values(), "discrepancy")
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
        summary[task if direction == "legacy_0_to_1" else f"{task}/{direction}"] = result
    return summary


def write_outputs(out, summary, files):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "typicality_vacuity.json").write_text(json.dumps(
        {"definition": __doc__.split("\n\n")[1].strip(), "inputs": files, "tasks": summary},
        indent=2, allow_nan=False) + "\n")
    with (out / "typicality_vacuity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "quantity", "arm", "value", "count", "n", "direction"])
        for result in summary.values():
            task, direction = result["task"], result["direction"]
            writer.writerow([task, "original_inside", "", result["original_inside"],
                             result["original_inside_count"], result["n_trials"], direction])
            for arm, values in result["inside_per_arm"].items():
                writer.writerow([task, "inside_after_optimization", arm, values["fraction"], values["count"],
                                 values["n"], direction])
            moved = result.get("moved_by_typicality")
            if moved:
                writer.writerow([task, "brought_inside_by_typicality", "typicality",
                                 moved["fraction_of_typicality_passes_owed_to_penalty"],
                                 moved["brought_inside_by_typicality"], moved["n_paired"], direction])


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
    def ratio_text(value):
        return "N/A (zero cutoff)" if value is None else f"{value:.3f}"
    for task, result in summary.items():
        print(f"\n{task}: {result['n_trials']} eligible trials from {result['n_subjects']} held-out subjects")
        print(f"  original already inside region: {result['original_inside_count']}/{result['n_trials']}"
              f" ({100 * result['original_inside']:.1f}%), median original/tau {ratio_text(result['original_margin_median'])}")
        for arm, values in result["inside_per_arm"].items():
            print(f"  {arm:<26} inside after optimization: {values['count']}/{values['n']}"
                  f" ({100 * values['fraction']:.1f}%), median discrepancy/tau "
                  f"{ratio_text(result['discrepancy_ratio_median_per_arm'][arm])}")
        moved = result.get("moved_by_typicality")
        if moved:
            print(f"  outside after base: {moved['outside_after_base']}/{moved['n_paired']};"
                  f" brought inside by the typicality penalty: {moved['brought_inside_by_typicality']};"
                  f" typicality identical to base (penalty never ran): "
                  f"{moved['typicality_identical_to_base']}/{moved['n_paired']}")


if __name__ == "__main__":
    main()
