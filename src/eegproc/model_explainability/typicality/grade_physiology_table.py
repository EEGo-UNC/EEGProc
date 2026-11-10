#!/usr/bin/env python3
"""Produce the graded-physiology column of the counterfactual results table.

Reads saved physiology JSON for every attempt. --report binary summarizes
the recorded checks per objective/family. Graded reporting accepts saved
exceedances, embedded or separate JSON component arrays, or legacy NPZ arrays.

Component arrays are rescored against source-only references and must match
archived binary fractions. Saved graded summaries are reported as saved;
binary-only summaries cannot supply distances. Missing grades are counted
and graded reporting exits nonzero instead of treating them as zero.

Expected layout (as archived by runner.py):

    <suite>/fold_NN/subject_N/trial_M/<objective>/attempt_0001/
        physiology.json  # checks, grades, and per-state compact components

Usage
-----
    python grade_physiology_table.py --suite <suite_root> --task arousal

Run from the EEGProc repo root; no TensorFlow or model loading is required.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def quantiles(values):
    array = np.asarray(sorted(values), dtype=float)
    return (float(np.median(array)),
            float(np.quantile(array, 0.25)),
            float(np.quantile(array, 0.75)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path,
                        help="suite root containing fold_NN directories")
    parser.add_argument("--task", default="arousal", choices=("arousal", "valence"))
    parser.add_argument("--which", default="all",
                        choices=("all", "counterfactual", "original", "reconstruction"),
                        help="all: real EEG and reconstruction once per trial, then every objective's counterfactual")
    parser.add_argument("--report", choices=("graded", "binary"), default="graded")
    parser.add_argument("--repo", default=Path("."), type=Path)
    parser.add_argument("--eeg", default=None, type=Path)
    parser.add_argument("--labels", default=None, type=Path)
    parser.add_argument("--fs", default=128.0, type=float)
    parser.add_argument("--quantile", default=0.95, type=float)
    parser.add_argument("--required-fraction", default=0.95, type=float)
    parser.add_argument("--tolerance", default=1e-9, type=float)
    parser.add_argument("--statistic", default="worst",
                        choices=("worst", "mean"),
                        help="per-trial summary: worst family, or mean across families")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))

    from eegproc.model_explainability.typicality.physiology import FAMILIES
    from eegproc.model_explainability.typicality.grading_archives import (
        BoundsMismatch, GradingUnavailable, SourceReferences, assess_attempt,
        attempt_directories, graded_score, read_summary,
    )

    references = SourceReferences(args)
    subject_pattern = re.compile(r"subject_(\d+)")
    per_objective = defaultdict(list)
    passes = defaultdict(list)
    binary = defaultdict(list)
    gate_failures = []
    unavailable = []
    origins = Counter()
    signals = (("original", "reconstruction", "counterfactual") if args.which == "all"
               else (args.which,))
    attempts = [(which, attempt) for which in signals
                for attempt in attempt_directories(args.suite, which)]
    seen_trials = set()
    print(f"found {len(attempts)} archived {args.which} signal attempts under {args.suite}\n")
    if not attempts:
        print("UNAVAILABLE: no physiology JSON summaries or component archives found.", file=sys.stderr)
        return 2

    for which, attempt in attempts:
        objective = attempt.parent.name
        match = next((subject_pattern.fullmatch(part) for part in reversed(attempt.parts)
                      if subject_pattern.fullmatch(part)), None)
        if match is None:
            unavailable.append((str(attempt), "No subject ID in attempt path"))
            continue
        held_out = int(match.group(1))
        if args.which == "all" and which != "counterfactual":
            # Include direction/fold paths in the identity; only objective copies
            # of the same trial are duplicates.
            trial_key = (which, attempt.parent.parent)
            if trial_key in seen_trials:
                continue
            objective = {"original": "real EEG (original)",
                         "reconstruction": "reconstruction R(Z)"}[which]
        try:
            if args.report == "binary":
                binary[objective].append(read_summary(attempt, which))
                origins["saved_binary_json"] += 1
                if args.which == "all" and which != "counterfactual":
                    seen_trials.add(trial_key)
                continue
            summary, origin = assess_attempt(attempt, which, held_out, references)
            value = graded_score(summary, args.statistic)
            origins[origin] += 1
            if value is not None:
                per_objective[objective].append(value)
                passes[objective].append(summary["available_checks_passed"])
                if args.which == "all" and which != "counterfactual":
                    seen_trials.add(trial_key)
        except BoundsMismatch:
            gate_failures.append(str(attempt))
        except (GradingUnavailable, OSError, ValueError) as error:
            unavailable.append((str(attempt), str(error)))

    if args.report == "binary":
        print(f"{args.task}: saved binary physiology checks from JSON")
        print("In-range fractions and pass percentages; graded distances are not inferred.\n")
        print(f"{'objective':<26}{'family':<24}{'n/total':>10}{'pass %':>10}"
              f"{'median':>10}{'Q1':>10}{'Q3':>10}")
        for objective in sorted(binary):
            summaries = binary[objective]
            for family in FAMILIES:
                checks = [summary["checks"][family] for summary in summaries]
                fractions = [check["fraction_in_range"] for check in checks
                             if check.get("fraction_in_range") is not None]
                flags = [check["passed"] for check in checks if check.get("passed") is not None]
                n = f"{len(fractions)}/{len(summaries)}"
                passed = f"{100 * sum(flags) / len(flags):.2f}" if flags else "N/A"
                values = [f"{value:.6f}" for value in quantiles(fractions)] if fractions else ["N/A"] * 3
                print(f"{objective:<26}{family:<24}{n:>10}{passed:>10}" +
                      "".join(f"{value:>10}" for value in values))
    else:
        label = "worst-family" if args.statistic == "worst" else "mean-family"
        print(f"{args.task}: median [Q1, Q3] {label} exceedance, half-interval widths\n")
        print("'all pass' = every available check family met the required fraction\n")
        print(f"{'signal / objective':<34}{'n':>5}{'median':>10}{'Q1':>10}{'Q3':>10}{'all pass':>10}")
        print("-" * 79)
        baseline_rows = [name for name in ("real EEG (original)", "reconstruction R(Z)")
                         if name in per_objective]
        for objective in baseline_rows + sorted(set(per_objective) - set(baseline_rows)):
            values = per_objective[objective]
            median, q1, q3 = quantiles(values)
            flags = [flag for flag in passes[objective] if flag is not None]
            rate = f"{100 * sum(flags) / len(flags):.1f}%" if flags else "N/A"
            print(f"{objective:<34}{len(values):>5}{median:>10.3f}{q1:>10.3f}{q3:>10.3f}{rate:>10}")
            print(f"{'':34}  -> LaTeX cell:  {median:.3f} [{q1:.3f}, {q3:.3f}]")
    print("\nInput sources: " + ", ".join(f"{name}={count}" for name, count in sorted(origins.items())))
    if origins["saved_json"]:
        print("Saved graded JSON summaries were not independently revalidated against raw EEG.")
    if unavailable:
        print(f"\nUNAVAILABLE on {len(unavailable)} attempts -- excluded from the report:")
        for path, reason in unavailable[:10]:
            print(f"  {path}: {reason}")
    if gate_failures:
        print(f"\nGATE FAILED on {len(gate_failures)} trials -- excluded from the quantiles above:")
        for path in gate_failures[:10]:
            print(f"  {path}")
        return 1
    return 2 if unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())
