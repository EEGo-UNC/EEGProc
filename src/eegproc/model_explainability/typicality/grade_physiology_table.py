#!/usr/bin/env python3
"""Produce the graded-physiology column of the counterfactual results table.

Walks a counterfactual suite, refits each fold's PhysiologicalReference from
source-subject EEG, scores every archived counterfactual against it, and prints
median [Q1, Q3] worst-family exceedance per objective.

Every trial passes through the same gate as grade_physiology.py: the refit
bounds must reproduce the archived fraction_in_range values. Trials that fail
the gate are counted and EXCLUDED from the quantiles rather than silently
folded in, and the run exits non-zero if any failed, because a disagreeing
reference is not the one the run used.

Expected layout (as archived by runner.py):

    <suite>/fold_NN/subject_N/trial_M/<objective>/attempt_0001/
        physiology.json
        physiology_counterfactual.npz

Usage
-----
    python grade_physiology_table.py --suite <suite_root> --task arousal

Run from the EEGProc repo root on the branch with physiology.deviation().
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


def stub_tensorflow():
    for name in ("tensorflow", "tensorflow.keras", "keras"):
        module = MagicMock()
        module.__name__ = name
        module.__spec__ = MagicMock()
        sys.modules.setdefault(name, module)


def quantiles(values):
    array = np.asarray(sorted(values), dtype=float)
    return (float(np.median(array)),
            float(np.quantile(array, 0.25)),
            float(np.quantile(array, 0.75)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path,
                        help="suite root containing fold_NN directories")
    parser.add_argument("--task", default="arousal", choices=("arousal", "valence"))
    parser.add_argument("--which", default="counterfactual",
                        choices=("counterfactual", "original", "reconstruction"))
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
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    stub_tensorflow()

    from eegproc.model_explainability.model_agnostic.sic_adapter import load_sic_raw_trials
    from eegproc.model_explainability.typicality.physiology import (
        signal_diagnostics, PhysiologicalReference, DEFAULT_BANDS, FAMILIES)

    if not hasattr(PhysiologicalReference, "deviation"):
        sys.exit("This checkout has no PhysiologicalReference.deviation().")

    config = {
        "raw_eeg_npy": str(args.eeg or repo / "datasets" / "dreamer_eeg.npy"),
        "raw_labels_npy": str(args.labels or repo / "datasets" / "dreamer_labels.npy"),
        "dataset": "dreamer", "label_dimension": args.task,
        "fs": args.fs, "window_sec": 1, "window_overlap": 0,
        "window_normalization": "global_rms",
        "label_threshold_mode": "global", "median_label": 3,
    }
    dataset = load_sic_raw_trials(config)
    print(f"dataset: {len(dataset.features)} trials, feature_order={dataset.feature_order!r}")

    def physical(index, values):
        if dataset.normalization_scale is None:
            return np.asarray(values)
        return (np.asarray(values) * dataset.normalization_scale[index]
                + dataset.normalization_offset[index])

    reference_cache = {}

    def reference_for(held_out):
        if held_out not in reference_cache:
            started = time.time()
            source = [i for i in range(len(dataset.features))
                      if int(dataset.subject_ids[i]) != held_out]
            diagnostics = [
                signal_diagnostics(physical(i, dataset.features[i]), fs=args.fs,
                                   n_channels=14, band_edges=DEFAULT_BANDS,
                                   feature_order=dataset.feature_order or "channel-major")
                for i in source
            ]
            reference_cache[held_out] = PhysiologicalReference.fit(
                diagnostics, quantile=args.quantile,
                required_fraction=args.required_fraction)
            print(f"  fit reference for held-out {held_out}: "
                  f"{len(source)} source trials ({time.time() - started:.1f}s)")
        return reference_cache[held_out]

    subject_pattern = re.compile(r"subject_(\d+)")
    per_objective = defaultdict(list)
    gate_failures = []
    missing = 0

    attempts = sorted(args.suite.rglob(f"attempt_*/physiology_{args.which}.npz"))
    print(f"found {len(attempts)} archived {args.which} files under {args.suite}\n")

    for npz_path in attempts:
        attempt = npz_path.parent
        objective = attempt.parent.parent.name
        match = subject_pattern.search(str(attempt))
        if match is None:
            missing += 1
            continue
        held_out = int(match.group(1))
        summary_path = attempt / "physiology.json"
        if not summary_path.exists():
            missing += 1
            continue

        reference = reference_for(held_out)
        archive = np.load(npz_path)
        values = {family: archive[family] for family in FAMILIES if family in archive.files}
        summary = reference.assess(values)

        recorded = json.loads(summary_path.read_text())[args.which]
        ok = True
        for family, archived in recorded["checks"].items():
            got = summary["checks"][family]["fraction_in_range"]
            want = archived["fraction_in_range"]
            if want is None or got is None:
                ok = ok and (want is got)
            elif abs(got - want) > args.tolerance:
                ok = False
        if not ok:
            gate_failures.append(str(attempt))
            continue

        key = ("worst_family_exceedance" if args.statistic == "worst"
               else "mean_family_exceedance")
        if summary[key] is not None:
            per_objective[objective].append(summary[key])

    print()
    label = "worst-family" if args.statistic == "worst" else "mean-family"
    print(f"{args.task}: median [Q1, Q3] {label} exceedance, half-interval widths\n")
    print(f"{'objective':<34}{'n':>5}{'median':>10}{'Q1':>10}{'Q3':>10}")
    print("-" * 69)
    for objective in sorted(per_objective):
        values = per_objective[objective]
        median, q1, q3 = quantiles(values)
        print(f"{objective:<34}{len(values):>5}{median:>10.3f}{q1:>10.3f}{q3:>10.3f}")
        print(f"{'':34}  -> LaTeX cell:  {median:.3f} [{q1:.2f}, {q3:.2f}]")

    if missing:
        print(f"\nskipped {missing} attempt directories with no physiology.json or subject id")
    if gate_failures:
        print(f"\nGATE FAILED on {len(gate_failures)} trials -- excluded from the quantiles above:")
        for path in gate_failures[:10]:
            print(f"  {path}")
        if len(gate_failures) > 10:
            print(f"  ... and {len(gate_failures) - 10} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
