#!/usr/bin/env python3
"""Rescore an archived counterfactual's physiological checks on a graded scale.

Refits the fold's PhysiologicalReference from source-subject EEG (exactly as
runner.py does), then scores an archived physiology_*.npz against it and reports
the graded deviation added in commit 832ca13.

The script REFUSES to print graded numbers unless the refit bounds first
reproduce the archived binary fractions in physiology.json. That check is the
whole point: if it fails, the bounds are not the ones the run used.

Usage
-----
    python grade_physiology.py \
        --attempt  runs/.../subject_0/trial_16/typicality/attempt_0001 \
        --held-out 0 \
        --task     arousal

Run it from the EEGProc repo root, on the branch with physiology.deviation()
(graded-physiological-checks or later), with datasets/dreamer_eeg.npy present.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


def stub_tensorflow():
    """sic_adapter imports TF at module scope but never uses it to load data."""
    for name in ("tensorflow", "tensorflow.keras", "keras"):
        module = MagicMock()
        module.__name__ = name
        module.__spec__ = MagicMock()
        sys.modules.setdefault(name, module)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attempt", required=True, type=Path,
                        help="attempt_0001 directory holding physiology*.npz and physiology.json")
    parser.add_argument("--held-out", required=True, type=int,
                        help="held-out subject id; the reference is fit on its LOSO complement")
    parser.add_argument("--task", default="arousal", choices=("arousal", "valence"))
    parser.add_argument("--which", default="counterfactual",
                        choices=("counterfactual", "original", "reconstruction"))
    parser.add_argument("--repo", default=Path("."), type=Path, help="EEGProc repo root")
    parser.add_argument("--eeg", default=None, type=Path)
    parser.add_argument("--labels", default=None, type=Path)
    parser.add_argument("--fs", default=128.0, type=float)
    parser.add_argument("--quantile", default=0.95, type=float)
    parser.add_argument("--required-fraction", default=0.95, type=float)
    parser.add_argument("--tolerance", default=1e-9, type=float,
                        help="max allowed difference against the archived fractions")
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    stub_tensorflow()

    from eegproc.model_explainability.model_agnostic.sic_adapter import load_sic_raw_trials
    from eegproc.model_explainability.typicality.physiology import (
        signal_diagnostics, PhysiologicalReference, DEFAULT_BANDS, FAMILIES)

    if not hasattr(PhysiologicalReference, "deviation"):
        sys.exit("This checkout has no PhysiologicalReference.deviation(). "
                 "Check out graded-physiological-checks (or later) first.")

    # ---- 1. rebuild the dataset exactly as the run's study.json describes ----
    config = {
        "raw_eeg_npy": str(args.eeg or repo / "datasets" / "dreamer_eeg.npy"),
        "raw_labels_npy": str(args.labels or repo / "datasets" / "dreamer_labels.npy"),
        "dataset": "dreamer", "label_dimension": args.task,
        "fs": args.fs, "window_sec": 1, "window_overlap": 0,
        "window_normalization": "global_rms",
        "label_threshold_mode": "global", "median_label": 3,
    }
    started = time.time()
    dataset = load_sic_raw_trials(config)
    print(f"dataset: {len(dataset.features)} trials, "
          f"feature_order={dataset.feature_order!r} ({time.time() - started:.1f}s)")

    def physical(index, values):
        if dataset.normalization_scale is None:
            return np.asarray(values)
        return (np.asarray(values) * dataset.normalization_scale[index]
                + dataset.normalization_offset[index])

    # ---- 2. refit the physiological reference on the LOSO complement ----
    source = [i for i in range(len(dataset.features))
              if int(dataset.subject_ids[i]) != args.held_out]
    if not source:
        sys.exit(f"No source trials for held-out subject {args.held_out}")
    started = time.time()
    diagnostics = [
        signal_diagnostics(physical(i, dataset.features[i]), fs=args.fs, n_channels=14,
                           band_edges=DEFAULT_BANDS,
                           feature_order=dataset.feature_order or "channel-major")
        for i in source
    ]
    reference = PhysiologicalReference.fit(diagnostics, quantile=args.quantile,
                                           required_fraction=args.required_fraction)
    print(f"reference: {len(source)} source trials from "
          f"{len(set(int(dataset.subject_ids[i]) for i in source))} subjects "
          f"({time.time() - started:.1f}s)")

    # ---- 3. score the archived candidate ----
    archive = np.load(args.attempt / f"physiology_{args.which}.npz")
    values = {family: archive[family] for family in FAMILIES if family in archive.files}
    summary = reference.assess(values)

    # ---- 4. GATE: the refit bounds must reproduce the archived fractions ----
    recorded = json.loads((args.attempt / "physiology.json").read_text())[args.which]
    failures = []
    for family, archived in recorded["checks"].items():
        got = summary["checks"][family]["fraction_in_range"]
        want = archived["fraction_in_range"]
        if want is None or got is None:
            if want is not got:
                failures.append(f"  {family}: archived {want!r}, recomputed {got!r}")
            continue
        if abs(got - want) > args.tolerance:
            failures.append(f"  {family}: archived {want!r}, recomputed {got!r}")
    if failures:
        print("\nBOUNDS DO NOT MATCH THE ARCHIVE -- graded numbers withheld:")
        print("\n".join(failures))
        sys.exit(1)
    print(f"gate passed: refit bounds reproduce every archived fraction "
          f"(|delta| <= {args.tolerance:g})\n")

    # ---- 5. report ----
    print(f"{args.which}: passed {summary['passed_count']}/{summary['available_count']} "
          f"evaluable checks ({summary['required_count']} families defined)\n")
    header = f"{'family':<24}{'in range':>10}{'mean exc':>11}{'med exc':>10}{'max exc':>10}{'z med':>9}{'z max':>9}"
    print(header)
    print("-" * len(header))
    for family in FAMILIES:
        check = summary["checks"][family]
        deviation = check.get("deviation")
        if deviation is None:
            print(f"{family:<24}{'--':>10}{'unscorable':>11}   ({check['reason']})")
            continue
        print(f"{family:<24}{check['fraction_in_range']:>10.4f}"
              f"{deviation['exceedance_mean']:>11.4f}{deviation['exceedance_median']:>10.4f}"
              f"{deviation['exceedance_max']:>10.4f}{deviation['z_median']:>9.4f}"
              f"{deviation['z_max']:>9.4f}")
    print()
    print(f"graded families      : {summary['graded_count']}")
    print(f"worst family mean exc: {summary['worst_family_exceedance']:.4f}")
    print(f"mean family exc      : {summary['mean_family_exceedance']:.4f}")
    print("\nunit: half-interval widths (z <= 1 == inside the source interval)")


if __name__ == "__main__":
    main()
