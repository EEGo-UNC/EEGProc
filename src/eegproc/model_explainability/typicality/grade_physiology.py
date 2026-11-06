#!/usr/bin/env python3
"""Rescore an archived counterfactual's physiological checks on a graded scale.

Reads physiology.json for saved binary checks, graded distances, or per-state
component values embedded by PhysiologicalReference.assess().
Optional physiology_<which>.json component arrays and legacy NPZ archives
can be rescored against a source-only reference.

Rescored components must reproduce archived binary fractions. Saved graded
JSON is reported with its provenance; binary-only JSON cannot supply graded
distances. Use --report binary to display the saved checks directly.

Usage
-----
    python grade_physiology.py \
        --attempt  runs/.../subject_0/trial_16/typicality/attempt_0001 \
        --held-out 0 \
        --task     arousal

Run from the EEGProc repo root. Raw EEG is needed only to rescore component
arrays; saved binary/graded JSON needs no model or TensorFlow.
"""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attempt", required=True, type=Path,
                        help="attempt_0001 directory holding physiology.json and optional component arrays")
    parser.add_argument("--held-out", required=True, type=int,
                        help="held-out subject id; the reference is fit on its LOSO complement")
    parser.add_argument("--task", default="arousal", choices=("arousal", "valence"))
    parser.add_argument("--which", default="counterfactual",
                        choices=("counterfactual", "original", "reconstruction"))
    parser.add_argument("--report", choices=("graded", "binary"), default="graded",
                        help="Graded distances, or the saved per-family binary checks.")
    parser.add_argument("--repo", default=Path("."), type=Path, help="EEGProc repo root")
    parser.add_argument("--eeg", default=None, type=Path)
    parser.add_argument("--labels", default=None, type=Path)
    parser.add_argument("--fs", default=128.0, type=float)
    parser.add_argument("--quantile", default=0.95, type=float)
    parser.add_argument("--required-fraction", default=0.95, type=float)
    parser.add_argument("--tolerance", default=1e-9, type=float,
                        help="max allowed difference against the archived fractions")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))

    from eegproc.model_explainability.typicality.physiology import FAMILIES
    from eegproc.model_explainability.typicality.grading_archives import (
        BoundsMismatch, GradingUnavailable, SourceReferences, assess_attempt, graded_score, read_summary,
    )

    if args.report == "binary":
        try:
            summary = read_summary(args.attempt, args.which)
        except GradingUnavailable as error:
            print(f"UNAVAILABLE: {error}", file=sys.stderr)
            return 2
        print(f"{args.which}: saved binary physiology checks from JSON")
        print("Graded distances are not inferred from pass fractions.\n")
        print(f"{'family':<24}{'passed':>12}{'in-range fraction':>20}")
        for family in FAMILIES:
            check = summary["checks"][family]
            fraction = check.get("fraction_in_range")
            label = "N/A" if fraction is None else f"{fraction:.6f}"
            passed = "N/A" if check.get("passed") is None else str(check["passed"])
            print(f"{family:<24}{passed:>12}{label:>20}")
        return 0

    try:
        summary, origin = assess_attempt(args.attempt, args.which, args.held_out, SourceReferences(args))
    except BoundsMismatch as error:
        print("BOUNDS DO NOT MATCH THE ARCHIVE -- graded numbers withheld:")
        print(error)
        return 1
    except (GradingUnavailable, OSError, ValueError) as error:
        print(f"UNAVAILABLE: {error}", file=sys.stderr)
        return 2
    if origin == "saved_json":
        print("Using saved graded JSON; reference bounds were not independently recomputed.")
    else:
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
    print(f"graded families      : {summary.get('graded_count', 'N/A')}")
    for label, statistic in (("worst family mean exc", "worst"), ("mean family exc", "mean")):
        value = graded_score(summary, statistic)
        print(f"{label}: " + ("N/A" if value is None else f"{value:.4f}"))
    print("\nunit: half-interval widths (z <= 1 == inside the source interval)")


if __name__ == "__main__":
    raise SystemExit(main())
