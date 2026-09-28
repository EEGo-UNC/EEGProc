"""Exact fold-paired alignment tests for completed ICLR counterfactual reports.

Inputs are the archived-score and raw-X report fold CSVs. Sign flipping is
performed over subjects, which are the independent units; trial counts are
reported descriptively and are never treated as independent observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def exact_paired_sign_flip(differences):
    """Two-sided exact randomization p for the mean within-fold difference."""
    all_values = np.asarray(differences, dtype=np.float64)
    if all_values.ndim != 1 or not np.isfinite(all_values).all():
        raise ValueError("Paired differences must be a finite vector")
    values = all_values[all_values != 0]
    n = len(values)
    if n == 0:
        return {"n_folds": len(all_values), "n_nonzero": 0,
                "mean_difference": 0.0, "p_two_sided": 1.0}
    if n > 30:
        raise ValueError("Exact sign flipping is limited to 30 nonzero folds")
    observed = abs(float(values.sum()))
    total = 1 << (n - 1)
    extreme = 0
    # Fix one sign: each omitted global sign reversal has the same absolute
    # statistic, so this still enumerates the exact two-sided distribution.
    for start in range(0, total, 1 << 15):
        masks = np.arange(start, min(start + (1 << 15), total), dtype=np.uint32)
        sums = np.full(len(masks), values[0], dtype=np.float64)
        for bit, value in enumerate(values[1:]):
            sums += np.where((masks >> bit) & 1, value, -value)
        extreme += int(np.count_nonzero(np.abs(sums) >= observed - 1e-12))
    return {"n_folds": len(all_values), "n_nonzero": n,
            "mean_difference": float(all_values.mean()),
            "p_two_sided": extreme / total}


def analyze(latent_folds, raw_folds):
    latent = _rows(latent_folds)
    raw = _rows(raw_folds)
    if len({(r["task"], r["fold_subject"]) for r in latent}) != len(latent):
        raise ValueError("Duplicate task/subject in latent fold report")
    if len({(r["task"], r["fold_subject"]) for r in raw}) != len(raw):
        raise ValueError("Duplicate task/subject in raw X fold report")
    if {(r["task"], r["fold_subject"]) for r in latent} != {
            (r["task"], r["fold_subject"]) for r in raw}:
        raise ValueError("Latent and raw reports must cover identical task/subject folds")
    output = {
        "test": "exact two-sided fold-paired sign flip of mean difference",
        "unit": "held-out subject",
        "scope": "Latent tests compare D(E(X)) and D(Zcf) under the same frozen fold Gaussian. Raw X uses a separate source-subject-out reference.",
        "limitations": [
            "D(Zcf) was explicitly optimized for low discrepancy, so latent p-values do not independently validate decoded EEG alignment.",
            "The raw X 95th-percentile threshold was fitted and calibrated on the same source trials; its coverage is descriptive, so no p-value is assigned to that coverage.",
            "No decoded R(Zcf) waveform discrepancy can be computed from compact runs without saved waveform arrays.",
        ],
        "tasks": {},
    }
    for task in sorted({r["task"] for r in latent}):
        task_latent = [r for r in latent if r["task"] == task]
        paired = [r for r in task_latent if int(r["n_latent_class1_flip"]) > 0]
        task_raw = [r for r in raw if r["task"] == task and r["status"] == "scored"]
        coverage_differences = [float(r["optimized_cf_inside_source_percent"]) -
                                float(r["real_x_inside_source_percent"]) for r in paired]
        percentile_differences = [float(r["optimized_cf_median_source_typical_percentile"]) -
                                  float(r["real_x_median_source_typical_percentile"]) for r in paired]
        normalized_differences = [float(r["optimized_cf_minus_real_x_median_over_threshold"])
                                  for r in paired]
        raw_rank_differences = [float(r["cross_subject_probability_heldout_discrepancy_higher"]) - 0.5
                                for r in task_raw if r["cross_subject_probability_heldout_discrepancy_higher"]]
        real_inside = sum(int(r["n_real_x_inside_source"]) for r in paired)
        real_total = sum(int(r["n_heldout_real_class1_sampled"]) for r in paired)
        cf_inside = sum(int(r["n_optimized_cf_inside_source"]) for r in paired)
        cf_total = sum(int(r["n_latent_class1_flip"]) for r in paired)
        output["tasks"][task] = {
            "n_folds": len(task_latent), "n_paired_counterfactual_folds": len(paired),
            "n_raw_x_scored_folds": len(task_raw),
            "latent_paired_coverage": {
                "real_x_encoded_inside": real_inside, "real_x_encoded_total": real_total,
                "counterfactual_optimized_inside": cf_inside,
                "counterfactual_optimized_total": cf_total,
                "real_x_encoded_percent": 100 * real_inside / real_total,
                "counterfactual_optimized_percent": 100 * cf_inside / cf_total,
                "fold_paired_test": exact_paired_sign_flip(coverage_differences),
            },
            "latent_source_percentile_fold_median_difference":
                exact_paired_sign_flip(percentile_differences),
            "latent_median_discrepancy_over_threshold_cf_minus_real":
                exact_paired_sign_flip(normalized_differences),
            "raw_x_crossfit_probability_heldout_higher_minus_half":
                exact_paired_sign_flip(raw_rank_differences),
        }
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latent_folds", type=Path)
    parser.add_argument("raw_folds", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.latent_folds, args.raw_folds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote paired p-values to {args.out}")


if __name__ == "__main__":
    main()
