"""Read saved physiology summaries and optional diagnostic component arrays."""

import json
import math
from pathlib import Path
import time

import numpy as np

from .physiology import DEFAULT_BANDS, FAMILIES, PhysiologicalReference, signal_diagnostics


class GradingUnavailable(ValueError):
    """The archive does not contain the requested graded information."""


class BoundsMismatch(ValueError):
    """Recomputed fractions disagree with the saved physiological checks."""


def read_summary(attempt, which):
    path = Path(attempt) / "physiology.json"
    try:
        summary = json.loads(path.read_text())[which]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise GradingUnavailable(f"Cannot read {which} summary from {path}: {error}") from error
    checks = summary.get("checks", {}) if isinstance(summary, dict) else {}
    if any(family not in checks for family in FAMILIES):
        raise GradingUnavailable(f"Incomplete physiological checks in {path}")
    for family in FAMILIES:
        check = checks[family]
        if not isinstance(check, dict):
            raise GradingUnavailable(f"Invalid {family} check in {path}")
        fraction = check.get("fraction_in_range")
        if fraction is not None and (not isinstance(fraction, (int, float))
                                     or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise GradingUnavailable(f"Invalid {family} fraction in {path}")
        if check.get("passed") not in (True, False, None):
            raise GradingUnavailable(f"Invalid {family} pass flag in {path}")
    return summary


def graded_score(summary, statistic):
    key = "worst_family_exceedance" if statistic == "worst" else "mean_family_exceedance"
    if key not in summary:
        means = []
        for check in summary["checks"].values():
            if check.get("fraction_in_range") is None:
                continue
            deviation = check.get("deviation")
            if not isinstance(deviation, dict) or "exceedance_mean" not in deviation:
                raise GradingUnavailable(
                    "physiology.json contains binary checks only or incomplete graded values; "
                    "graded distances were not saved. Use --report binary for the saved checks. "
                    "Graded reporting requires saved exceedances or per-component diagnostics."
                )
            value = deviation["exceedance_mean"]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise GradingUnavailable("Invalid saved per-family exceedance_mean")
            means.append(value)
        if not means:
            raise GradingUnavailable("No saved graded values are available in physiology.json")
        return max(means) if statistic == "worst" else math.fsum(means) / len(means)
    value = summary[key]
    if value is not None and (not isinstance(value, (int, float))
                              or not math.isfinite(value) or value < 0):
        raise GradingUnavailable(f"Invalid saved {key}")
    return value


def attempt_directories(suite, which):
    paths = set()
    for name in ("physiology.json", f"physiology_{which}.json", f"physiology_{which}.npz"):
        paths.update(path.parent for path in Path(suite).rglob(f"attempt_*/{name}"))
    return sorted(paths)


def assess_attempt(attempt, which, held_out, references):
    """Recompute from components when available; otherwise use saved grades."""
    attempt = Path(attempt)
    recorded = read_summary(attempt, which)
    component_json = attempt / f"physiology_{which}.json"
    component_npz = attempt / f"physiology_{which}.npz"
    if component_json.is_file():
        try:
            document = json.loads(component_json.read_text())
            values = {family: np.asarray(document[family], dtype=float) for family in FAMILIES}
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise GradingUnavailable(f"Invalid diagnostic components in {component_json}: {error}") from error
    elif component_npz.is_file():
        with np.load(component_npz, allow_pickle=False) as archive:
            values = {family: archive[family] for family in FAMILIES if family in archive.files}
        if len(values) != len(FAMILIES):
            raise GradingUnavailable(f"Incomplete diagnostic components in {component_npz}")
    else:
        graded_score(recorded, "worst")
        graded_score(recorded, "mean")
        return recorded, "saved_json"

    reference = references(held_out)
    if any(values[family].shape != reference.lower[family].shape for family in FAMILIES):
        raise GradingUnavailable(f"Diagnostic component shapes disagree with the reference in {attempt}")
    summary = reference.assess(values)
    failures = []
    for family in FAMILIES:
        got = summary["checks"][family]["fraction_in_range"]
        want = recorded["checks"][family].get("fraction_in_range")
        if (want is None or got is None) and want is not got:
            failures.append(f"{family}: archived {want!r}, recomputed {got!r}")
        elif want is not None and got is not None and abs(got - want) > references.args.tolerance:
            failures.append(f"{family}: archived {want!r}, recomputed {got!r}")
    if failures:
        raise BoundsMismatch("; ".join(failures))
    return summary, "recomputed_components"


class SourceReferences:
    """Load raw EEG and refit source references only when components need them."""

    def __init__(self, args):
        self.args = args
        self.dataset = None
        self.cache = {}

    def __call__(self, held_out):
        if held_out in self.cache:
            return self.cache[held_out]
        if self.dataset is None:
            from .grading_data import load_grading_trials
            args = self.args
            repo = args.repo.resolve()
            self.dataset = load_grading_trials({
                "raw_eeg_npy": str(args.eeg or repo / "datasets/dreamer_eeg.npy"),
                "raw_labels_npy": str(args.labels or repo / "datasets/dreamer_labels.npy"),
                "dataset": "dreamer", "label_dimension": args.task,
                "fs": args.fs, "window_sec": 1, "window_overlap": 0,
                "window_normalization": "global_rms",
                "label_threshold_mode": "global", "median_label": 3,
            })
            print(f"dataset: {len(self.dataset.features)} trials, "
                  f"feature_order={self.dataset.feature_order!r}")
        started = time.time()
        dataset = self.dataset
        source = np.flatnonzero(dataset.subject_ids != held_out)
        if not len(source):
            raise GradingUnavailable(f"No source trials for held-out subject {held_out}")
        diagnostics = [signal_diagnostics(
            dataset.features[i] * dataset.normalization_scale[i] + dataset.normalization_offset[i],
            fs=self.args.fs, n_channels=14, band_edges=DEFAULT_BANDS,
            feature_order=dataset.feature_order or "channel-major",
        ) for i in source]
        self.cache[held_out] = PhysiologicalReference.fit(
            diagnostics, quantile=self.args.quantile, required_fraction=self.args.required_fraction)
        print(f"reference: held-out {held_out}, {len(source)} source trials "
              f"({time.time() - started:.1f}s)")
        return self.cache[held_out]
