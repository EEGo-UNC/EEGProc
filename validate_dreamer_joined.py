#!/usr/bin/env python3
"""
Audit a DREAMER joined CSV before using it in EEGProc.

This script is deliberately stricter than a normal CSV reader. It checks:

- expected DREAMER schema (23 subjects x 18 trials)
- baseline/stimulus presence and sample counts
- sample_idx ordering, gaps, duplicates, and reopened blocks
- numeric/finite EEG values in all 14 Emotiv channels
- constant, duplicated, or suspiciously correlated EEG channels
- exact duplicate stimulus recordings across trials
- valence/arousal availability and consistency within each trial
- binary class balance using the EEGProc convention: score >= 3 -> class 1
- whether labels are actually present on stimulus rows, as required by the
  current prepare_datasets.py loader
- rough spectral power, including whether meaningful >30 Hz content remains

The file is processed in chunks, so it can audit a large long-form CSV without
holding the entire dataset in memory.

Examples
--------
python validate_dreamer_joined.py datasets/dreamer_joined.csv

python validate_dreamer_joined.py datasets/dreamer_joint.csv \
    --report-json dreamer_audit.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


EEG_COLUMNS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

REQUIRED_COLUMNS = {
    "subject_id",
    "trial_id",
    "segment",
    "sample_idx",
    "valence",
    "arousal",
    *EEG_COLUMNS,
}

SEGMENT_ALIASES = {
    "baseline": "baseline",
    "base": "baseline",
    "stimulus": "stimulus",
    "stimuli": "stimulus",
    "label": "label",
}

SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "ERROR": 2}


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class SegmentStats:
    count: int = 0
    min_idx: int | None = None
    max_idx: int | None = None
    first_idx: int | None = None
    last_idx: int | None = None
    continuity_errors: int = 0
    duplicate_or_reverse_steps: int = 0
    digest: Any = field(
        default_factory=lambda: hashlib.blake2b(digest_size=16),
        repr=False,
    )

    def update_indices(self, values: np.ndarray) -> None:
        if values.size == 0:
            return

        values = np.asarray(values, dtype=np.int64)
        first = int(values[0])
        last = int(values[-1])

        if self.first_idx is None:
            self.first_idx = first
        elif self.last_idx is not None:
            boundary_step = first - self.last_idx
            if boundary_step != 1:
                self.continuity_errors += 1
                if boundary_step <= 0:
                    self.duplicate_or_reverse_steps += 1

        if values.size > 1:
            steps = np.diff(values)
            self.continuity_errors += int(np.sum(steps != 1))
            self.duplicate_or_reverse_steps += int(np.sum(steps <= 0))

        self.count += int(values.size)
        self.min_idx = first if self.min_idx is None else min(self.min_idx, int(values.min()))
        self.max_idx = last if self.max_idx is None else max(self.max_idx, int(values.max()))
        self.last_idx = last


class Audit:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        **details: Any,
    ) -> None:
        self.findings.append(Finding(severity, code, message, details))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "ERROR"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "WARNING"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a long-form DREAMER joined CSV."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--expected-subjects", type=int, default=23)
    parser.add_argument("--expected-trials", type=int, default=18)
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--minimum-stimulus-seconds", type=float, default=60.0)
    parser.add_argument("--median-label", type=float, default=3.0)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument(
        "--spectral-trials",
        type=int,
        default=3,
        help="Number of stimulus recordings to retain for rough PSD checks.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=50_000,
        help="Maximum EEG rows retained for correlation diagnostics.",
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a nonzero status for warnings as well as errors.",
    )
    return parser.parse_args()


def clean_label_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.removeprefix("[")
        .str.removesuffix("]")
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def safe_int_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    bad = numeric.isna() | ~np.isfinite(numeric)
    non_integer = (~bad) & (np.abs(numeric - np.round(numeric)) > 1e-9)
    bad = bad | non_integer
    out = numeric.round().fillna(-1).astype(np.int64)
    return out, bad


def class_summary(
    labels_by_trial: dict[tuple[int, int], dict[str, set[float]]],
    label_name: str,
    threshold: float,
) -> dict[str, Any]:
    values = []
    by_subject: dict[int, list[int]] = defaultdict(list)

    for (subject_id, _trial_id), label_sets in sorted(labels_by_trial.items()):
        unique = label_sets[label_name]
        if len(unique) != 1:
            continue
        raw = next(iter(unique))
        cls = int(raw >= threshold)
        values.append(cls)
        by_subject[subject_id].append(cls)

    if not values:
        return {
            "n_trials": 0,
            "class_0": 0,
            "class_1": 0,
            "majority_fraction": None,
            "subjects_single_class": [],
        }

    arr = np.asarray(values, dtype=np.int64)
    counts = np.bincount(arr, minlength=2)
    single_class_subjects = [
        subject
        for subject, subject_labels in by_subject.items()
        if len(set(subject_labels)) < 2
    ]
    return {
        "n_trials": int(arr.size),
        "class_0": int(counts[0]),
        "class_1": int(counts[1]),
        "majority_fraction": float(counts.max() / counts.sum()),
        "subjects_single_class": single_class_subjects,
    }


def spectral_summary(
    captures: dict[tuple[int, int], list[np.ndarray]],
    fs: float,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from scipy.signal import welch
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"scipy.signal.welch unavailable: {exc}"

    bands = {
        "delta_1_4": (1.0, 4.0),
        "theta_4_8": (4.0, 8.0),
        "alpha_8_13": (8.0, 13.0),
        "beta_13_30": (13.0, 30.0),
        "gamma_30_43": (30.0, 43.0),
    }

    trial_results = []
    for key, pieces in captures.items():
        if not pieces:
            continue
        signal = np.concatenate(pieces, axis=0)
        if signal.shape[0] < int(fs * 8):
            continue

        frequencies, psd = welch(
            signal,
            fs=fs,
            nperseg=min(1024, signal.shape[0]),
            axis=0,
            detrend="constant",
        )
        mean_psd = np.mean(psd, axis=1)
        total_mask = (frequencies >= 1.0) & (frequencies <= 43.0)
        total_power = float(np.trapz(mean_psd[total_mask], frequencies[total_mask]))
        if total_power <= 0 or not np.isfinite(total_power):
            continue

        fractions: dict[str, float] = {}
        for name, (low, high) in bands.items():
            mask = (frequencies >= low) & (frequencies < high)
            power = float(np.trapz(mean_psd[mask], frequencies[mask]))
            fractions[name] = power / total_power

        high_mask = (frequencies >= 30.0) & (frequencies <= 43.0)
        high_power = float(np.trapz(mean_psd[high_mask], frequencies[high_mask]))
        fractions["above_30_fraction_of_1_43"] = high_power / total_power
        trial_results.append({"subject_trial": list(key), **fractions})

    if not trial_results:
        return None, "No sufficiently long captured stimulus trial for PSD analysis."

    aggregate = {}
    metric_names = [name for name in trial_results[0] if name != "subject_trial"]
    for metric in metric_names:
        aggregate[metric] = float(np.median([row[metric] for row in trial_results]))

    return {
        "median_across_captured_trials": aggregate,
        "captured_trials": trial_results,
    }, None


def main() -> int:
    args = parse_args()
    audit = Audit()

    if not args.csv_path.is_file():
        print(f"ERROR: file not found: {args.csv_path}", file=sys.stderr)
        return 2

    header = pd.read_csv(args.csv_path, nrows=0)
    columns = set(header.columns)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        print(f"ERROR: missing required columns: {missing}", file=sys.stderr)
        return 2

    unexpected_duplicate_headers = header.columns[header.columns.duplicated()].tolist()
    if unexpected_duplicate_headers:
        audit.add(
            "ERROR",
            "duplicate_headers",
            "CSV contains duplicate column names.",
            columns=unexpected_duplicate_headers,
        )

    segment_stats: dict[tuple[int, int, str], SegmentStats] = defaultdict(SegmentStats)
    labels_by_trial: dict[tuple[int, int], dict[str, set[float]]] = defaultdict(
        lambda: {"valence": set(), "arousal": set()}
    )
    label_row_counts: dict[tuple[int, int], int] = defaultdict(int)
    stimulus_rows_with_both_labels: dict[tuple[int, int], int] = defaultdict(int)
    stimulus_row_counts: dict[tuple[int, int], int] = defaultdict(int)

    seen_subjects: set[int] = set()
    trials_by_subject: dict[int, set[int]] = defaultdict(set)
    unknown_segments: set[str] = set()

    feature_count = np.zeros(len(EEG_COLUMNS), dtype=np.int64)
    feature_sum = np.zeros(len(EEG_COLUMNS), dtype=np.float64)
    feature_sumsq = np.zeros(len(EEG_COLUMNS), dtype=np.float64)
    feature_min = np.full(len(EEG_COLUMNS), np.inf, dtype=np.float64)
    feature_max = np.full(len(EEG_COLUMNS), -np.inf, dtype=np.float64)
    feature_nonfinite = np.zeros(len(EEG_COLUMNS), dtype=np.int64)

    total_rows = 0
    signal_rows = 0
    zero_rows = 0
    identical_consecutive_rows = 0
    malformed_id_rows = 0
    signal_rows_with_missing_eeg = 0
    label_rows_with_eeg = 0
    out_of_range_label_values: dict[str, set[float]] = {
        "valence": set(),
        "arousal": set(),
    }

    correlation_sample: list[np.ndarray] = []
    correlation_sample_count = 0

    capture_limit = max(0, int(round(args.minimum_stimulus_seconds * args.fs)))
    spectral_captures: dict[tuple[int, int], list[np.ndarray]] = {}
    spectral_capture_counts: dict[tuple[int, int], int] = defaultdict(int)

    active_block: tuple[int, int, str] | None = None
    closed_blocks: set[tuple[int, int, str]] = set()
    reopened_blocks: set[tuple[int, int, str]] = set()

    usecols = list(header.columns)
    reader = pd.read_csv(
        args.csv_path,
        usecols=usecols,
        chunksize=args.chunk_size,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        subject_values, bad_subject = safe_int_series(chunk["subject_id"])
        trial_values, bad_trial = safe_int_series(chunk["trial_id"])
        sample_values, bad_sample = safe_int_series(chunk["sample_idx"])
        bad_ids = bad_subject | bad_trial | bad_sample
        malformed_id_rows += int(bad_ids.sum())

        chunk = chunk.copy()
        chunk["subject_id"] = subject_values
        chunk["trial_id"] = trial_values
        chunk["sample_idx"] = sample_values

        normalized_segment = chunk["segment"].astype("string").str.strip().str.lower()
        mapped_segment = normalized_segment.map(SEGMENT_ALIASES)
        unknown_mask = mapped_segment.isna()
        unknown_segments.update(normalized_segment[unknown_mask].dropna().unique().tolist())
        chunk["_segment"] = mapped_segment.fillna("unknown")

        valid_structure = ~bad_ids & ~unknown_mask
        valid_chunk = chunk.loc[valid_structure].copy()
        if valid_chunk.empty:
            continue

        subjects = valid_chunk["subject_id"].unique()
        seen_subjects.update(int(x) for x in subjects)
        for subject_id, group in valid_chunk.groupby("subject_id", sort=False):
            trials_by_subject[int(subject_id)].update(
                int(x) for x in group["trial_id"].unique()
            )

        # Check that each segment block is contiguous in the file.
        block_columns = valid_chunk[["subject_id", "trial_id", "_segment"]]
        previous_tuple = active_block
        for row in block_columns.itertuples(index=False, name=None):
            block = (int(row[0]), int(row[1]), str(row[2]))
            if block != previous_tuple:
                if previous_tuple is not None:
                    closed_blocks.add(previous_tuple)
                if block in closed_blocks:
                    reopened_blocks.add(block)
                previous_tuple = block
        active_block = previous_tuple

        eeg_numeric = valid_chunk[EEG_COLUMNS].apply(pd.to_numeric, errors="coerce")
        eeg_array = eeg_numeric.to_numpy(dtype=np.float64, copy=False)
        finite_matrix = np.isfinite(eeg_array)

        is_signal = valid_chunk["_segment"].isin(["baseline", "stimulus"]).to_numpy()
        is_label = (valid_chunk["_segment"] == "label").to_numpy()

        signal_rows += int(is_signal.sum())
        if np.any(is_signal):
            signal_finite = finite_matrix[is_signal]
            signal_values = eeg_array[is_signal]
            missing_rows = ~np.all(signal_finite, axis=1)
            signal_rows_with_missing_eeg += int(missing_rows.sum())

            valid_signal_values = signal_values[np.all(signal_finite, axis=1)]
            if valid_signal_values.size:
                feature_count += valid_signal_values.shape[0]
                feature_sum += valid_signal_values.sum(axis=0)
                feature_sumsq += np.square(valid_signal_values).sum(axis=0)
                feature_min = np.minimum(feature_min, valid_signal_values.min(axis=0))
                feature_max = np.maximum(feature_max, valid_signal_values.max(axis=0))
                zero_rows += int(
                    np.sum(np.all(np.abs(valid_signal_values) <= 1e-12, axis=1))
                )

                remaining = args.sample_rows - correlation_sample_count
                if remaining > 0:
                    take = min(remaining, valid_signal_values.shape[0])
                    if take > 0:
                        positions = np.linspace(
                            0,
                            valid_signal_values.shape[0] - 1,
                            num=take,
                            dtype=np.int64,
                        )
                        correlation_sample.append(valid_signal_values[positions])
                        correlation_sample_count += take

        feature_nonfinite += np.sum(~finite_matrix[is_signal], axis=0).astype(np.int64)

        if np.any(is_label):
            label_eeg = finite_matrix[is_label]
            label_rows_with_eeg += int(np.sum(np.any(label_eeg, axis=1)))

        valence = clean_label_series(valid_chunk["valence"])
        arousal = clean_label_series(valid_chunk["arousal"])
        valid_chunk["_valence"] = valence
        valid_chunk["_arousal"] = arousal

        for label_name, values in (("valence", valence), ("arousal", arousal)):
            finite = values[np.isfinite(values)]
            invalid_range = finite[(finite < 1.0) | (finite > 5.0)]
            out_of_range_label_values[label_name].update(
                float(x) for x in invalid_range.unique()
            )

        label_subset = valid_chunk[
            ["subject_id", "trial_id", "_segment", "_valence", "_arousal"]
        ]
        for (subject_id, trial_id), group in label_subset.groupby(
            ["subject_id", "trial_id"],
            sort=False,
        ):
            key = (int(subject_id), int(trial_id))
            for label_name, internal_name in (
                ("valence", "_valence"),
                ("arousal", "_arousal"),
            ):
                finite_values = group[internal_name].dropna().to_numpy(dtype=np.float64)
                labels_by_trial[key][label_name].update(
                    float(np.round(x, 8)) for x in np.unique(finite_values)
                )

            label_row_counts[key] += int(np.sum(group["_segment"] == "label"))
            stimulus_mask_group = group["_segment"] == "stimulus"
            stimulus_row_counts[key] += int(stimulus_mask_group.sum())
            both = (
                stimulus_mask_group
                & group["_valence"].notna()
                & group["_arousal"].notna()
            )
            stimulus_rows_with_both_labels[key] += int(both.sum())

        # Per-segment ordering, digest, and consecutive-row checks.
        valid_chunk["_row_position"] = np.arange(len(valid_chunk), dtype=np.int64)
        for (subject_id, trial_id, segment), group in valid_chunk.groupby(
            ["subject_id", "trial_id", "_segment"],
            sort=False,
        ):
            key = (int(subject_id), int(trial_id), str(segment))
            indices = group["sample_idx"].to_numpy(dtype=np.int64)
            stats = segment_stats[key]
            stats.update_indices(indices)

            if segment in {"baseline", "stimulus"}:
                group_positions = group["_row_position"].to_numpy(dtype=np.int64)
                group_eeg = eeg_array[group_positions]
                good = np.all(np.isfinite(group_eeg), axis=1)
                group_eeg_good = group_eeg[good].astype("<f4", copy=False)
                if group_eeg_good.size:
                    stats.digest.update(group_eeg_good.tobytes(order="C"))
                    if group_eeg_good.shape[0] > 1:
                        identical_consecutive_rows += int(
                            np.sum(
                                np.all(
                                    group_eeg_good[1:] == group_eeg_good[:-1],
                                    axis=1,
                                )
                            )
                        )

                if segment == "stimulus" and args.spectral_trials > 0:
                    trial_key = (int(subject_id), int(trial_id))
                    if (
                        trial_key in spectral_captures
                        or len(spectral_captures) < args.spectral_trials
                    ):
                        spectral_captures.setdefault(trial_key, [])
                        remaining = capture_limit - spectral_capture_counts[trial_key]
                        if remaining > 0 and group_eeg_good.size:
                            piece = group_eeg_good[:remaining].astype(np.float64)
                            spectral_captures[trial_key].append(piece)
                            spectral_capture_counts[trial_key] += piece.shape[0]

        if chunk_number % 10 == 0:
            print(
                f"Processed {total_rows:,} rows "
                f"({len(seen_subjects)} subjects found)...",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Structural findings
    # ------------------------------------------------------------------
    if total_rows == 0:
        audit.add("ERROR", "empty_csv", "CSV contains no data rows.")

    if malformed_id_rows:
        audit.add(
            "ERROR",
            "malformed_identifiers",
            "Some subject_id, trial_id, or sample_idx values are missing, nonnumeric, or nonintegral.",
            rows=malformed_id_rows,
        )

    if unknown_segments:
        audit.add(
            "ERROR",
            "unknown_segments",
            "Unknown segment names were found.",
            values=sorted(unknown_segments),
        )

    expected_subject_ids = set(range(1, args.expected_subjects + 1))
    if seen_subjects != expected_subject_ids:
        audit.add(
            "ERROR",
            "subject_set",
            "Subject IDs do not match the expected DREAMER set.",
            expected=sorted(expected_subject_ids),
            found=sorted(seen_subjects),
            missing=sorted(expected_subject_ids - seen_subjects),
            unexpected=sorted(seen_subjects - expected_subject_ids),
        )

    expected_trial_ids = set(range(1, args.expected_trials + 1))
    for subject_id in sorted(seen_subjects):
        found_trials = trials_by_subject[subject_id]
        if found_trials != expected_trial_ids:
            audit.add(
                "ERROR",
                "trial_set",
                f"Subject {subject_id} does not have the expected trial IDs.",
                subject_id=subject_id,
                expected=sorted(expected_trial_ids),
                found=sorted(found_trials),
                missing=sorted(expected_trial_ids - found_trials),
                unexpected=sorted(found_trials - expected_trial_ids),
            )

    if reopened_blocks:
        audit.add(
            "ERROR",
            "reopened_blocks",
            "Some subject/trial/segment blocks reappear after another block began; the CSV is not properly grouped.",
            examples=[list(x) for x in sorted(reopened_blocks)[:20]],
            count=len(reopened_blocks),
        )

    expected_trial_keys = {
        (subject_id, trial_id)
        for subject_id in expected_subject_ids
        for trial_id in expected_trial_ids
    }
    found_trial_keys = set(labels_by_trial) | {
        (subject_id, trial_id)
        for subject_id, trial_id, _segment in segment_stats
    }

    missing_trial_keys = sorted(expected_trial_keys - found_trial_keys)
    if missing_trial_keys:
        audit.add(
            "ERROR",
            "missing_subject_trials",
            "Entire subject/trial combinations are missing.",
            examples=[list(x) for x in missing_trial_keys[:20]],
            count=len(missing_trial_keys),
        )

    minimum_stimulus_samples = int(
        math.ceil(args.minimum_stimulus_seconds * args.fs)
    )
    stimulus_lengths = []
    baseline_lengths = []
    continuity_examples = []

    for trial_key in sorted(found_trial_keys):
        subject_id, trial_id = trial_key
        baseline_key = (subject_id, trial_id, "baseline")
        stimulus_key = (subject_id, trial_id, "stimulus")

        if baseline_key not in segment_stats:
            audit.add(
                "WARNING",
                "missing_baseline",
                f"Subject {subject_id}, trial {trial_id} has no baseline segment.",
                subject_id=subject_id,
                trial_id=trial_id,
            )
        else:
            baseline_lengths.append(segment_stats[baseline_key].count)

        if stimulus_key not in segment_stats:
            audit.add(
                "ERROR",
                "missing_stimulus",
                f"Subject {subject_id}, trial {trial_id} has no stimulus segment.",
                subject_id=subject_id,
                trial_id=trial_id,
            )
            continue

        stimulus_stat = segment_stats[stimulus_key]
        stimulus_lengths.append(stimulus_stat.count)
        if stimulus_stat.count < minimum_stimulus_samples:
            audit.add(
                "ERROR",
                "short_stimulus",
                f"Subject {subject_id}, trial {trial_id} is shorter than the required contiguous 60 seconds.",
                subject_id=subject_id,
                trial_id=trial_id,
                samples=stimulus_stat.count,
                required=minimum_stimulus_samples,
            )

    for key, stats in segment_stats.items():
        if key[2] not in {"baseline", "stimulus"}:
            continue
        if stats.continuity_errors:
            continuity_examples.append(
                {
                    "subject_id": key[0],
                    "trial_id": key[1],
                    "segment": key[2],
                    "count": stats.count,
                    "min_idx": stats.min_idx,
                    "max_idx": stats.max_idx,
                    "continuity_errors": stats.continuity_errors,
                    "duplicate_or_reverse_steps": stats.duplicate_or_reverse_steps,
                }
            )

        if stats.first_idx not in {0, 1}:
            audit.add(
                "WARNING",
                "unusual_sample_start",
                "A signal segment does not start at sample_idx 0 or 1.",
                subject_id=key[0],
                trial_id=key[1],
                segment=key[2],
                first_idx=stats.first_idx,
            )

    if continuity_examples:
        audit.add(
            "ERROR",
            "sample_index_continuity",
            "Some signal segments have duplicate, missing, reversed, or out-of-order sample_idx values.",
            examples=continuity_examples[:20],
            affected_segments=len(continuity_examples),
        )

    # ------------------------------------------------------------------
    # Label findings
    # ------------------------------------------------------------------
    for trial_key in sorted(found_trial_keys):
        label_sets = labels_by_trial[trial_key]
        for label_name in ("valence", "arousal"):
            unique = label_sets[label_name]
            if len(unique) == 0:
                audit.add(
                    "ERROR",
                    "missing_trial_label",
                    f"{label_name} is missing for subject {trial_key[0]}, trial {trial_key[1]}.",
                    subject_id=trial_key[0],
                    trial_id=trial_key[1],
                    label=label_name,
                )
            elif len(unique) > 1:
                audit.add(
                    "ERROR",
                    "inconsistent_trial_label",
                    f"{label_name} changes within subject {trial_key[0]}, trial {trial_key[1]}.",
                    subject_id=trial_key[0],
                    trial_id=trial_key[1],
                    label=label_name,
                    values=sorted(unique),
                )

        stimulus_count = stimulus_row_counts[trial_key]
        labeled_stimulus_count = stimulus_rows_with_both_labels[trial_key]
        if stimulus_count > 0 and labeled_stimulus_count == 0:
            audit.add(
                "ERROR",
                "labels_not_on_stimulus_rows",
                "This trial has labels elsewhere in the CSV but none on stimulus rows. The current prepare_datasets.py loader reads labels from the first stimulus row, so this format will produce missing labels.",
                subject_id=trial_key[0],
                trial_id=trial_key[1],
                label_rows=label_row_counts[trial_key],
                stimulus_rows=stimulus_count,
            )
        elif 0 < labeled_stimulus_count < stimulus_count:
            audit.add(
                "WARNING",
                "partially_labeled_stimulus",
                "Only some stimulus rows carry valence/arousal. The current loader only needs the first row, but inconsistent placement is fragile.",
                subject_id=trial_key[0],
                trial_id=trial_key[1],
                labeled_stimulus_rows=labeled_stimulus_count,
                stimulus_rows=stimulus_count,
            )

    for label_name, values in out_of_range_label_values.items():
        if values:
            audit.add(
                "ERROR",
                "label_range",
                f"{label_name} contains values outside DREAMER's expected 1-5 range.",
                label=label_name,
                values=sorted(values),
            )

    valence_summary = class_summary(
        labels_by_trial,
        "valence",
        args.median_label,
    )
    arousal_summary = class_summary(
        labels_by_trial,
        "arousal",
        args.median_label,
    )

    # ------------------------------------------------------------------
    # EEG findings
    # ------------------------------------------------------------------
    if signal_rows_with_missing_eeg:
        audit.add(
            "ERROR",
            "missing_eeg_values",
            "Signal rows contain missing, nonnumeric, or infinite EEG values.",
            rows=signal_rows_with_missing_eeg,
            per_channel_nonfinite={
                channel: int(count)
                for channel, count in zip(EEG_COLUMNS, feature_nonfinite)
                if count
            },
        )

    if label_rows_with_eeg:
        audit.add(
            "WARNING",
            "label_rows_contain_eeg",
            "Rows marked segment=label contain numeric EEG values.",
            rows=label_rows_with_eeg,
        )

    if zero_rows:
        severity = "ERROR" if zero_rows > max(10, signal_rows * 0.001) else "WARNING"
        audit.add(
            severity,
            "all_zero_rows",
            "All-zero EEG sample rows were found.",
            rows=zero_rows,
            fraction=(zero_rows / signal_rows if signal_rows else None),
        )

    if identical_consecutive_rows:
        severity = (
            "ERROR"
            if identical_consecutive_rows > max(100, signal_rows * 0.001)
            else "WARNING"
        )
        audit.add(
            severity,
            "identical_consecutive_samples",
            "Exactly identical consecutive 14-channel EEG rows were found.",
            pairs=identical_consecutive_rows,
            fraction=(identical_consecutive_rows / signal_rows if signal_rows else None),
        )

    means = np.divide(
        feature_sum,
        feature_count,
        out=np.full_like(feature_sum, np.nan),
        where=feature_count > 0,
    )
    variances = np.divide(
        feature_sumsq,
        feature_count,
        out=np.full_like(feature_sumsq, np.nan),
        where=feature_count > 0,
    ) - np.square(means)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances)

    constant_channels = [
        EEG_COLUMNS[i]
        for i, std in enumerate(stds)
        if not np.isfinite(std) or std <= 1e-12
    ]
    if constant_channels:
        audit.add(
            "ERROR",
            "constant_channels",
            "One or more EEG channels are constant or contain no valid values.",
            channels=constant_channels,
        )

    correlation_result: dict[str, Any] = {}
    if correlation_sample:
        sample_matrix = np.concatenate(correlation_sample, axis=0)
        if sample_matrix.shape[0] >= 10:
            corr = np.corrcoef(sample_matrix, rowvar=False)
            suspicious_pairs = []
            for i in range(len(EEG_COLUMNS)):
                for j in range(i + 1, len(EEG_COLUMNS)):
                    value = float(corr[i, j])
                    if np.isfinite(value) and abs(value) >= 0.99999:
                        suspicious_pairs.append(
                            {
                                "channel_a": EEG_COLUMNS[i],
                                "channel_b": EEG_COLUMNS[j],
                                "correlation": value,
                            }
                        )
            correlation_result = {
                "sample_rows": int(sample_matrix.shape[0]),
                "near_duplicate_pairs": suspicious_pairs,
            }
            if suspicious_pairs:
                audit.add(
                    "ERROR",
                    "near_duplicate_channels",
                    "Some EEG channel pairs are almost perfectly correlated, suggesting duplicated or incorrectly merged columns.",
                    pairs=suspicious_pairs,
                )

    stimulus_hashes: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for (subject_id, trial_id, segment), stats in segment_stats.items():
        if segment == "stimulus" and stats.count:
            stimulus_hashes[stats.digest.hexdigest()].append((subject_id, trial_id))

    exact_duplicate_trials = [
        [list(key) for key in keys]
        for keys in stimulus_hashes.values()
        if len(keys) > 1
    ]
    if exact_duplicate_trials:
        audit.add(
            "ERROR",
            "duplicate_stimulus_recordings",
            "Exact duplicate stimulus EEG recordings were found across different subject/trial combinations.",
            groups=exact_duplicate_trials[:20],
            count=len(exact_duplicate_trials),
        )

    if args.spectral_trials > 0:
        spectral, spectral_error = spectral_summary(spectral_captures, args.fs)
    else:
        spectral, spectral_error = None, None

    if spectral_error:
        audit.add(
            "WARNING",
            "spectral_check_unavailable",
            spectral_error,
        )
    elif spectral is not None:
        high_fraction = spectral["median_across_captured_trials"][
            "above_30_fraction_of_1_43"
        ]
        if high_fraction < 1e-4:
            audit.add(
                "WARNING",
                "little_power_above_30hz",
                "The captured raw EEG has almost no 30-43 Hz power. Gamma-band features may be meaningless because the source may already be low-pass filtered near 30 Hz.",
                median_fraction=high_fraction,
            )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    feature_summary = {
        channel: {
            "count": int(feature_count[i]),
            "mean": float(means[i]) if np.isfinite(means[i]) else None,
            "std": float(stds[i]) if np.isfinite(stds[i]) else None,
            "min": float(feature_min[i]) if np.isfinite(feature_min[i]) else None,
            "max": float(feature_max[i]) if np.isfinite(feature_max[i]) else None,
            "nonfinite": int(feature_nonfinite[i]),
        }
        for i, channel in enumerate(EEG_COLUMNS)
    }

    def length_summary(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "median": None, "max": None}
        arr = np.asarray(values, dtype=np.int64)
        return {
            "count": int(arr.size),
            "min": int(arr.min()),
            "median": float(np.median(arr)),
            "max": int(arr.max()),
        }

    report = {
        "file": str(args.csv_path),
        "rows": total_rows,
        "signal_rows": signal_rows,
        "subjects_found": sorted(seen_subjects),
        "trials_per_subject": {
            str(subject): sorted(trials)
            for subject, trials in sorted(trials_by_subject.items())
        },
        "baseline_length_samples": length_summary(baseline_lengths),
        "stimulus_length_samples": length_summary(stimulus_lengths),
        "label_class_summary_score_ge_threshold_is_class_1": {
            "threshold": args.median_label,
            "valence": valence_summary,
            "arousal": arousal_summary,
        },
        "feature_summary": feature_summary,
        "correlation_diagnostics": correlation_result,
        "spectral_diagnostics": spectral,
        "findings": [finding.as_dict() for finding in audit.findings],
        "status": (
            "FAIL"
            if audit.errors
            else "WARN"
            if audit.warnings
            else "PASS"
        ),
    }

    print("\n" + "=" * 78)
    print("DREAMER CSV AUDIT")
    print("=" * 78)
    print(f"File:       {args.csv_path}")
    print(f"Rows:       {total_rows:,}")
    print(f"Subjects:   {len(seen_subjects)}")
    print(f"Trial keys: {len(found_trial_keys)}")
    print(
        "Stimulus lengths: "
        f"min={report['stimulus_length_samples']['min']}, "
        f"median={report['stimulus_length_samples']['median']}, "
        f"max={report['stimulus_length_samples']['max']}"
    )
    print(
        "Arousal classes (score >= "
        f"{args.median_label:g} is class 1): "
        f"{arousal_summary['class_0']} / {arousal_summary['class_1']}, "
        f"majority={arousal_summary['majority_fraction']}"
    )
    print(
        "Valence classes (score >= "
        f"{args.median_label:g} is class 1): "
        f"{valence_summary['class_0']} / {valence_summary['class_1']}, "
        f"majority={valence_summary['majority_fraction']}"
    )

    print("\nFindings:")
    if not audit.findings:
        print("  [PASS] No structural or numerical problems detected.")
    else:
        for finding in sorted(
            audit.findings,
            key=lambda item: (-SEVERITY_ORDER[item.severity], item.code),
        ):
            print(
                f"  [{finding.severity}] {finding.code}: {finding.message}"
            )
            if finding.details:
                compact = json.dumps(finding.details, sort_keys=True)
                if len(compact) > 500:
                    compact = compact[:497] + "..."
                print(f"      {compact}")

    print("\nFinal status:", report["status"])

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"JSON report: {args.report_json}")

    if audit.errors:
        return 2
    if args.strict and audit.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
