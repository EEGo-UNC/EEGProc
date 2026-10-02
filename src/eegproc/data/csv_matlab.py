"""Readers for AMIGOS, DREAMER and preprocessed DEAP releases."""
from __future__ import annotations

from contextlib import closing
from fractions import Fraction
from pathlib import Path
import pickle
import re
import sqlite3
import tempfile
import warnings

import numpy as np
import pandas as pd

from ._csv_common import (
    EEG_CHANNELS, KEYS, RATINGS, load_mat, mat_cells, numeric,
    require_columns, samples_by_channels, signal_frames, skip_empty,
)


def amigos_frames(root: Path, chunksize: int, baseline_samples: int = 640):
    """Read original AMIGOS preprocessed MATLAB cells, one participant at a time."""
    def subject(path):
        match = re.fullmatch(r"Data_Preprocessed_P(\d+)\.mat", path.name)
        if not match:
            raise ValueError(f"Unrecognized AMIGOS filename: {path.name}")
        return int(match[1])

    files = [root] if root.is_file() else list(root.rglob("Data_Preprocessed_P*.mat"))
    if not files:
        raise FileNotFoundError(f"No Data_Preprocessed_P*.mat files under {root}")
    seen = set()
    for path in sorted(files, key=subject):
        sid = subject(path)
        if sid in seen:
            raise ValueError(f"Duplicate AMIGOS participant {sid}")
        seen.add(sid)
        contents = load_mat(path)
        trials = mat_cells(contents["joined_data"])
        labels = mat_cells(contents["labels_selfassessment"])
        if len(trials) != len(labels):
            raise ValueError(f"{path}: signal and label trial counts differ")
        for tid, (raw, rating) in enumerate(zip(trials, labels), 1):
            source = f"{path.name}, trial {tid}"
            if np.asarray(raw).size == 0:
                skip_empty(source)
                continue
            signal = samples_by_channels(raw, (16, 17), source)[:, :16]
            rating = numeric(rating).ravel()
            if rating.size < 3:
                raise ValueError(f"{source}: expected at least arousal, valence, dominance")
            if len(signal) <= baseline_samples:
                raise ValueError(f"{source}: no samples after the {baseline_samples}-sample baseline")
            label = dict(zip(RATINGS, rating[[1, 0, 2]]))
            for segment, block in [("baseline", signal[:baseline_samples]),
                                   ("stimulus", signal[baseline_samples:])]:
                yield from signal_frames(block, sid, tid, segment, EEG_CHANNELS + ["ECG1", "ECG2"],
                                         label, chunksize)


def _csv_path(root, name):
    path = root / f"{name}.csv"
    if not path.is_file():
        path = root / f"{name}.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Expected {name}.csv or {name}.csv.gz in {root}")
    return path


def dreamer_csv_frames(root: Path, chunksize: int):
    """Join existing DREAMER exports on disk, preserving EEG row order/indices."""
    # SQLite keeps the large ECG lookup and EEG table out of Python's RAM.
    with tempfile.TemporaryDirectory(prefix="eegproc-dreamer-") as tmp:
        with closing(sqlite3.connect(str(Path(tmp) / "join.sqlite"))) as db:
            specs = [("eeg", KEYS + EEG_CHANNELS), ("ecg", KEYS + ["ECG1", "ECG2"]),
                     ("labels", KEYS[:2] + RATINGS)]
            for table, columns in specs:
                path = _csv_path(root, f"dreamer_{table}")
                rows = 0
                for frame in pd.read_csv(path, chunksize=chunksize):
                    require_columns(frame, columns, path)
                    key_cols = KEYS[:2] if table == "labels" else KEYS
                    if frame[key_cols].isna().any().any():
                        raise ValueError(f"{path}: empty join keys")
                    frame[columns].to_sql(table, db, if_exists="append", index=False)
                    rows += len(frame)
                if not rows:
                    raise ValueError(f"{path}: no rows")
                keys = KEYS[:2] if table == "labels" else KEYS
                try:
                    db.execute(f'CREATE UNIQUE INDEX {table}_keys ON {table} ({", ".join(keys)})')
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"{path}: duplicate join keys") from exc
            trial_join = " AND ".join(f"eeg.{k} = labels.{k}" for k in KEYS[:2])
            ecg_join = " AND ".join(f"eeg.{k} = ecg.{k}" for k in KEYS)
            missing = db.execute(f"SELECT COUNT(*) FROM eeg LEFT JOIN labels ON {trial_join} "
                                 "WHERE labels.subject_id IS NULL").fetchone()[0]
            if missing:
                raise ValueError(f"DREAMER: {missing} EEG rows have no trial label")
            missing = db.execute(f"SELECT COUNT(*) FROM eeg LEFT JOIN ecg ON {ecg_join} "
                                 "WHERE ecg.subject_id IS NULL").fetchone()[0]
            if missing:
                warnings.warn(f"DREAMER: {missing} EEG rows have no matching ECG; writing blanks")
            columns = KEYS + EEG_CHANNELS + ["ECG1", "ECG2"] + RATINGS
            selected = [f'eeg."{k}"' for k in KEYS + EEG_CHANNELS]
            selected += ["ecg.ECG1", "ecg.ECG2"] + [f"labels.{k}" for k in RATINGS]
            cursor = db.execute(f'SELECT {", ".join(selected)} FROM eeg '
                                f'LEFT JOIN ecg ON {ecg_join} LEFT JOIN labels ON {trial_join} '
                                'ORDER BY eeg.rowid')
            while rows := cursor.fetchmany(chunksize):
                yield pd.DataFrame.from_records(rows, columns=columns)


def dreamer_mat_frames(path: Path, chunksize: int):
    """Read DREAMER.mat, aligning ECG to EEG time using the stored sample rates."""
    from scipy.signal import resample_poly

    dreamer = load_mat(path, struct_as_record=False, squeeze_me=False)["DREAMER"][0, 0]
    eeg_fs = float(numeric(dreamer.EEG_SamplingRate).item())
    ecg_fs = float(numeric(dreamer.ECG_SamplingRate).item())
    if eeg_fs <= 0 or ecg_fs <= 0:
        raise ValueError("DREAMER sampling rates must be positive")
    ratio = Fraction(eeg_fs / ecg_fs).limit_denominator(10000)
    for sid, subject_cell in enumerate(mat_cells(dreamer.Data), 1):
        subject = np.asarray(subject_cell).item()
        eeg = subject.EEG[0, 0]
        ecg = subject.ECG[0, 0]
        ratings = [numeric(getattr(subject, name)).ravel()
                   for name in ["ScoreValence", "ScoreArousal", "ScoreDominance"]]
        for segment in ["baseline", "stimuli"]:
            eeg_trials = mat_cells(getattr(eeg, segment))
            ecg_trials = mat_cells(getattr(ecg, segment))
            if len(eeg_trials) != len(ecg_trials) or any(len(r) != len(eeg_trials) for r in ratings):
                raise ValueError(f"DREAMER participant {sid}: trial counts differ")
            for tid, (eeg_trial, ecg_trial) in enumerate(zip(eeg_trials, ecg_trials), 1):
                source = f"DREAMER participant {sid}, trial {tid}, {segment}"
                eeg_trial = samples_by_channels(eeg_trial, (14,), source)
                ecg_trial = samples_by_channels(ecg_trial, (2,), source)
                aligned = resample_poly(ecg_trial, ratio.numerator, ratio.denominator, axis=0)
                if abs(len(aligned) - len(eeg_trial)) > 1:
                    raise ValueError(f"{source}: EEG/ECG durations disagree")
                if len(aligned) < len(eeg_trial):
                    aligned = np.pad(aligned, ((0, 1), (0, 0)), constant_values=np.nan)
                signal = np.column_stack([eeg_trial, aligned[:len(eeg_trial)]])
                label = dict(zip(RATINGS, [r[tid - 1] for r in ratings]))
                yield from signal_frames(signal, sid, tid, "stimulus" if segment == "stimuli" else segment,
                                         EEG_CHANNELS + ["ECG1", "ECG2"], label, chunksize)


def deap_frames(root: Path, chunksize: int):
    """Read official preprocessed Python DEAP files (trusted pickle inputs only)."""
    channels = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3",
                "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6",
                "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"]
    files = [root] if root.is_file() else sorted(root.glob("s*.dat"))
    if not files:
        raise FileNotFoundError(f"No DEAP s*.dat files in {root}")
    for path in files:
        match = re.fullmatch(r"s(\d+)\.dat", path.name)
        if not match:
            raise ValueError(f"Unrecognized DEAP filename: {path.name}")
        with path.open("rb") as handle:
            contents = pickle.load(handle, encoding="latin1")
        signal, labels = np.asarray(contents["data"]), np.asarray(contents["labels"])
        if signal.ndim != 3 or signal.shape[1] < 32 or labels.shape != (len(signal), 4):
            raise ValueError(f"{path}: invalid DEAP data/labels shape")
        n_samples = signal.shape[2]
        if n_samples not in (8064, 7680):
            raise ValueError(f"{path}: expected 8064 samples (baseline included) or 7680 (removed)")
        baseline = n_samples - 7680
        for tid, (trial, rating) in enumerate(zip(signal, labels), 1):
            for segment, block in [("baseline", trial[:32, :baseline].T),
                                   ("stimulus", trial[:32, baseline:].T)]:
                yield from signal_frames(block, int(match[1]), tid, segment, channels,
                                         dict(zip(RATINGS + ["liking"], rating)), chunksize)
