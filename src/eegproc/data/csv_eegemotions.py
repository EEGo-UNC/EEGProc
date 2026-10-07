"""Stream EEGEmotions-27 raw text and attach its participant/emotion metadata."""
from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from ._csv_common import EEG_CHANNELS, require_columns

LABEL_COLUMNS = {"cowen": "Emo_Label_Cowen(27)", "ekman": "Emo_Label_Ekman(6)"}
OUTPUT_LABELS = {"cowen": "emo_label_cowen_27", "ekman": "emo_label_ekman_6"}


def parse_filename(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)_(\d+)(?:\.0+)?\.txt", path.name)
    if not match:
        raise ValueError(f"Invalid EEGEmotions filename: {path.name}")
    return int(match[1]), int(match[2])


def _integers(values: pd.Series, source) -> pd.Series:
    result = pd.to_numeric(values, errors="raise")
    if result.isna().any() or not np.isfinite(result).all() or (result % 1 != 0).any():
        raise ValueError(f"{source}: IDs must be finite integers")
    return result.astype("int64")


def _label_lookup(path: Path, kind: str):
    other = "ekman" if kind == "cowen" else "cowen"
    columns = ["ParticipantID", LABEL_COLUMNS[kind], LABEL_COLUMNS[other]]
    labels = pd.read_csv(path, usecols=columns)
    for column in columns:
        labels[column] = _integers(labels[column], path)
    grouped = labels.groupby(columns[:2])[columns[2]]
    conflicts = int((grouped.nunique() > 1).sum())
    if conflicts:
        warnings.warn(f"{path}: {conflicts} label groups have multiple mappings; using their "
                      "mode (lowest label on a tie), matching the original conversion script")
    return grouped.agg(lambda values: values.mode().iloc[0]).to_dict()


def _raw_chunks(path: Path, chunksize: int):
    with path.open(encoding="utf-8-sig") as handle:
        first = next((line.strip() for line in handle if line.strip()), "")
    if not first:
        raise ValueError(f"Empty EEG file: {path}")
    sep = "," if "," in first else r"\s+"
    tokens = [t.strip() for t in first.split(",")] if sep == "," else first.split()
    try:
        [float(token) for token in tokens]
        has_header = False
    except ValueError:
        has_header = True
    with pd.read_csv(path, sep=sep, header=0 if has_header else None,
                     chunksize=chunksize, encoding="utf-8-sig") as reader:
        for frame in reader:
            if has_header:
                frame.columns = frame.columns.str.strip()
                require_columns(frame, EEG_CHANNELS, path)
                frame = frame[EEG_CHANNELS].copy()
            else:
                if len(frame.columns) != len(EEG_CHANNELS):
                    raise ValueError(f"{path}: headerless input needs exactly 14 columns; "
                                     "add channel headers if extra columns are present")
                frame.columns = EEG_CHANNELS
            yield frame.apply(pd.to_numeric, errors="raise").reset_index(drop=True)


def eegemotions_frames(root: Path, chunksize: int, *, labels_path: Path | None = None,
                       source_label_kind: str = "cowen"):
    """Preserve file emotion IDs verbatim, including zero if present in a release."""
    if source_label_kind not in LABEL_COLUMNS:
        raise ValueError("source_label_kind must be 'cowen' or 'ekman'")
    raw_dir = root / "eeg_raw"
    files = sorted(raw_dir.glob("*.txt"), key=parse_filename)
    if not files:
        raise FileNotFoundError(f"No EEG text files in {raw_dir}")
    participants_path = root / "participants_info.csv"
    participants = pd.read_csv(participants_path).rename(columns={
        "Participant_ID": "subject_id", "Participant ID": "subject_id", "participant_id": "subject_id",
        "Age": "age", "Gender": "gender", "Nation": "nation",
    })
    require_columns(participants, ["subject_id", "age", "gender", "nation"], participants_path)
    participants["subject_id"] = _integers(participants["subject_id"], participants_path)
    if participants.subject_id.duplicated().any():
        raise ValueError(f"{participants_path}: duplicate participant IDs")
    metadata = participants.set_index("subject_id")[["age", "gender", "nation"]].to_dict("index")
    if labels_path is None:
        labels_path = root / "training" / "eeg_features_extracted.csv"
    # The primary emotion label comes from the filename; the table supplies the other system.
    lookup = _label_lookup(labels_path, source_label_kind) if labels_path.is_file() else {}
    if not labels_path.is_file():
        warnings.warn(f"{labels_path} not found; the secondary emotion label will be blank")
    seen, missing_participants, missing_labels = set(), set(), set()
    other = "ekman" if source_label_kind == "cowen" else "cowen"
    for path in files:
        sid, tid = parse_filename(path)
        if (sid, tid) in seen:
            raise ValueError(f"Duplicate EEGEmotions subject/emotion: {(sid, tid)}")
        seen.add((sid, tid))
        meta = metadata.get(sid)
        if meta is None:
            missing_participants.add(sid)
            meta = dict.fromkeys(["age", "gender", "nation"], np.nan)
        secondary = lookup.get((sid, tid), np.nan)
        if pd.isna(secondary):
            missing_labels.add((sid, tid))
        offset = 0
        for frame in _raw_chunks(path, chunksize):
            n = len(frame)
            frame.insert(0, "sample_idx", np.arange(offset, offset + n))
            frame.insert(0, "segment", "eeg_raw")
            frame.insert(0, "trial_id", tid)
            frame.insert(0, "subject_id", sid)
            for name, value in meta.items():
                frame[name] = value
            frame["source_file"] = path.name
            frame["source_file_label"] = tid
            values = {source_label_kind: tid, other: secondary}
            for kind in ["cowen", "ekman"]:
                frame[OUTPUT_LABELS[kind]] = values[kind]
            offset += n
            yield frame
    if missing_participants:
        warnings.warn(f"EEGEmotions: missing demographics for {len(missing_participants)} participants")
    if lookup and missing_labels:
        warnings.warn(f"EEGEmotions: no secondary label for {len(missing_labels)} subject/emotion pairs")
