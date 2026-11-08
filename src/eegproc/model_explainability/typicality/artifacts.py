"""Atomic result files and streamed traces, readable without TensorFlow."""

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, payload):
    _atomic(path, lambda handle: handle.write((json.dumps(
        payload, indent=2, allow_nan=False, default=_json_default) + "\n").encode()))


def summary_path(path):
    """Prefer compact JSON summaries, with read-only support for old archives."""
    path = Path(path)
    return path.with_suffix(".json") if path.with_suffix(".json").is_file() else path.with_suffix(".npz")


def read_summary(path):
    path = summary_path(path)
    if path.suffix == ".json":
        return {name: np.asarray(value) for name, value in json.loads(path.read_text()).items()}
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    fields = list(fieldnames or dict.fromkeys(key for row in rows for key in row))
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic(path, lambda handle: handle.write(stream.getvalue().encode()))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((values.dtype.str, values.shape)).encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


class TrialRecorder:
    """One attempt directory with per-step scalars and endpoint metrics.

    Step s is the state BEFORE update s+1. Every finite step is written to
    the scalar history. Signal, latent, gradient, and optimizer tensors are
    never persisted.
    Completed trials can be reused; unfinished trials retain all attempts.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.handle = (self.directory / "history.jsonl").open("x", encoding="utf-8")
        self.rows = []

    def record(self, row):
        self.handle.write(json.dumps(row, allow_nan=False, default=_json_default) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.rows.append(dict(row))

    def close(self):
        if self.handle.closed:
            return
        self.handle.close()
        write_csv(self.directory / "history.csv", self.rows)

    def finish(self, result, metadata):
        self.close()
        summary = {**metadata, **result["summary"], "status": "completed"}
        write_json(self.directory / "result.json", summary)
        # The commit marker comes last, after every required artifact is durable.
        artifacts = {str(p.relative_to(self.directory)): file_sha256(p)
                     for p in sorted(self.directory.rglob("*")) if p.is_file()}
        write_json(self.directory / "complete.json", {"schema_version": 1, "sha256": artifacts})
        return summary

    def fail(self, error, metadata):
        self.close()
        write_json(self.directory / "result.json", {
            **metadata, "status": "error", "error_type": type(error).__name__,
            "error": str(error), "steps_recorded": len(self.rows),
        })


def completed_attempt(directory):
    """Verify the last committed attempt before reusing its outputs."""
    directory = Path(directory)
    for attempt in sorted(directory.glob("attempt_*"), reverse=True):
        marker = attempt / "complete.json"
        if not marker.is_file():
            continue
        payload = json.loads(marker.read_text())
        for name, digest in payload["sha256"].items():
            path = attempt / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"Saved artifact changed or is incomplete: {path}")
        return attempt
    return None


def next_attempt(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.glob("attempt_*"))
    number = max([int(p.name.split("_")[-1]) for p in existing] + [0]) + 1
    return directory / f"attempt_{number:04d}"
