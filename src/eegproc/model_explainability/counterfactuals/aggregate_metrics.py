"""Collect independent counterfactual array-task outputs into one JSON file."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _parse_csv_value(value: str):
    if value == "":
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    try:
        integer = int(value)
    except ValueError:
        pass
    else:
        if str(integer) == value or (value.startswith("+") and str(integer) == value[1:]):
            return integer
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else None


def _read_history(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: _parse_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _flatten_numeric(value, prefix: str, destination: dict[str, list[float]]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(nested, path, destination)
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _flatten_numeric(nested, f"{prefix}[{index}]", destination)
        return
    if isinstance(value, bool):
        destination.setdefault(prefix, []).append(float(value))
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            destination.setdefault(prefix, []).append(number)


def _describe(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    return {
        "count": len(numbers),
        "mean": mean,
        "standard_deviation": math.sqrt(variance),
        "minimum": min(numbers),
        "maximum": max(numbers),
    }


def _collect_trial_records(root: Path, subject_id: int) -> tuple[list[dict], list[str]]:
    records_by_trial = {}
    read_errors = []
    pattern = "trial_*/subject_*_trial_*/result.json"
    for result_path in sorted(root.glob(pattern)):
        try:
            metrics = _read_json(result_path)
            result_subject = int(metrics["subject_id"])
            trial_id = int(metrics["trial_id"])
            if result_subject != subject_id:
                continue
            if trial_id in records_by_trial:
                raise ValueError(
                    f"Duplicate completed result for subject {subject_id}, trial {trial_id}."
                )

            trial_directory = result_path.parent
            task_directory = trial_directory.parent
            history_path = trial_directory / "history.csv"
            settings_path = task_directory / "settings.json"
            history = _read_history(history_path)
            settings = _read_json(settings_path)
            artifacts = sorted(
                str(path.relative_to(root))
                for path in trial_directory.iterdir()
                if path.is_file()
            )
            records_by_trial[trial_id] = {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "metrics": metrics,
                "history": history,
                "settings": settings,
                "artifacts": artifacts,
            }
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            read_errors.append(f"{result_path}: {exc}")
    return [records_by_trial[key] for key in sorted(records_by_trial)], read_errors


def build_metrics_payload(
    root: Path,
    *,
    subject_id: int = 0,
    expected_trial_ids: Iterable[int] = range(18),
) -> dict:
    """Return all per-trial metrics/history plus cross-trial summaries."""
    root = Path(root)
    expected = sorted({int(value) for value in expected_trial_ids})
    if not expected or any(value < 0 for value in expected):
        raise ValueError("expected_trial_ids must contain nonnegative integers.")

    records, read_errors = _collect_trial_records(root, int(subject_id))
    completed = [record["trial_id"] for record in records]
    unexpected = sorted(set(completed) - set(expected))
    missing = sorted(set(expected) - set(completed))

    numeric_values: dict[str, list[float]] = {}
    for record in records:
        _flatten_numeric(record["metrics"], "", numeric_values)
    numeric_summaries = {
        name: _describe(values) for name, values in sorted(numeric_values.items())
    }

    decoder_paths = sorted(
        {
            path
            for record in records
            for path in record["metrics"].get("decoded_trials", {})
        }
    )
    decoded_success_rate = {}
    for path in decoder_paths:
        successes = [
            bool(record["metrics"]["decoded_trials"][path]["counterfactual"]["success"])
            for record in records
            if path in record["metrics"].get("decoded_trials", {})
        ]
        decoded_success_rate[path] = sum(successes) / len(successes)

    latent_successes = [
        bool(record["metrics"]["latent_counterfactual"]["success"])
        for record in records
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root.resolve()),
        "subject_id": int(subject_id),
        "expected_trial_ids": expected,
        "completed_trial_ids": completed,
        "missing_trial_ids": missing,
        "unexpected_trial_ids": unexpected,
        "n_expected_trials": len(expected),
        "n_completed_trials": len(completed),
        "complete": not missing and not unexpected and not read_errors,
        "read_errors": read_errors,
        "aggregate": {
            "latent_success_rate": (
                sum(latent_successes) / len(latent_successes)
                if latent_successes
                else None
            ),
            "decoded_success_rate": decoded_success_rate,
            "numeric_metric_summaries": numeric_summaries,
        },
        "trials": records,
    }


def write_metrics_json(
    root: Path,
    *,
    output: Path | None = None,
    subject_id: int = 0,
    expected_trial_ids: Iterable[int] = range(18),
) -> dict:
    """Lock, rescan completed tasks, and atomically replace the combined JSON."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(output) if output is not None else root / "all_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        payload = build_metrics_payload(
            root,
            subject_id=subject_id,
            expected_trial_ids=expected_trial_ids,
        )
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine independently completed counterfactual trial tasks into "
            "one atomic JSON metrics file."
        )
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subject-id", type=int, default=0)
    parser.add_argument("--expected-trials", type=int, default=18)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero if any expected trial result is missing.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.subject_id < 0:
        raise ValueError("--subject-id must be nonnegative.")
    if args.expected_trials < 1:
        raise ValueError("--expected-trials must be positive.")
    output = args.output or args.run_root / "all_metrics.json"
    payload = write_metrics_json(
        args.run_root,
        output=output,
        subject_id=args.subject_id,
        expected_trial_ids=range(args.expected_trials),
    )
    print(
        f"Saved {payload['n_completed_trials']}/{payload['n_expected_trials']} "
        f"trial metrics to {output}."
    )
    if payload["missing_trial_ids"]:
        print(f"Missing trials: {payload['missing_trial_ids']}")
    return int(args.require_complete and not payload["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
