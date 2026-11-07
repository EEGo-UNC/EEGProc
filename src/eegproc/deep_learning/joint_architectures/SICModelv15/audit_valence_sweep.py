#!/usr/bin/env python3
"""Read-only audit of source shards after a quota failure or storage transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from collect_valence_sweep_shards import SHARD_TARGETS
from sic_v15_valence_sweep import configurations, select_oracle


def audit_suite(suite, configs, verify_checkpoints=False):
    suite = Path(suite).resolve()
    if not suite.is_dir():
        raise ValueError(f"Suite directory does not exist: {suite}")
    records = configurations(configs)
    shards, missing = [], []
    for index in range(len(records)):
        for shard_index, targets in enumerate(SHARD_TARGETS):
            task_id = 3 * index + shard_index
            path = suite / f"configuration_{index}" / f"shard_{shard_index}"
            if not path.is_dir():
                missing.append(task_id)
                continue
            row = {"task_id": task_id, "path": str(path), "complete_epoch_files": False,
                   "execution_status": "missing", "ready_for_collector": False, "errors": []}
            try:
                spec = json.loads((path / "task.json").read_text())
                expected = configurations(configs, seed=spec["seed"])[index]
                if spec["configuration"] != expected or tuple(spec["target_subjects"]) != targets:
                    raise ValueError("Task configuration or held-out users do not match this shard.")
                epochs = int(spec["source_epochs"])
                metrics = [json.loads(item.read_text()) for item in sorted(path.glob("subject_*/epoch_*.json"))]
                summary = select_oracle(metrics, targets, epochs)
                for metric in metrics:
                    checkpoint = (path / metric["checkpoint"]).resolve()
                    if not checkpoint.is_relative_to(path) or not checkpoint.is_file():
                        raise ValueError(f"Missing or invalid checkpoint: {checkpoint}")
                    if verify_checkpoints:
                        with zipfile.ZipFile(checkpoint) as archive:
                            required = {"config.json", "metadata.json", "model.weights.h5"}
                            if not required <= set(archive.namelist()):
                                raise ValueError(f"Incomplete Keras archive: {checkpoint}")
                            bad = archive.testzip()
                            if bad is not None:
                                raise ValueError(f"Corrupt checkpoint member: {checkpoint}: {bad}")
                row.update(epoch_files=len(metrics), expected_epoch_files=len(targets) * epochs,
                           complete_epoch_files=summary["status"] == "completed")
                execution_path = path / "execution.json"
                if execution_path.is_file():
                    execution = json.loads(execution_path.read_text())
                    row["execution_status"] = execution.get("status", "unknown")
                    row["ready_for_collector"] = (row["complete_epoch_files"] and
                        execution.get("status") == "completed" and not execution.get("failures") and
                        not execution.get("interrupted") and not execution.get("unstarted_subjects"))
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
                row["errors"].append(str(error))
            shards.append(row)
    return {"suite": str(suite), "checkpoint_check": "archive CRC" if verify_checkpoints else "existence only",
            "missing_tasks": missing, "shards": shards,
            "note": "No files were changed. Complete epoch files with missing execution.json require review; "
                    "this audit does not declare a failed job successful or resume training."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--configs", type=Path,
                        default=Path(__file__).with_name("sic_v15_valence_focal_scale_recon_configs.json"))
    parser.add_argument("--verify-checkpoints", action="store_true",
                        help="Read every Keras archive and verify CRCs; potentially substantial I/O.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit_suite(args.suite, args.configs, args.verify_checkpoints)
    except (OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Suite: {result['suite']}\nCheckpoint check: {result['checkpoint_check']}")
        for row in result["shards"]:
            print(f"Task {row['task_id']}: epochs {row.get('epoch_files', '?')}/"
                  f"{row.get('expected_epoch_files', '?')}; execution={row['execution_status']}; "
                  f"collector-ready={row['ready_for_collector']}")
            for error in row["errors"]:
                print(f"  {error}")
        print("Missing tasks: " + ", ".join(map(str, result["missing_tasks"])))
        print(result["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
