#!/usr/bin/env python3
"""Combine source-fit shards before selecting one oracle epoch over 23 users."""

import argparse
import json
from pathlib import Path

from sic_v15_valence_sweep import select_oracle, summarize, validate_checkpoints, write_json


SHARD_TARGETS = (tuple(range(0, 8)), tuple(range(8, 16)), tuple(range(16, 23)))
MATCH_FIELDS = ("configuration", "source_epochs", "source_batch_size", "seed",
                "gpus_per_fold", "deterministic_training", "eeg", "labels",
                "preprocessing", "metrics_only")


def collect(output_dir):
    """Validate shards and collect scores, optionally referencing saved models.

    Shard-local best epochs are never used for the final result. Every
    subject/epoch row enters the shared selector with equal subject weight.
    Only small metric JSON files are copied; any model files stay in their shards.
    """
    output = Path(output_dir).resolve()
    merged_rows = []
    baseline = None
    source_shards = []
    for shard_number, expected_targets in enumerate(SHARD_TARGETS):
        shard = output / f"shard_{shard_number}"
        if not (shard / "task.json").is_file() or not (shard / "execution.json").is_file():
            raise ValueError(f"Incomplete source shard: {shard}")
        spec = json.loads((shard / "task.json").read_text())
        execution = json.loads((shard / "execution.json").read_text())
        if execution.get("status") != "completed" or execution.get("failures") or execution.get("interrupted"):
            raise ValueError(f"Failed or incomplete source shard: {shard}")
        if tuple(spec["target_subjects"]) != expected_targets:
            raise ValueError(f"Unexpected held-out users in {shard}")
        # Older checkpoint-backed suites predate this flag.
        spec.setdefault("metrics_only", False)
        if baseline is None:
            baseline = spec
        elif any(spec.get(key) != baseline.get(key) for key in MATCH_FIELDS):
            raise ValueError(f"Source shard configuration/budget mismatch: {shard}")
        rows = [json.loads(path.read_text()) for path in sorted(shard.glob("subject_*/epoch_*.json"))]
        result = select_oracle(rows, list(expected_targets), int(spec["source_epochs"]),
                               metrics_only=spec["metrics_only"])
        if result["status"] != "completed":
            raise ValueError(f"Missing subject/epoch results in {shard}")
        checkpoints = validate_checkpoints(shard, rows, metrics_only=spec["metrics_only"])
        if spec["metrics_only"]:
            merged_rows.extend(rows)
        else:
            merged_rows.extend({**row, "checkpoint": str(checkpoint.relative_to(output))}
                               for row, checkpoint in zip(rows, checkpoints))
        source_shards.append(str(shard))

    # Validate the complete global set before publishing any aggregate files.
    targets = list(range(23))
    selected = select_oracle(merged_rows, targets, int(baseline["source_epochs"]),
                             metrics_only=baseline["metrics_only"])
    if selected["status"] != "completed":
        raise ValueError("The source shards do not cover every user and epoch.")
    spec = {key: value for key, value in baseline.items() if key not in ("deadline", "started")}
    spec.update(target_subjects=targets, source_shards=source_shards)
    write_json(output / "task.json", spec)
    for row in merged_rows:
        write_json(output / f"subject_{int(row['target_subject']):02d}" / f"epoch_{int(row['epoch']):03d}.json", row)
    write_json(output / "execution.json", {
        "status": "completed", "failures": [], "interrupted": False,
        "unstarted_subjects": [], "source_shards": source_shards,
    })
    result = summarize(output)
    print(f"Selected common oracle epoch {result['best_epoch']} across all 23 users; "
          f"mean balanced_accuracy={result['selection_score']:.12f}; "
          f"metrics_only={baseline['metrics_only']}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        collect(args.output_dir)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Cannot select a global oracle epoch: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
