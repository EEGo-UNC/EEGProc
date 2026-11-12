#!/usr/bin/env python3
"""Check sharded DREAMER valence checkpoints and prepare a CFO model manifest."""

import argparse
import json
from pathlib import Path
import sys


SHARD_SUBJECTS = (range(0, 8), range(8, 16), range(16, 23))
ALL_SUBJECTS = set(range(23))
SHARED_SETTINGS = ("configuration", "seed", "eeg", "labels", "preprocessing")


def read_json(path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error


def build_manifest(configuration_dir, epoch):
    shards = [configuration_dir / f"shard_{index}" for index in range(len(SHARD_SUBJECTS))]
    specs = [read_json(shard / "task.json") for shard in shards]
    metrics_only = [shard.name for shard, spec in zip(shards, specs) if spec.get("metrics_only") is True]
    if metrics_only:
        raise ValueError(
            f"Metrics-only output in {', '.join(metrics_only)}. These shards saved predictions "
            "without model weights. Rerun the selected configuration with checkpoint saving enabled."
        )

    config = specs[0]["configuration"]
    hyperparameters = config["hyperparameters"]
    if not hyperparameters.get("use_decoder") or hyperparameters["reconstruction_loss_weight"] <= 0:
        raise ValueError(f"{config['name']} needs a trained decoder for EEG counterfactuals.")

    entries = []
    for shard, subjects, spec in zip(shards, SHARD_SUBJECTS, specs):
        if sorted(spec["target_subjects"]) != list(subjects):
            raise ValueError(f"{shard}: unexpected held-out subjects")
        if any(spec[key] != specs[0][key] for key in SHARED_SETTINGS):
            raise ValueError(f"{shard}: configuration, seed, data, or preprocessing differs across shards")
        for subject in subjects:
            epoch_file = Path(f"subject_{subject:02d}") / f"epoch_{epoch:03d}"
            record_path = shard / epoch_file.with_suffix(".json")
            record = read_json(record_path)
            if (record.get("target_subject"), record.get("epoch")) != (subject, epoch):
                raise ValueError(f"{record_path}: subject or epoch does not match its location")
            relative_checkpoint = epoch_file.with_suffix(".keras")
            checkpoint = shard / relative_checkpoint
            if record.get("checkpoint") != relative_checkpoint.as_posix():
                raise ValueError(f"{record_path}: no matching checkpoint recorded")
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise ValueError(f"Missing or empty checkpoint: {checkpoint}")
            entries.append({
                "target_subject": subject,
                "stage": "zero_shot_source_model",
                "source_subject_ids": sorted(ALL_SUBJECTS - {subject}),
                "path": str(checkpoint.resolve()),
            })
    return {"epoch": epoch, "configuration": config["name"], "models": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration_dir", type=Path, help="Sweep directory containing shard_0 through shard_2")
    parser.add_argument("--epoch", type=int, default=48, help="One epoch shared across all subjects (default: 48)")
    parser.add_argument("--output", type=Path, help="Write a new manifest here. Omit to check availability only.")
    args = parser.parse_args()
    if args.epoch < 1:
        parser.error("--epoch must be positive")

    try:
        manifest = build_manifest(args.configuration_dir, args.epoch)
        if args.output:
            with args.output.open("x") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"Checkpoint check failed: {error}", file=sys.stderr)
        return 1

    print(f"Found {len(manifest['models'])} {manifest['configuration']} checkpoints at epoch {args.epoch}.")
    if args.output:
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
