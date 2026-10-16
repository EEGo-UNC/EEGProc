#!/usr/bin/env python3
"""Fixed-configuration DREAMER SICv15 jobs with common-epoch oracle selection.

The controller never imports TensorFlow. Each fold starts in a fresh process
with its GPU mask installed before importing the model. Oracle labels are used
for checkpoint selection only, never for gradients or MI adjacency estimation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_CONFIGS = Path(__file__).with_name("sic_v15_valence_configs.json")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configurations(path=DEFAULT_CONFIGS):
    document = json.loads(Path(path).read_text())
    records = []
    names = set()
    fingerprints = set()
    for index, entry in enumerate(document["configurations"]):
        name = entry["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            raise ValueError(f"Invalid configuration name: {name!r}")
        unknown = set(entry["overrides"]) - set(document["base"])
        if unknown:
            raise ValueError(f"Unknown overrides: {sorted(unknown)}")
        config = {**document["base"], **entry["overrides"]}
        if any(isinstance(value, dict) for value in config.values()):
            raise ValueError("Each array task must contain one fixed configuration, not a grid.")
        fingerprint = json.dumps(config, sort_keys=True)
        if name in names or fingerprint in fingerprints:
            raise ValueError("Duplicate configuration name or hyperparameters.")
        names.add(name)
        fingerprints.add(fingerprint)
        records.append({"index": index, "name": name, "hyperparameters": config})
    if not records:
        raise ValueError("The configuration manifest is empty.")
    return records


def prepare_data(eeg_path, labels_path, cache, expected_subjects=23):
    """Match the full run: 1s windows, no overlap, per-window global RMS.

    Inputs are the already converted, band-stacked arrays used by SICv15:
    EEG (subject, trial, 42, sample); ratings (subject, trial, dimension).
    Trial order is preserved, and only complete 128-sample windows are used.
    """
    import numpy as np

    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    ratings = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if eeg.ndim != 4 or eeg.shape[2] != 42 or eeg.shape[3] < 128:
        raise ValueError(f"Expected band-stacked EEG (subjects, trials, 42, samples>=128), got {eeg.shape}.")
    if ratings.ndim != 3 or ratings.shape[:2] != eeg.shape[:2] or ratings.shape[2] < 1:
        raise ValueError("Ratings must align with the EEG subject/trial axes.")
    n_subjects, n_trials, _, n_samples = eeg.shape
    if n_subjects != expected_subjects:
        raise ValueError(f"Expected {expected_subjects} source subjects, found {n_subjects}.")
    if not np.isfinite(ratings[..., 0]).all() or not ((ratings[..., 0] >= 1) & (ratings[..., 0] <= 5)).all():
        raise ValueError("Valence ratings must be finite and within DREAMER's 1–5 scale.")
    y = (ratings[..., 0] >= 3).astype(np.int64).reshape(-1)
    for subject in range(n_subjects):
        if len(np.unique(y[subject * n_trials:(subject + 1) * n_trials])) != 2:
            raise ValueError(f"Subject {subject} does not have both valence classes.")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    n_windows = n_samples // 128
    x = np.lib.format.open_memmap(cache / "features.npy", mode="w+", dtype=np.float32,
                                 shape=(n_subjects * n_trials, n_windows, 128, 42))
    for subject in range(n_subjects):
        for trial in range(n_trials):
            # Cast before normalization, matching the historical data loader.
            windows = np.asarray(eeg[subject, trial, :, :n_windows * 128], dtype=np.float32)
            windows = windows.reshape(42, n_windows, 128).transpose(1, 2, 0)
            if not np.isfinite(windows).all():
                raise ValueError(f"Nonfinite EEG in subject {subject}, trial {trial}.")
            rms = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=(1, 2), keepdims=True))
            x[subject * n_trials + trial] = windows.astype(np.float64) / np.maximum(rms, 1e-6)
    x.flush()
    del x
    np.save(cache / "labels.npy", y)
    np.save(cache / "subjects.npy", np.repeat(np.arange(n_subjects), n_trials))
    np.save(cache / "trials.npy", np.tile(np.arange(n_trials), n_subjects))
    return list(range(n_subjects))


def select_oracle(rows, subjects, epochs):
    """Maximize the unweighted subject mean at one shared epoch; ties go early."""
    subjects = list(subjects)
    if not subjects or len(subjects) != len(set(subjects)) or epochs < 1:
        raise ValueError("Expected distinct subjects and at least one epoch.")
    indexed = {}
    for row in rows:
        key = (int(row["target_subject"]), int(row["epoch"]))
        score = float(row["balanced_accuracy"])
        if key[0] not in subjects or not 1 <= key[1] <= epochs:
            raise ValueError(f"Unexpected subject/epoch: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate subject/epoch: {key}")
        if not math.isfinite(score) or not 0 <= score <= 1 or row["decision_threshold"] != 0.5:
            raise ValueError(f"Invalid oracle metric: {row}")
        indexed[key] = row
    missing = [[subject, epoch] for subject in subjects for epoch in range(1, epochs + 1)
               if (subject, epoch) not in indexed]
    means = []
    for epoch in range(1, epochs + 1):
        if all((subject, epoch) in indexed for subject in subjects):
            means.append({"epoch": epoch, "mean_balanced_accuracy":
                          math.fsum(indexed[subject, epoch]["balanced_accuracy"] for subject in subjects) / len(subjects)})
    result = {"status": "incomplete" if missing else "completed", "missing_subject_epochs": missing,
              "epoch_means": means, "selection_rule": "max_epoch(mean_subject(trial_balanced_accuracy_at_0.5))",
              "oracle": True, "evaluation_note": "Held-out labels select the epoch; this is an oracle estimate, not unbiased test performance."}
    # Do not rank a configuration that ran a smaller epoch or subject budget.
    if not missing:
        best = max(means, key=lambda row: row["mean_balanced_accuracy"])
        result.update(best_epoch=best["epoch"], selection_score=best["mean_balanced_accuracy"],
                      selected_models={str(subject): indexed[subject, best["epoch"]]["checkpoint"] for subject in subjects})
    return result


def summarize(output):
    output = Path(output)
    spec = json.loads((output / "task.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted(output.glob("subject_*/epoch_*.json"))]
    for row in rows:
        if not (output / row["checkpoint"]).is_file():
            raise ValueError(f"Missing checkpoint for saved epoch: {row['checkpoint']}")
    result = select_oracle(rows, spec["target_subjects"], spec["source_epochs"])
    result.update(configuration=spec["configuration"], source_epochs=spec["source_epochs"],
                  target_subjects=spec["target_subjects"], decision_threshold=0.5)
    write_json(output / "oracle_summary.json", result)
    with (output / "oracle_epoch_means.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "mean_balanced_accuracy"])
        writer.writeheader()
        writer.writerows(result["epoch_means"])
    if result["status"] == "completed":
        write_json(output / "oracle_models.json", {"epoch": result["best_epoch"],
                   "mean_balanced_accuracy": result["selection_score"], "models": result["selected_models"]})
    return result


def worker(spec_path, subject):
    # CUDA_VISIBLE_DEVICES has already been set by the controller.
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import balanced_accuracy_score
    from eegproc.deep_learning.cross_validation.probabilities import _predict_probabilities, _predict_labels
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model

    spec_path = Path(spec_path).resolve()
    output = spec_path.parent
    spec = json.loads(spec_path.read_text())
    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != spec["gpus_per_fold"]:
        raise RuntimeError(f"Expected {spec['gpus_per_fold']} GPUs, found {len(physical)}.")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(spec["seed"] + subject)
    arrays = {name: np.load(output / "data" / f"{name}.npy", mmap_mode="r")
              for name in ("features", "labels", "subjects", "trials")}
    mask = arrays["subjects"] == subject
    source = ~mask
    config = spec["configuration"]["hyperparameters"]
    model = build_sic_model(input_shape=tuple(arrays["features"].shape[1:]),
                            training_features=arrays["features"][source], training_labels=arrays["labels"][source],
                            training_subject_ids=arrays["subjects"][source], training_trial_ids=arrays["trials"][source],
                            **config)
    if spec["gpus_per_fold"] > 1:
        model.configure_fold_devices(tuple(device.name for device in tf.config.list_logical_devices("GPU")))
    folder = output / f"subject_{subject:02d}"
    folder.mkdir()

    class SaveOracleEpoch(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.batches = 0
            self.started = time.monotonic()

        def on_train_batch_end(self, batch, logs=None):
            self.batches += 1
            if time.time() >= spec["deadline"]:
                self.model.stop_training = True

        def on_epoch_end(self, epoch, logs=None):
            # Never label a partial epoch as a completed training epoch.
            if self.batches != config["mldg_steps_per_epoch"]:
                return
            if not math.isfinite(float((logs or {}).get("loss", float("nan")))):
                raise ValueError("Training produced a nonfinite loss.")
            probabilities = _predict_probabilities(self.model, arrays["features"][mask], batch_size=1,
                                                    n_prediction_latent_samples=0, latent_sampling_seed=None)
            predictions = _predict_labels(probabilities, decision_threshold=0.5)
            score = float(balanced_accuracy_score(arrays["labels"][mask], predictions))
            number = epoch + 1
            checkpoint = folder / f"epoch_{number:03d}.keras"
            temporary = folder / f"epoch_{number:03d}.tmp.keras"
            self.model.save(temporary)
            temporary.replace(checkpoint)
            row = {"target_subject": subject, "epoch": number, "balanced_accuracy": score,
                   "decision_threshold": 0.5, "checkpoint": str(checkpoint.relative_to(output)),
                   "epoch_seconds": time.monotonic() - self.started,
                   "y_true": arrays["labels"][mask].tolist(), "probabilities": probabilities.tolist(),
                   "trial_ids": arrays["trials"][mask].tolist()}
            write_json(folder / f"epoch_{number:03d}.json", row)
            print(f"ORACLE subject={subject} epoch={number} balanced_accuracy={score:.6f}", flush=True)
            if time.time() >= spec["deadline"]:
                self.model.stop_training = True

    model.fit(model.prepare_fit_inputs(arrays["features"][source], arrays["subjects"][source]),
              arrays["labels"][source], epochs=spec["source_epochs"], verbose=2,
              callbacks=[tf.keras.callbacks.TerminateOnNaN(), SaveOracleEpoch()])
    completed = len(list(folder.glob("epoch_*.json"))) == spec["source_epochs"]
    return 0 if completed else 2


def run(args):
    records = configurations(args.configs)
    if not 0 <= args.config_index < len(records):
        raise ValueError(f"Configuration index must be in 0..{len(records)-1}.")
    if args.source_epochs < 1 or args.time_budget_seconds <= 0:
        raise ValueError("Epoch count and time budget must be positive.")
    masks = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if len(masks) != 4 or len(set(masks)) != 4 or any(not mask.strip() for mask in masks):
        raise ValueError("Exactly four distinct allocated GPUs are required.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + args.time_budget_seconds
    available = prepare_data(args.eeg, args.labels, output / "data")
    targets = available if args.target_subjects is None else args.target_subjects
    if not targets or len(targets) != len(set(targets)) or not set(targets) <= set(available):
        raise ValueError("Target subjects must be unique IDs from the DREAMER data.")
    spec = {"configuration": records[args.config_index], "source_epochs": args.source_epochs,
            "target_subjects": targets, "seed": 42, "gpus_per_fold": 2, "deadline": deadline,
            "eeg": str(Path(args.eeg).resolve()), "labels": str(Path(args.labels).resolve()),
            "time_budget_seconds": args.time_budget_seconds, "started": started,
            "preprocessing": "128 Hz; 1s nonoverlapping windows; global RMS per window; valence >= 3"}
    write_json(output / "task.json", spec)
    pending = list(targets)
    active = {}
    failures = []
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or active:
            for slot in range(2):
                if slot not in active and pending and time.time() < deadline and not interrupted and not failures:
                    subject = pending.pop(0)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(masks[2*slot:2*slot+2]), PYTHONUNBUFFERED="1")
                    log = (output / f"subject_{subject:02d}.log").open("w")
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker",
                                                "--spec", str(output / "task.json"), "--subject", str(subject)],
                                               env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[slot] = (process, subject, log)
                    print(f"Started subject {subject} on GPU pair {env['CUDA_VISIBLE_DEVICES']}", flush=True)
            for slot, (process, subject, log) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del active[slot]
                    if code != 0:
                        failures.append({"subject": subject, "returncode": code})
            # A five-minute margin normally lets the final batch, inference and
            # checkpoint finish. The Slurm six-hour limit remains the hard cap.
            if interrupted or time.time() >= deadline + 240:
                break
            if not active and (failures or time.time() >= deadline):
                break
            if pending or active:
                time.sleep(1)
    finally:
        for process, subject, log in active.values():
            process.terminate()
        for process, subject, log in active.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            failures.append({"subject": subject, "returncode": process.returncode})
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    summary = summarize(output)
    write_json(output / "execution.json", {"elapsed_seconds": time.time() - started, "failures": failures,
               "unstarted_subjects": pending, "interrupted": interrupted, "status": summary["status"]})
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "completed" and not failures and not interrupted else 2


def rank(suite):
    groups = {}
    excluded = []
    for path in sorted(Path(suite).glob("*/oracle_summary.json")):
        result = json.loads(path.read_text())
        execution = path.with_name("execution.json")
        execution = json.loads(execution.read_text()) if execution.exists() else {}
        if result["status"] != "completed" or execution.get("failures") or execution.get("interrupted"):
            excluded.append(str(path.parent))
            continue
        # Full LOSO and screening subsets, or different budgets, must not mix.
        key = (tuple(sorted(result["target_subjects"])), result["source_epochs"])
        groups.setdefault(key, []).append({"configuration": result["configuration"]["name"],
                  "best_epoch": result["best_epoch"], "score": result["selection_score"], "directory": str(path.parent)})
    return {"leaderboards": [{"subjects": list(key[0]), "source_epochs": key[1],
            "ranking": sorted(rows, key=lambda row: (-row["score"], row["configuration"]))}
            for key, rows in groups.items()], "excluded_incomplete_or_failed": excluded}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    listing.add_argument("--count", action="store_true")
    launch = commands.add_parser("run")
    launch.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    launch.add_argument("--config-index", type=int, required=True)
    launch.add_argument("--output-dir", type=Path, required=True)
    launch.add_argument("--eeg", type=Path, required=True)
    launch.add_argument("--labels", type=Path, required=True)
    launch.add_argument("--source-epochs", type=int, default=6)
    launch.add_argument("--time-budget-seconds", type=int, default=21000)
    launch.add_argument("--target-subjects", nargs="+", type=int)
    fold = commands.add_parser("worker", help=argparse.SUPPRESS)
    fold.add_argument("--spec", type=Path, required=True)
    fold.add_argument("--subject", type=int, required=True)
    report = commands.add_parser("summarize")
    report.add_argument("output_dir", type=Path)
    ranking = commands.add_parser("rank")
    ranking.add_argument("suite", type=Path)
    args = parser.parse_args(argv)
    if args.command == "list":
        records = configurations(args.configs)
        print(len(records) if args.count else json.dumps(records, indent=2))
    elif args.command == "run":
        return run(args)
    elif args.command == "worker":
        return worker(args.spec, args.subject)
    elif args.command == "summarize":
        result = summarize(args.output_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "completed" else 2
    elif args.command == "rank":
        print(json.dumps(rank(args.suite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
