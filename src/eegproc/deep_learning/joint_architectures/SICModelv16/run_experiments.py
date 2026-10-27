#!/usr/bin/env python3
"""Single-GPU SICv16 LOSO experiments; all model execution happens on the cluster.

Reuse v15's array preparation, manifest validation and common-epoch selector.
The controller imports neither TensorFlow nor the model. Each fold is isolated
in a new process. Source-validation subjects select ordinary checkpoints;
test labels select only the explicitly separate common-epoch oracle result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
# Load the existing pure-Python runner by path. Importing its package would
# execute SICModelv15/__init__.py and initialize TensorFlow in the controller.
_shared_spec = importlib.util.spec_from_file_location("_sic_v15_oracle",
    Path(__file__).resolve().parent.parent / "SICModelv15" / "sic_v15_valence_sweep.py")
_shared = importlib.util.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(_shared)
configurations = _shared.configurations
select_oracle = _shared.select_oracle
write_json = _shared.write_json


def prepare_data(*args, **kwargs):
    from eegproc.deep_learning.joint_architectures.SICModelv16.data import prepare_data as prepare
    return prepare(*args, **kwargs)


def progress_for(output, subject, epochs, **kwargs):
    from eegproc.deep_learning.joint_architectures.SICModelv16.recovery import checkpoint_progress
    return checkpoint_progress(output, subject, epochs, **kwargs)

DEFAULT_CONFIGS = Path(__file__).with_name("configurations.json")


def prediction_diagnostics(labels, probabilities):
    """Describe fixed-threshold predictions without changing model selection."""
    positive = [float(row[1]) for row in probabilities]
    if not positive or len(labels) != len(positive):
        raise ValueError("Prediction diagnostics require aligned, nonempty labels/probabilities.")
    predicted = [int(value >= 0.5) for value in positive]
    return {"predicted_class_1_fraction": sum(predicted) / len(predicted),
            "true_class_1_fraction": sum(int(y) for y in labels) / len(labels),
            "class_1_probability_min": min(positive),
            "class_1_probability_max": max(positive),
            "single_class_predictions": len(set(predicted)) == 1}


def summarize(output):
    output = Path(output)
    spec = json.loads((output / "task.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted(output.glob("subject_*/epoch_*.json"))]
    for row in rows:
        if not (output / row["checkpoint"]).is_file():
            raise ValueError(f"Missing checkpoint: {row['checkpoint']}")
    oracle = select_oracle(rows, spec["target_subjects"], spec["source_epochs"])
    oracle.update(configuration=spec["configuration"], target_subjects=spec["target_subjects"],
                  source_epochs=spec["source_epochs"], seed=spec["seed"])
    write_json(output / "oracle_summary.json", oracle)
    if oracle["status"] == "completed":
        write_json(output / "oracle_models.json", {
            "epoch": oracle["best_epoch"], "models": oracle["selected_models"],
            "mean_balanced_accuracy": oracle["selection_score"], "oracle": True})
    selected = []
    for subject in spec["target_subjects"]:
        fold = [r for r in rows if r["target_subject"] == subject]
        if len(fold) != spec["source_epochs"]:
            continue
        # All epochs have identical validation subjects, at fixed threshold .5.
        # A subject mean gives each validation subject equal weight.
        best = max(sorted(fold, key=lambda r: r["epoch"]),
                   key=lambda r: r["validation_mean_balanced_accuracy"])
        selected.append({"target_subject": subject, "epoch": best["epoch"],
            "checkpoint": best["checkpoint"], "test_metrics": best["test_metrics"],
            "prediction_diagnostics": prediction_diagnostics(best["y_true"], best["probabilities"]),
            "validation_mean_balanced_accuracy": best["validation_mean_balanced_accuracy"]})
    validation = {"oracle": False, "selection_rule": "per-fold source-validation subject mean balanced accuracy; earliest tie",
                  "status": "completed" if len(selected) == len(spec["target_subjects"]) else "incomplete",
                  "folds": selected}
    if validation["status"] == "completed":
        validation["mean_test_balanced_accuracy"] = math.fsum(
            r["test_metrics"]["balanced_accuracy"] for r in selected) / len(selected)
        validation["mean_test_accuracy"] = math.fsum(
            r["test_metrics"]["accuracy"] for r in selected) / len(selected)
    write_json(output / "validation_selected_summary.json", validation)
    return {"oracle": oracle, "validation_selected": validation}


def fold_subjects(subjects, target, validation_count, seed):
    """Split whole subjects deterministically, independently of model settings."""
    import numpy as np
    source = np.asarray(sorted(set(int(s) for s in subjects) - {int(target)}))
    if not 1 <= validation_count < len(source):
        raise ValueError("Reserve at least one validation subject and one training subject.")
    validation = np.random.default_rng(seed + int(target)).choice(
        source, size=validation_count, replace=False)
    return sorted(set(source.tolist()) - set(validation.tolist())), sorted(validation.tolist())


def worker(spec_path, subject):
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    from eegproc.deep_learning.joint_architectures.SICModelv16.sic_model import build_sic_model

    spec_path = Path(spec_path).resolve()
    output = spec_path.parent
    spec = json.loads(spec_path.read_text())
    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != 1:
        raise RuntimeError(f"Exactly one allocated GPU is required, found {len(physical)}.")
    tf.config.experimental.set_memory_growth(physical[0], True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(spec["seed"] + subject)
    arrays = {name: np.load(output / "data" / f"{name}.npy", mmap_mode="r")
              for name in ("features", "labels", "subjects", "trials")}
    if spec.get("data_subjects") is not None and set(np.unique(arrays["subjects"]).tolist()) != set(spec["data_subjects"]):
        raise ValueError("Prepared arrays contain subjects outside the configured data pool.")
    train_ids, validation_ids = fold_subjects(arrays["subjects"], subject,
                                             spec["validation_subjects"], spec["seed"])
    train = np.isin(arrays["subjects"], train_ids)
    validation = np.isin(arrays["subjects"], validation_ids)
    test = arrays["subjects"] == subject
    config = spec["configuration"]["hyperparameters"]
    if config.get("classification_level", "trial") != "trial":
        raise ValueError("This runner operates on complete trials; use a separate protocol for windows.")
    folder = output / f"subject_{subject:02d}"
    progress = progress_for(output, subject, spec["source_epochs"], verify_latest=True)
    if progress["complete"]:
        return 0
    if progress["orphan_checkpoints"]:
        raise ValueError(f"Unpaired saved models require inspection before resume; preserved: {progress['orphan_checkpoints']}")
    initial_epoch = progress["completed_epochs"]
    expected_batches = (config.get("mldg_steps_per_epoch") if config.get("training_method") == "mldg" else
                        math.ceil(int(train.sum()) / spec["batch_size"]))
    if expected_batches is None:
        raise ValueError("Resume requires an explicit MLDG steps_per_epoch.")
    if progress["checkpoint"]:
        with warnings.catch_warnings(record=True) as load_warnings:
            warnings.simplefilter("always")
            model = tf.keras.models.load_model(progress["checkpoint"], compile=True)
        if any("optimizer" in str(w.message).lower() and "skipping" in str(w.message).lower()
               for w in load_warnings):
            raise ValueError("Keras skipped optimizer state restoration; refusing to restart optimizer moments.")
        if int(model.main_optimizer.iterations.numpy()) != initial_epoch * expected_batches:
            raise ValueError("Restored optimizer step count does not match the completed epoch budget.")
        model.set_source_training_metadata(arrays["subjects"][train], arrays["trials"][train])
        write_json(folder / f"resume_{time.time_ns()}.json", {
            "initial_epoch": initial_epoch, "checkpoint": progress["checkpoint"],
            "optimizer_iterations": int(model.main_optimizer.iterations.numpy()),
            "note": "Weights and optimizer restored. Future shuffle/dropout draws need not match an uninterrupted run."})
        print(f"RESUME subject={subject} after epoch={initial_epoch}; optimizer state restored", flush=True)
    else:
        model = build_sic_model(input_shape=tuple(arrays["features"].shape[1:]),
            training_subject_ids=arrays["subjects"][train],
            training_trial_ids=arrays["trials"][train], **config)
    folder.mkdir(exist_ok=True)
    write_json(folder / "runtime.json", {"tensorflow": tf.__version__, "numpy": np.__version__,
        "gpu": tf.config.experimental.get_device_details(physical[0]),
        "python": sys.version, "train_trials": int(train.sum()),
        "validation_trials": int(validation.sum()), "test_trials": int(test.sum())})
    split = {"test_subject": subject, "train_subjects": train_ids, "validation_subjects": validation_ids}
    if (folder / "split.json").exists() and json.loads((folder / "split.json").read_text()) != split:
        raise ValueError("Saved subject split differs from the current split; refusing resume.")
    write_json(folder / "split.json", split)
    if expected_batches is None:
        raise ValueError("Set mldg_steps_per_epoch explicitly for MLDG experiments.")

    def evaluate(mask):
        features = arrays["features"][mask]
        # Explicit softmax: model outputs logits, never guess their interpretation.
        probabilities = np.concatenate([tf.nn.softmax(model(features[i:i+1], training=False), -1).numpy()
                                        for i in range(len(features))])
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite predictions.")
        labels = arrays["labels"][mask]
        predicted = (probabilities[:, 1] >= 0.5).astype(int)
        metrics = dict(accuracy=float(accuracy_score(labels, predicted)),
            balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
            macro_f1=float(f1_score(labels, predicted, average="macro", zero_division=0)),
            brier_score=float(np.mean((probabilities[:, 1] - labels) ** 2)),
            roc_auc=float(roc_auc_score(labels, probabilities[:, 1])))
        return metrics, probabilities

    class SaveEpoch(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.batches, self.started = 0, time.monotonic()

        def on_train_batch_end(self, batch, logs=None):
            self.batches += 1
            if time.time() >= spec["deadline"]:
                self.model.stop_training = True

        def on_epoch_end(self, epoch, logs=None):
            if self.batches != expected_batches:
                return  # A partial epoch cannot enter either selection path.
            log_values = {key: float(value) for key, value in (logs or {}).items()}
            if "loss" not in log_values or not all(math.isfinite(v) for v in log_values.values()):
                raise ValueError("Nonfinite training metrics.")
            validation_metrics, validation_probabilities = evaluate(validation)
            test_metrics, test_probabilities = evaluate(test)
            test_diagnostics = prediction_diagnostics(arrays["labels"][test], test_probabilities)
            validation_labels = arrays["labels"][validation]
            validation_subjects = arrays["subjects"][validation]
            validation_scores = [float(balanced_accuracy_score(
                validation_labels[validation_subjects == s],
                (validation_probabilities[validation_subjects == s, 1] >= 0.5).astype(int)))
                for s in validation_ids]
            number = epoch + 1
            checkpoint = folder / f"epoch_{number:03d}.keras"
            temporary = folder / f"epoch_{number:03d}.tmp.keras"
            if checkpoint.exists():
                raise ValueError(f"Refusing to overwrite an existing checkpoint: {checkpoint}")
            if temporary.exists():
                temporary.rename(folder / f"interrupted-{time.time_ns()}-{temporary.name}")
            self.model.save(temporary)
            temporary.replace(checkpoint)
            write_json(folder / f"epoch_{number:03d}.json", {
                "target_subject": subject, "epoch": number, "decision_threshold": 0.5,
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_metrics": test_metrics, "validation_metrics": validation_metrics,
                "test_prediction_diagnostics": test_diagnostics,
                "validation_mean_balanced_accuracy": math.fsum(validation_scores) / len(validation_scores),
                "checkpoint": str(checkpoint.relative_to(output)),
                "epoch_seconds": time.monotonic() - self.started, "training_metrics": log_values,
                "gpu_memory_bytes": tf.config.experimental.get_memory_info("GPU:0"),
                "y_true": arrays["labels"][test].tolist(), "probabilities": test_probabilities.tolist(),
                "trial_ids": arrays["trials"][test].tolist(),
                "validation_labels": validation_labels.tolist(),
                "validation_subject_ids": validation_subjects.tolist(),
                "validation_probabilities": validation_probabilities.tolist()})
            print(f"SAVED epoch={number} test_BA={test_metrics['balanced_accuracy']:.5f} "
                  f"validation_BA={math.fsum(validation_scores)/len(validation_scores):.5f}", flush=True)
            if test_diagnostics["single_class_predictions"]:
                print(f"DIAGNOSTIC subject={subject} epoch={number}: all held-out predictions "
                      f"are class {int(test_diagnostics['predicted_class_1_fraction'])} at threshold 0.5; "
                      "reported metrics remain unchanged.", flush=True)
            if time.time() >= spec["deadline"]:
                self.model.stop_training = True

    model.fit(model.prepare_fit_inputs(arrays["features"][train], arrays["subjects"][train]),
              arrays["labels"][train], batch_size=spec["batch_size"], epochs=spec["source_epochs"],
              initial_epoch=initial_epoch,
              verbose=2, callbacks=[tf.keras.callbacks.TerminateOnNaN(), SaveEpoch()])
    return 0 if len(list(folder.glob("epoch_*.json"))) == spec["source_epochs"] else 2


def run(args):
    records = configurations(args.configs)
    if not 0 <= args.config_index < len(records):
        raise ValueError("Invalid configuration index.")
    if min(args.source_epochs, args.batch_size, args.time_budget_seconds) < 1:
        raise ValueError("Epochs, batch size and time budget must be positive.")
    output = Path(args.output_dir).resolve()
    resuming = bool(getattr(args, "resume", False))
    output.mkdir(parents=True, exist_ok=resuming)
    started = time.time()
    old_spec = json.loads((output / "task.json").read_text()) if resuming else None
    if resuming:
        checks = {"configuration": records[args.config_index], "source_epochs": args.source_epochs,
                  "seed": args.seed, "batch_size": args.batch_size, "validation_subjects": args.validation_subjects}
        if any(old_spec.get(key) != value for key, value in checks.items()):
            raise ValueError("Resume configuration/budget differs from task.json.")
        if old_spec["target_subjects"] != args.target_subjects:
            raise ValueError("Resume target set differs from the original run.")
        if old_spec.get("data_subjects") != getattr(args, "data_subjects", None):
            raise ValueError("Resume data pool differs from the original run.")
        for name in ("features", "labels", "subjects", "trials"):
            if not (output / "data" / f"{name}.npy").is_file():
                raise ValueError(f"Prepared {name}.npy is missing; refusing to silently rebuild resume data.")
        for path, metadata in old_spec["input_files"].items():
            current = Path(path).stat()
            if current.st_size != metadata["size"] or current.st_mtime_ns != metadata["mtime_ns"]:
                raise ValueError(f"Source dataset changed since the saved run: {path}")
        available = old_spec.get("data_subjects") or list(range(23))
        write_json(output / f"previous-task-{time.time_ns()}.json", old_spec)
        if (output / "execution.json").exists():
            previous = (output / "execution.json").read_bytes()
            (output / f"previous-execution-{time.time_ns()}.json").write_bytes(previous)
    else:
        write_json(output / "execution.json", {"status": "preparing_data", "started": started})
        try:
            available = prepare_data(args.eeg, args.labels, output / "data",
                                     data_subjects=getattr(args, "data_subjects", None))
        except Exception as exc:
            write_json(output / "execution.json", {"status": "failed", "stage": "data_preparation",
                "error_type": type(exc).__name__, "error": str(exc)})
            (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
    targets = available if args.target_subjects is None else args.target_subjects
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(available):
        raise ValueError("Invalid target subjects.")
    fold_subjects(available, targets[0], args.validation_subjects, args.seed)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    spec = dict(configuration=records[args.config_index], source_epochs=args.source_epochs,
        data_subjects=getattr(args, "data_subjects", None),
        target_subjects=targets, seed=args.seed, batch_size=args.batch_size,
        validation_subjects=args.validation_subjects, deadline=started + args.time_budget_seconds,
        started=started, git_revision=revision, working_tree_dirty=dirty,
        manifest_sha256=hashlib.sha256(args.configs.read_bytes()).hexdigest(),
        input_files={str(p.resolve()): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                     for p in (args.eeg, args.labels)},
        preprocessing="SIC v15: complete ordered trials; 128-sample nonoverlapping windows; global window RMS; valence >= 3")
    write_json(output / "task.json", spec)
    write_json(output / "execution.json", {"status": "running", "resumed": resuming, "started": started})
    process = None
    interrupted = False
    failures, completed = [], []

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        if process is not None and process.poll() is None:
            process.terminate()

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for subject in targets:
            if interrupted or time.time() >= spec["deadline"]:
                break
            if resuming and progress_for(output, subject, args.source_epochs)["complete"]:
                completed.append(subject)
                print(f"SKIP subject={subject}: all {args.source_epochs} epochs already saved", flush=True)
                continue
            with (output / f"subject_{subject:02d}.log").open("a" if resuming else "w") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker",
                    "--spec", str(output / "task.json"), "--subject", str(subject)],
                    stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
                timeout = max(1, spec["deadline"] + 180 - time.time())
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    code = 2
            if code:
                failures.append({"subject": subject, "returncode": code})
                break
            completed.append(subject)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    summary = summarize(output)
    write_json(output / "execution.json", {"failures": failures, "interrupted": interrupted,
        "completed_subjects": completed, "elapsed_seconds": time.time() - started,
        "status": summary["oracle"]["status"]})
    print(json.dumps(summary, indent=2))
    return 0 if summary["oracle"]["status"] == "completed" and not failures and not interrupted else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    launch.add_argument("--config-index", type=int, required=True)
    launch.add_argument("--output-dir", type=Path, required=True)
    launch.add_argument("--eeg", type=Path, required=True)
    launch.add_argument("--labels", type=Path, required=True)
    launch.add_argument("--source-epochs", type=int, default=50)
    launch.add_argument("--batch-size", type=int, default=1)
    launch.add_argument("--validation-subjects", type=int, default=2)
    launch.add_argument("--seed", type=int, default=42)
    launch.add_argument("--time-budget-seconds", type=int, default=85000)
    launch.add_argument("--target-subjects", nargs="+", type=int)
    launch.add_argument("--data-subjects", nargs="+", type=int,
                        help="Restrict the entire data pool, including training and validation.")
    launch.add_argument("--resume", action="store_true")
    fold = commands.add_parser("worker")
    fold.add_argument("--spec", type=Path, required=True)
    fold.add_argument("--subject", type=int, required=True)
    report = commands.add_parser("summarize")
    report.add_argument("output", type=Path)
    listing = commands.add_parser("list")
    listing.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "worker":
        return worker(args.spec, args.subject)
    if args.command == "summarize":
        result = summarize(args.output)
        print(json.dumps(result, indent=2))
        return 0 if result["oracle"]["status"] == "completed" else 2
    print(json.dumps(configurations(args.configs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
