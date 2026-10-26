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

DEFAULT_CONFIGS = Path(__file__).with_name("configurations.json")


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
    train_ids, validation_ids = fold_subjects(arrays["subjects"], subject,
                                             spec["validation_subjects"], spec["seed"])
    train = np.isin(arrays["subjects"], train_ids)
    validation = np.isin(arrays["subjects"], validation_ids)
    test = arrays["subjects"] == subject
    config = spec["configuration"]["hyperparameters"]
    if config.get("classification_level", "trial") != "trial":
        raise ValueError("This runner operates on complete trials; use a separate protocol for windows.")
    model = build_sic_model(input_shape=tuple(arrays["features"].shape[1:]),
        training_subject_ids=arrays["subjects"][train],
        training_trial_ids=arrays["trials"][train], **config)
    folder = output / f"subject_{subject:02d}"
    folder.mkdir(exist_ok=False)
    write_json(folder / "runtime.json", {"tensorflow": tf.__version__, "numpy": np.__version__,
        "gpu": tf.config.experimental.get_device_details(physical[0]),
        "python": sys.version, "train_trials": int(train.sum()),
        "validation_trials": int(validation.sum()), "test_trials": int(test.sum())})
    write_json(folder / "split.json", {"test_subject": subject,
        "train_subjects": train_ids, "validation_subjects": validation_ids})
    expected_batches = (model.mldg_steps_per_epoch if model.use_mldg else
                        math.ceil(int(train.sum()) / spec["batch_size"]))
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
            validation_labels = arrays["labels"][validation]
            validation_subjects = arrays["subjects"][validation]
            validation_scores = [float(balanced_accuracy_score(
                validation_labels[validation_subjects == s],
                (validation_probabilities[validation_subjects == s, 1] >= 0.5).astype(int)))
                for s in validation_ids]
            number = epoch + 1
            checkpoint = folder / f"epoch_{number:03d}.keras"
            temporary = folder / f"epoch_{number:03d}.tmp.keras"
            self.model.save(temporary)
            temporary.replace(checkpoint)
            write_json(folder / f"epoch_{number:03d}.json", {
                "target_subject": subject, "epoch": number, "decision_threshold": 0.5,
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_metrics": test_metrics, "validation_metrics": validation_metrics,
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
            if time.time() >= spec["deadline"]:
                self.model.stop_training = True

    model.fit(model.prepare_fit_inputs(arrays["features"][train], arrays["subjects"][train]),
              arrays["labels"][train], batch_size=spec["batch_size"], epochs=spec["source_epochs"],
              verbose=2, callbacks=[tf.keras.callbacks.TerminateOnNaN(), SaveEpoch()])
    return 0 if len(list(folder.glob("epoch_*.json"))) == spec["source_epochs"] else 2


def run(args):
    records = configurations(args.configs)
    if not 0 <= args.config_index < len(records):
        raise ValueError("Invalid configuration index.")
    if min(args.source_epochs, args.batch_size, args.time_budget_seconds) < 1:
        raise ValueError("Epochs, batch size and time budget must be positive.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write_json(output / "execution.json", {"status": "preparing_data", "started": started})
    try:
        available = prepare_data(args.eeg, args.labels, output / "data")
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
        target_subjects=targets, seed=args.seed, batch_size=args.batch_size,
        validation_subjects=args.validation_subjects, deadline=started + args.time_budget_seconds,
        started=started, git_revision=revision, working_tree_dirty=dirty,
        manifest_sha256=hashlib.sha256(args.configs.read_bytes()).hexdigest(),
        input_files={str(p.resolve()): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                     for p in (args.eeg, args.labels)},
        preprocessing="SIC v15: complete ordered trials; 128-sample nonoverlapping windows; global window RMS; valence >= 3")
    write_json(output / "task.json", spec)
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
            with (output / f"subject_{subject:02d}.log").open("w") as log:
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
