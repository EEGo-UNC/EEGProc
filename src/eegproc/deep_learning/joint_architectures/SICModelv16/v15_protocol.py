"""v15 MLDG/common-epoch/calibration protocol with v16's replacement EEG pair.

Controller/reporting imports are framework-free. Each target is isolated on the
same two-GPU fold groups used by v15. Old validation-selected runs are handled
only by the explicitly named legacy runner, never resumed into this protocol.
"""
from __future__ import annotations

import argparse
import hashlib
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
from eegproc.deep_learning.joint_architectures.SICModelv16.run_experiments import (
    configurations, prediction_diagnostics, prepare_data, progress_for, select_oracle, write_json,
)

PROTOCOL = "v15_mldg_common_epoch_calibration"
DEFAULT_EPOCHS = 12
DEFAULT_CALIBRATION_LEVELS = ((3, 6),)


def metrics_for(labels, probabilities):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite predictions.")
    predicted = (probabilities[:, 1] >= 0.5).astype(int)
    return dict(accuracy=float(accuracy_score(labels, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
        macro_f1=float(f1_score(labels, predicted, average="macro", zero_division=0)),
        brier_score=float(np.mean((probabilities[:, 1] - labels) ** 2)),
        roc_auc=float(roc_auc_score(labels, probabilities[:, 1])))


def configure_devices(spec):
    import tensorflow as tf
    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != spec["gpus_per_fold"]:
        raise RuntimeError(f"Expected {spec['gpus_per_fold']} isolated GPUs, found {len(physical)}.")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    return tuple(device.name for device in tf.config.list_logical_devices("GPU"))


def _arrays(output):
    import numpy as np
    return {name: np.load(output / "data" / f"{name}.npy", mmap_mode="r")
            for name in ("features", "labels", "subjects", "trials")}


def worker(spec_path, subject):
    import numpy as np
    import tensorflow as tf
    from eegproc.deep_learning.joint_architectures.SICModelv16.sic_model import build_sic_model
    spec_path = Path(spec_path).resolve()
    output = spec_path.parent
    spec = json.loads(spec_path.read_text())
    devices = configure_devices(spec)
    tf.keras.utils.set_random_seed(spec["seed"] + subject)
    arrays = _arrays(output)
    source = arrays["subjects"] != subject
    target = ~source
    folder = output / f"subject_{subject:02d}"
    folder.mkdir(exist_ok=True)
    progress = progress_for(output, subject, spec["source_epochs"], verify_latest=True)
    if progress["orphan_checkpoints"]:
        raise ValueError("Unpaired checkpoints require inspection; all originals are preserved.")
    if progress["complete"]:
        return 0
    config = spec["configuration"]["hyperparameters"]
    initial_epoch = progress["completed_epochs"]
    if progress["checkpoint"]:
        model = tf.keras.models.load_model(progress["checkpoint"], compile=True)
        if int(model.main_optimizer.iterations.numpy()) != initial_epoch * config["mldg_steps_per_epoch"]:
            raise ValueError("Optimizer steps do not match saved MLDG epochs; refusing weights-only resume.")
        model.set_source_training_metadata(arrays["subjects"][source], arrays["trials"][source])
        write_json(folder / f"resume_{time.time_ns()}.json", {
            "checkpoint": progress["checkpoint"], "initial_epoch": initial_epoch,
            "note": "Weights and optimizer restored; subsequent random draws may differ from an uninterrupted run."})
    else:
        model = build_sic_model(tuple(arrays["features"].shape[1:]),
            training_subject_ids=arrays["subjects"][source],
            training_trial_ids=arrays["trials"][source], **config)
    model.configure_fold_devices(devices, deterministic_execution=True)
    write_json(folder / "split.json", {"test_subject": subject,
        "train_subjects": sorted(np.unique(arrays["subjects"][source]).tolist()),
        "validation_subjects": [], "protocol": PROTOCOL})

    class SaveEpoch(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.batches = 0
            self.started = time.monotonic()

        def on_train_batch_end(self, batch, logs=None):
            self.batches += 1
            if time.time() >= spec["deadline"]:
                self.model.stop_training = True

        def on_epoch_end(self, epoch, logs=None):
            if self.batches != config["mldg_steps_per_epoch"]:
                return
            values = {key: float(value) for key, value in (logs or {}).items()}
            if "loss" not in values or not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Nonfinite source training metrics.")
            features = arrays["features"][target]
            probabilities = np.concatenate([tf.nn.softmax(self.model(features[i:i+1], training=False), -1).numpy()
                                            for i in range(len(features))])
            scores = metrics_for(arrays["labels"][target], probabilities)
            number = epoch + 1
            checkpoint = folder / f"epoch_{number:03d}.keras"
            if checkpoint.exists():
                raise ValueError(f"Refusing to overwrite {checkpoint}")
            temporary = folder / f"epoch_{number:03d}.tmp.keras"
            if temporary.exists():
                temporary.rename(folder / f"interrupted-{time.time_ns()}-{temporary.name}")
            self.model.save(temporary)
            temporary.replace(checkpoint)
            write_json(folder / f"epoch_{number:03d}.json", {
                "target_subject": subject, "epoch": number, "decision_threshold": 0.5,
                "balanced_accuracy": scores["balanced_accuracy"], "test_metrics": scores,
                "checkpoint": str(checkpoint.relative_to(output)), "training_metrics": values,
                "epoch_seconds": time.monotonic() - self.started,
                "y_true": arrays["labels"][target].tolist(), "probabilities": probabilities.tolist(),
                "trial_ids": arrays["trials"][target].tolist()})
            print(f"ORACLE subject={subject} epoch={number} balanced_accuracy={scores['balanced_accuracy']:.6f}", flush=True)
            if time.time() >= spec["deadline"]:
                self.model.stop_training = True

    model.fit(model.prepare_fit_inputs(arrays["features"][source], arrays["subjects"][source]),
        arrays["labels"][source], batch_size=spec["batch_size"], epochs=spec["source_epochs"],
        initial_epoch=initial_epoch, verbose=2, callbacks=[tf.keras.callbacks.TerminateOnNaN(), SaveEpoch()])
    return 0 if progress_for(output, subject, spec["source_epochs"])["complete"] else 2


def summarize(output):
    output = Path(output)
    spec = json.loads((output / "task.json").read_text())
    if spec.get("protocol") != PROTOCOL:
        raise ValueError("This is not a restored v15-protocol run.")
    rows = [json.loads(path.read_text()) for path in sorted(output.glob("subject_*/epoch_*.json"))]
    for row in rows:
        if not (output / row["checkpoint"]).is_file():
            raise ValueError(f"Missing checkpoint: {row['checkpoint']}")
    oracle = select_oracle(rows, spec["target_subjects"], spec["source_epochs"])
    oracle.update(protocol=PROTOCOL, configuration=spec["configuration"],
                  target_subjects=spec["target_subjects"], source_epochs=spec["source_epochs"], seed=spec["seed"])
    if oracle["status"] == "completed":
        selected = [row for row in rows if row["epoch"] == oracle["best_epoch"]]
        oracle["folds"] = [{"target_subject": row["target_subject"], "epoch": row["epoch"],
            "checkpoint": row["checkpoint"], "test_metrics": row["test_metrics"],
            "prediction_diagnostics": prediction_diagnostics(row["y_true"], row["probabilities"])} for row in selected]
        write_json(output / "oracle_models.json", {"epoch": oracle["best_epoch"],
            "models": oracle["selected_models"], "mean_balanced_accuracy": oracle["selection_score"], "oracle": True})
    write_json(output / "oracle_summary.json", oracle)
    return {"oracle": oracle}


def calibration_worker(spec_path, subject):
    import tensorflow as tf
    from eegproc.deep_learning.joint_architectures.SICModelv16 import sic_model  # register Keras types
    from eegproc.deep_learning.cross_validation import subject_calibration_cv
    from eegproc.deep_learning.joint_architectures.sic_common.config import SIC_CLASSIFICATION_METRICS
    from eegproc.deep_learning.joint_architectures.sic_common.training import _save_calibration_artifacts
    spec_path = Path(spec_path).resolve()
    output = spec_path.parent
    spec = json.loads(spec_path.read_text())
    configure_devices(spec)
    tf.keras.utils.set_random_seed(spec["seed"] + subject)
    # The controller publishes immutable selection before parallel calibration.
    # Workers only read it, avoiding concurrent writes to shared report files.
    oracle = json.loads((output / "oracle_summary.json").read_text())
    if oracle["status"] != "completed" or oracle.get("protocol") != PROTOCOL:
        raise ValueError("Every source fold/epoch must finish before calibration.")
    model = tf.keras.models.load_model(output / oracle["selected_models"][str(subject)], compile=False)
    arrays = _arrays(output)
    result = subject_calibration_cv(model_builder_function=None, pretrained_model=model,
        feature_array=arrays["features"], label_array=arrays["labels"],
        subject_id_array=arrays["subjects"], trial_id_array=arrays["trials"],
        source_epochs=0, source_batch_size=spec["batch_size"],
        training_seed=spec["seed"], deterministic_training=True,
        calibration_epochs=spec["calibration_epochs"], calibration_batch_size=spec["calibration_batch_size"],
        calibration_levels=[tuple(level) for level in spec["calibration_levels"]],
        calibration_learning_rate=1e-4, calibration_optimizer="adamw", calibration_weight_decay=0.0,
        calibration_seed=spec["seed"], stratify_calibration=True,
        validation_subjects_per_fold=0, early_stopping_patience=None, restore_best_weights=False,
        evaluation_level="trial", metrics=SIC_CLASSIFICATION_METRICS, decision_threshold=0.5,
        log_predictions=True, source_use_class_weight=False, calibration_use_class_weight=False,
        calibration_fit_kwargs={"callbacks": [tf.keras.callbacks.TerminateOnNaN()]},
        n_jobs=1, target_subjects=[subject], verbose=0)
    expected = next(row["test_metrics"]["balanced_accuracy"] for row in oracle["folds"]
                    if row["target_subject"] == subject)
    actual = result["overall"]["zero_shot_all_trials_mean_scores"]["balanced_accuracy"]
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"Reloaded zero-shot balanced accuracy differs: {actual} != {expected}")
    folder = output / "calibration" / f"subject_{subject:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    _save_calibration_artifacts(folder, model_config=spec["configuration"]["hyperparameters"], results=result)
    write_json(folder / "result.json", {"target_subject": subject, "selected_epoch": oracle["best_epoch"],
        "protocol": PROTOCOL, "overall": result["overall"]})
    return 0


def publish_final(output):
    output = Path(output)
    spec = json.loads((output / "task.json").read_text())
    summary = summarize(output)
    oracle = summary["oracle"]
    if oracle["status"] != "completed":
        return summary
    calibrated = []
    for subject in spec["target_subjects"]:
        path = output / "calibration" / f"subject_{subject:02d}" / "result.json"
        if not path.is_file():
            return summary
        result = json.loads(path.read_text())
        if result["selected_epoch"] != oracle["best_epoch"] or result["protocol"] != PROTOCOL:
            raise ValueError("Calibration result does not match selected source models.")
        calibrated.append(result)
    means = {metric: math.fsum(row["test_metrics"][metric] for row in oracle["folds"]) / len(oracle["folds"])
             for metric in oracle["folds"][0]["test_metrics"]}
    levels = {}
    for shots, folds in spec["calibration_levels"]:
        values = [row["overall"]["calibration_levels"][str(shots)]["calibrated_mean_scores"] for row in calibrated]
        levels[str(shots)] = {"folds": folds, "calibrated_mean_scores": {
            key: math.fsum(value[key] for value in values) / len(values) for key in values[0]}}
    final = {"status": "completed", "protocol": PROTOCOL, "oracle": True,
        "selected_epoch": oracle["best_epoch"], "mean_accuracy": means["accuracy"],
        "mean_balanced_accuracy": means["balanced_accuracy"], "zero_shot_mean_scores": means,
        "calibration_levels": levels, "subjects": spec["target_subjects"]}
    write_json(output / "final_metrics.json", final)
    summary["final"] = final
    return summary


def launch_stage(output, spec, command, subjects):
    """v15's isolated, synchronous two-GPU folds; two folds in parallel."""
    if not subjects:
        return [], False
    masks = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    required = spec["n_jobs"] * spec["gpus_per_fold"]
    if len(masks) != required or len(set(masks)) != required or any(not mask.strip() for mask in masks):
        raise ValueError(f"Allocate exactly {required} distinct GPUs for {spec['n_jobs']} fold workers.")
    pending, active, failures = list(subjects), {}, []
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or active:
            for slot in range(spec["n_jobs"]):
                if slot in active or not pending or interrupted or failures or time.time() >= spec["deadline"]:
                    continue
                subject = pending.pop(0)
                first = slot * spec["gpus_per_fold"]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(masks[first:first+spec["gpus_per_fold"]]),
                    PYTHONUNBUFFERED="1", PYTHONHASHSEED=str(spec["seed"]),
                    TF_DETERMINISTIC_OPS="1", TF_CUDNN_DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                log = (output / f"{command}_subject_{subject:02d}.log").open("a")
                try:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), command,
                        "--spec", str(output / "task.json"), "--subject", str(subject)],
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active[slot] = (process, subject, log)
            for slot, (process, subject, log) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del active[slot]
                    if code:
                        failures.append({"subject": subject, "returncode": code})
            if interrupted or time.time() >= spec["deadline"] + 180:
                break
            if not active and (failures or time.time() >= spec["deadline"]):
                break
            if pending or active:
                time.sleep(1)
    finally:
        for process, subject, log in active.values():
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            failures.append({"subject": subject, "returncode": process.returncode})
        for sig, handler in old.items():
            signal.signal(sig, handler)
    return failures, interrupted


def run(args):
    records = configurations(args.configs)
    if not 0 <= args.config_index < len(records):
        raise ValueError("Invalid configuration index.")
    config = records[args.config_index]["hyperparameters"]
    if (config.get("training_method") != "mldg" or not config.get("use_recurrent_classifier")
            or not config.get("use_variational_head") or not config.get("use_subject_adversarial")):
        raise ValueError("Restored protocol requires v15 MLDG, recurrent/VC heads and the subject adversary.")
    if getattr(args, "validation_subjects", 0) != 0 or getattr(args, "data_subjects", None) is not None:
        raise ValueError("v15 uses all 22 source subjects; no validation holdout or restricted data pool.")
    source_epochs = int(args.source_epochs)
    batch_size = int(args.batch_size)
    calibration_epochs = int(getattr(args, "calibration_epochs", 1))
    calibration_batch_size = int(getattr(args, "calibration_batch_size", 64))
    n_jobs, gpus_per_fold = int(getattr(args, "n_jobs", 2)), int(getattr(args, "gpus_per_fold", 2))
    if min(source_epochs, batch_size, calibration_epochs, calibration_batch_size, n_jobs, gpus_per_fold, args.time_budget_seconds) < 1:
        raise ValueError("Training, calibration, device and time budgets must be positive.")
    levels = [list(level) for level in (getattr(args, "calibration_levels", None) or DEFAULT_CALIBRATION_LEVELS)]
    if (any(len(level) != 2 or min(level) < 1 or level[0] >= 18 for level in levels)
            or len({level[0] for level in levels}) != len(levels)):
        raise ValueError("Calibration levels require distinct shots in 1..17 and positive fold counts.")
    output = Path(args.output_dir).resolve()
    resuming = bool(getattr(args, "resume", False))
    output.mkdir(parents=True, exist_ok=resuming)
    immutable = dict(protocol=PROTOCOL, configuration=records[args.config_index], source_epochs=source_epochs,
        batch_size=batch_size, seed=args.seed, target_subjects=args.target_subjects or list(range(23)),
        validation_subjects=0, data_subjects=None, calibration_epochs=calibration_epochs,
        calibration_batch_size=calibration_batch_size, calibration_levels=levels,
        n_jobs=n_jobs, gpus_per_fold=gpus_per_fold)
    if (not immutable["target_subjects"] or len(set(immutable["target_subjects"])) != len(immutable["target_subjects"])
            or not set(immutable["target_subjects"]) <= set(range(23))):
        raise ValueError("Invalid target subjects.")
    started = time.time()
    if resuming:
        previous = json.loads((output / "task.json").read_text())
        if any(previous.get(key) != value for key, value in immutable.items()):
            raise ValueError("Resume protocol/settings differ; use a new run for the restored v15 protocol.")
        for path, metadata in previous["input_files"].items():
            current = Path(path).stat()
            if current.st_size != metadata["size"] or current.st_mtime_ns != metadata["mtime_ns"]:
                raise ValueError("Input dataset changed since the saved run.")
        for name in ("features", "labels", "subjects", "trials"):
            if not (output / "data" / f"{name}.npy").is_file():
                raise ValueError("Prepared resume data is missing; refusing to rebuild it.")
        write_json(output / f"previous-task-{time.time_ns()}.json", previous)
    else:
        write_json(output / "execution.json", {"status": "preparing_data", "protocol": PROTOCOL})
        try:
            prepare_data(args.eeg, args.labels, output / "data")
        except Exception as exc:
            write_json(output / "execution.json", {"status": "failed", "stage": "data_preparation",
                "protocol": PROTOCOL, "error_type": type(exc).__name__, "error": str(exc)})
            raise
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    spec = dict(immutable, git_revision=revision, working_tree_dirty=dirty, started=started, deadline=started + args.time_budget_seconds,
        input_files={str(path.resolve()): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                     for path in (args.eeg, args.labels)},
        manifest_sha256=hashlib.sha256(args.configs.read_bytes()).hexdigest(),
        preprocessing="v15: complete ordered trials; 128-sample windows; global window RMS; valence >= 3",
        reference="SICModelv15/sic_v15_valence_configs.json: baseline; new encoder/decoder topology only")
    write_json(output / "task.json", spec)
    write_json(output / "execution.json", {"status": "source_training", "protocol": PROTOCOL})
    remaining = [subject for subject in spec["target_subjects"]
                 if not progress_for(output, subject, source_epochs)["complete"]]
    failures, interrupted = launch_stage(output, spec, "worker", remaining)
    summary = summarize(output)
    if not failures and not interrupted and summary["oracle"]["status"] == "completed":
        write_json(output / "execution.json", {"status": "calibrating", "protocol": PROTOCOL})
        remaining = [subject for subject in spec["target_subjects"]
            if not (output / "calibration" / f"subject_{subject:02d}" / "result.json").is_file()]
        failures, interrupted = launch_stage(output, spec, "calibrate", remaining)
        summary = publish_final(output)
    complete = "final" in summary and not failures and not interrupted
    write_json(output / "execution.json", {"status": "completed" if complete else "incomplete",
        "protocol": PROTOCOL, "failures": failures, "interrupted": interrupted,
        "elapsed_seconds": time.time() - started})
    print(json.dumps(summary, indent=2))
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "calibrate"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    args = parser.parse_args()
    return (worker if args.command == "worker" else calibration_worker)(args.spec, args.subject)


if __name__ == "__main__":
    raise SystemExit(main())
