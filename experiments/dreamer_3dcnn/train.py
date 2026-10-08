"""Train one tracked DREAMER 3D-CNN run.

The script is deliberately config-driven: Longleaf receives exactly the JSON
created by ``tracker.py plan`` and writes all portable artifacts beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TARGET_INDEX = {"valence": 0, "arousal": 1}


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_revision(project_dir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_windows(
    eeg: np.ndarray,
    ratings: np.ndarray,
    *,
    target: str,
    threshold: float,
    window_samples: int,
    overlap: float,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert ``(subject, trial, feature, time)`` arrays to rank-3 windows."""
    if eeg.ndim != 4 or ratings.ndim != 3:
        raise ValueError(f"Expected rank-4 EEG and rank-3 labels; got {eeg.shape}, {ratings.shape}.")
    if eeg.shape[:2] != ratings.shape[:2] or eeg.shape[2] != 42:
        raise ValueError(f"Unexpected DREAMER array shapes: EEG={eeg.shape}, labels={ratings.shape}.")
    if target not in TARGET_INDEX:
        raise ValueError(f"Unknown target {target!r}.")
    if window_samples < 1 or window_samples > eeg.shape[-1]:
        raise ValueError("window_samples must fit within one DREAMER trial.")
    if not 0 <= overlap < 1:
        raise ValueError("window_overlap must be in [0, 1).")
    if normalization not in {"none", "window_zscore"}:
        raise ValueError("normalization must be 'none' or 'window_zscore'.")

    hop = max(1, round(window_samples * (1.0 - overlap)))
    starts = range(0, eeg.shape[-1] - window_samples + 1, hop)
    starts = tuple(starts)
    n_subjects, n_trials = eeg.shape[:2]
    rows = n_subjects * n_trials * len(starts)
    features = np.empty((rows, window_samples, eeg.shape[2]), dtype=np.float32)
    labels = np.empty(rows, dtype=np.int64)
    subject_ids = np.empty(rows, dtype=np.int64)
    trial_ids = np.empty(rows, dtype=np.int64)
    cursor = 0
    for subject in range(n_subjects):
        for trial in range(n_trials):
            label = int(ratings[subject, trial, TARGET_INDEX[target]] >= threshold)
            for start in starts:
                window = np.asarray(
                    eeg[subject, trial, :, start : start + window_samples].T,
                    dtype=np.float32,
                )
                if normalization == "window_zscore":
                    mean = window.mean(axis=0, keepdims=True, dtype=np.float32)
                    std = window.std(axis=0, keepdims=True, dtype=np.float32)
                    window = (window - mean) / (std + np.float32(1e-6))
                features[cursor] = window
                labels[cursor] = label
                subject_ids[cursor] = subject
                trial_ids[cursor] = subject * n_trials + trial
                cursor += 1
    return features, labels, subject_ids, trial_ids


def build_model_factory(config: dict[str, Any]):
    import tensorflow as tf

    from eegproc.deep_learning.unsupervised.Convolutions.CNN3D import (
        MTLFuseNet3DCNNEncoder,
    )

    def build_model(**overrides):
        settings = {**config, **overrides}
        tf.keras.utils.set_random_seed(int(settings["seed"]))
        inputs = tf.keras.layers.Input((int(settings["window_samples"]), 42), name="eeg_window")
        encoder = MTLFuseNet3DCNNEncoder(
            timesteps=int(settings["window_samples"]),
            t_down=int(np.prod(settings["temporal_pool_sizes"])) if settings["temporal_pool_sizes"] else 1,
            conv_filters=tuple(settings["conv_filters"]),
            temporal_kernel_size=int(settings["temporal_kernel_size"]),
            spatial_kernel_size=int(settings["spatial_kernel_size"]),
            spatial_pool_sizes=tuple(settings["spatial_pool_sizes"]),
            temporal_pool_sizes=tuple(settings["temporal_pool_sizes"]),
            emb_dim=int(settings["emb_dim"]),
            dropout=float(settings["dropout"]),
        )
        x = encoder(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D(name="temporal_average")(x)
        x = tf.keras.layers.Dense(int(settings["hidden_units"]), activation="relu", name="hidden")(x)
        x = tf.keras.layers.Dropout(float(settings["dropout"]), name="classifier_dropout")(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax", name="class_probabilities")(x)
        model = tf.keras.Model(inputs, outputs, name="dreamer_3dcnn")
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    build_model._sequence_hyperparameter_depths = {
        "conv_filters": 1,
        "spatial_pool_sizes": 1,
        "temporal_pool_sizes": 1,
    }
    return build_model


def run(config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    import tensorflow as tf

    from eegproc.deep_learning.cross_validation import loso_cv

    tf.keras.utils.set_random_seed(int(config["seed"]))
    eeg_path = data_dir / "dreamer_eeg.npy"
    labels_path = data_dir / "dreamer_labels.npy"
    eeg = np.load(eeg_path, mmap_mode="r")
    ratings = np.load(labels_path, mmap_mode="r")
    features, labels, subjects, trials = make_windows(
        eeg,
        ratings,
        target=str(config["target"]),
        threshold=float(config["label_threshold"]),
        window_samples=int(config["window_samples"]),
        overlap=float(config["window_overlap"]),
        normalization=str(config["normalization"]),
    )
    print(
        f"Assembled {len(features)} windows: shape={features.shape}, "
        f"positive_rate={labels.mean():.4f}",
        flush=True,
    )
    results = loso_cv(
        build_model_factory(config),
        features,
        labels,
        subjects,
        trials,
        n_epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        evaluation_level="trial",
        selection_level="trial",
        selection_metric="balanced_accuracy",
        metrics=("accuracy", "f1", "macro_f1", "balanced_accuracy", "roc_auc", "brier_score", "ece"),
        log_predictions=False,
        validation_subjects_per_fold=int(config["validation_subjects"]),
        validation_seed=int(config["seed"]),
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_monitor="val_loss",
        restore_best_weights=True,
        verbose=2,
        n_jobs=1,
        cpus_per_worker=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        max_folds=config["max_folds"],
    )
    results["run_metadata"] = {
        "target": config["target"],
        "n_windows": int(len(features)),
        "feature_shape": list(features.shape),
        "positive_rate": float(labels.mean()),
        "tensorflow_version": tf.__version__,
    }
    return results


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    manifest_path = args.output_dir / "manifest.json"
    manifest = read_json(manifest_path)
    project_dir = Path(__file__).resolve().parents[2]
    manifest.update(
        {
            "status": "running",
            "started_at": now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": platform.node(),
            "git_revision": git_revision(project_dir),
            "python_version": platform.python_version(),
        }
    )
    write_json(manifest_path, manifest)
    try:
        results = run(config, args.data_dir)
        write_json(args.output_dir / "result.json", results)
        manifest.update({"status": "completed", "completed_at": now()})
        write_json(manifest_path, manifest)
        print(f"Run complete: {manifest['run_id']}")
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(manifest_path, manifest)
        (args.output_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
