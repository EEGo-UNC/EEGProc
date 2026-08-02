"""LOSO training entry point for the GCN-only EEGProc baseline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from eegproc.deep_learning.cross_val import loso_cv
from eegproc.deep_learning.joint_architectures.joint_v2_vae_vc.joint_v2_autoencoder_vc_train import (
    load_joint_v2_training_data,
)

from .model import build_gcn_softmax_classifier


MODEL_HPARAM_KEYS = {
    "learning_rate",
    "gcn_units",
    "temporal_pool_sizes",
    "t_down",
    "emb_dim",
    "dropout",
    "activation",
    "use_batch_norm",
    "temporal_readout",
    "classifier_units",
    "classifier_dropout",
    "l2_weight",
    "clipnorm",
}


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _flatten_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            row[key] = json.dumps(value, default=_json_default)
        elif isinstance(value, np.generic):
            row[key] = value.item()
        else:
            row[key] = value
    return row


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    rows = [_flatten_for_csv(record) for record in records]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_hyperparameters(raw_json: str | None, json_file: str | None) -> dict:
    if raw_json and json_file:
        raise ValueError("Use only one of --hyperparameters-json or --hyperparameters-file.")
    if json_file:
        with Path(json_file).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif raw_json:
        payload = json.loads(raw_json)
    else:
        payload = {
            "epochs": [300],
            "batch_size": [8],
            "learning_rate": [1e-4],
            "gcn_units": [[32, 16]],
            "temporal_pool_sizes": [[2]],
            "t_down": [2],
            "emb_dim": [32],
            "dropout": [0.30],
            "activation": ["relu"],
            "use_batch_norm": [False],
            "temporal_readout": ["mean_max"],
            "classifier_units": [64],
            "classifier_dropout": [0.50],
            "l2_weight": [1e-4],
            "clipnorm": [1.0],
        }
    if not isinstance(payload, dict):
        raise TypeError("Hyperparameters must decode to a JSON object.")
    return payload


def _as_class_ids(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels.astype(np.int64, copy=False)
    if labels.ndim == 2 and labels.shape[1] == 1:
        return labels[:, 0].astype(np.int64, copy=False)
    if labels.ndim == 2 and labels.shape[1] > 1:
        return np.argmax(labels, axis=1).astype(np.int64, copy=False)
    raise ValueError(f"Unsupported label shape: {labels.shape}.")


def _infer_n_classes(labels: np.ndarray) -> int:
    class_ids = _as_class_ids(labels)
    if class_ids.size == 0:
        raise ValueError("The dataset contains no labels.")
    n_classes = int(np.max(class_ids)) + 1
    if n_classes < 2:
        raise ValueError(f"Expected at least two classes; got {n_classes}.")
    return n_classes


def _load_arrays(args: argparse.Namespace):
    prepared = [
        args.features_npy,
        args.labels_npy,
        args.subjects_npy,
        args.trials_npy,
    ]
    if any(prepared):
        if not all(prepared):
            raise ValueError(
                "Prepared-array mode requires --features-npy, --labels-npy, "
                "--subjects-npy, and --trials-npy together."
            )
        return tuple(np.load(Path(path), allow_pickle=False) for path in prepared)

    loader_kwargs: dict[str, Any] = {
        "label_dimension": args.label_dimension,
        "window_size_sec": args.window_sec,
        "fs": args.fs,
        "overlap": args.window_overlap,
        "median_label": args.median_label,
        "zscore": not args.no_zscore,
    }
    signature = inspect.signature(load_joint_v2_training_data)
    if "dataset" in signature.parameters:
        loader_kwargs["dataset"] = args.dataset
    if args.raw_eeg_npy:
        loader_kwargs["eeg_path"] = args.raw_eeg_npy
    if args.raw_labels_npy:
        loader_kwargs["labels_path"] = args.raw_labels_npy

    arrays = load_joint_v2_training_data(**loader_kwargs)
    if len(arrays) != 4:
        raise RuntimeError(
            "The current load_joint_v2_training_data must return features, labels, "
            "subject IDs, and trial IDs. Update EEGProc's joint data loader or use "
            "the four prepared-array arguments."
        )
    return arrays


def _collect_best_epochs(payload: Any) -> list[int]:
    epochs: list[int] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "best_epoch" and value is not None:
                try:
                    epoch = int(value)
                except (TypeError, ValueError):
                    continue
                if epoch > 0:
                    epochs.append(epoch)
            else:
                epochs.extend(_collect_best_epochs(value))
    elif isinstance(payload, list):
        for value in payload:
            epochs.extend(_collect_best_epochs(value))
    return epochs


def _selected_config(cv_results: dict, hyperparameters: dict) -> dict:
    for key in ("best_config", "selected_config", "best_hyperparameters"):
        best_config = cv_results.get(key)
        if isinstance(best_config, dict):
            return dict(best_config)

    # A one-configuration run does not need the CV utility to report best_config.
    single: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        if key in {"gcn_units", "temporal_pool_sizes"}:
            # [[32, 16]] represents one sequence-valued candidate.
            if isinstance(value, list) and len(value) == 1:
                single[key] = value[0]
            else:
                raise RuntimeError(
                    "CV results did not report best_config for a multi-candidate "
                    f"sequence hyperparameter {key!r}."
                )
        elif isinstance(value, list) and len(value) == 1:
            single[key] = value[0]
        else:
            raise RuntimeError(
                "CV results did not report best_config and the supplied grid has "
                f"multiple candidates for {key!r}."
            )
    return single


def _fit_final_model(
    *,
    selected_config: dict,
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_channels: int,
    n_bands: int,
    epochs: int,
    output_dir: Path,
    verbose: int,
) -> None:
    fit_config = dict(selected_config)
    batch_size = int(fit_config.pop("batch_size", 8))
    fit_config.pop("epochs", None)
    unknown = set(fit_config) - MODEL_HPARAM_KEYS
    if unknown:
        raise ValueError(f"Unknown final-model hyperparameters: {sorted(unknown)}")

    model = build_gcn_softmax_classifier(
        input_shape=tuple(features.shape[1:]),
        n_classes=n_classes,
        n_channels=n_channels,
        n_bands=n_bands,
        **fit_config,
    )
    class_ids = _as_class_ids(labels)
    classes, counts = np.unique(class_ids, return_counts=True)
    class_weight = {
        int(class_id): len(class_ids) / (len(classes) * int(count))
        for class_id, count in zip(classes, counts)
    }
    history = model.fit(
        features,
        labels,
        epochs=int(epochs),
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.TerminateOnNaN()],
        verbose=verbose,
    )
    model.save(output_dir / "final_model.keras")
    model.save_weights(output_dir / "final_model.weights.h5")
    history_rows = [
        {"epoch": index + 1, **{key: values[index] for key, values in history.history.items()}}
        for index in range(len(history.history.get("loss", [])))
    ]
    _write_records_csv(output_dir / "final_history.csv", history_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a deterministic GCN-only softmax baseline with EEGProc LOSO CV."
    )
    parser.add_argument("--dataset", default="dreamer")
    parser.add_argument("--label-dimension", choices=("valence", "arousal"), default="arousal")
    parser.add_argument("--raw-eeg-npy")
    parser.add_argument("--raw-labels-npy")
    parser.add_argument("--features-npy")
    parser.add_argument("--labels-npy")
    parser.add_argument("--subjects-npy")
    parser.add_argument("--trials-npy")
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--window-overlap", type=float, default=0.0)
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--median-label", type=float, default=3.0)
    parser.add_argument("--no-zscore", action="store_true")
    parser.add_argument("--n-channels", type=int, default=14)
    parser.add_argument("--n-bands", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-subjects", type=int, default=4)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--selection-level", choices=("window", "trial"), default="trial")
    parser.add_argument("--selection-metric", choices=("loss", "accuracy", "f1", "precision", "recall"), default="f1")
    parser.add_argument("--early-stopping-monitor", default="val_trial_f1")
    parser.add_argument("--early-stopping-mode", choices=("auto", "min", "max"), default="max")
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.005)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hyperparameters-json")
    parser.add_argument("--hyperparameters-file")
    parser.add_argument("--out-dir", default="runs/gcn_softmax")
    parser.add_argument("--run-name", default="gcn_softmax")
    parser.add_argument("--final-epochs", type=int)
    parser.add_argument("--no-final-model", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    feature_array, label_array, subject_id_array, trial_id_array = _load_arrays(args)
    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    lengths = {
        len(feature_array), len(label_array), len(subject_id_array), len(trial_id_array)
    }
    if len(lengths) != 1:
        raise ValueError("Features, labels, subject IDs, and trial IDs must align.")
    if feature_array.ndim != 3:
        raise ValueError(
            "Expected window tensors shaped (windows, timesteps, features); "
            f"got {feature_array.shape}."
        )
    expected_features = args.n_channels * args.n_bands
    if feature_array.shape[-1] != expected_features:
        raise ValueError(
            f"The data have {feature_array.shape[-1]} features per timestep, but "
            f"--n-channels {args.n_channels} * --n-bands {args.n_bands} = "
            f"{expected_features}."
        )

    n_classes = _infer_n_classes(label_array)
    hyperparameters = _load_hyperparameters(
        args.hyperparameters_json, args.hyperparameters_file
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "command.json", {"argv": os.sys.argv, "args": vars(args)})
    _write_json(run_dir / "hyperparameters.json", hyperparameters)
    _write_json(
        run_dir / "dataset_summary.json",
        {
            "feature_shape": feature_array.shape,
            "n_subjects": len(np.unique(subject_id_array)),
            "n_trials": len(set(zip(subject_id_array.tolist(), trial_id_array.tolist()))),
            "class_counts_window": {
                int(key): int(value)
                for key, value in zip(*np.unique(_as_class_ids(label_array), return_counts=True))
            },
        },
    )

    def model_builder_function(**hparams) -> tf.keras.Model:
        unknown = set(hparams) - MODEL_HPARAM_KEYS
        if unknown:
            raise ValueError(f"Unknown GCN-only hyperparameters: {sorted(unknown)}")
        return build_gcn_softmax_classifier(
            input_shape=tuple(feature_array.shape[1:]),
            n_classes=n_classes,
            n_channels=args.n_channels,
            n_bands=args.n_bands,
            **hparams,
        )

    model_builder_function._sequence_hyperparameter_depths = {
        "gcn_units": 1,
        "temporal_pool_sizes": 1,
    }

    cv_results = loso_cv(
        model_builder_function=model_builder_function,
        feature_array=feature_array,
        label_array=label_array,
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        hyperparameters=hyperparameters,
        evaluation_level="trial",
        selection_level=args.selection_level,
        selection_metric=args.selection_metric,
        maximize_metric=args.selection_metric not in {"loss"},
        metrics=("accuracy", "f1", "precision", "recall"),
        log_predictions=True,
        n_prediction_latent_samples=0,
        validation_subjects_per_fold=args.validation_subjects,
        validation_seed=args.validation_seed,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        restore_best_weights=True,
        verbose=args.verbose,
        extra_fit_kwargs={"callbacks": [tf.keras.callbacks.TerminateOnNaN()]},
        n_jobs=args.n_jobs,
        cpus_per_worker=args.cpus_per_worker,
        max_folds=args.max_folds,
    )

    _write_json(run_dir / "loso_cv_results.json", cv_results)
    outer_folds = cv_results.get("outer_fold_results", [])
    if isinstance(outer_folds, list):
        _write_records_csv(run_dir / "loso_cv_folds.csv", outer_folds)

    selected_config = _selected_config(cv_results, hyperparameters)
    _write_json(run_dir / "selected_config.json", selected_config)

    if not args.no_final_model:
        best_epochs = _collect_best_epochs(cv_results)
        if args.final_epochs is not None:
            final_epochs = max(1, int(args.final_epochs))
        elif best_epochs:
            final_epochs = max(1, int(round(float(np.median(best_epochs)))))
        else:
            final_epochs = int(selected_config.get("epochs", args.epochs))
        _write_json(
            run_dir / "final_training_plan.json",
            {"final_epochs": final_epochs, "cv_best_epochs": best_epochs},
        )
        _fit_final_model(
            selected_config=selected_config,
            features=feature_array,
            labels=label_array,
            n_classes=n_classes,
            n_channels=args.n_channels,
            n_bands=args.n_bands,
            epochs=final_epochs,
            output_dir=run_dir,
            verbose=args.verbose,
        )

    print(f"GCN-only run complete: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
