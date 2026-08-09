"""Training entry point for corrected joint_v5_sts.

Architecture:
    1-second channel-band waveform
        -> shared fixed-MI GCN
        -> spectral GRU across theta/alpha/beta
        -> classifier

Training is WINDOW LEVEL. Trial IDs are retained so cross_val can aggregate
window probabilities and select/report TRIAL-LEVEL metrics. There is no
trial-embedding average inside the neural model.

v5.0 excludes BiLSTM, VAE, decoder, fusion, temporal pooling,
subject-adversarial, SupCon, and MLDG machinery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import csv
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from itertools import product

try:
    from .joint_sts_cli import parse_args
    from .joint_sts_model import build_joint_sts_model
except ImportError:
    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from joint_sts_cli import parse_args
    from joint_sts_model import build_joint_sts_model

# Reuse v4's dataset construction/grouping ONLY. No v4 model code is used.
try:
    from ..joint_v4_sts.joint_sts_model_train import (
        load_joint_v4_training_data,
        _class_ids,
        _n_classes,
        _selected_epochs,
        _fold_rows,
    )
except ImportError:
    from eegproc.deep_learning.joint_architectures.joint_v4_sts.joint_sts_model_train import (
        load_joint_v4_training_data,
        _class_ids,
        _n_classes,
        _selected_epochs,
        _fold_rows,
    )

try:
    from ..joint_v2_data import get_dataset_config
except ImportError:
    from eegproc.deep_learning.joint_architectures.joint_v2_data import get_dataset_config

try:
    from ...cross_val import fixed_loso_cv, loso_cv
except ImportError:
    from eegproc.deep_learning.cross_val import fixed_loso_cv, loso_cv


@dataclass(slots=True)
class JointV5TrainingConfig:
    output_dir: Path = Path("runs/joint_v5_sts")
    run_name: str = "dreamer_valence_joint_v5_sts"
    dataset: str = "dreamer"
    n_channels: int = 14
    n_bands: int = 3
    # Neural model trains one prediction per 1-second window.
    classification_level: str = "window"
    # cross_val aggregates window probabilities by (subject, trial).
    evaluation_level: str = "trial"

    batch_size: int = 16
    cv_max_epochs: int = 100
    optimizer_name: str = "adamw"
    classification_learning_rate: float = 1e-4
    weight_decay: float = 5e-5

    # Corrected v5 uses the complete 1-second waveform as each GCN node's
    # feature vector, so there is no temporal pooling/downsampling.
    t_down: int = 1
    temporal_pool_sizes: tuple[int, ...] = ()
    gcn_units: tuple[int, ...] = (32,)
    gcn_dropout: float = 0.10
    gcn_activation: str = "relu"
    gcn_use_batch_norm: bool = False
    spectral_gru_units: int = 384
    spectral_gru_dropout: float = 0.0

    mi_n_neighbors: int = 3
    mi_random_state: int = 42
    mi_zero_diagonal: bool = False
    mi_band_reduction: str = "mean"
    mi_max_observations: int | None = 50000

    classification_hidden_units: int = 128
    classification_dropout: float = 0.30
    activation: str = "relu"
    focal_gamma: float = 0.0
    focal_alpha: tuple[float, ...] | None = None
    use_class_weight: bool = False

    selection_level: str = "trial"
    selection_metric: str = "accuracy"
    decision_thresholds: tuple[float, ...] = (0.5,)
    threshold_selection_level: str = "trial"
    threshold_selection_metric: str = "accuracy"

    validation_subjects_per_fold: int = 4
    validation_seed: int | None = 42
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = "val_accuracy"
    early_stopping_mode: str = "max"

    final_epochs: int | None = None
    final_epoch_strategy: str = "median"
    run_no_validation_loso_before_final: bool = True
    max_folds: int | None = None

    prediction_diagnostics: bool = True
    prediction_diagnostics_every_n_epochs: int = 1
    prediction_diagnostics_max_samples: int = 256
    prediction_diagnostics_threshold_tolerance: float = 0.01
    prediction_diagnostics_seed: int | None = 42

    n_jobs: int = 1
    cpus_per_worker: int | None = None
    outer_verbose: int = 0
    final_verbose: int = 2
    seed: int | None = 42

    save_full_model: bool = True
    save_weights: bool = True
    save_adjacency_matrices: bool = True
    label_threshold_mode: str = "global"
    window_normalization: str = "global_rms"
    hyperparameters: dict = field(default_factory=dict)

    # Reverse-engineering mode: intentionally select each LOSO fold's best
    # checkpoint using the held-out test subject itself. These numbers are
    # oracle/test-leaky and must never be presented as unbiased LOSO results.
    oracle_test_epoch_selection: bool = False
    oracle_metric: str = "accuracy"
    oracle_every_n_epochs: int = 1
    oracle_save_fold_weights: bool = False


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def _configure_logger(run_dir):
    logger = logging.getLogger(f"eegproc.joint_v5_sts.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _save_adjacencies(model, run_dir):
    nested = model.get_adjacency_matrices()
    flat = {}
    for key, value in nested.items():
        flat[key] = np.asarray(value.numpy() if tf.is_tensor(value) else value)
    if flat:
        np.savez_compressed(run_dir / "adjacency_matrices.npz", **flat)



def _probability_metrics(y_true, probabilities, threshold=0.5):
    """Classification metrics from class-probability rows."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) != len(y_true):
        raise ValueError(
            "probabilities must be (n_samples,n_classes) and align with y_true."
        )

    n_classes = probabilities.shape[1]
    if n_classes == 2:
        y_pred = (probabilities[:, 1] >= float(threshold)).astype(np.int64)
        binary_f1 = f1_score(y_true, y_pred, zero_division=0)
        binary_precision = precision_score(y_true, y_pred, zero_division=0)
        binary_recall = recall_score(y_true, y_pred, zero_division=0)
    else:
        y_pred = np.argmax(probabilities, axis=1).astype(np.int64)
        binary_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        binary_precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        binary_recall = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )

    clipped = np.clip(probabilities, 1e-7, 1.0)
    loss = float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))

    return {
        "loss": loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(binary_f1),
        "precision": float(binary_precision),
        "recall": float(binary_recall),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pred_class_1_fraction": (
            float(np.mean(y_pred == 1)) if n_classes == 2 else None
        ),
        "n_samples": int(len(y_true)),
    }


def _aggregate_probabilities_by_trial(y_window, trial_ids, probabilities):
    """Average window probabilities within each held-out trial."""
    y_window = np.asarray(y_window, dtype=np.int64).reshape(-1)
    trial_ids = np.asarray(trial_ids).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if not (len(y_window) == len(trial_ids) == len(probabilities)):
        raise ValueError("y_window, trial_ids, and probabilities must align.")

    unique_trials = np.unique(trial_ids)
    trial_y = []
    trial_probabilities = []

    for trial_id in unique_trials:
        mask = trial_ids == trial_id
        labels = np.unique(y_window[mask])
        if len(labels) != 1:
            raise ValueError(
                f"Trial {trial_id!r} contains multiple labels: {labels.tolist()}."
            )
        trial_y.append(int(labels[0]))
        trial_probabilities.append(np.mean(probabilities[mask], axis=0))

    return (
        np.asarray(trial_y, dtype=np.int64),
        np.asarray(trial_probabilities, dtype=np.float64),
        unique_trials,
    )


class OracleTestEpochSelector(tf.keras.callbacks.Callback):
    """Select a checkpoint by evaluating the HELD-OUT TEST SUBJECT each epoch.

    This callback is intentionally test-leaky. It exists only to investigate
    whether test-subject epoch selection can reproduce MTLFuseNet-scale scores.
    """

    def __init__(
        self,
        *,
        X_test,
        y_test,
        trial_ids_test,
        fold_number,
        batch_size,
        oracle_metric="accuracy",
        threshold=0.5,
        every_n_epochs=1,
    ):
        super().__init__()
        if every_n_epochs <= 0:
            raise ValueError("oracle_every_n_epochs must be positive.")
        self.X_test = np.asarray(X_test, dtype=np.float32)
        self.y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)
        self.trial_ids_test = np.asarray(trial_ids_test).reshape(-1)
        self.fold_number = int(fold_number)
        self.batch_size = int(batch_size)
        self.oracle_metric = str(oracle_metric)
        self.threshold = float(threshold)
        self.every_n_epochs = int(every_n_epochs)

        self.best_epoch = None
        self.best_score = -np.inf
        self.best_trial_loss = np.inf
        self.best_weights = None
        self.best_trial_metrics = None
        self.best_window_metrics = None
        self.epoch_rows = []

    def _predict_probabilities(self):
        logits = self.model.predict(
            self.X_test,
            batch_size=self.batch_size,
            verbose=0,
        )
        logits = np.asarray(logits)
        if logits.ndim != 2:
            raise ValueError(f"Expected rank-2 logits, got {logits.shape}.")
        return tf.nn.softmax(logits, axis=-1).numpy()

    def on_epoch_end(self, epoch, logs=None):
        epoch_number = int(epoch) + 1
        total_epochs = int(self.params.get("epochs", epoch_number))
        if (
            epoch_number % self.every_n_epochs != 0
            and epoch_number != total_epochs
        ):
            return

        probabilities = self._predict_probabilities()
        window_metrics = _probability_metrics(
            self.y_test,
            probabilities,
            threshold=self.threshold,
        )
        trial_y, trial_probabilities, _ = _aggregate_probabilities_by_trial(
            self.y_test,
            self.trial_ids_test,
            probabilities,
        )
        trial_metrics = _probability_metrics(
            trial_y,
            trial_probabilities,
            threshold=self.threshold,
        )

        score = float(trial_metrics[self.oracle_metric])
        trial_loss = float(trial_metrics["loss"])
        improved = (
            score > self.best_score + 1e-12
            or (
                abs(score - self.best_score) <= 1e-12
                and trial_loss < self.best_trial_loss - 1e-12
            )
        )

        row = {
            "fold": self.fold_number,
            "epoch": epoch_number,
            "oracle_metric": self.oracle_metric,
            "oracle_score": score,
            **{f"test_trial_{k}": v for k, v in trial_metrics.items()},
            **{f"test_window_{k}": v for k, v in window_metrics.items()},
        }
        self.epoch_rows.append(row)

        if logs is not None:
            logs[f"oracle_test_trial_{self.oracle_metric}"] = score
            logs["oracle_test_trial_accuracy"] = trial_metrics["accuracy"]
            logs["oracle_test_trial_balanced_accuracy"] = trial_metrics[
                "balanced_accuracy"
            ]

        marker = " BEST" if improved else ""
        pred1 = trial_metrics.get("pred_class_1_fraction")
        pred1_text = "" if pred1 is None else f" | pred1={pred1:.4f}"
        print(
            f"[ORACLE][Fold {self.fold_number}][Epoch {epoch_number}/{total_epochs}] "
            f"test_trial_accuracy={trial_metrics['accuracy']:.4f} | "
            f"balanced_accuracy={trial_metrics['balanced_accuracy']:.4f} | "
            f"{self.oracle_metric}={score:.4f}{pred1_text}{marker}",
            flush=True,
        )

        if improved:
            self.best_score = score
            self.best_trial_loss = trial_loss
            self.best_epoch = epoch_number
            self.best_weights = [np.array(w, copy=True) for w in self.model.get_weights()]
            self.best_trial_metrics = dict(trial_metrics)
            self.best_window_metrics = dict(window_metrics)

    def on_train_end(self, logs=None):
        del logs
        if self.best_weights is None:
            raise RuntimeError("Oracle callback never evaluated the test subject.")
        self.model.set_weights(self.best_weights)
        print(
            f"[ORACLE][Fold {self.fold_number}] restored TEST-SELECTED epoch "
            f"{self.best_epoch} ({self.oracle_metric}={self.best_score:.4f})",
            flush=True,
        )



def _expand_oracle_hyperparameters(hyperparameters):
    """Expand EEGProc-style JSON grids, including None and sequence-valued HPs."""
    if not hyperparameters:
        return [{}]

    keys = list(hyperparameters)
    value_lists = []
    sequence_value_keys = {"gcn_units", "focal_alpha"}

    for key in keys:
        value = hyperparameters[key]
        if value is None:
            values = [None]
        elif not isinstance(value, (list, tuple)):
            values = [value]
        elif key in sequence_value_keys and value and all(
            not isinstance(item, (list, tuple)) for item in value
        ):
            # e.g. gcn_units=[32] means one architecture value, not a
            # one-element search over scalar 32. Existing EEGProc grids often
            # use [[32]], which is also handled correctly below.
            values = [list(value)]
        else:
            values = list(value)

        if not values:
            raise ValueError(f"Hyperparameter {key!r} has an empty search list.")
        value_lists.append(values)

    return [dict(zip(keys, combo)) for combo in product(*value_lists)]

def _run_oracle_loso(
    *,
    X,
    y,
    subjects,
    trials,
    config,
    builder,
    run_dir,
    logger,
):
    """Strict LOSO training with intentionally test-leaky epoch selection."""
    if config.oracle_every_n_epochs <= 0:
        raise ValueError("oracle_every_n_epochs must be positive.")
    if config.oracle_metric not in {"accuracy", "balanced_accuracy", "f1", "macro_f1"}:
        raise ValueError(f"Unsupported oracle_metric={config.oracle_metric!r}.")

    y_ids = _class_ids(y)
    unique_subjects = np.sort(np.unique(subjects))
    if config.max_folds is not None:
        unique_subjects = unique_subjects[: int(config.max_folds)]

    configs = _expand_oracle_hyperparameters(config.hyperparameters)

    logger.warning("=" * 80)
    logger.warning("ORACLE TEST-SUBJECT EPOCH SELECTION IS ENABLED")
    logger.warning(
        "Each held-out LOSO subject is evaluated after every oracle interval and "
        "selects its own best checkpoint. Results are intentionally test-leaky."
    )
    logger.warning("There is NO validation set; all 22 non-test subjects train the fold.")
    logger.warning("These results must NOT be reported as unbiased LOSO performance.")
    logger.warning("=" * 80)

    print(
        "\nORACLE LOSO: held-out test subject chooses the best epoch "
        "(INTENTIONALLY LEAKY)\n",
        flush=True,
    )
    print(f"Configurations: {len(configs)}", flush=True)
    print(f"LOSO folds per configuration: {len(unique_subjects)}", flush=True)
    print(f"Oracle metric: test-trial {config.oracle_metric}", flush=True)
    print(f"Oracle evaluation interval: every {config.oracle_every_n_epochs} epoch(s)", flush=True)
    print("Validation subjects: 0", flush=True)

    all_config_results = []
    all_epoch_rows = []
    threshold = float(config.decision_thresholds[0])

    if config.n_jobs != 1:
        logger.warning(
            "Oracle v5 currently runs folds sequentially for deterministic "
            "test-epoch checkpoint restoration; ignoring n_jobs=%d.",
            config.n_jobs,
        )

    for config_index, hp in enumerate(configs, start=1):
        print("\n" + "#" * 80, flush=True)
        print(
            f"ORACLE configuration {config_index}/{len(configs)}: {hp}",
            flush=True,
        )
        fold_rows = []

        for fold_number, test_subject in enumerate(unique_subjects, start=1):
            test_mask = subjects == test_subject
            train_mask = ~test_mask

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test_ids = y_ids[test_mask]
            test_trial_ids = trials[test_mask]

            epochs = int(hp.get("epochs", config.cv_max_epochs))
            current_batch_size = int(hp.get("batch_size", config.batch_size))
            model_hp = {
                k: v for k, v in hp.items() if k not in {"epochs", "batch_size"}
            }

            print(
                f"\n[ORACLE Fold {fold_number}/{len(unique_subjects)}] "
                f"test subject={test_subject!r} | train_windows={len(X_train)} | "
                f"test_windows={len(X_test)} | validation=0",
                flush=True,
            )

            tf.keras.backend.clear_session()
            model = builder(**model_hp)

            oracle_callback = OracleTestEpochSelector(
                X_test=X_test,
                y_test=y_test_ids,
                trial_ids_test=test_trial_ids,
                fold_number=fold_number,
                batch_size=current_batch_size,
                oracle_metric=config.oracle_metric,
                threshold=threshold,
                every_n_epochs=config.oracle_every_n_epochs,
            )

            callbacks = [tf.keras.callbacks.TerminateOnNaN(), oracle_callback]

            class_weight = None
            if config.use_class_weight:
                train_ids = _class_ids(y_train)
                classes, counts = np.unique(train_ids, return_counts=True)
                class_weight = {
                    int(c): len(train_ids) / (len(classes) * count)
                    for c, count in zip(classes, counts)
                }

            fit_kwargs = dict(
                epochs=epochs,
                batch_size=current_batch_size,
                verbose=0,
                callbacks=callbacks,
                shuffle=True,
            )
            if class_weight is not None:
                fit_kwargs["class_weight"] = class_weight

            history = model.fit(X_train, y_train, **fit_kwargs)

            if oracle_callback.best_trial_metrics is None:
                raise RuntimeError("Oracle fold completed without a selected epoch.")

            fold_row = {
                "config_index": config_index,
                "fold": fold_number,
                "test_subject": (
                    test_subject.item()
                    if isinstance(test_subject, np.generic)
                    else test_subject
                ),
                "n_train_windows": int(len(X_train)),
                "n_test_windows": int(len(X_test)),
                "n_validation_windows": 0,
                "epochs_requested": epochs,
                "epochs_ran": int(len(history.history.get("loss", []))),
                "oracle_best_epoch": int(oracle_callback.best_epoch),
                "oracle_metric": config.oracle_metric,
                "oracle_score": float(oracle_callback.best_score),
                "decision_threshold": threshold,
                **{
                    f"trial_{k}": v
                    for k, v in oracle_callback.best_trial_metrics.items()
                },
                **{
                    f"window_{k}": v
                    for k, v in oracle_callback.best_window_metrics.items()
                },
            }
            fold_rows.append(fold_row)
            all_epoch_rows.extend(
                {"config_index": config_index, **row}
                for row in oracle_callback.epoch_rows
            )

            print(
                f"[ORACLE Fold {fold_number}] BEST epoch={oracle_callback.best_epoch} | "
                f"trial_accuracy={fold_row['trial_accuracy']:.4f} | "
                f"trial_balanced_accuracy={fold_row['trial_balanced_accuracy']:.4f}",
                flush=True,
            )

            if config.oracle_save_fold_weights:
                fold_dir = _ensure_dir(run_dir / "oracle_fold_weights")
                model.save_weights(
                    fold_dir
                    / f"config_{config_index:02d}_fold_{fold_number:02d}_"
                    f"subject_{test_subject}.weights.h5"
                )

            if config.save_adjacency_matrices:
                adjacency_dir = _ensure_dir(run_dir / "oracle_fold_adjacencies")
                nested = model.get_adjacency_matrices()
                if nested:
                    np.savez_compressed(
                        adjacency_dir
                        / f"config_{config_index:02d}_fold_{fold_number:02d}_"
                        f"subject_{test_subject}.npz",
                        **{
                            k: np.asarray(v.numpy() if tf.is_tensor(v) else v)
                            for k, v in nested.items()
                        },
                    )

            del model
            tf.keras.backend.clear_session()

        scores = np.asarray(
            [row["oracle_score"] for row in fold_rows], dtype=np.float64
        )
        config_result = {
            "config_index": config_index,
            "config": hp,
            "oracle_metric": config.oracle_metric,
            "mean_oracle_score": float(np.mean(scores)),
            "std_oracle_score": float(np.std(scores)),
            "folds": fold_rows,
        }
        all_config_results.append(config_result)
        print(
            f"\nORACLE configuration {config_index} complete: "
            f"mean {config.oracle_metric}={np.mean(scores):.6f} ± {np.std(scores):.6f}",
            flush=True,
        )

    best_config_result = max(
        all_config_results,
        key=lambda result: result["mean_oracle_score"],
    )

    result = {
        "protocol": "ORACLE_TEST_SUBJECT_EPOCH_SELECTION_INTENTIONALLY_LEAKY",
        "warning": (
            "Held-out LOSO test subjects were evaluated during training and used "
            "to select checkpoints. These scores are not unbiased generalization estimates."
        ),
        "validation_subjects_per_fold": 0,
        "oracle_metric": config.oracle_metric,
        "oracle_every_n_epochs": config.oracle_every_n_epochs,
        "decision_threshold": threshold,
        "n_configurations": len(all_config_results),
        "n_folds": len(unique_subjects),
        "best_config_index": best_config_result["config_index"],
        "best_config": best_config_result["config"],
        "best_mean_oracle_score": best_config_result["mean_oracle_score"],
        "best_std_oracle_score": best_config_result["std_oracle_score"],
        "selected_folds": best_config_result["folds"],
        "config_results": all_config_results,
    }

    _write_json(run_dir / "oracle_loso_results.json", result)
    _write_csv(run_dir / "oracle_loso_folds.csv", best_config_result["folds"])
    _write_csv(run_dir / "oracle_epoch_history.csv", all_epoch_rows)
    _write_json(
        run_dir / "training_summary.json",
        {
            "run_dir": str(run_dir),
            "architecture": "mtl_fixed_mi_gcn_then_spectral_gru_then_classifier",
            "protocol": result["protocol"],
            "warning": result["warning"],
            "best_config": result["best_config"],
            "best_mean_oracle_score": result["best_mean_oracle_score"],
            "best_std_oracle_score": result["best_std_oracle_score"],
        },
    )

    logger.warning(
        "ORACLE run complete: test-trial %s=%.6f ± %.6f. TEST-LEAKY RESULT.",
        config.oracle_metric,
        result["best_mean_oracle_score"],
        result["best_std_oracle_score"],
    )
    return result

def train_joint_v5_sts(X, y, subjects, trials, config: JointV5TrainingConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(config.output_dir / f"{config.run_name}_{timestamp}")
    logger = _configure_logger(run_dir)

    if config.seed is not None:
        tf.keras.utils.set_random_seed(config.seed)
        np.random.seed(config.seed)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    subjects = np.asarray(subjects).reshape(-1)
    trials = np.asarray(trials).reshape(-1)
    if config.classification_level != "window":
        raise ValueError(
            "Corrected v5 trains at window level only. "
            "Use trial-level aggregation in cross_val for reporting."
        )
    if config.evaluation_level != "trial":
        raise ValueError("Corrected v5 baseline expects trial-level evaluation.")
    if config.t_down != 1 or tuple(config.temporal_pool_sizes):
        raise ValueError(
            "Corrected v5 requires t_down=1 and temporal_pool_sizes=()."
        )
    if X.ndim != 3:
        raise ValueError(
            "Corrected v5 expects window samples shaped (N,T,F); "
            f"got {X.shape}."
        )
    if X.shape[-1] != config.n_channels * config.n_bands:
        raise ValueError(
            f"features={X.shape[-1]} but n_channels*n_bands="
            f"{config.n_channels * config.n_bands}."
        )

    # Remove architecture keys from legacy v5 hyperparameter JSONs. They are
    # fixed by the corrected representation and should not appear as fake
    # selected hyperparameters in CV output.
    sanitized_hparams = dict(config.hyperparameters)
    for fixed_key in ("classification_level", "t_down", "temporal_pool_sizes"):
        if fixed_key in sanitized_hparams:
            logger.info(
                "Ignoring legacy hyperparameter %s=%r; corrected v5 fixes "
                "classification_level='window', t_down=1, temporal_pool_sizes=().",
                fixed_key,
                sanitized_hparams[fixed_key],
            )
            sanitized_hparams.pop(fixed_key)
    config.hyperparameters = sanitized_hparams

    _write_json(run_dir / "training_config.json", asdict(config))
    logger.info("Architecture: full-window shared fixed-MI GCN -> spectral GRU -> classifier")
    logger.info(
        "Per-window tensor interpretation: (T,C*Bands) -> "
        "(Bands,C,T) -> GCN -> band sequence -> GRU"
    )
    logger.info(
        "GCN output per band: channels x %d; GRU output: %d-D direct latent",
        config.gcn_units[-1],
        config.spectral_gru_units,
    )
    logger.info("Training level: window")
    logger.info("Primary evaluation/selection level: trial")
    logger.info("Input tensor: %s", X.shape)
    logger.info("No temporal pooling/downsampling: t_down=1")
    logger.info("MI adjacency is adapted inside EACH model.fit from fold-training data only.")
    logger.info("BiLSTM/VAE/fusion/adversarial/SupCon/MLDG: all disabled in v5.0")

    allowed = {
        "learning_rate", "classification_learning_rate", "optimizer", "optimizer_name",
        "weight_decay", "t_down", "temporal_pool_sizes", "gcn_units",
        "gcn_dropout", "gcn_activation", "gcn_use_batch_norm",
        "spectral_gru_units", "spectral_gru_dropout", "mi_n_neighbors",
        "mi_random_state", "mi_zero_diagonal", "mi_band_reduction",
        "mi_max_observations", "classification_hidden_units",
        "classification_dropout", "activation", "focal_gamma", "focal_alpha",
        "use_class_weight", "classification_level",
    }

    def builder(**h):
        unknown = set(h) - allowed - {"epochs", "batch_size"}
        if unknown:
            raise ValueError(f"Unknown v5 hyperparameters: {sorted(unknown)}")
        return build_joint_sts_model(
            input_shape=tuple(X.shape[1:]),
            classification_level="window",
            n_classes=_n_classes(y),
            n_channels=config.n_channels,
            n_bands=config.n_bands,
            t_down=1,
            temporal_pool_sizes=(),
            gcn_units=h.get("gcn_units", config.gcn_units),
            gcn_dropout=float(h.get("gcn_dropout", config.gcn_dropout)),
            gcn_activation=str(h.get("gcn_activation", config.gcn_activation)),
            gcn_use_batch_norm=bool(h.get("gcn_use_batch_norm", config.gcn_use_batch_norm)),
            spectral_gru_units=int(h.get("spectral_gru_units", config.spectral_gru_units)),
            spectral_gru_dropout=float(h.get("spectral_gru_dropout", config.spectral_gru_dropout)),
            mi_n_neighbors=int(h.get("mi_n_neighbors", config.mi_n_neighbors)),
            mi_random_state=int(h.get("mi_random_state", config.mi_random_state)),
            mi_zero_diagonal=bool(h.get("mi_zero_diagonal", config.mi_zero_diagonal)),
            mi_band_reduction=str(h.get("mi_band_reduction", config.mi_band_reduction)),
            mi_max_observations=h.get("mi_max_observations", config.mi_max_observations),
            classification_hidden_units=int(h.get("classification_hidden_units", config.classification_hidden_units)),
            classification_dropout=float(h.get("classification_dropout", config.classification_dropout)),
            activation=str(h.get("activation", config.activation)),
            focal_gamma=float(h.get("focal_gamma", config.focal_gamma)),
            focal_alpha=h.get("focal_alpha", config.focal_alpha),
            use_class_weight=bool(h.get("use_class_weight", config.use_class_weight)),
            optimizer_name=str(h.get("optimizer", h.get("optimizer_name", config.optimizer_name))),
            classification_learning_rate=float(h.get("learning_rate", h.get("classification_learning_rate", config.classification_learning_rate))),
            weight_decay=float(h.get("weight_decay", config.weight_decay)),
        )

    builder._sequence_hyperparameter_depths = {
        "gcn_units": 1,
        "focal_alpha": 1,
    }

    if config.oracle_test_epoch_selection:
        return _run_oracle_loso(
            X=X,
            y=y,
            subjects=subjects,
            trials=trials,
            config=config,
            builder=builder,
            run_dir=run_dir,
            logger=logger,
        )

    cv = loso_cv(
        model_builder_function=builder,
        feature_array=X,
        label_array=y,
        subject_id_array=subjects,
        trial_id_array=trials,
        n_epochs=config.cv_max_epochs,
        batch_size=config.batch_size,
        hyperparameters=config.hyperparameters,
        evaluation_level=config.evaluation_level,
        selection_level=config.selection_level,
        selection_metric=config.selection_metric,
        metrics=("accuracy","f1","precision","recall","macro_f1","macro_precision","macro_recall","balanced_accuracy"),
        log_predictions=True,
        n_prediction_latent_samples=0,
        latent_sampling_seed=None,
        decision_thresholds=config.decision_thresholds,
        threshold_selection_metric=config.threshold_selection_metric,
        threshold_selection_level=config.threshold_selection_level,
        prediction_diagnostics=config.prediction_diagnostics,
        prediction_diagnostics_every_n_epochs=config.prediction_diagnostics_every_n_epochs,
        prediction_diagnostics_max_samples=config.prediction_diagnostics_max_samples,
        prediction_diagnostics_threshold_tolerance=config.prediction_diagnostics_threshold_tolerance,
        prediction_diagnostics_seed=config.prediction_diagnostics_seed,
        validation_subjects_per_fold=(config.validation_subjects_per_fold if config.use_early_stopping else 0),
        validation_seed=config.validation_seed,
        early_stopping_patience=(config.early_stopping_patience if config.use_early_stopping else None),
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_monitor=config.early_stopping_monitor,
        early_stopping_mode=config.early_stopping_mode,
        restore_best_weights=True,
        verbose=config.outer_verbose,
        extra_fit_kwargs={"callbacks": [tf.keras.callbacks.TerminateOnNaN()]},
        n_jobs=config.n_jobs,
        cpus_per_worker=config.cpus_per_worker,
        max_folds=config.max_folds,
    )

    _write_json(run_dir / "loso_cv_results.json", cv)
    _write_csv(run_dir / "loso_cv_folds.csv", _fold_rows(cv))

    selected = dict(cv["best_config"])
    selected_cap = int(selected.get("epochs", config.cv_max_epochs))
    if config.final_epochs is not None:
        final_epochs, best_epochs = int(config.final_epochs), []
    elif config.use_early_stopping:
        final_epochs, best_epochs = _selected_epochs(
            cv, config.final_epoch_strategy, selected_cap
        )
    else:
        final_epochs, best_epochs = selected_cap, []

    final_batch_size = int(selected.get("batch_size", config.batch_size))
    final_h = {k: v for k, v in selected.items() if k not in {"epochs", "batch_size"}}
    thresholds = [
        float(row["decision_threshold"])
        for row in _fold_rows(cv)
        if row.get("decision_threshold") is not None
    ]
    final_threshold = float(
        np.median(thresholds) if thresholds else config.decision_thresholds[0]
    )

    if config.run_no_validation_loso_before_final:
        fixed = fixed_loso_cv(
            model_builder_function=builder,
            feature_array=X,
            label_array=y,
            subject_id_array=subjects,
            trial_id_array=trials,
            fixed_config=final_h,
            n_epochs=final_epochs,
            batch_size=final_batch_size,
            evaluation_level=config.evaluation_level,
            selection_level=config.selection_level,
            selection_metric="balanced_accuracy",
            maximize_metric=True,
            metrics=("accuracy","f1","precision","recall","macro_f1","macro_precision","macro_recall","balanced_accuracy"),
            log_predictions=True,
            n_prediction_latent_samples=0,
            latent_sampling_seed=None,
            decision_threshold=final_threshold,
            prediction_diagnostics=config.prediction_diagnostics,
            prediction_diagnostics_every_n_epochs=config.prediction_diagnostics_every_n_epochs,
            prediction_diagnostics_max_samples=config.prediction_diagnostics_max_samples,
            prediction_diagnostics_threshold_tolerance=config.prediction_diagnostics_threshold_tolerance,
            prediction_diagnostics_seed=config.prediction_diagnostics_seed,
            verbose=config.outer_verbose,
            extra_fit_kwargs={"callbacks": [tf.keras.callbacks.TerminateOnNaN()]},
            n_jobs=config.n_jobs,
            cpus_per_worker=config.cpus_per_worker,
            max_folds=config.max_folds,
        )
        _write_json(run_dir / "no_validation_loso_results.json", fixed)

    model = builder(**final_h)
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(str(run_dir / "final_training_history.csv")),
    ]
    class_weight = None
    if config.use_class_weight:
        classes, counts = np.unique(_class_ids(y), return_counts=True)
        class_weight = {
            int(c): len(y) / (len(classes) * count)
            for c, count in zip(classes, counts)
        }
    fit_kwargs = dict(
        epochs=final_epochs,
        batch_size=final_batch_size,
        verbose=config.final_verbose,
        callbacks=callbacks,
    )
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight
    history = model.fit(X, y, **fit_kwargs)
    final_eval = model.evaluate(X, y, verbose=0, return_dict=True)

    if config.save_weights:
        model.save_weights(run_dir / "final_model.weights.h5")
    if config.save_full_model:
        model.save(run_dir / "final_model.keras")
    if config.save_adjacency_matrices:
        _save_adjacencies(model, run_dir)

    summary = {
        "run_dir": str(run_dir),
        "architecture": "mtl_fixed_mi_gcn_then_spectral_gru_then_classifier",
        "classification_level": "window",
        "evaluation_level": config.evaluation_level,
        "selected_final_config": selected,
        "selected_final_epochs": final_epochs,
        "selected_final_batch_size": final_batch_size,
        "cv_best_epochs": best_epochs,
        "final_decision_threshold": final_threshold,
        "use_class_weight": config.use_class_weight,
        "final_class_weight": class_weight,
        "final_fit_history": history.history,
        "final_full_dataset_metrics": final_eval,
    }
    _write_json(run_dir / "training_summary.json", summary)
    logger.info("Saved v5 artifacts to %s", run_dir)
    return summary


def main(argv=None):
    args = parse_args(argv)
    hparams = json.loads(args.hyperparameters_json) if args.hyperparameters_json else {}

    config = JointV5TrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        dataset=args.dataset,
        n_channels=args.n_channels,
        n_bands=args.n_bands,
        classification_level="window",
        evaluation_level="trial",
        batch_size=args.batch_size,
        cv_max_epochs=args.epochs,
        optimizer_name=args.optimizer,
        classification_learning_rate=args.classification_learning_rate,
        weight_decay=args.weight_decay,
        t_down=1,
        temporal_pool_sizes=(),
        gcn_units=tuple(args.gcn_units),
        gcn_dropout=args.gcn_dropout,
        gcn_activation=args.gcn_activation,
        gcn_use_batch_norm=args.gcn_use_batch_norm,
        spectral_gru_units=args.spectral_gru_units,
        spectral_gru_dropout=args.spectral_gru_dropout,
        mi_n_neighbors=args.mi_n_neighbors,
        mi_random_state=args.mi_random_state,
        mi_zero_diagonal=args.mi_zero_diagonal,
        mi_band_reduction=args.mi_band_reduction,
        mi_max_observations=args.mi_max_observations,
        classification_hidden_units=args.classification_hidden_units,
        classification_dropout=args.classification_dropout,
        activation=args.activation,
        focal_gamma=args.focal_gamma,
        focal_alpha=None if args.focal_alpha is None else tuple(args.focal_alpha),
        use_class_weight=args.use_class_weight,
        selection_level=args.selection_level,
        selection_metric=args.selection_metric,
        decision_thresholds=tuple(sorted(map(float, args.decision_thresholds))),
        threshold_selection_level=args.threshold_selection_level,
        threshold_selection_metric=args.threshold_selection_metric,
        validation_subjects_per_fold=(
            0 if args.oracle_test_epoch_selection else args.validation_subjects
        ),
        validation_seed=(
            args.validation_seed if args.validation_seed is not None else args.seed
        ),
        use_early_stopping=(
            False if args.oracle_test_epoch_selection else not args.no_early_stopping
        ),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        final_epochs=args.final_epochs,
        final_epoch_strategy=args.final_epoch_strategy,
        run_no_validation_loso_before_final=(
            False
            if args.oracle_test_epoch_selection
            else not args.skip_no_validation_loso_before_final
        ),
        max_folds=args.max_folds,
        prediction_diagnostics=not args.no_prediction_diagnostics,
        prediction_diagnostics_every_n_epochs=args.prediction_diagnostics_every,
        prediction_diagnostics_max_samples=args.prediction_diagnostics_samples,
        prediction_diagnostics_threshold_tolerance=args.prediction_threshold_tolerance,
        prediction_diagnostics_seed=args.prediction_diagnostics_seed,
        n_jobs=args.n_jobs,
        cpus_per_worker=args.cpus_per_worker,
        outer_verbose=args.outer_verbose,
        final_verbose=args.final_verbose,
        seed=args.seed,
        save_full_model=not args.no_save_full_model,
        save_weights=not args.no_save_weights,
        save_adjacency_matrices=not args.no_save_adjacency_matrices,
        label_threshold_mode=args.label_threshold_mode,
        window_normalization=args.window_normalization,
        hyperparameters=hparams,
        oracle_test_epoch_selection=args.oracle_test_epoch_selection,
        oracle_metric=args.oracle_metric,
        oracle_every_n_epochs=args.oracle_every,
        oracle_save_fold_weights=args.oracle_save_fold_weights,
    )

    if args.oracle_test_epoch_selection:
        print(
            "\n*** ORACLE MODE ENABLED: validation set removed; the held-out "
            "test subject selects the best epoch. Results are intentionally "
            "test-leaky. ***\n",
            flush=True,
        )

    if not np.isclose(float(args.window_sec), 1.0):
        raise ValueError(
            "Corrected joint_v5_sts uses the paper-style 1-second segment as "
            "the GCN node-feature waveform. Set --window-sec 1.0."
        )
    if not np.isclose(float(args.window_overlap), 0.0):
        raise ValueError(
            "The v5.0 baseline uses non-overlapping 1-second segments. "
            "Set --window-overlap 0.0."
        )

    if args.classification_level != "window":
        print(
            "joint_v5_sts: overriding --classification-level "
            f"{args.classification_level!r}; v5.0 trains windows and reports "
            "trial-level metrics via cross_val.",
            flush=True,
        )
    if int(args.t_down) != 1 or tuple(args.temporal_pool_sizes):
        print(
            "joint_v5_sts: ignoring temporal pooling CLI values; corrected "
            "v5.0 fixes t_down=1 and temporal_pool_sizes=().",
            flush=True,
        )

    dataset_config = get_dataset_config(args.dataset)
    eeg_path = args.raw_eeg_npy or dataset_config.eeg_path
    labels_path = args.raw_labels_npy or dataset_config.labels_path

    X, y, subjects, trials = load_joint_v4_training_data(
        eeg_path=eeg_path,
        labels_path=labels_path,
        label_dimension=args.label_dimension,
        window_size_sec=1.0,
        fs=args.fs,
        overlap=0.0,
        median_label=args.median_label,
        window_normalization=args.window_normalization,
        label_threshold_mode=args.label_threshold_mode,
        dataset=dataset_config,
    )

    expected_samples = int(round(float(args.fs)))
    if X.ndim != 3 or X.shape[1] != expected_samples:
        raise ValueError(
            "Expected one-second window tensor (N, fs, features); "
            f"with fs={args.fs:g}, expected T={expected_samples}, got {X.shape}."
        )

    # IMPORTANT: do NOT group windows into rank-4 trial tensors. cross_val keeps
    # trial IDs and averages window probabilities only for trial-level metrics.
    train_joint_v5_sts(X, y, subjects, trials, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
