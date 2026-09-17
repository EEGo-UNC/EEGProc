"""EEGProc training entry point for MTLFuseNet on cached DREAMER trials.

The original replication stores each training window as three model inputs:

    X_ST : (9, 9, 128)      spatio-temporal grid
    DE   : (3, 14)          differential-entropy features
    adj  : (3, 14, 14)      per-band graph adjacency

EEGProc's shared cross-validation code expects one NumPy feature tensor that it
can index by fold. This module therefore packs those three arrays into one
rank-3 tensor shaped ``(windows, 1, packed_features)``. ``EEGProcMTLFuseNet``
unpacks the tensor inside the model, preserving the original architecture while
allowing the existing LOSO, trial aggregation, thresholding, diagnostics,
parallel workers, and result schemas to be reused.

The adapter also makes inference deterministic by using ``z_mean`` when
``n_prediction_latent_samples=0``. Positive latent-sample counts use Monte Carlo
sampling through ``predict_mc_probabilities``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import argparse
import csv
import glob
import json
import logging
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from .losses import focal_loss, triplet_center_loss
    from .mtl_model import MTLFuseNet
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from losses import focal_loss, triplet_center_loss
    from mtl_model import MTLFuseNet

try:
    from ...cross_val import PredictionDiagnostics, loso_cv
except ImportError:
    resolved = Path(__file__).resolve()
    src_root = next(
        (parent for parent in resolved.parents if parent.name == "src"),
        resolved.parent,
    )
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from eegproc.deep_learning.cross_val import PredictionDiagnostics, loso_cv


X_ST_SHAPE = (9, 9, 128)
DE_SHAPE = (3, 14)
ADJ_SHAPE = (3, 14, 14)
X_ST_SIZE = int(np.prod(X_ST_SHAPE))
DE_SIZE = int(np.prod(DE_SHAPE))
ADJ_SIZE = int(np.prod(ADJ_SHAPE))
PACKED_SIZE = X_ST_SIZE + DE_SIZE + ADJ_SIZE


@dataclass(slots=True)
class MTLFuseNetTrainingConfig:
    """Complete run, model, and LOSO configuration for MTLFuseNet."""

    output_dir: Path = Path("runs") / "mtlfusenet"
    run_name: str = "mtlfusenet"
    processed_dir: Path = Path("processed_trials")
    task: str = "valence"

    # Fit and optimizer.
    optimizer_name: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    cv_max_epochs: int = 50
    final_epochs: int | None = None
    final_epoch_strategy: str = "median"
    seed: int | None = 42

    # LOSO and reporting.
    selection_metric: str = "accuracy"
    selection_level: str = "trial"
    maximize_metric: bool | None = None
    max_folds: int | None = None
    n_jobs: int = 1
    cpus_per_worker: int | None = None
    outer_verbose: int = 0
    final_verbose: int = 2

    # Validation and checkpoint selection.
    validation_subjects_per_fold: int = 2
    validation_seed: int | None = 42
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = "val_accuracy"
    early_stopping_mode: str = "max"

    # Inference and diagnostics.
    prediction_latent_samples: int = 0
    latent_sampling_seed: int | None = 42
    prediction_batch_size: int = 128
    decision_thresholds: tuple[float, ...] = (0.5,)
    threshold_selection_metric: str = "accuracy"
    threshold_selection_level: str = "trial"
    prediction_diagnostics: bool = True
    prediction_diagnostics_every_n_epochs: int = 1
    prediction_diagnostics_max_samples: int = 256
    prediction_diagnostics_threshold_tolerance: float = 0.01
    prediction_diagnostics_seed: int | None = 42

    # Paper architecture defaults for DREAMER.
    num_classes: int = 2
    vae_latent: int = 128
    gcn_dim: int = 32
    gru_units: int = 384
    beta1: float = 0.7
    beta2: float = 0.2
    beta3: float = 0.1
    focal_alpha: float = 0.7
    focal_gamma: float = 2.0
    tc_margin: float = 1.0
    dropout: float = 0.2

    # Artifacts.
    save_weights: bool = True
    save_full_model: bool = False
    save_final_history_csv: bool = True

    # Flat EEGProc hyperparameter grid.
    hyperparameters: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cached-data loading and reversible input packing
# ---------------------------------------------------------------------------


def _validate_trial_sample(sample: dict[str, Any], path: Path, task: str) -> int:
    required = {
        "X_ST",
        "DE",
        "adj",
        task,
        "subject_id",
        "trial_id",
        "num_win",
    }
    missing = required - set(sample)
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")

    x_st = np.asarray(sample["X_ST"])
    de = np.asarray(sample["DE"])
    adj = np.asarray(sample["adj"])
    n = int(sample["num_win"])

    if x_st.shape != (n, *X_ST_SHAPE):
        raise ValueError(
            f"{path}: X_ST must have shape {(n, *X_ST_SHAPE)}, got {x_st.shape}."
        )
    if de.shape != (n, *DE_SHAPE):
        raise ValueError(
            f"{path}: DE must have shape {(n, *DE_SHAPE)}, got {de.shape}."
        )
    if adj.shape != ADJ_SHAPE:
        raise ValueError(
            f"{path}: adj must have shape {ADJ_SHAPE}, got {adj.shape}."
        )
    if n < 1:
        raise ValueError(f"{path}: num_win must be positive.")
    if int(sample[task]) not in (0, 1):
        raise ValueError(f"{path}: {task} label must be 0 or 1.")
    if not np.isfinite(x_st).all():
        raise ValueError(f"{path}: X_ST contains NaN or Inf.")
    if not np.isfinite(de).all():
        raise ValueError(f"{path}: DE contains NaN or Inf.")
    if not np.isfinite(adj).all():
        raise ValueError(f"{path}: adj contains NaN or Inf.")
    return n


def _pack_trial_windows(sample: dict[str, Any], output: np.ndarray, start: int) -> int:
    """Pack one cached trial directly into a preallocated feature tensor."""
    n = int(sample["num_win"])
    stop = start + n
    flat = output[start:stop, 0, :]

    flat[:, :X_ST_SIZE] = np.asarray(sample["X_ST"], dtype=np.float32).reshape(
        n, X_ST_SIZE
    )
    de_start = X_ST_SIZE
    adj_start = de_start + DE_SIZE
    flat[:, de_start:adj_start] = np.asarray(
        sample["DE"], dtype=np.float32
    ).reshape(n, DE_SIZE)
    flat[:, adj_start:] = np.broadcast_to(
        np.asarray(sample["adj"], dtype=np.float32),
        (n, *ADJ_SHAPE),
    ).reshape(n, ADJ_SIZE)
    return stop


def load_mtlfusenet_training_data(
    processed_dir: str | Path,
    task: str = "valence",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached trials and return EEGProc-aligned flat window arrays."""
    if task not in {"valence", "arousal"}:
        raise ValueError("task must be valence or arousal.")

    processed_dir = Path(processed_dir)
    paths = sorted(
        Path(path)
        for path in glob.glob(str(processed_dir / "subj*_trial*.pkl"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No subj*_trial*.pkl files were found in {processed_dir}. "
            "Run mtl_preprocess.py first."
        )

    metadata: list[tuple[Path, int, int, int, int]] = []
    total_windows = 0
    observed_pairs: set[tuple[int, int]] = set()

    # First pass validates files and obtains the exact allocation size.
    for path in paths:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        n = _validate_trial_sample(sample, path, task)
        subject_id = int(sample["subject_id"])
        trial_id = int(sample["trial_id"])
        pair = (subject_id, trial_id)
        if pair in observed_pairs:
            raise ValueError(f"Duplicate cached subject/trial pair: {pair}.")
        observed_pairs.add(pair)
        metadata.append((path, n, int(sample[task]), subject_id, trial_id))
        total_windows += n

    feature_array = np.empty((total_windows, 1, PACKED_SIZE), dtype=np.float32)
    label_array = np.empty(total_windows, dtype=np.int64)
    subject_id_array = np.empty(total_windows, dtype=np.int64)
    trial_id_array = np.empty(total_windows, dtype=np.int64)

    cursor = 0
    for path, n, label, subject_id, trial_id in metadata:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        stop = _pack_trial_windows(sample, feature_array, cursor)
        label_array[cursor:stop] = label
        subject_id_array[cursor:stop] = subject_id
        trial_id_array[cursor:stop] = trial_id
        cursor = stop

    if cursor != total_windows:
        raise RuntimeError(
            f"Packed {cursor} windows, but allocation expected {total_windows}."
        )

    return feature_array, label_array, subject_id_array, trial_id_array


def unpack_mtlfusenet_inputs(
    packed_inputs: tf.Tensor | np.ndarray,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Unpack ``(B, 1, PACKED_SIZE)`` or ``(B, PACKED_SIZE)`` inputs."""
    packed = tf.convert_to_tensor(packed_inputs, dtype=tf.float32)
    if packed.shape.rank == 3:
        if packed.shape[1] not in (1, None):
            raise ValueError(
                "Packed MTLFuseNet input rank 3 must have singleton axis 1; "
                f"got {packed.shape}."
            )
        packed = packed[:, 0, :]
    elif packed.shape.rank != 2:
        raise ValueError(
            "Packed MTLFuseNet inputs must have rank 2 or 3; "
            f"got {packed.shape}."
        )

    tf.debugging.assert_equal(
        tf.shape(packed)[-1],
        PACKED_SIZE,
        message="Incorrect packed MTLFuseNet feature dimension.",
    )
    x_st = tf.reshape(packed[:, :X_ST_SIZE], (-1, *X_ST_SHAPE))
    de_start = X_ST_SIZE
    adj_start = de_start + DE_SIZE
    de = tf.reshape(packed[:, de_start:adj_start], (-1, *DE_SHAPE))
    adj = tf.reshape(packed[:, adj_start:], (-1, *ADJ_SHAPE))
    return x_st, de, adj


# ---------------------------------------------------------------------------
# Model adapter for EEGProc cross_val.py
# ---------------------------------------------------------------------------


class EEGProcMTLFuseNet(MTLFuseNet):
    """MTLFuseNet adapter accepting EEGProc's single packed feature tensor."""

    def __init__(self, prediction_batch_size: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.prediction_batch_size = int(prediction_batch_size)
        if self.prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive.")

        # EEGProc training scripts conventionally monitor val_accuracy.
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )

    def _forward(
        self,
        packed_inputs,
        *,
        training: bool = False,
        sample_latent: bool | None = None,
    ) -> dict[str, tf.Tensor]:
        x_st, de, adj = unpack_mtlfusenet_inputs(packed_inputs)

        z_mean, z_log_var = self.encoder(x_st, training=training)
        if sample_latent is None:
            sample_latent = bool(training)
        z = self.sampling([z_mean, z_log_var]) if sample_latent else z_mean
        recon = self.decoder(z, training=training)

        node = de[..., tf.newaxis]
        agg = tf.einsum("bkij,bkjf->bkif", adj, node)
        h = tf.tensordot(agg, self.gcn_W, axes=[[3], [0]]) + self.gcn_b
        h = tf.nn.relu(h)
        batch = tf.shape(h)[0]
        sequence = tf.reshape(h, (batch, 3, 14 * self.gcn_dim))
        z_ss = self.gru(sequence, training=training)

        z_sst = tf.concat([z, z_ss], axis=-1)
        probabilities = self.classifier(
            self.dropout(z_sst, training=training)
        )
        return {
            "probabilities": probabilities,
            "y_pred": probabilities,
            "Z_SST": z_sst,
            "classification_latent": z_sst,
            "recon": recon,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
        }

    def call(
        self,
        inputs,
        training: bool = False,
        sample_latent: bool | None = None,
        include_reconstruction: bool = True,
    ):
        del include_reconstruction  # retained for cross_val diagnostic compatibility
        return self._forward(
            inputs,
            training=training,
            sample_latent=sample_latent,
        )

    def compute_losses(self, inputs, labels, out):
        x_st, _de, _adj = unpack_mtlfusenet_inputs(inputs)
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        focal = tf.reduce_mean(
            focal_loss(
                labels,
                out["probabilities"],
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
        )
        tc = triplet_center_loss(
            out["Z_SST"],
            labels,
            self.centers,
            margin=self.tc_margin,
        )
        recon_mse = tf.reduce_mean(tf.square(x_st - out["recon"]))
        kl = -0.5 * tf.reduce_mean(
            1
            + out["z_log_var"]
            - tf.square(out["z_mean"])
            - tf.exp(out["z_log_var"])
        )
        vae = recon_mse + kl
        total = self.beta1 * focal + self.beta2 * tc + self.beta3 * vae
        return total, focal, tc, vae

    def train_step(self, data):
        inputs, labels, _sample_weight = tf.keras.utils.unpack_x_y_sample_weight(
            data
        )
        with tf.GradientTape() as tape:
            out = self(inputs, training=True, sample_latent=True)
            total, focal, tc, vae = self.compute_losses(inputs, labels, out)
        gradients = tape.gradient(total, self.trainable_variables)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_variables)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(gradient_pairs)
        return self._update_trackers(
            total,
            focal,
            tc,
            vae,
            labels,
            out["probabilities"],
        )

    def test_step(self, data):
        inputs, labels, _sample_weight = tf.keras.utils.unpack_x_y_sample_weight(
            data
        )
        out = self(inputs, training=False, sample_latent=False)
        total, focal, tc, vae = self.compute_losses(inputs, labels, out)
        return self._update_trackers(
            total,
            focal,
            tc,
            vae,
            labels,
            out["probabilities"],
        )

    def fit(self, *args, **kwargs):
        # cross_val.py computes inverse-frequency class weights by default.
        # MTLFuseNet already contains focal weighting, and its custom train_step
        # does not consume sample weights. Remove the keyword explicitly.
        kwargs.pop("class_weight", None)
        return super().fit(*args, **kwargs)

    def _batched_outputs(
        self,
        inputs,
        *,
        sample_latent: bool,
        batch_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        batch_size = int(batch_size or self.prediction_batch_size)
        dataset = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)
        collected: dict[str, list[np.ndarray]] = {}
        for batch in dataset:
            output = self(
                batch,
                training=False,
                sample_latent=sample_latent,
            )
            for key, value in output.items():
                if key == "recon":
                    continue
                collected.setdefault(key, []).append(value.numpy())
        return {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in collected.items()
        }

    def predict_proba(self, inputs):
        return self._batched_outputs(
            inputs,
            sample_latent=False,
        )["probabilities"]

    def predict_diagnostics(self, inputs, batch_size: int | None = None):
        return self._batched_outputs(
            inputs,
            sample_latent=False,
            batch_size=batch_size,
        )

    def predict_mc_probabilities(
        self,
        inputs,
        n_samples: int,
        seed: int | tuple[int, int] | None = None,
    ) -> dict[str, np.ndarray]:
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        if isinstance(seed, tuple):
            seed_value = int(seed[0]) * 1_000_003 + int(seed[1])
        else:
            seed_value = None if seed is None else int(seed)

        draws: list[np.ndarray] = []
        for sample_index in range(int(n_samples)):
            if seed_value is not None:
                tf.random.set_seed(seed_value + sample_index)
            draws.append(
                self._batched_outputs(
                    inputs,
                    sample_latent=True,
                )["probabilities"]
            )
        return {"probability_samples": np.stack(draws, axis=0)}


# ---------------------------------------------------------------------------
# Run helpers and artifact writing
# ---------------------------------------------------------------------------


def _ensure_dir(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"eegproc.mtlfusenet.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if tf.is_tensor(value):
        return value.numpy().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fold_records(cv_results: dict) -> list[dict]:
    for key in ("fold_results", "outer_fold_results"):
        if cv_results.get(key):
            return list(cv_results[key])
    if cv_results.get("best_config_result", {}).get("fold_results"):
        return list(cv_results["best_config_result"]["fold_results"])
    return []


def _grid_summary_rows(cv_results: dict) -> list[dict]:
    rows: list[dict] = []
    for result in cv_results.get("config_results", []):
        row = {
            "config_index": result.get("config_index"),
            "is_selected": int(
                result.get("config_index") == cv_results.get("best_config_index")
            ),
            "selection_level": result.get("selection_level"),
            "selection_metric": result.get("selection_metric"),
            "selection_score": result.get("selection_score"),
            "selection_score_std": result.get("selection_score_std"),
            "n_folds": result.get("n_folds"),
            "config": json.dumps(
                result.get("config", {}),
                sort_keys=True,
                default=_json_default,
            ),
        }
        for prefix in ("window", "trial"):
            for metric, value in result.get(f"{prefix}_mean_scores", {}).items():
                row[f"{prefix}_{metric}_mean"] = value
            for metric, value in result.get(f"{prefix}_std_scores", {}).items():
                row[f"{prefix}_{metric}_std"] = value
        rows.append(row)
    return rows


def _select_final_epochs(
    cv_results: dict,
    strategy: str,
    fallback: int,
) -> tuple[int, list[int]]:
    best_epochs = [
        int(row["best_epoch"])
        for row in _fold_records(cv_results)
        if row.get("best_epoch") is not None and int(row["best_epoch"]) >= 1
    ]
    if not best_epochs:
        return max(1, int(fallback)), []
    if strategy == "median":
        selected = int(np.rint(np.median(best_epochs)))
    elif strategy == "mean":
        selected = int(np.rint(np.mean(best_epochs)))
    elif strategy == "max":
        selected = int(np.max(best_epochs))
    else:
        raise ValueError("final_epoch_strategy must be median, mean, or max.")
    return max(1, selected), best_epochs


def _make_optimizer(name: str, learning_rate: float, weight_decay: float):
    name = str(name).lower()
    if name == "adam":
        if weight_decay != 0.0:
            raise ValueError("weight_decay must be 0 when optimizer=adam.")
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer must be adam or adamw.")


def _model_hparameter_keys() -> set[str]:
    return {
        "optimizer",
        "optimizer_name",
        "learning_rate",
        "weight_decay",
        "vae_latent",
        "gcn_dim",
        "gru_units",
        "beta1",
        "beta2",
        "beta3",
        "focal_alpha",
        "focal_gamma",
        "tc_margin",
        "dropout",
        "prediction_batch_size",
    }


def train_mtlfusenet(
    training_config: MTLFuseNetTrainingConfig | None = None,
) -> dict:
    """Train MTLFuseNet through EEGProc's shared LOSO implementation."""
    config = training_config or MTLFuseNetTrainingConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(
        config.output_dir / f"{config.run_name}_{config.task}_{timestamp}"
    )
    logger = _configure_logger(run_dir)

    if config.seed is not None:
        tf.keras.utils.set_random_seed(config.seed)
        np.random.seed(config.seed)

    logger.info("Loading cached MTLFuseNet trials from %s", config.processed_dir)
    features, labels, subjects, trials = load_mtlfusenet_training_data(
        config.processed_dir,
        task=config.task,
    )

    logger.info("Packed feature tensor: %s", features.shape)
    logger.info("Packed feature width: %d", PACKED_SIZE)
    logger.info("Windows: %d", len(features))
    logger.info("Subjects: %d", len(np.unique(subjects)))
    logger.info(
        "Subject-trial groups: %d",
        len(set(zip(subjects.tolist(), trials.tolist()))),
    )
    values, counts = np.unique(labels, return_counts=True)
    class_distribution = {
        int(value): int(count) for value, count in zip(values, counts)
    }
    logger.info("Class counts: %s", class_distribution)
    logger.info("Task: %s", config.task)
    logger.info(
        "Fold-local Keras class_weight is intentionally removed; "
        "classification uses the replicated focal-loss implementation."
    )
    logger.info(
        "Selection: %s_%s",
        config.selection_level,
        config.selection_metric,
    )
    logger.info(
        "Inference: %s",
        "posterior mean"
        if config.prediction_latent_samples == 0
        else f"MC mean over {config.prediction_latent_samples} samples",
    )

    _write_json(run_dir / "training_config.json", asdict(config))
    _write_json(
        run_dir / "packed_input_layout.json",
        {
            "packed_shape": [None, 1, PACKED_SIZE],
            "X_ST": {"slice": [0, X_ST_SIZE], "shape": list(X_ST_SHAPE)},
            "DE": {
                "slice": [X_ST_SIZE, X_ST_SIZE + DE_SIZE],
                "shape": list(DE_SHAPE),
            },
            "adj": {
                "slice": [X_ST_SIZE + DE_SIZE, PACKED_SIZE],
                "shape": list(ADJ_SHAPE),
            },
        },
    )

    allowed_hparams = _model_hparameter_keys()

    def model_builder(**hparams) -> EEGProcMTLFuseNet:
        unknown = set(hparams) - allowed_hparams
        if unknown:
            raise ValueError(
                f"Unknown MTLFuseNet hyperparameter(s): {sorted(unknown)}"
            )
        optimizer_name = str(
            hparams.get(
                "optimizer",
                hparams.get("optimizer_name", config.optimizer_name),
            )
        )
        learning_rate = float(hparams.get("learning_rate", config.learning_rate))
        weight_decay = float(hparams.get("weight_decay", config.weight_decay))
        beta1 = float(hparams.get("beta1", config.beta1))
        beta2 = float(hparams.get("beta2", config.beta2))
        beta3 = float(hparams.get("beta3", config.beta3))
        dropout = float(hparams.get("dropout", config.dropout))
        if min(beta1, beta2, beta3) < 0.0:
            raise ValueError("beta1, beta2, and beta3 must be non-negative.")
        if not np.isclose(beta1 + beta2 + beta3, 1.0, atol=1e-6):
            raise ValueError("beta1 + beta2 + beta3 must equal 1.0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        model = EEGProcMTLFuseNet(
            num_classes=config.num_classes,
            vae_latent=int(hparams.get("vae_latent", config.vae_latent)),
            gcn_dim=int(hparams.get("gcn_dim", config.gcn_dim)),
            gru_units=int(hparams.get("gru_units", config.gru_units)),
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            focal_alpha=float(
                hparams.get("focal_alpha", config.focal_alpha)
            ),
            focal_gamma=float(
                hparams.get("focal_gamma", config.focal_gamma)
            ),
            tc_margin=float(hparams.get("tc_margin", config.tc_margin)),
            dropout=dropout,
            prediction_batch_size=int(
                hparams.get(
                    "prediction_batch_size",
                    config.prediction_batch_size,
                )
            ),
            name="mtlfusenet",
        )
        model.compile(
            optimizer=_make_optimizer(
                optimizer_name,
                learning_rate,
                weight_decay,
            )
        )
        return model

    common_cv_kwargs = {
        "model_builder_function": model_builder,
        "feature_array": features,
        "label_array": labels,
        "subject_id_array": subjects,
        "trial_id_array": trials,
        "n_epochs": config.cv_max_epochs,
        "batch_size": config.batch_size,
        "hyperparameters": config.hyperparameters,
        "evaluation_level": "window",
        "selection_level": config.selection_level,
        "selection_metric": config.selection_metric,
        "maximize_metric": config.maximize_metric,
        "metrics": (
            "accuracy",
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
        ),
        "log_predictions": True,
        "n_prediction_latent_samples": config.prediction_latent_samples,
        "latent_sampling_seed": config.latent_sampling_seed,
        "decision_thresholds": config.decision_thresholds,
        "threshold_selection_metric": config.threshold_selection_metric,
        "threshold_selection_level": config.threshold_selection_level,
        "prediction_diagnostics": config.prediction_diagnostics,
        "prediction_diagnostics_every_n_epochs": (
            config.prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": (
            config.prediction_diagnostics_max_samples
        ),
        "prediction_diagnostics_threshold_tolerance": (
            config.prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": config.prediction_diagnostics_seed,
        "validation_subjects_per_fold": (
            config.validation_subjects_per_fold
            if config.use_early_stopping
            else 0
        ),
        "validation_seed": config.validation_seed,
        "early_stopping_patience": (
            config.early_stopping_patience
            if config.use_early_stopping
            else None
        ),
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "early_stopping_monitor": config.early_stopping_monitor,
        "early_stopping_mode": config.early_stopping_mode,
        "restore_best_weights": True,
        "verbose": config.outer_verbose,
        "extra_fit_kwargs": {
            "callbacks": [tf.keras.callbacks.TerminateOnNaN()]
        },
        "n_jobs": config.n_jobs,
        "cpus_per_worker": config.cpus_per_worker,
    }

    logger.info("Starting EEGProc LOSO")
    cv_results = loso_cv(
        **common_cv_kwargs,
        max_folds=config.max_folds,
    )

    _write_json(run_dir / "cv_results.json", cv_results)
    _write_json(run_dir / "loso_cv_results.json", cv_results)
    _write_csv(run_dir / "grid_search_summary.csv", _grid_summary_rows(cv_results))
    _write_csv(
        run_dir / "prediction_diagnostics.csv",
        cv_results.get("prediction_diagnostics_log", []),
    )
    _write_csv(
        run_dir / "window_predictions.csv",
        cv_results.get("window_prediction_log", []),
    )
    _write_csv(
        run_dir / "trial_predictions.csv",
        cv_results.get("trial_prediction_log", []),
    )

    fold_rows: list[dict] = []
    for fold in _fold_records(cv_results):
        row = dict(fold)
        for key in (
            "prediction_log",
            "window_prediction_log",
            "trial_prediction_log",
            "variational_interval_log",
            "window_variational_interval_log",
            "trial_variational_interval_log",
            "prediction_diagnostics_log",
        ):
            row.pop(key, None)
        for key in ("left_out_subjects", "outer_test_subjects", "validation_subjects"):
            if key in row and isinstance(row[key], (list, tuple, np.ndarray)):
                row[key] = ",".join(map(str, row[key]))
        if "inner_fold_results" in row:
            row["inner_fold_results"] = json.dumps(
                row["inner_fold_results"],
                default=_json_default,
            )
        fold_rows.append(row)
    _write_csv(run_dir / "loso_cv_folds.csv", fold_rows)

    selected_config = dict(cv_results.get("best_config", {}))
    if not selected_config and config.hyperparameters:
        raise RuntimeError("Cross-validation did not return best_config.")
    _write_json(run_dir / "selected_config.json", selected_config)

    configured_epoch_cap = int(
        selected_config.get("epochs", config.cv_max_epochs)
    )
    if config.final_epochs is not None:
        final_epochs = max(1, int(config.final_epochs))
        cv_best_epochs: list[int] = []
    elif config.use_early_stopping:
        final_epochs, cv_best_epochs = _select_final_epochs(
            cv_results,
            config.final_epoch_strategy,
            configured_epoch_cap,
        )
    else:
        final_epochs = configured_epoch_cap
        cv_best_epochs = []

    final_batch_size = int(selected_config.get("batch_size", config.batch_size))
    final_hparams = {
        key: value
        for key, value in selected_config.items()
        if key not in {"epochs", "batch_size"}
    }
    logger.info("Selected config: %s", selected_config)
    logger.info("Final epochs: %d", final_epochs)
    logger.info("Final batch size: %d", final_batch_size)

    final_model = model_builder(**final_hparams)
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.TerminateOnNaN()
    ]
    diagnostics_callback: PredictionDiagnostics | None = None
    if config.prediction_diagnostics:
        diagnostics_callback = PredictionDiagnostics(
            X_train=features,
            y_train=labels,
            fold_number=None,
            batch_size=final_batch_size,
            every_n_epochs=config.prediction_diagnostics_every_n_epochs,
            max_samples=config.prediction_diagnostics_max_samples,
            threshold_tolerance=(
                config.prediction_diagnostics_threshold_tolerance
            ),
            seed=config.prediction_diagnostics_seed,
        )
        callbacks.append(diagnostics_callback)
    if config.save_final_history_csv:
        callbacks.insert(
            0,
            tf.keras.callbacks.CSVLogger(
                str(run_dir / "final_training_history.csv")
            ),
        )

    final_history = final_model.fit(
        features,
        labels,
        epochs=final_epochs,
        batch_size=final_batch_size,
        verbose=config.final_verbose,
        callbacks=callbacks,
    )
    if diagnostics_callback is not None:
        _write_csv(
            run_dir / "final_prediction_diagnostics.csv",
            diagnostics_callback.history,
        )

    final_eval = final_model.evaluate(
        features,
        labels,
        batch_size=final_batch_size,
        verbose=0,
        return_dict=True,
    )
    if config.save_weights:
        final_model.save_weights(run_dir / "final_model.weights.h5")
    if config.save_full_model:
        final_model.save(run_dir / "final_model.keras")

    final_summary = {
        "run_dir": str(run_dir),
        "architecture": "mtlfusenet_vae_gcn_gru",
        "task": config.task,
        "packed_input_size": PACKED_SIZE,
        "n_windows": int(len(features)),
        "n_subjects": int(len(np.unique(subjects))),
        "n_trials": int(len(set(zip(subjects.tolist(), trials.tolist())))),
        "class_distribution": class_distribution,
        "selected_final_config": selected_config,
        "selected_final_epochs": int(final_epochs),
        "selected_final_batch_size": int(final_batch_size),
        "cv_best_epochs": cv_best_epochs,
        "prediction_latent_samples": config.prediction_latent_samples,
        "cv_results": cv_results,
        "final_fit_history": final_history.history,
        "final_full_dataset_metrics": final_eval,
    }
    _write_json(run_dir / "training_summary.json", final_summary)
    logger.info("Final full-data metrics: %s", final_eval)
    logger.info("Saved MTLFuseNet artifacts to %s", run_dir)
    return final_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_bool_pair(
    parser: argparse.ArgumentParser,
    positive_flag: str,
    negative_flag: str,
    destination: str,
    default: bool,
    positive_help: str,
    negative_help: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        positive_flag,
        dest=destination,
        action="store_true",
        help=positive_help,
    )
    group.add_argument(
        negative_flag,
        dest=destination,
        action="store_false",
        help=negative_help,
    )
    parser.set_defaults(**{destination: default})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MTLFuseNet with EEGProc LOSO and trial-level metrics."
    )
    parser.add_argument("--processed-dir", default="processed_trials")
    parser.add_argument("--task", choices=("valence", "arousal"), default="valence")
    parser.add_argument("--out-dir", default="runs/mtlfusenet")
    parser.add_argument("--run-name", default="mtlfusenet")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument(
        "--final-epoch-strategy",
        choices=("median", "mean", "max"),
        default="median",
    )

    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cpus-per-worker", type=int, default=None)
    parser.add_argument("--outer-verbose", type=int, default=0)
    parser.add_argument("--final-verbose", type=int, default=2)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "loss",
            "joint_loss",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
        ),
        default="accuracy",
    )
    parser.add_argument(
        "--selection-level",
        choices=("window", "trial"),
        default="trial",
    )

    parser.add_argument("--validation-subjects", type=int, default=2)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-monitor", default="val_accuracy")
    parser.add_argument(
        "--early-stopping-mode",
        choices=("auto", "min", "max"),
        default="max",
    )

    parser.add_argument("--prediction-latent-samples", type=int, default=0)
    parser.add_argument("--latent-sampling-seed", type=int, default=42)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument(
        "--decision-thresholds",
        type=float,
        nargs="+",
        default=[0.5],
    )
    parser.add_argument(
        "--threshold-selection-metric",
        choices=("accuracy", "f1", "balanced_accuracy", "binary_f1"),
        default="accuracy",
    )
    parser.add_argument(
        "--threshold-selection-level",
        choices=("window", "trial"),
        default="trial",
    )
    parser.add_argument("--no-prediction-diagnostics", action="store_true")
    parser.add_argument("--prediction-diagnostics-every", type=int, default=1)
    parser.add_argument("--prediction-diagnostics-samples", type=int, default=256)
    parser.add_argument("--prediction-threshold-tolerance", type=float, default=0.01)
    parser.add_argument("--prediction-diagnostics-seed", type=int, default=42)

    parser.add_argument("--vae-latent", type=int, default=128)
    parser.add_argument("--gcn-dim", type=int, default=32)
    parser.add_argument("--gru-units", type=int, default=384)
    parser.add_argument("--beta1", type=float, default=0.7)
    parser.add_argument("--beta2", type=float, default=0.2)
    parser.add_argument("--beta3", type=float, default=0.1)
    parser.add_argument("--focal-alpha", type=float, default=0.7)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tc-margin", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--hyperparameters-json", default=None)
    _add_bool_pair(
        parser,
        "--save-weights",
        "--no-save-weights",
        "save_weights",
        True,
        "Save final Keras weights.",
        "Do not save final Keras weights.",
    )
    _add_bool_pair(
        parser,
        "--save-full-model",
        "--no-save-full-model",
        "save_full_model",
        False,
        "Save the complete custom Keras model.",
        "Do not save the complete custom Keras model.",
    )
    _add_bool_pair(
        parser,
        "--save-final-history-csv",
        "--no-save-final-history-csv",
        "save_final_history_csv",
        True,
        "Save final fit history as CSV.",
        "Do not save final fit history as CSV.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace, hyperparameters: dict) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "prediction_batch_size": args.prediction_batch_size,
        "vae_latent": args.vae_latent,
        "gcn_dim": args.gcn_dim,
        "gru_units": args.gru_units,
        "prediction_diagnostics_every": args.prediction_diagnostics_every,
        "prediction_diagnostics_samples": args.prediction_diagnostics_samples,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError("--max-folds must be positive.")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive.")
    if args.cpus_per_worker is not None and args.cpus_per_worker < 1:
        raise ValueError("--cpus-per-worker must be positive.")
    if args.validation_subjects < 0:
        raise ValueError("--validation-subjects must be non-negative.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative.")
    if args.prediction_latent_samples < 0:
        raise ValueError("--prediction-latent-samples must be non-negative.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.optimizer == "adam" and args.weight_decay != 0.0:
        raise ValueError("Use --optimizer adamw when weight decay is nonzero.")
    for name, value in (
        ("beta1", args.beta1),
        ("beta2", args.beta2),
        ("beta3", args.beta3),
        ("focal_alpha", args.focal_alpha),
        ("focal_gamma", args.focal_gamma),
        ("tc_margin", args.tc_margin),
    ):
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if not np.isclose(args.beta1 + args.beta2 + args.beta3, 1.0, atol=1e-6):
        raise ValueError("--beta1 + --beta2 + --beta3 must equal 1.0.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    thresholds = [float(value) for value in args.decision_thresholds]
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Decision thresholds must lie strictly between 0 and 1.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Decision thresholds must not contain duplicates.")
    if not isinstance(hyperparameters, dict):
        raise ValueError("--hyperparameters-json must decode to an object.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hyperparameters = (
        json.loads(args.hyperparameters_json)
        if args.hyperparameters_json
        else {}
    )
    _validate_args(args, hyperparameters)

    validation_seed = (
        args.validation_seed
        if args.validation_seed is not None
        else args.seed
    )
    config = MTLFuseNetTrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        processed_dir=Path(args.processed_dir),
        task=args.task,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        cv_max_epochs=args.epochs,
        final_epochs=args.final_epochs,
        final_epoch_strategy=args.final_epoch_strategy,
        seed=args.seed,
        selection_metric=args.selection_metric,
        selection_level=args.selection_level,
        max_folds=args.max_folds,
        n_jobs=args.n_jobs,
        cpus_per_worker=args.cpus_per_worker,
        outer_verbose=args.outer_verbose,
        final_verbose=args.final_verbose,
        validation_subjects_per_fold=args.validation_subjects,
        validation_seed=validation_seed,
        use_early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        prediction_latent_samples=args.prediction_latent_samples,
        latent_sampling_seed=args.latent_sampling_seed,
        prediction_batch_size=args.prediction_batch_size,
        decision_thresholds=tuple(sorted(map(float, args.decision_thresholds))),
        threshold_selection_metric=args.threshold_selection_metric,
        threshold_selection_level=args.threshold_selection_level,
        prediction_diagnostics=not args.no_prediction_diagnostics,
        prediction_diagnostics_every_n_epochs=args.prediction_diagnostics_every,
        prediction_diagnostics_max_samples=args.prediction_diagnostics_samples,
        prediction_diagnostics_threshold_tolerance=args.prediction_threshold_tolerance,
        prediction_diagnostics_seed=args.prediction_diagnostics_seed,
        vae_latent=args.vae_latent,
        gcn_dim=args.gcn_dim,
        gru_units=args.gru_units,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tc_margin=args.tc_margin,
        dropout=args.dropout,
        save_weights=args.save_weights,
        save_full_model=args.save_full_model,
        save_final_history_csv=args.save_final_history_csv,
        hyperparameters=hyperparameters,
    )
    train_mtlfusenet(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
