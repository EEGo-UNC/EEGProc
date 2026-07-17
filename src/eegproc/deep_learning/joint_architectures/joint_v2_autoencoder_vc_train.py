"""Training entry point for the joint VAE + BiLSTM + variational classifier.

This module keeps the v2 model file focused on architecture only. It provides
ordinary leave-one-subject-out cross-validation, flat hyperparameter search
across the complete joint CNN-VAE/BiLSTM model, DREAMER-backed data loading
(see ``joint_v2_data.py``), structured logging, and final model saving.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

try:
    from .joint_v2_autoencoder_vc import JointAutoencoderVariationalClassifierV2
    from .joint_v2_data import (
        DEFAULT_DREAMER_EEG_PATH,
        DEFAULT_DREAMER_LABELS_PATH,
        DREAMER_FS,
        DREAMER_MEDIAN_LABEL,
        build_joint_v2_dataset,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))

    from joint_v2_autoencoder_vc import JointAutoencoderVariationalClassifierV2
    from joint_v2_data import (
        DEFAULT_DREAMER_EEG_PATH,
        DEFAULT_DREAMER_LABELS_PATH,
        DREAMER_FS,
        DREAMER_MEDIAN_LABEL,
        build_joint_v2_dataset,
    )

try:
    from ..cross_val import loso_cv
    from ..supervised.rnn_architectures import BiLSTMClassifier
    from ..supervised.variational_classifier import VariationalClassifier
    from ..unsupervised.Convolutions.CNN1D import CNN1DDecoder, CNN1DEncoder
except ImportError:
    SRC_ROOT = Path(__file__).resolve().parents[3]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from eegproc.deep_learning.cross_val import loso_cv
    from eegproc.deep_learning.supervised.rnn_architectures import (
        BiLSTMClassifier,
    )
    from eegproc.deep_learning.supervised.variational_classifier import (
        VariationalClassifier,
    )
    from eegproc.deep_learning.unsupervised.Convolutions.CNN1D import (
        CNN1DDecoder,
        CNN1DEncoder,
    )


@dataclass(slots=True)
class JointV2TrainingConfig:
    """Tunable settings for the joint-model training pipeline."""

    output_dir: Path = Path("runs") / "joint_autoencoder_vc_v2"
    run_name: str = "joint_autoencoder_vc_v2"
    learning_rate: float = 1e-3
    batch_size: int = 32
    cv_max_epochs: int = 50
    final_epoch_strategy: str = "median"
    final_epochs: int | None = None
    selection_metric: str = "f1"
    selection_level: str = "trial"
    maximize_metric: bool | None = None
    prediction_latent_samples: int = 0
    latent_sampling_seed: int | None = None
    n_outer_subjects_to_leave_out: int = 2
    n_inner_subjects_to_leave_out: int = 1
    outer_verbose: int = 0
    inner_verbose: int = 0
    final_verbose: int = 1
    inner_early_stopping_patience: int = 5
    inner_early_stopping_min_delta: float = 0.0
    use_inner_early_stopping: bool = True
    save_full_model: bool = True
    save_weights: bool = True
    save_final_history_csv: bool = True
    seed: int | None = None
    bilstm_units: int = 64
    n_bilstm_layers: int = 2
    bilstm_dropout: float = 0.10
    bilstm_kwargs: dict = field(default_factory=dict)
    encoder_kwargs: dict = field(default_factory=dict)
    decoder_kwargs: dict = field(default_factory=dict)
    classifier_kwargs: dict = field(default_factory=dict)
    model_kwargs: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    n_jobs: int = 4
    cpus_per_worker: int = 2
    max_folds: int | None = None


def load_joint_v2_training_data(
    eeg_path: str | Path = DEFAULT_DREAMER_EEG_PATH,
    labels_path: str | Path = DEFAULT_DREAMER_LABELS_PATH,
    label_dimension: str = "valence",
    window_size_sec: float = 4.0,
    fs: float = DREAMER_FS,
    overlap: float = 0.0,
    median_label: float = DREAMER_MEDIAN_LABEL,
    zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load windowed data plus aligned subject and trial identifiers.

    ``joint_v2_data.build_joint_v2_dataset`` historically returned only
    features, labels, and subject IDs. When that older interface is present,
    this wrapper reconstructs trial IDs from the raw four-dimensional array
    and the exact windowing parameters. A newer four-array return value is
    accepted directly.
    """
    dataset_arrays = build_joint_v2_dataset(
        eeg_path=eeg_path,
        labels_path=labels_path,
        label_dimension=label_dimension,
        window_size_sec=window_size_sec,
        fs=fs,
        overlap=overlap,
        median_label=median_label,
        zscore=zscore,
    )

    if len(dataset_arrays) == 4:
        feature_array, label_array, subject_id_array, trial_id_array = dataset_arrays
        return feature_array, label_array, subject_id_array, trial_id_array

    if len(dataset_arrays) != 3:
        raise ValueError(
            "build_joint_v2_dataset must return either three arrays "
            "(features, labels, subject_ids) or four arrays with trial_ids."
        )

    feature_array, label_array, subject_id_array = dataset_arrays

    # build_joint_v2_dataset emits windows in subject-major, trial-major order.
    # mmap_mode reads only the array metadata/shape here, not another full copy.
    raw_eeg = np.load(Path(eeg_path), mmap_mode="r", allow_pickle=False)
    if raw_eeg.ndim != 4:
        raise ValueError(
            "Expected raw EEG shaped "
            "(n_subjects, n_trials, n_channels, n_samples); got "
            f"{raw_eeg.shape}."
        )

    n_subjects, n_trials, _n_channels, n_samples = raw_eeg.shape
    window_size = int(round(window_size_sec * fs))
    if window_size <= 0:
        raise ValueError(
            f"window_size_sec * fs must produce a positive size; got {window_size}."
        )
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
    if n_samples < window_size:
        raise ValueError(
            f"Raw trials contain {n_samples} samples, shorter than "
            f"window_size={window_size}."
        )

    hop = max(1, int(round(window_size * (1.0 - overlap))))
    n_windows_per_trial = 1 + (n_samples - window_size) // hop
    per_subject_trial_ids = np.repeat(
        np.arange(n_trials, dtype=np.int64),
        n_windows_per_trial,
    )
    trial_id_array = np.tile(per_subject_trial_ids, n_subjects)

    if len(trial_id_array) != len(feature_array):
        raise ValueError(
            "Generated trial IDs do not align with the windowed dataset. "
            f"Generated {len(trial_id_array)} IDs for {len(feature_array)} windows. "
            "If joint_v2_data changed its window ordering or count, update it "
            "to return trial IDs directly."
        )

    return feature_array, label_array, subject_id_array, trial_id_array

def _load_numpy_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def _ensure_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_run_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"eegproc.joint_v2.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _grid_summary_rows(cv_results: dict) -> list[dict]:
    """Flatten configuration-level LOSO results for a compact CSV summary."""
    rows: list[dict] = []
    for config_result in cv_results.get("config_results", []):
        row = {
            "config_index": int(config_result["config_index"]),
            "is_selected": int(
                config_result["config_index"] == cv_results.get("best_config_index")
            ),
            "selection_level": config_result.get("selection_level"),
            "selection_metric": config_result.get("selection_metric"),
            "selection_score": config_result.get("selection_score"),
            "selection_score_std": config_result.get("selection_score_std"),
            "n_folds": config_result.get("n_folds"),
            "config": json.dumps(
                config_result.get("config", {}),
                sort_keys=True,
                default=_json_default,
            ),
        }

        for prefix in ("window", "trial"):
            mean_scores = config_result.get(f"{prefix}_mean_scores", {})
            std_scores = config_result.get(f"{prefix}_std_scores", {})
            for metric_name, metric_value in mean_scores.items():
                row[f"{prefix}_{metric_name}_mean"] = metric_value
            for metric_name, metric_value in std_scores.items():
                row[f"{prefix}_{metric_name}_std"] = metric_value

        rows.append(row)
    return rows


def _positive_int_tuple(
    name: str,
    value,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    """Normalize a sequence of positive integer layer settings."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple, got {value!r}.")
    if not value and not allow_empty:
        raise ValueError(f"{name} must be non-empty, got {value!r}.")
    normalized = tuple(int(item) for item in value)
    if any(item < 1 for item in normalized):
        raise ValueError(f"{name} values must all be >= 1, got {normalized!r}.")
    return normalized


def _nonnegative_int_tuple(name: str, value) -> tuple[int, ...]:
    """Normalize a sequence of non-negative integer layer indices."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple, got {value!r}.")
    normalized = tuple(int(item) for item in value)
    if any(item < 0 for item in normalized):
        raise ValueError(f"{name} values must all be >= 0, got {normalized!r}.")
    return normalized


def _normalize_encoder_configuration(encoder_config: dict) -> dict:
    """Validate and normalize one CNN encoder configuration from the grid."""
    config = dict(encoder_config)
    config["conv_filters"] = _positive_int_tuple(
        "conv_filters", config["conv_filters"]
    )
    config["kernel_sizes"] = _positive_int_tuple(
        "kernel_sizes", config["kernel_sizes"]
    )
    config["pool_after_layers"] = _nonnegative_int_tuple(
        "pool_after_layers", config["pool_after_layers"]
    )
    config["pool_sizes"] = _positive_int_tuple(
        "pool_sizes",
        config["pool_sizes"],
        allow_empty=True,
    )

    n_conv_layers = len(config["conv_filters"])
    if len(config["kernel_sizes"]) != n_conv_layers:
        raise ValueError(
            "conv_filters and kernel_sizes must describe the same number of "
            f"convolutional layers. Got {config['conv_filters']!r} and "
            f"{config['kernel_sizes']!r}."
        )
    if len(config["pool_after_layers"]) != len(config["pool_sizes"]):
        raise ValueError(
            "pool_after_layers and pool_sizes must have the same length. Got "
            f"{config['pool_after_layers']!r} and {config['pool_sizes']!r}."
        )
    if len(set(config["pool_after_layers"])) != len(config["pool_after_layers"]):
        raise ValueError(
            "pool_after_layers cannot contain duplicate layer indices: "
            f"{config['pool_after_layers']!r}."
        )
    if any(index >= n_conv_layers for index in config["pool_after_layers"]):
        raise ValueError(
            "pool_after_layers contains an index outside the convolutional "
            f"stack of length {n_conv_layers}: {config['pool_after_layers']!r}."
        )

    config["t_down"] = int(config["t_down"])
    config["emb_dim"] = int(config["emb_dim"])
    config["dropout"] = float(config["dropout"])
    config["use_batch_norm"] = bool(config["use_batch_norm"])

    if config["t_down"] < 1:
        raise ValueError(f"t_down must be >= 1, got {config['t_down']}.")
    if config["emb_dim"] < 1:
        raise ValueError(f"emb_dim must be >= 1, got {config['emb_dim']}.")
    if not 0.0 <= config["dropout"] < 1.0:
        raise ValueError(
            f"Encoder dropout must be in [0, 1), got {config['dropout']}."
        )

    return config

def build_joint_autoencoder_variational_classifier_v2(
    input_shape: tuple[int, int],
    n_classes: int = 2,
    learning_rate: float = 1e-3,
    ae_loss_weight: float = 0.5,
    vc_loss_weight: float = 0.5,
    vae_beta: float = 1.0,
    vc_alpha: float = 1.0,
    vc_beta: float = 1.0,
    vc_gamma: float = 0.0,
    vc_lambda: float = 1.0,
    update_discriminator: bool = False,
    bilstm_units: int = 64,
    n_bilstm_layers: int = 2,
    bilstm_dropout: float = 0.10,
    bilstm_kwargs: dict | None = None,
    encoder_kwargs: dict | None = None,
    decoder_kwargs: dict | None = None,
    classifier_kwargs: dict | None = None,
    model_name: str = "joint_autoencoder_variational_classifier_v2",
) -> JointAutoencoderVariationalClassifierV2:
    """Build and compile the CNN-VAE + BiLSTM + VC model.

    The CNN output is projected to learned ``z_mean`` and ``z_log_var``
    sequences inside ``JointAutoencoderVariationalClassifierV2``. Training
    uses reparameterized samples; evaluation uses the posterior mean.

    Encoder settings supplied through ``encoder_kwargs`` and the BiLSTM
    settings are all rebuilt for every grid configuration. This allows flat
    LOSO search over convolution filters, kernels, pooling, embedding width,
    encoder dropout, VAE beta, loss weights, and recurrent settings together.
    """
    timesteps, n_features = input_shape

    encoder_defaults = {
        "timesteps": timesteps,
        "n_features": n_features,
        "t_down": 2,
        "conv_filters": (16, 32),
        "kernel_sizes": (5, 3),
        "pool_after_layers": (0,),
        "pool_sizes": (2,),
        "emb_dim": 16,
        "dropout": 0.1,
        "use_batch_norm": False,
    }
    if encoder_kwargs:
        encoder_defaults.update(encoder_kwargs)
    encoder_defaults = _normalize_encoder_configuration(encoder_defaults)

    classifier_defaults = {"n_classes": n_classes}
    if classifier_kwargs:
        classifier_defaults.update(classifier_kwargs)

    encoder = CNN1DEncoder(**encoder_defaults)

    # Build the encoder once so the recurrent input dimensions are known.
    dummy_input = tf.zeros(
        shape=(1, timesteps, n_features),
        dtype=tf.float32,
    )
    latent_sequence = encoder(dummy_input, training=False)
    if latent_sequence.shape.rank != 3:
        raise ValueError(
            "CNN1DEncoder must return a rank-3 sequence shaped "
            "(batch, latent_timesteps, latent_features); got "
            f"{latent_sequence.shape}."
        )

    latent_timesteps = latent_sequence.shape[1]
    latent_features = latent_sequence.shape[2]
    if latent_timesteps is None or latent_features is None:
        raise ValueError(
            "The CNN encoder must expose static latent timestep and feature "
            "dimensions so the BiLSTM feature extractor can be built."
        )

    decoder = CNN1DDecoder.from_encoder(
        encoder,
        **(decoder_kwargs or {}),
    )

    recurrent_defaults = {
        "lstm_units": int(bilstm_units),
        "n_bilstm_layers": int(n_bilstm_layers),
        "dropout": float(bilstm_dropout),
        "name": "joint_bilstm",
    }
    if bilstm_kwargs:
        reserved = {"timesteps", "n_features", "n_classes"}
        conflicting = reserved.intersection(bilstm_kwargs)
        if conflicting:
            raise ValueError(
                "bilstm_kwargs cannot override dimensions supplied by the "
                f"joint model: {sorted(conflicting)}"
            )
        recurrent_defaults.update(bilstm_kwargs)

    classification_model = BiLSTMClassifier(
        timesteps=int(latent_timesteps),
        n_features=int(latent_features),
        n_classes=n_classes,
        **recurrent_defaults,
    ).build_feature_extractor()

    variational_classifier = VariationalClassifier(**classifier_defaults)

    model = JointAutoencoderVariationalClassifierV2(
        encoder=encoder,
        decoder=decoder,
        classification_model=classification_model,
        variational_classifier=variational_classifier,
        latent_features=int(latent_features),
        ae_loss_weight=ae_loss_weight,
        vc_loss_weight=vc_loss_weight,
        vae_beta=vae_beta,
        vc_alpha=vc_alpha,
        vc_beta=vc_beta,
        vc_gamma=vc_gamma,
        vc_lambda=vc_lambda,
        update_discriminator=update_discriminator,
        name=model_name,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        )
    )
    return model


def train_joint_autoencoder_variational_classifier_v2(
    feature_array: np.ndarray | None = None,
    label_array: np.ndarray | None = None,
    subject_id_array: np.ndarray | None = None,
    trial_id_array: np.ndarray | None = None,
    data_loader: Callable[
        [], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] | None = None,
    training_config: JointV2TrainingConfig | None = None,
    model_builder_function: Callable[[], tf.keras.Model] | None = None,
) -> dict:
    """Train the joint v2 model with ordinary LOSO CV and final saving."""

    training_config = training_config or JointV2TrainingConfig()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_path(
        training_config.output_dir / f"{training_config.run_name}_{run_timestamp}"
    )
    logger = _configure_run_logger(run_dir)

    if training_config.seed is not None:
        tf.keras.utils.set_random_seed(training_config.seed)
        np.random.seed(training_config.seed)

    if data_loader is not None:
        feature_array, label_array, subject_id_array, trial_id_array = data_loader()
    elif any(
        array is None
        for array in (feature_array, label_array, subject_id_array, trial_id_array)
    ):
        (
            feature_array,
            label_array,
            subject_id_array,
            trial_id_array,
        ) = load_joint_v2_training_data()

    feature_array = np.asarray(feature_array, dtype=np.float32)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    if feature_array.ndim != 3:
        raise ValueError(
            f"feature_array must have shape (n_windows, timesteps, n_features); got {feature_array.shape}."
        )

    input_lengths = (
        len(feature_array),
        len(label_array),
        len(subject_id_array),
        len(trial_id_array),
    )
    if len(set(input_lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            f"must align. Got lengths {input_lengths}."
        )

    if model_builder_function is None:
        encoder_hparam_keys = {
            "t_down",
            "conv_filters",
            "kernel_sizes",
            "pool_after_layers",
            "pool_sizes",
            "emb_dim",
            "dropout",
            "use_batch_norm",
        }
        bilstm_hparam_keys = {
            "bilstm_units",
            "bilstm_layers",
            "bilstm_dropout",
        }
        model_hparam_keys = {
            "learning_rate",
            "ae_loss_weight",
            "vc_loss_weight",
            "vae_beta",
            "vc_alpha",
            "vc_beta",
            "vc_gamma",
            "vc_lambda",
            "update_discriminator",
            *encoder_hparam_keys,
            *bilstm_hparam_keys,
            "bilstm_kwargs",
            "encoder_kwargs",
            "decoder_kwargs",
            "classifier_kwargs",
        }

        def model_builder_function(**hparams) -> tf.keras.Model:
            unknown_hparams = set(hparams) - model_hparam_keys
            if unknown_hparams:
                raise ValueError(
                    f"Unknown hyperparameter(s): {sorted(unknown_hparams)}"
                )

            bilstm_kwargs = dict(training_config.bilstm_kwargs)
            bilstm_kwargs.update(hparams.get("bilstm_kwargs", {}))

            encoder_kwargs = dict(training_config.encoder_kwargs)
            encoder_kwargs.update(
                {key: hparams[key] for key in encoder_hparam_keys if key in hparams}
            )
            encoder_kwargs.update(hparams.get("encoder_kwargs", {}))

            decoder_kwargs = dict(training_config.decoder_kwargs)
            decoder_kwargs.update(hparams.get("decoder_kwargs", {}))

            classifier_kwargs = dict(training_config.classifier_kwargs)
            classifier_kwargs.update(hparams.get("classifier_kwargs", {}))

            return build_joint_autoencoder_variational_classifier_v2(
                input_shape=tuple(feature_array.shape[1:]),
                n_classes=int(np.max(label_array)) + 1,
                learning_rate=float(
                    hparams.get("learning_rate", training_config.learning_rate)
                ),
                ae_loss_weight=float(
                    hparams.get(
                        "ae_loss_weight",
                        training_config.model_kwargs.get("ae_loss_weight", 0.5),
                    )
                ),
                vc_loss_weight=float(
                    hparams.get(
                        "vc_loss_weight",
                        training_config.model_kwargs.get("vc_loss_weight", 0.5),
                    )
                ),
                vae_beta=float(
                    hparams.get(
                        "vae_beta",
                        training_config.model_kwargs.get("vae_beta", 1.0),
                    )
                ),
                vc_alpha=float(
                    hparams.get(
                        "vc_alpha",
                        training_config.model_kwargs.get("vc_alpha", 1.0),
                    )
                ),
                vc_beta=float(
                    hparams.get(
                        "vc_beta",
                        training_config.model_kwargs.get("vc_beta", 1.0),
                    )
                ),
                vc_gamma=float(
                    hparams.get(
                        "vc_gamma",
                        training_config.model_kwargs.get("vc_gamma", 0.0),
                    )
                ),
                vc_lambda=float(
                    hparams.get(
                        "vc_lambda",
                        training_config.model_kwargs.get("vc_lambda", 1.0),
                    )
                ),
                update_discriminator=bool(
                    hparams.get(
                        "update_discriminator",
                        training_config.model_kwargs.get("update_discriminator", False),
                    )
                ),
                bilstm_units=int(
                    hparams.get("bilstm_units", training_config.bilstm_units)
                ),
                n_bilstm_layers=int(
                    hparams.get(
                        "bilstm_layers",
                        training_config.n_bilstm_layers,
                    )
                ),
                bilstm_dropout=float(
                    hparams.get(
                        "bilstm_dropout",
                        training_config.bilstm_dropout,
                    )
                ),
                bilstm_kwargs=bilstm_kwargs,
                encoder_kwargs=encoder_kwargs,
                decoder_kwargs=decoder_kwargs,
                classifier_kwargs=classifier_kwargs,
            )

    logger.info("Starting joint-model v2 training run in %s", run_dir)
    logger.info("Feature shape: %s", feature_array.shape)
    logger.info("Unique subjects: %d", len(np.unique(subject_id_array)))
    logger.info(
        "Unique subject/trial pairs: %d",
        len(set(zip(subject_id_array.tolist(), trial_id_array.tolist()))),
    )
    _write_json(run_dir / "training_config.json", asdict(training_config))

    cv_results = loso_cv(
        model_builder_function=model_builder_function,
        feature_array=feature_array,
        label_array=label_array,
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        n_epochs=training_config.cv_max_epochs,
        batch_size=training_config.batch_size,
        hyperparameters=training_config.hyperparameters,
        evaluation_level="trial",
        selection_level=training_config.selection_level,
        selection_metric=training_config.selection_metric,
        maximize_metric=training_config.maximize_metric,
        metrics=("accuracy", "f1", "precision", "recall"),
        log_predictions=True,
        n_prediction_latent_samples=training_config.prediction_latent_samples,
        latent_sampling_seed=training_config.latent_sampling_seed,
        verbose=training_config.outer_verbose,
        extra_fit_kwargs={"callbacks": [tf.keras.callbacks.TerminateOnNaN()]},
        n_jobs=training_config.n_jobs,
        cpus_per_worker=training_config.cpus_per_worker,
        max_folds=training_config.max_folds,
    )

    fold_rows: list[dict] = []
    for fold_result in cv_results["outer_fold_results"]:
        row = dict(fold_result)
        test_subjects = row.pop("outer_test_subjects", row.pop("left_out_subjects", []))
        row["outer_test_subjects"] = ",".join(map(str, test_subjects))
        row["inner_fold_results"] = json.dumps(
            row["inner_fold_results"], default=_json_default
        )
        fold_rows.append(row)

    _write_json(run_dir / "loso_cv_results.json", cv_results)
    _write_csv(run_dir / "loso_cv_folds.csv", fold_rows)
    _write_csv(
        run_dir / "grid_search_summary.csv",
        _grid_summary_rows(cv_results),
    )

    if "best_config" not in cv_results:
        raise RuntimeError(
            "loso_cv did not return best_config; the final model cannot be built."
        )
    selected_final_config = dict(cv_results["best_config"])
    _write_json(run_dir / "selected_config.json", selected_final_config)
    selected_final_epochs = (
        max(1, int(training_config.final_epochs))
        if training_config.final_epochs is not None
        else max(
            1,
            int(
                selected_final_config.get(
                    "epochs",
                    training_config.cv_max_epochs,
                )
            ),
        )
    )
    selected_final_batch_size = int(
        selected_final_config.get("batch_size", training_config.batch_size)
    )
    selected_final_model_hparams = {
        key: value
        for key, value in selected_final_config.items()
        if key not in {"epochs", "batch_size"}
    }

    logger.info("Selected final config: %s", selected_final_config)
    logger.info(
        "Selected %d epochs for the final full-data fit.", selected_final_epochs
    )

    final_model = model_builder_function(**selected_final_model_hparams)
    final_callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if training_config.save_final_history_csv:
        final_callbacks.insert(
            0,
            tf.keras.callbacks.CSVLogger(str(run_dir / "final_training_history.csv")),
        )

    final_history = final_model.fit(
        feature_array,
        label_array,
        epochs=selected_final_epochs,
        batch_size=selected_final_batch_size,
        verbose=training_config.final_verbose,
        callbacks=final_callbacks,
    )

    final_eval = final_model.evaluate(
        feature_array,
        label_array,
        verbose=0,
        return_dict=True,
    )

    if training_config.save_weights:
        final_model.save_weights(run_dir / "final_model.weights.h5")

    if training_config.save_full_model:
        final_model.save(run_dir / "final_model.keras")

    final_summary = {
        "run_dir": str(run_dir),
        "selected_final_config": selected_final_config,
        "selected_final_epochs": selected_final_epochs,
        "selected_final_batch_size": selected_final_batch_size,
        "loso_cv": cv_results,
        "final_fit_history": final_history.history,
        "final_full_dataset_metrics": final_eval,
    }

    _write_json(run_dir / "training_summary.json", final_summary)
    logger.info("Final full-data metrics: %s", final_eval)
    logger.info("Saved run artifacts to %s", run_dir)

    return final_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train JointAutoencoderVariationalClassifierV2 with flat LOSO "
            "hyperparameter search over the CNN-VAE and classifier."
        )
    )
    parser.add_argument("--out-dir", default="runs/joint_autoencoder_vc_v2")
    parser.add_argument("--run-name", default="joint_autoencoder_vc_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help=(
            "Optionally run only the first N sorted LOSO folds. "
            "Use 1 for an end-to-end smoke test; omit for complete LOSO."
        ),
    )
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument(
        "--final-epoch-strategy",
        choices=("median", "mean", "max"),
        default="median",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("loss", "accuracy", "f1", "precision", "recall"),
        default="f1",
        help="Metric used to rank complete LOSO configurations (default: f1).",
    )
    parser.add_argument(
        "--selection-level",
        choices=("window", "trial"),
        default="trial",
        help="Prediction level used for hyperparameter selection (default: trial).",
    )
    parser.add_argument(
        "--prediction-latent-samples",
        type=int,
        default=0,
        help=(
            "Number of q(z|x) samples averaged for every CV prediction. "
            "Use 0 for deterministic z_mean inference, 1 for one random draw, "
            "or values such as 5/30 for Monte Carlo averaging (default: 0)."
        ),
    )
    parser.add_argument(
        "--latent-sampling-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible Monte Carlo latent prediction.",
    )
    parser.add_argument("--outer-verbose", type=int, default=0)
    parser.add_argument("--inner-verbose", type=int, default=0)
    parser.add_argument("--final-verbose", type=int, default=1)
    parser.add_argument("--no-inner-early-stopping", action="store_true")
    parser.add_argument("--inner-patience", type=int, default=5)
    parser.add_argument("--inner-min-delta", type=float, default=0.0)
    parser.add_argument("--no-save-full-model", action="store_true")
    parser.add_argument("--no-save-weights", action="store_true")
    parser.add_argument("--no-save-final-history-csv", action="store_true")
    parser.add_argument("--ae-loss-weight", type=float, default=0.5)
    parser.add_argument("--vc-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--vae-beta",
        type=float,
        default=1.0,
        help=(
            "Positive KL weight for the autoencoder posterior "
            "q(z_ae|x) against N(0, I) (default: 1.0)."
        ),
    )
    parser.add_argument("--vc-alpha", type=float, default=1.0)
    parser.add_argument("--vc-beta", type=float, default=1.0)
    parser.add_argument("--vc-gamma", type=float, default=0.0)
    parser.add_argument("--vc-lambda", type=float, default=1.0)
    parser.add_argument("--update-discriminator", action="store_true")
    parser.add_argument(
        "--bilstm-units",
        type=int,
        default=64,
        help="Hidden units per BiLSTM direction/layer (default: 64).",
    )
    parser.add_argument(
        "--bilstm-layers",
        type=int,
        default=2,
        help="Number of stacked bidirectional LSTM layers (default: 2).",
    )
    parser.add_argument(
        "--bilstm-dropout",
        type=float,
        default=0.10,
        help="Dropout after each BiLSTM block (default: 0.10).",
    )
    parser.add_argument(
        "--hyperparameters-json",
        default=None,
        help=(
            "Cartesian hyperparameter grid passed to cross_val.loso_cv. Scalar "
            "settings use candidate lists. Sequence-valued encoder settings use "
            "a flat list for one architecture or a nested list for multiple "
            "architectures. Example: "
            '{"epochs":[300],"vae_beta":[0.1,1.0],'
            '"conv_filters":[[16,32],[32,64]],"kernel_sizes":[5,3],'
            '"emb_dim":[8,16],"bilstm_units":[128]}.'
        ),
    )
    parser.add_argument("--features-npy", default=None)
    parser.add_argument("--labels-npy", default=None)
    parser.add_argument("--subjects-npy", default=None)
    parser.add_argument(
        "--trials-npy",
        default=None,
        help=(
            "Trial ID array aligned with --features-npy. Required with "
            "pre-windowed feature, label, and subject arrays."
        ),
    )
    parser.add_argument(
        "--raw-eeg-npy",
        default=None,
        help=(
            "Path to a pre-converted *_eeg.npy file, shape "
            "(n_subjects, n_trials, n_channels, n_samples) (see "
            "STSNet/prepare_datasets.py). Defaults to the bundled DREAMER "
            "array if neither this nor --features-npy/--labels-npy/"
            "--subjects-npy are given."
        ),
    )
    parser.add_argument(
        "--raw-labels-npy",
        default=None,
        help=(
            "Path to the matching *_labels.npy file, shape "
            "(n_subjects, n_trials, n_label_dims)."
        ),
    )
    parser.add_argument(
        "--label-dimension",
        choices=("valence", "arousal"),
        default="valence",
        help="Which label dimension to classify (default: valence).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=4.0,
        help="Window length in seconds for raw-signal segmentation (default: 4.0).",
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.0,
        help="Fractional overlap in [0, 1) between consecutive windows (default: 0.0).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DREAMER_FS,
        help="Sampling frequency in Hz (default: 128, matching STSNet's DREAMER config).",
    )
    parser.add_argument(
        "--median-label",
        type=float,
        default=DREAMER_MEDIAN_LABEL,
        help="Median-split threshold for label binarization (default: 3).",
    )
    parser.add_argument(
        "--no-zscore",
        action="store_true",
        help="Disable per-subject, per-channel z-scoring of the raw EEG before windowing.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of concurrent outer-fold worker processes. "
            "Use 1 for sequential execution."
        ),
    )

    parser.add_argument(
        "--cpus-per-worker",
        type=int,
        default=None,
        help=(
            "Maximum TensorFlow CPU threads assigned to each worker. "
            "For example, 4 jobs with 2 CPUs each requires approximately 8 CPUs."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    parsed_hyperparameters = (
        json.loads(args.hyperparameters_json)
        if args.hyperparameters_json
        else {}
    )
    if not isinstance(parsed_hyperparameters, dict):
        raise ValueError("--hyperparameters-json must decode to a JSON object.")
    if args.prediction_latent_samples < 0:
        raise ValueError("--prediction-latent-samples must be >= 0.")

    config = JointV2TrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        cv_max_epochs=args.epochs,
        final_epoch_strategy=args.final_epoch_strategy,
        final_epochs=args.final_epochs,
        selection_metric=args.selection_metric,
        selection_level=args.selection_level,
        prediction_latent_samples=args.prediction_latent_samples,
        latent_sampling_seed=args.latent_sampling_seed,
        outer_verbose=args.outer_verbose,
        inner_verbose=args.inner_verbose,
        final_verbose=args.final_verbose,
        inner_early_stopping_patience=args.inner_patience,
        inner_early_stopping_min_delta=args.inner_min_delta,
        use_inner_early_stopping=not args.no_inner_early_stopping,
        save_full_model=not args.no_save_full_model,
        save_weights=not args.no_save_weights,
        save_final_history_csv=not args.no_save_final_history_csv,
        seed=args.seed,
        bilstm_units=args.bilstm_units,
        n_bilstm_layers=args.bilstm_layers,
        bilstm_dropout=args.bilstm_dropout,
        model_kwargs={
            "ae_loss_weight": args.ae_loss_weight,
            "vc_loss_weight": args.vc_loss_weight,
            "vae_beta": args.vae_beta,
            "vc_alpha": args.vc_alpha,
            "vc_beta": args.vc_beta,
            "vc_gamma": args.vc_gamma,
            "vc_lambda": args.vc_lambda,
            "update_discriminator": args.update_discriminator,
        },
        hyperparameters=parsed_hyperparameters,
        n_jobs=args.n_jobs,
        cpus_per_worker=args.cpus_per_worker,
        max_folds=args.max_folds,
    )

    feature_array = label_array = subject_id_array = trial_id_array = None
    windowed_paths = (
        args.features_npy,
        args.labels_npy,
        args.subjects_npy,
        args.trials_npy,
    )
    if any(path is not None for path in windowed_paths) and not all(
        path is not None for path in windowed_paths
    ):
        raise ValueError(
            "Pre-windowed input requires --features-npy, --labels-npy, "
            "--subjects-npy, and --trials-npy together."
        )

    if all(path is not None for path in windowed_paths):
        feature_array = _load_numpy_array(args.features_npy)
        label_array = _load_numpy_array(args.labels_npy)
        subject_id_array = _load_numpy_array(args.subjects_npy)
        trial_id_array = _load_numpy_array(args.trials_npy)
        data_loader = None
    else:
        eeg_path = args.raw_eeg_npy or DEFAULT_DREAMER_EEG_PATH
        labels_path = args.raw_labels_npy or DEFAULT_DREAMER_LABELS_PATH
        data_loader = lambda: load_joint_v2_training_data(
            eeg_path=eeg_path,
            labels_path=labels_path,
            label_dimension=args.label_dimension,
            window_size_sec=args.window_sec,
            fs=args.fs,
            overlap=args.window_overlap,
            median_label=args.median_label,
            zscore=not args.no_zscore,
        )

    train_joint_autoencoder_variational_classifier_v2(
        feature_array=feature_array,
        label_array=label_array,
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        data_loader=data_loader,
        training_config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
