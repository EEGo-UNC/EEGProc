"""Training entry point for JointAutoencoderVariationalClassifierV2.

This module keeps the v2 model file focused on architecture only. It provides
nested LNSO cross-validation, structured logging, tunable hyperparameters,
DREAMER-backed data loading (see ``joint_v2_data.py``), and final model
saving.
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
    from .joint_autoencoder_vc_v2 import JointAutoencoderVariationalClassifierV2
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

    from joint_autoencoder_vc_v2 import JointAutoencoderVariationalClassifierV2
    from joint_v2_data import (
        DEFAULT_DREAMER_EEG_PATH,
        DEFAULT_DREAMER_LABELS_PATH,
        DREAMER_FS,
        DREAMER_MEDIAN_LABEL,
        build_joint_v2_dataset,
    )

try:
    from ..cross_validation import nested_lnso_cv
    from ..supervised.variational_classifier import VariationalClassifier
    from ..unsupervised.Convolutions.CNN1D import CNN1DDecoder, CNN1DEncoder
except ImportError:
    SRC_ROOT = Path(__file__).resolve().parents[3]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from eegproc.deep_learning.cross_validation import nested_lnso_cv
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
    encoder_kwargs: dict = field(default_factory=dict)
    decoder_kwargs: dict = field(default_factory=dict)
    classifier_kwargs: dict = field(default_factory=dict)
    model_kwargs: dict = field(default_factory=dict)


def load_joint_v2_training_data(
    eeg_path: str | Path = DEFAULT_DREAMER_EEG_PATH,
    labels_path: str | Path = DEFAULT_DREAMER_LABELS_PATH,
    label_dimension: str = "valence",
    window_size_sec: float = 4.0,
    fs: float = DREAMER_FS,
    overlap: float = 0.0,
    median_label: float = DREAMER_MEDIAN_LABEL,
    zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Default data loader for the joint-model training pipeline.

    Wraps ``joint_v2_data.build_joint_v2_dataset``, which adapts STSNet's
    DREAMER loading/label-binarization conventions
    (``STSNet/prepare_datasets.py``, ``STSNet/train_eval.py``) but windows
    the *raw* EEG signal (rather than STSNet's covariance/SPD features)
    since this model's decoder reconstructs raw amplitudes directly.

    By default this loads the pre-converted DREAMER arrays already present
    in ``eegproc/supervised/stsnet/data/`` (produced by STSNet's
    ``prepare_datasets.py``), using STSNet's DREAMER conventions: fs=128 Hz,
    4-second windows (512 samples, 15 windows per 60 s trial), median-split
    threshold of 3 (DREAMER's 1-5 Likert midpoint), and per-subject,
    per-channel z-scoring (see ``joint_v2_data.zscore_subject_eeg`` for why
    this is added on top of STSNet's pipeline).

    Parameters
    ----------
    eeg_path, labels_path : str or Path
        Paths to the pre-converted ``*_eeg.npy`` / ``*_labels.npy`` files.
    label_dimension : {"valence", "arousal"}, default="valence"
        Which label dimension to classify.
    window_size_sec : float, default=4.0
        Window length in seconds.
    fs : float, default=128
        Sampling frequency in Hz.
    overlap : float, default=0.0
        Fractional overlap between consecutive windows within a trial.
    median_label : float, default=3
        Median-split threshold for binarizing the chosen label dimension.
    zscore : bool, default=True
        Whether to z-score each subject's EEG per channel before windowing.

    Returns
    -------
    feature_array : np.ndarray, shape (n_windows, timesteps, n_channels)
    label_array : np.ndarray, shape (n_windows,)
    subject_id_array : np.ndarray, shape (n_windows,)
    """

    return build_joint_v2_dataset(
        eeg_path=eeg_path,
        labels_path=labels_path,
        label_dimension=label_dimension,
        window_size_sec=window_size_sec,
        fs=fs,
        overlap=overlap,
        median_label=median_label,
        zscore=zscore,
    )


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

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_final_epochs(
    outer_fold_results: list[dict],
    strategy: str = "median",
    fixed_epochs: int | None = None,
) -> int:
    if fixed_epochs is not None:
        return max(1, int(fixed_epochs))

    candidate_epochs = [int(fold["best_inner_epochs"]) for fold in outer_fold_results]
    if not candidate_epochs:
        return 1

    if strategy == "mean":
        return max(1, int(round(float(np.mean(candidate_epochs)))))
    if strategy == "max":
        return max(1, int(max(candidate_epochs)))

    return max(1, int(round(float(np.median(candidate_epochs)))))


def build_joint_autoencoder_variational_classifier_v2(
    input_shape: tuple[int, int],
    n_classes: int = 2,
    learning_rate: float = 1e-3,
    ae_loss_weight: float = 0.5,
    vc_loss_weight: float = 0.5,
    vc_alpha: float = 1.0,
    vc_beta: float = 0.0,
    vc_gamma: float = 1e-4,
    vc_lambda: float = 0.0,
    update_discriminator: bool = False,
    encoder_kwargs: dict | None = None,
    decoder_kwargs: dict | None = None,
    classifier_kwargs: dict | None = None,
    model_name: str = "joint_autoencoder_variational_classifier_v2",
) -> JointAutoencoderVariationalClassifierV2:
    """Build and compile a default v2 joint model."""

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

    classifier_defaults = {"n_classes": n_classes}
    if classifier_kwargs:
        classifier_defaults.update(classifier_kwargs)

    encoder = CNN1DEncoder(**encoder_defaults)
    decoder = CNN1DDecoder.from_encoder(encoder, **(decoder_kwargs or {}))
    variational_classifier = VariationalClassifier(**classifier_defaults)

    model = JointAutoencoderVariationalClassifierV2(
        encoder=encoder,
        decoder=decoder,
        variational_classifier=variational_classifier,
        ae_loss_weight=ae_loss_weight,
        vc_loss_weight=vc_loss_weight,
        vc_alpha=vc_alpha,
        vc_beta=vc_beta,
        vc_gamma=vc_gamma,
        vc_lambda=vc_lambda,
        update_discriminator=update_discriminator,
        name=model_name,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def train_joint_autoencoder_variational_classifier_v2(
    feature_array: np.ndarray | None = None,
    label_array: np.ndarray | None = None,
    subject_id_array: np.ndarray | None = None,
    data_loader: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    training_config: JointV2TrainingConfig | None = None,
    model_builder_function: Callable[[], tf.keras.Model] | None = None,
) -> dict:
    """Train the joint v2 model with nested LNSO selection and final saving."""

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
        feature_array, label_array, subject_id_array = data_loader()
    elif (
        feature_array is None
        or label_array is None
        or subject_id_array is None
    ):
        feature_array, label_array, subject_id_array = load_joint_v2_training_data()

    feature_array = np.asarray(feature_array, dtype=np.float32)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)

    if feature_array.ndim != 3:
        raise ValueError(
            f"feature_array must have shape (n_windows, timesteps, n_features); got {feature_array.shape}."
        )

    if model_builder_function is None:
        model_builder_function = lambda: build_joint_autoencoder_variational_classifier_v2(
            input_shape=tuple(feature_array.shape[1:]),
            n_classes=int(np.max(label_array)) + 1,
            learning_rate=training_config.learning_rate,
            ae_loss_weight=float(training_config.model_kwargs.get("ae_loss_weight", 0.5)),
            vc_loss_weight=float(training_config.model_kwargs.get("vc_loss_weight", 0.5)),
            vc_alpha=float(training_config.model_kwargs.get("vc_alpha", 1.0)),
            vc_beta=float(training_config.model_kwargs.get("vc_beta", 0.0)),
            vc_gamma=float(training_config.model_kwargs.get("vc_gamma", 1e-4)),
            vc_lambda=float(training_config.model_kwargs.get("vc_lambda", 0.0)),
            update_discriminator=bool(
                training_config.model_kwargs.get("update_discriminator", False)
            ),
            encoder_kwargs=training_config.encoder_kwargs,
            decoder_kwargs=training_config.decoder_kwargs,
            classifier_kwargs=training_config.classifier_kwargs,
        )

    logger.info("Starting joint-model v2 training run in %s", run_dir)
    logger.info("Feature shape: %s", feature_array.shape)
    logger.info("Unique subjects: %d", len(np.unique(subject_id_array)))
    _write_json(run_dir / "training_config.json", asdict(training_config))

    inner_callbacks: list[tf.keras.callbacks.Callback] = []
    if training_config.use_inner_early_stopping:
        inner_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=training_config.inner_early_stopping_patience,
                min_delta=training_config.inner_early_stopping_min_delta,
                restore_best_weights=True,
            )
        )

    nested_results = nested_lnso_cv(
        model_builder_function=model_builder_function,
        feature_array=feature_array,
        label_array=label_array,
        subject_id_array=subject_id_array,
        n_outer_subjects_to_leave_out=training_config.n_outer_subjects_to_leave_out,
        n_inner_subjects_to_leave_out=training_config.n_inner_subjects_to_leave_out,
        n_epochs=training_config.cv_max_epochs,
        batch_size=training_config.batch_size,
        verbose=training_config.outer_verbose,
        inner_verbose=training_config.inner_verbose,
        extra_fit_kwargs={},
        inner_extra_fit_kwargs={"callbacks": inner_callbacks} if inner_callbacks else {},
    )

    fold_rows: list[dict] = []
    for fold_result in nested_results["outer_fold_results"]:
        row = dict(fold_result)
        row["outer_test_subjects"] = ",".join(map(str, row["outer_test_subjects"]))
        row["inner_fold_results"] = json.dumps(
            row["inner_fold_results"], default=_json_default
        )
        fold_rows.append(row)

    _write_json(run_dir / "nested_cv_results.json", nested_results)
    _write_csv(run_dir / "nested_cv_outer_folds.csv", fold_rows)

    selected_final_epochs = _select_final_epochs(
        nested_results["outer_fold_results"],
        strategy=training_config.final_epoch_strategy,
        fixed_epochs=training_config.final_epochs,
    )

    logger.info("Selected %d epochs for the final full-data fit.", selected_final_epochs)

    final_model = model_builder_function()
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
        batch_size=training_config.batch_size,
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
        "selected_final_epochs": selected_final_epochs,
        "nested_cv": nested_results,
        "final_fit_history": final_history.history,
        "final_full_dataset_metrics": final_eval,
    }

    _write_json(run_dir / "training_summary.json", final_summary)
    logger.info("Final full-data metrics: %s", final_eval)
    logger.info("Saved run artifacts to %s", run_dir)

    return final_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train JointAutoencoderVariationalClassifierV2 with nested LNSO CV."
    )
    parser.add_argument("--out-dir", default="runs/joint_autoencoder_vc_v2")
    parser.add_argument("--run-name", default="joint_autoencoder_vc_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--outer-subjects", type=int, default=2)
    parser.add_argument("--inner-subjects", type=int, default=1)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument(
        "--final-epoch-strategy",
        choices=("median", "mean", "max"),
        default="median",
    )
    parser.add_argument("--seed", type=int, default=None)
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
    parser.add_argument("--vc-alpha", type=float, default=1.0)
    parser.add_argument("--vc-beta", type=float, default=0.0)
    parser.add_argument("--vc-gamma", type=float, default=1e-4)
    parser.add_argument("--vc-lambda", type=float, default=0.0)
    parser.add_argument("--update-discriminator", action="store_true")
    parser.add_argument("--features-npy", default=None)
    parser.add_argument("--labels-npy", default=None)
    parser.add_argument("--subjects-npy", default=None)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = JointV2TrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        cv_max_epochs=args.epochs,
        final_epoch_strategy=args.final_epoch_strategy,
        final_epochs=args.final_epochs,
        n_outer_subjects_to_leave_out=args.outer_subjects,
        n_inner_subjects_to_leave_out=args.inner_subjects,
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
        model_kwargs={
            "ae_loss_weight": args.ae_loss_weight,
            "vc_loss_weight": args.vc_loss_weight,
            "vc_alpha": args.vc_alpha,
            "vc_beta": args.vc_beta,
            "vc_gamma": args.vc_gamma,
            "vc_lambda": args.vc_lambda,
            "update_discriminator": args.update_discriminator,
        },
    )

    feature_array = label_array = subject_id_array = None
    if args.features_npy and args.labels_npy and args.subjects_npy:
        feature_array = _load_numpy_array(args.features_npy)
        label_array = _load_numpy_array(args.labels_npy)
        subject_id_array = _load_numpy_array(args.subjects_npy)
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
        data_loader=data_loader,
        training_config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
