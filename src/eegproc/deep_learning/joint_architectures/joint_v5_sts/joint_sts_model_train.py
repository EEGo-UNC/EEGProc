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
        validation_subjects_per_fold=args.validation_subjects,
        validation_seed=args.validation_seed if args.validation_seed is not None else args.seed,
        use_early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        final_epochs=args.final_epochs,
        final_epoch_strategy=args.final_epoch_strategy,
        run_no_validation_loso_before_final=not args.skip_no_validation_loso_before_final,
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
