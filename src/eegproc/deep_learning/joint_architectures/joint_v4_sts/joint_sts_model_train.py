"""Training entry point for joint_v4_sts.

Architecture:
    band-separated GCN -> BiLSTM -> classifier

The training protocol keeps EEGProc's LOSO/trial-level evaluation conventions
but deliberately removes v3's VAE, decoder, feature fusion, subject adversary,
SupCon, and alternating optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

try:
    from .joint_sts_cli import (
        json_default, parse_args, validate_args, write_csv, write_json
    )
    from .joint_sts_model import JointSTSModel, build_joint_sts_model
except ImportError:
    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from joint_sts_cli import json_default, parse_args, validate_args, write_csv, write_json
    from joint_sts_model import JointSTSModel, build_joint_sts_model

try:
    from ..joint_v2_data import (
        DatasetConfig,
        DREAMER_FS,
        DREAMER_MEDIAN_LABEL,
        build_joint_v2_dataset,
        get_dataset_config,
    )
except ImportError:
    from eegproc.deep_learning.joint_architectures.joint_v2_data import (
        DatasetConfig,
        DREAMER_FS,
        DREAMER_MEDIAN_LABEL,
        build_joint_v2_dataset,
        get_dataset_config,
    )

try:
    from ...cross_val import PredictionDiagnostics, fixed_loso_cv, loso_cv
except ImportError:
    from eegproc.deep_learning.cross_val import (
        PredictionDiagnostics, fixed_loso_cv, loso_cv
    )


@dataclass(slots=True)
class JointV4TrainingConfig:
    output_dir: Path = Path("runs/joint_v4_sts")
    run_name: str = "dreamer_valence_joint_v4_sts"
    dataset: str = "dreamer"
    n_channels: int = 14
    n_bands: int = 3

    batch_size: int = 64
    cv_max_epochs: int = 100
    optimizer_name: str = "adamw"
    classification_learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    t_down: int = 2
    temporal_pool_sizes: tuple[int, ...] = (2,)
    gcn_units: tuple[int, ...] = (128, 64)
    spectral_emb_dim: int = 128
    gcn_dropout: float = 0.20
    gcn_activation: str = "relu"
    gcn_use_batch_norm: bool = False
    graph_self_loop_bias: float = 2.0
    graph_identity_mix: float = 0.0
    graph_adjacency_reg_weight: float = 1e-4

    bilstm_units: int = 256
    n_bilstm_layers: int = 1
    bilstm_dropout: float = 0.30
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

    validation_subjects_per_fold: int = 2
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
    final_verbose: int = 1
    seed: int | None = 42

    save_full_model: bool = True
    save_weights: bool = True
    save_final_history_csv: bool = True
    save_adjacency_matrices: bool = True

    label_threshold_mode: str = "global"
    window_normalization: str = "global_rms"
    hyperparameters: dict = field(default_factory=dict)


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_logger(run_dir):
    logger = logging.getLogger(f"eegproc.joint_v4_sts.{run_dir.name}")
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


def _flatten_grouped_trials_to_windows(features, labels, subjects, trials):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    subjects = np.asarray(subjects).reshape(-1)
    trials = np.asarray(trials).reshape(-1)
    if features.ndim != 4:
        return features, labels, subjects, trials
    n_trials, n_windows, timesteps, n_features = features.shape
    return (
        features.reshape(n_trials * n_windows, timesteps, n_features),
        np.repeat(labels, n_windows, axis=0),
        np.repeat(subjects, n_windows),
        np.repeat(trials, n_windows),
    )


def _normalize_each_window(features, mode="global_rms", epsilon=1e-6):
    x = np.asarray(features, dtype=np.float32)
    if mode == "none":
        return x
    if mode == "global_rms":
        rms = np.sqrt(np.mean(np.square(x, dtype=np.float64), axis=(1,2), keepdims=True))
        return (x.astype(np.float64) / np.maximum(rms, epsilon)).astype(np.float32)
    if mode == "feature_zscore":
        mean = np.mean(x, axis=1, keepdims=True, dtype=np.float64)
        std = np.std(x, axis=1, keepdims=True, dtype=np.float64)
        return ((x.astype(np.float64)-mean) / np.maximum(std, epsilon)).astype(np.float32)
    raise ValueError(f"Unknown normalization mode: {mode}")


def _subject_median_window_labels(labels_path, label_dimension, subjects, trials):
    raw = np.load(Path(labels_path), allow_pickle=False)
    dim = {"valence": 0, "arousal": 1}[label_dimension]
    subjects = np.asarray(subjects).reshape(-1)
    trials = np.asarray(trials).reshape(-1)
    out = np.empty(len(subjects), dtype=np.int64)
    for subject_row, subject_id in enumerate(sorted(np.unique(subjects).tolist())):
        mask = subjects == subject_id
        unique_trials = sorted(np.unique(trials[mask]).tolist())
        ratings = raw[subject_row, :, dim].astype(np.float64)
        binary = (ratings >= np.median(ratings)).astype(np.int64)
        mapping = {trial_id: int(binary[i]) for i, trial_id in enumerate(unique_trials)}
        out[mask] = np.asarray([mapping[t] for t in trials[mask]], dtype=np.int64)
    return out


def load_joint_v4_training_data(
    eeg_path,
    labels_path,
    label_dimension="valence",
    window_size_sec=4.0,
    fs=DREAMER_FS,
    overlap=0.0,
    median_label=DREAMER_MEDIAN_LABEL,
    window_normalization="global_rms",
    label_threshold_mode="global",
    dataset="dreamer",
):
    arrays = build_joint_v2_dataset(
        eeg_path=eeg_path,
        labels_path=labels_path,
        label_dimension=label_dimension,
        window_size_sec=window_size_sec,
        fs=fs,
        overlap=overlap,
        median_label=median_label,
        zscore=False,
        dataset=dataset,
    )
    if len(arrays) == 4:
        features, labels, subjects, trials = arrays
    elif len(arrays) == 3:
        features, labels, subjects = arrays
        raw_eeg = np.load(Path(eeg_path), mmap_mode="r", allow_pickle=False)
        n_subjects, n_trials, _, n_samples = raw_eeg.shape
        window_size = int(round(window_size_sec * fs))
        hop = max(1, int(round(window_size * (1.0-overlap))))
        n_windows = 1 + (n_samples-window_size)//hop
        trials = np.tile(
            np.repeat(np.arange(n_trials, dtype=np.int64), n_windows),
            n_subjects,
        )
    else:
        raise ValueError("build_joint_v2_dataset must return 3 or 4 arrays.")

    features, labels, subjects, trials = _flatten_grouped_trials_to_windows(
        features, labels, subjects, trials
    )
    if label_threshold_mode == "subject_median":
        labels = _subject_median_window_labels(
            labels_path, label_dimension, subjects, trials
        )
    features = _normalize_each_window(features, window_normalization)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(subjects),
        np.asarray(trials),
    )


def _class_ids(y):
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:,0].astype(np.int64)
    return np.argmax(y, axis=1).astype(np.int64)


def _n_classes(y):
    ids = _class_ids(y)
    return int(np.max(ids)) + 1


def _selected_epochs(cv_results, strategy, fallback):
    fold_rows = cv_results.get("fold_results") or cv_results.get("outer_fold_results")
    if not fold_rows:
        fold_rows = cv_results.get("best_config_result", {}).get("fold_results", [])
    vals = [
        int(row["best_epoch"])
        for row in fold_rows
        if row.get("best_epoch") is not None and int(row["best_epoch"]) >= 1
    ]
    if not vals:
        return int(fallback), []
    if strategy == "median":
        return max(1, int(np.rint(np.median(vals)))), vals
    if strategy == "mean":
        return max(1, int(np.rint(np.mean(vals)))), vals
    return max(vals), vals


def _fold_rows(cv_results):
    return list(
        cv_results.get("fold_results")
        or cv_results.get("outer_fold_results")
        or cv_results.get("best_config_result", {}).get("fold_results", [])
    )


def _save_adjacencies(model, run_dir):
    nested = model.get_adjacency_matrices()
    flat = {}
    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}__{key}" if prefix else key, child)
        else:
            flat[prefix] = np.asarray(value.numpy() if tf.is_tensor(value) else value)
    visit("", nested)
    if flat:
        np.savez_compressed(run_dir / "adjacency_matrices.npz", **flat)


def train_joint_v4_sts(
    feature_array,
    label_array,
    subject_id_array,
    trial_id_array,
    config: JointV4TrainingConfig,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(config.output_dir / f"{config.run_name}_{timestamp}")
    logger = _configure_logger(run_dir)

    if config.seed is not None:
        tf.keras.utils.set_random_seed(config.seed)
        np.random.seed(config.seed)

    X = np.asarray(feature_array, dtype=np.float32)
    y = np.asarray(label_array)
    subjects = np.asarray(subject_id_array).reshape(-1)
    trials = np.asarray(trial_id_array).reshape(-1)
    if X.ndim != 3:
        raise ValueError(f"Expected (windows,timesteps,features), got {X.shape}.")
    if X.shape[-1] != config.n_channels * config.n_bands:
        raise ValueError(
            f"features={X.shape[-1]} but {config.n_channels}*{config.n_bands}="
            f"{config.n_channels*config.n_bands}."
        )

    write_json(run_dir / "training_config.json", asdict(config))
    logger.info("Architecture: band-separated GCN -> BiLSTM -> classifier")
    logger.info("Target: DREAMER valence")
    logger.info("Input: %s", X.shape)
    logger.info("Bands have independent GCN stacks and adjacency matrices.")
    logger.info("No VAE, decoder, feature-fusion posterior, subject adversary, or SupCon.")

    allowed = {
        "learning_rate", "classification_learning_rate", "optimizer", "optimizer_name",
        "weight_decay", "t_down", "temporal_pool_sizes", "gcn_units",
        "spectral_emb_dim", "gcn_dropout", "gcn_activation", "gcn_use_batch_norm",
        "graph_self_loop_bias", "graph_identity_mix", "graph_adjacency_reg_weight",
        "bilstm_units", "bilstm_layers", "n_bilstm_layers", "bilstm_dropout",
        "classification_hidden_units", "classification_dropout", "activation",
        "focal_gamma", "focal_alpha",
    }

    def builder(**h):
        unknown = set(h) - allowed - {"epochs", "batch_size"}
        if unknown:
            raise ValueError(f"Unknown v4 hyperparameters: {sorted(unknown)}")
        return build_joint_sts_model(
            input_shape=tuple(X.shape[1:]),
            n_classes=_n_classes(y),
            n_channels=config.n_channels,
            n_bands=config.n_bands,
            t_down=int(h.get("t_down", config.t_down)),
            temporal_pool_sizes=h.get("temporal_pool_sizes", config.temporal_pool_sizes),
            gcn_units=h.get("gcn_units", config.gcn_units),
            spectral_emb_dim=int(h.get("spectral_emb_dim", config.spectral_emb_dim)),
            gcn_dropout=float(h.get("gcn_dropout", config.gcn_dropout)),
            gcn_activation=str(h.get("gcn_activation", config.gcn_activation)),
            gcn_use_batch_norm=bool(h.get("gcn_use_batch_norm", config.gcn_use_batch_norm)),
            graph_self_loop_bias=float(h.get("graph_self_loop_bias", config.graph_self_loop_bias)),
            graph_identity_mix=float(h.get("graph_identity_mix", config.graph_identity_mix)),
            graph_adjacency_reg_weight=float(
                h.get("graph_adjacency_reg_weight", config.graph_adjacency_reg_weight)
            ),
            bilstm_units=int(h.get("bilstm_units", config.bilstm_units)),
            n_bilstm_layers=int(h.get("bilstm_layers", h.get("n_bilstm_layers", config.n_bilstm_layers))),
            bilstm_dropout=float(h.get("bilstm_dropout", config.bilstm_dropout)),
            classification_hidden_units=int(
                h.get("classification_hidden_units", config.classification_hidden_units)
            ),
            classification_dropout=float(
                h.get("classification_dropout", config.classification_dropout)
            ),
            activation=str(h.get("activation", config.activation)),
            focal_gamma=float(h.get("focal_gamma", config.focal_gamma)),
            focal_alpha=h.get("focal_alpha", config.focal_alpha),
            optimizer_name=str(h.get("optimizer", h.get("optimizer_name", config.optimizer_name))),
            classification_learning_rate=float(
                h.get("learning_rate", h.get(
                    "classification_learning_rate", config.classification_learning_rate
                ))
            ),
            weight_decay=float(h.get("weight_decay", config.weight_decay)),
        )

    builder._sequence_hyperparameter_depths = {
        "gcn_units": 1,
        "temporal_pool_sizes": 1,
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
        evaluation_level="trial",
        selection_level=config.selection_level,
        selection_metric=config.selection_metric,
        metrics=(
            "accuracy","f1","precision","recall",
            "macro_f1","macro_precision","macro_recall","balanced_accuracy",
        ),
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
        validation_subjects_per_fold=(
            config.validation_subjects_per_fold if config.use_early_stopping else 0
        ),
        validation_seed=config.validation_seed,
        early_stopping_patience=(
            config.early_stopping_patience if config.use_early_stopping else None
        ),
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_monitor=config.early_stopping_monitor,
        early_stopping_mode=config.early_stopping_mode,
        restore_best_weights=True,
        verbose=config.outer_verbose,
        extra_fit_kwargs={"callbacks":[tf.keras.callbacks.TerminateOnNaN()]},
        n_jobs=config.n_jobs,
        cpus_per_worker=config.cpus_per_worker,
        max_folds=config.max_folds,
    )

    write_json(run_dir / "loso_cv_results.json", cv)
    write_csv(run_dir / "loso_cv_folds.csv", _fold_rows(cv))

    selected = dict(cv["best_config"])
    selected_epochs_cap = int(selected.get("epochs", config.cv_max_epochs))
    if config.final_epochs is not None:
        final_epochs, best_epochs = int(config.final_epochs), []
    elif config.use_early_stopping:
        final_epochs, best_epochs = _selected_epochs(
            cv, config.final_epoch_strategy, selected_epochs_cap
        )
    else:
        final_epochs, best_epochs = selected_epochs_cap, []

    final_batch_size = int(selected.get("batch_size", config.batch_size))
    final_h = {k:v for k,v in selected.items() if k not in {"epochs","batch_size"}}

    fold_thresholds = [
        float(row["decision_threshold"])
        for row in _fold_rows(cv)
        if row.get("decision_threshold") is not None
    ]
    final_threshold = float(
        np.median(fold_thresholds) if fold_thresholds else config.decision_thresholds[0]
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
            evaluation_level="trial",
            selection_level=config.selection_level,
            selection_metric="balanced_accuracy",
            maximize_metric=True,
            metrics=(
                "accuracy","f1","precision","recall",
                "macro_f1","macro_precision","macro_recall","balanced_accuracy",
            ),
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
            extra_fit_kwargs={"callbacks":[tf.keras.callbacks.TerminateOnNaN()]},
            n_jobs=config.n_jobs,
            cpus_per_worker=config.cpus_per_worker,
            max_folds=config.max_folds,
        )
        write_json(run_dir / "no_validation_loso_results.json", fixed)

    model = builder(**final_h)
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    if config.save_final_history_csv:
        callbacks.insert(0, tf.keras.callbacks.CSVLogger(
            str(run_dir / "final_training_history.csv")
        ))

    class_weight = None
    if config.use_class_weight:
        classes, counts = np.unique(_class_ids(y), return_counts=True)
        class_weight = {
            int(c): len(y)/(len(classes)*count)
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
        "architecture": "band_separated_gcn_then_bilstm_then_classifier",
        "target": "valence",
        "selected_final_config": selected,
        "selected_final_epochs": final_epochs,
        "selected_final_batch_size": final_batch_size,
        "cv_best_epochs": best_epochs,
        "final_decision_threshold": final_threshold,
        "use_class_weight": config.use_class_weight,
        "final_class_weight": class_weight,
        "cv_results": cv,
        "final_fit_history": history.history,
        "final_full_dataset_metrics": final_eval,
    }
    write_json(run_dir / "training_summary.json", summary)
    logger.info("Saved v4 artifacts to %s", run_dir)
    return summary


def main(argv=None):
    args = parse_args(argv)
    hparams = json.loads(args.hyperparameters_json) if args.hyperparameters_json else {}
    validate_args(args, hparams)

    config = JointV4TrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        dataset=args.dataset,
        n_channels=args.n_channels,
        n_bands=args.n_bands,
        batch_size=args.batch_size,
        cv_max_epochs=args.epochs,
        optimizer_name=args.optimizer,
        classification_learning_rate=args.classification_learning_rate,
        weight_decay=args.weight_decay,
        t_down=args.t_down,
        temporal_pool_sizes=tuple(args.temporal_pool_sizes),
        gcn_units=tuple(args.gcn_units),
        spectral_emb_dim=args.spectral_emb_dim,
        gcn_dropout=args.gcn_dropout,
        gcn_activation=args.gcn_activation,
        gcn_use_batch_norm=args.gcn_use_batch_norm,
        graph_self_loop_bias=args.graph_self_loop_bias,
        graph_identity_mix=args.graph_identity_mix,
        graph_adjacency_reg_weight=args.graph_adjacency_reg_weight,
        bilstm_units=args.bilstm_units,
        n_bilstm_layers=args.bilstm_layers,
        bilstm_dropout=args.bilstm_dropout,
        classification_hidden_units=args.classification_hidden_units,
        classification_dropout=args.classification_dropout,
        activation=args.activation,
        focal_gamma=args.focal_gamma,
        focal_alpha=None if args.focal_alpha is None else tuple(args.focal_alpha),
        use_class_weight=args.use_class_weight,
        selection_level=args.selection_level,
        selection_metric=args.selection_metric,
        decision_thresholds=tuple(sorted(map(float,args.decision_thresholds))),
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
        save_final_history_csv=not args.no_save_final_history_csv,
        save_adjacency_matrices=not args.no_save_adjacency_matrices,
        label_threshold_mode=args.label_threshold_mode,
        window_normalization=args.window_normalization,
        hyperparameters=hparams,
    )

    dataset_config = get_dataset_config(args.dataset)
    eeg_path = args.raw_eeg_npy or dataset_config.eeg_path
    labels_path = args.raw_labels_npy or dataset_config.labels_path

    X, y, subjects, trials = load_joint_v4_training_data(
        eeg_path=eeg_path,
        labels_path=labels_path,
        label_dimension=args.label_dimension,
        window_size_sec=args.window_sec,
        fs=args.fs,
        overlap=args.window_overlap,
        median_label=args.median_label,
        window_normalization=args.window_normalization,
        label_threshold_mode=args.label_threshold_mode,
        dataset=dataset_config,
    )

    train_joint_v4_sts(X, y, subjects, trials, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
