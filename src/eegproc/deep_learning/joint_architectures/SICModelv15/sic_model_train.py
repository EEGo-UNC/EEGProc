"""Training and reporting entry point for SIC: Subject Invariant Calibrator.

Protocol
--------
For every hyperparameter configuration and target subject:
  1. pretrain one source model on all other subjects;
  2. evaluate the untouched source model on all target trials (0 calibration);
  3. for every requested (shots, folds) calibration pair, construct seeded
     target-only calibration/evaluation splits;
  4. for each split, restore the same source weights, evaluate paired zero-shot,
     fine-tune only the configured recurrent/VC classifier suffix, and
     re-evaluate;
  5. aggregate folds within each subject and shot level, then across subjects.

The expensive source model is fit once per target subject. Every calibration
fit uses a fresh optimizer and restored source weights. Grid ranking
can use either the zero-shot LOSO aggregate or the post-calibration aggregate;
calibration is always run and reported regardless of the selection level.

``remove_median_label`` is a data hyperparameter. When true, every trial whose
original target rating equals ``median_label`` is removed in full before SIC
splitting, including every window from that trial. It may be fixed or searched
through the same JSON Cartesian grid and is never forwarded to the model.

Source optimization is selected with the model hyperparameter
``training_method``: ``"erm"`` for ordinary joint training, ``"vrex"`` for
subject-risk variance regularization, or ``"mldg"`` for first-order
subject-episodic meta-learning.  MLDG defaults to four rotating virtual-unseen
subjects and every remaining source subject in meta-train.  Each episode samples
complete trials and applies one persistent outer update.

SIC uses independent parallel deterministic branches. The complete GCN-GRU and
raw-EEG BiLSTM feature vectors are concatenated without an encoder projection or
bottleneck. In trial mode, every encoded timestep from every ordered window is
passed to a GRU/BiGRU sequence summarizer; no temporal or cross-window mean is
taken. That recurrent state goes directly to the sole VariationalClassifier
logits head and to the subject adversary. Optional deterministic decoders
reconstruct the original EEG independently from the GCN-GRU and BiLSTM feature
sequences. When both branches are active, a trainable convex scalar also creates
one joint reconstruction in the original EEG space. There is no dense classifier
stack, encoder VAE, or encoder KL loss.

CLI construction, JSON/grid decoding, validation, and conversion to the runtime
configuration live in ``sic_model_args.py`` so this module stays focused on
data preparation, training, evaluation, and reporting.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from functools import partial
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from eegproc.deep_learning.cross_validation import subject_calibration_cv
from eegproc.deep_learning.joint_architectures.joint_models_data import get_dataset_config
from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import SIC_BUILDER_API_VERSION, build_sic_model
from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_args import (
    SIC_CLASSIFICATION_METRICS,
    SIC_DATA_HYPERPARAMETERS,
    SIC_SELECTION_OVERALL_KEYS,
    SICTrainingConfig,
    calibration_levels_from_args,
    calibration_plan as _calibration_plan,
    configuration_training_method as _configuration_training_method,
    expand_cartesian_grid,
    metric_mode as _metric_mode,
    parse_cli,
    selected_calibration_shots as _selected_calibration_shots,
    selection_score_source as _selection_score_source,
    split_data_hyperparameters as _split_data_hyperparameters,
    training_config_from_args,
)
from eegproc.deep_learning.joint_architectures.sic_common.training import (
    _json_fingerprint,
    _ensure_dir,
    _configure_logger,
    _json_default,
    _write_json,
    _write_csv,
    _class_ids,
    _n_classes,
    _flatten_grouped_trials_to_windows,
    _group_windows_into_trials,
    _normalize_each_window,
    _original_target_window_ratings,
    _subject_median_window_labels,
    _group_consistent_trial_values,
    _reporting_rating_values,
    _rating_count_key,
    _trial_distribution_summary,
    _log_trial_distribution,
    _apply_median_label_ablation,
    load_sic_training_data,
    _subject_summary_rows,
    _calibration_fold_rows,
    _selection_score_from_calibration_results,
    _configuration_summary_row,
    _save_calibration_artifacts,
    _source_checkpoint_settings,
)


def _run_subject_calibration_configuration(
    *,
    builder,
    X,
    y,
    subjects,
    trials,
    model_config: dict,
    config: SICTrainingConfig,
    results_dir: Path,
) -> dict:
    """Run the common LOSO + three-fold calibration evaluator once."""
    source_monitor, source_monitor_mode = _source_checkpoint_settings(config)
    return subject_calibration_cv(
        model_builder_function=builder,
        feature_array=X,
        label_array=y,
        subject_id_array=subjects,
        trial_id_array=trials,
        fixed_config=model_config,
        training_seed=config.seed,
        deterministic_training=config.deterministic_training,
        source_epochs=config.source_epochs,
        source_batch_size=config.source_batch_size,
        calibration_epochs=config.calibration_epochs,
        calibration_batch_size=config.calibration_batch_size,
        calibration_trials=config.calibration_trials,
        calibration_folds=config.calibration_folds,
        calibration_levels=_calibration_plan(config),
        calibration_selection_shots=_selected_calibration_shots(config),
        calibration_learning_rate=config.calibration_learning_rate,
        calibration_optimizer=config.calibration_optimizer,
        calibration_weight_decay=config.calibration_weight_decay,
        calibration_seed=config.calibration_seed,
        stratify_calibration=config.stratify_calibration,
        validation_subjects_per_fold=config.validation_subjects,
        validation_seed=config.validation_seed,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_monitor=source_monitor,
        early_stopping_mode=source_monitor_mode,
        restore_best_weights=config.restore_best_weights,
        evaluation_level=config.classification_level,
        metrics=SIC_CLASSIFICATION_METRICS,
        ece_bins=config.ece_bins,
        decision_threshold=config.decision_threshold,
        prediction_diagnostics=config.prediction_diagnostics,
        prediction_diagnostics_metric=config.prediction_diagnostics_metric,
        prediction_diagnostics_every_n_epochs=(
            config.prediction_diagnostics_every_n_epochs
        ),
        prediction_diagnostics_max_samples=(
            config.prediction_diagnostics_max_samples
        ),
        prediction_diagnostics_threshold_tolerance=(
            config.prediction_diagnostics_threshold_tolerance
        ),
        prediction_diagnostics_seed=config.prediction_diagnostics_seed,
        log_predictions=True,
        source_use_class_weight=config.source_use_class_weight,
        calibration_use_class_weight=config.calibration_use_class_weight,
        source_fit_kwargs={"callbacks": [tf.keras.callbacks.TerminateOnNaN()]},
        calibration_fit_kwargs={
            "callbacks": [tf.keras.callbacks.TerminateOnNaN()]
        },
        verbose=config.verbose,
        n_jobs=config.n_jobs,
        gpu_ids=config.gpu_ids,
        gpus_per_fold=config.gpus_per_fold,
        cpus_per_worker=config.cpus_per_worker,
        max_subjects=config.max_subjects,
        target_subjects=config.target_subjects,
        source_model_output_dir=results_dir / "loso_zero_shot_models",
    )


def train_sic_loso_validation(
    feature_array,
    label_array,
    subject_id_array,
    trial_id_array,
    config: SICTrainingConfig,
    original_rating_array=None,
):
    """Full Cartesian SIC search with calibration nested inside every LOSO fold.

    Every configuration runs the same target-subject loop. The source model is
    evaluated zero-shot, then restored independently for each of the three
    calibration folds. ``hyperparameter_selection_level`` changes only which
    aggregate ranks configurations; it never disables calibration or its
    reports.
    """
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
    original_ratings = (
        None
        if original_rating_array is None
        else np.asarray(original_rating_array, dtype=np.float64).reshape(-1)
    )

    expected_rank = 4 if config.classification_level == "trial" else 3
    if X.ndim != expected_rank:
        raise ValueError(
            f"SIC {config.classification_level} mode expects rank {expected_rank}; "
            f"got {X.shape}."
        )
    if X.shape[-1] != config.n_channels * config.n_bands:
        raise ValueError(
            f"features={X.shape[-1]} but {config.n_channels}*{config.n_bands}="
            f"{config.n_channels * config.n_bands}."
        )

    model_config = dict(config.model_config)
    model_config.setdefault("classification_level", config.classification_level)
    model_config.setdefault("n_classes", _n_classes(y))
    model_config.setdefault("n_channels", config.n_channels)
    model_config.setdefault("n_bands", config.n_bands)

    # ``training_method`` is the primary selector. Preserve legacy JSON files
    # that still express V-REx with use_vrex=true.
    if "training_method" not in model_config and "use_vrex" not in model_config:
        model_config["training_method"] = "erm"

    grid_configurations, grid_dimensions = expand_cartesian_grid(model_config)
    if not grid_configurations:  # Defensive: product() always yields at least one.
        raise ValueError("Hyperparameter grid produced no configurations.")
    grid_records = [
        {"configuration_id": index, "hyperparameters": candidate}
        for index, candidate in enumerate(grid_configurations, start=1)
    ]

    _write_json(run_dir / "training_config.json", asdict(config))
    _write_json(run_dir / "model_config.json", model_config)
    _write_json(
        run_dir / "hyperparameter_grid.json",
        {
            "search_type": "full_cartesian_product",
            "hyperparameter_selection_level": (
                config.hyperparameter_selection_level
            ),
            "evaluation_level": config.classification_level,
            "selection_metric": config.selection_metric,
            "maximize_metric": _metric_mode(config.selection_metric) == "max",
            "selection_score_source": _selection_score_source(config),
            "calibration_always_runs": True,
            "calibration_plan": [
                {"shots": shots, "folds": folds}
                for shots, folds in _calibration_plan(config)
            ],
            "calibration_selection_shots": _selected_calibration_shots(config),
            "data_hyperparameters": sorted(SIC_DATA_HYPERPARAMETERS),
            "calibration_aggregation": {
                "within_subject": "mean of each shot level's calibration-fold metrics",
                "across_subjects": "mean of subject-level aggregates",
            },
            "dimensions": grid_dimensions,
            "n_configurations": len(grid_configurations),
            "configurations": grid_records,
        },
    )

    logger.info("Model: SIC (Subject Invariant Calibrator)")
    logger.info(
        "Training protocol: configuration -> LOSO subject -> calibration plan %s",
        _calibration_plan(config),
    )
    logger.info("Classification level: %s", config.classification_level)
    if config.classification_level == "trial":
        logger.info(
            "Trial classifier grid: type=%s units=%s layers=%s dropout=%s; "
            "ordered window embeddings are recurrently summarized without "
            "cross-window averaging",
            model_config.get("classifier_rnn_type", "bigru"),
            model_config.get("classifier_rnn_units", 128),
            model_config.get("n_classifier_rnn_layers", "inferred"),
            model_config.get("classifier_rnn_dropout", 0.20),
        )
    logger.info(
        "Decoder grid: enabled=%s independent_branches=true "
        "joint_fusion=when_both_branches_active "
        "reconstruction_loss_weight=%s auxiliary_weight=%s initial_alpha=%s "
        "dropout=%s; frozen during calibration",
        model_config.get("use_decoder", True),
        model_config.get("reconstruction_loss_weight", 0.10),
        model_config.get("joint_reconstruction_auxiliary_weight", 0.25),
        model_config.get("joint_reconstruction_initial_alpha", 0.5),
        model_config.get("decoder_dropout", 0.10),
    )
    logger.info(
        "Each target: one source fit + independent fine-tunes for %s",
        _calibration_plan(config),
    )
    logger.info(
        "Hyperparameter-grid selection: level=%s metric=%s source=%s",
        config.hyperparameter_selection_level,
        config.selection_metric,
        _selection_score_source(config),
    )
    logger.info("Calibration runs and is saved for every configuration.")
    logger.info(
        "Source checkpoint selection: %d source-validation subjects, "
        "monitor=%s mode=%s patience=%s restore_best=%s",
        config.validation_subjects,
        *_source_checkpoint_settings(config),
        config.early_stopping_patience,
        config.restore_best_weights,
    )
    logger.info(
        "LOSO target data never enters source fitting, source checkpoint "
        "selection, or MI-adjacency estimation."
    )
    logger.info(
        "Full Cartesian grid: dimensions=%s total_configurations=%d",
        grid_dimensions or {"fixed_configuration": 1},
        len(grid_configurations),
    )
    for index, candidate in enumerate(grid_configurations, start=1):
        logger.info(
            "Grid configuration %d/%d: %s",
            index,
            len(grid_configurations),
            _json_fingerprint(candidate),
        )

    progress_path = run_dir / "hyperparameter_search_progress.json"
    completed_summaries: list[dict] = []
    failed_configurations: list[dict] = []

    for index, candidate in enumerate(grid_configurations, start=1):
        configuration_dir = _ensure_dir(run_dir / f"configuration_{index:04d}")
        logger.info(
            "Starting configuration %d/%d: %s",
            index,
            len(grid_configurations),
            _json_fingerprint(candidate),
        )
        _write_json(configuration_dir / "model_config.json", candidate)
        try:
            if config.seed is not None:
                tf.keras.utils.set_random_seed(config.seed)
                np.random.seed(config.seed)
            builder_config, data_config = _split_data_hyperparameters(candidate)
            (
                candidate_X,
                candidate_y,
                candidate_subjects,
                candidate_trials,
                data_summary,
            ) = _apply_median_label_ablation(
                X,
                y,
                subjects,
                trials,
                original_ratings,
                remove_median_label=data_config["remove_median_label"],
                median_label=config.median_label,
            )
            logger.info(
                "Configuration %d dataset: remove_median_label=%s "
                "median_label=%s removed_trials=%d removed_samples=%d "
                "retained_trials=%d retained_samples=%d",
                index,
                data_summary["remove_median_label"],
                config.median_label,
                data_summary["removed_trials"],
                data_summary["removed_samples"],
                data_summary["retained_trials"],
                data_summary.get("retained_samples", len(candidate_X)),
            )
            _log_trial_distribution(
                logger,
                data_summary,
                context=f"Configuration {index}",
            )
            _write_json(configuration_dir / "dataset_ablation.json", data_summary)
            builder = partial(
                build_sic_model,
                input_shape=tuple(candidate_X.shape[1:]),
            )
            results = _run_subject_calibration_configuration(
                builder=builder,
                X=candidate_X,
                y=candidate_y,
                subjects=candidate_subjects,
                trials=candidate_trials,
                model_config=builder_config,
                config=config,
                results_dir=configuration_dir,
            )
            _save_calibration_artifacts(
                configuration_dir,
                model_config=candidate,
                results=results,
            )
            overall = dict(results.get("overall", {}))
            selection_score = _selection_score_from_calibration_results(
                results,
                selection_level=config.hyperparameter_selection_level,
                selection_metric=config.selection_metric,
                calibration_selection_shots=_selected_calibration_shots(config),
            )
            summary = {
                "configuration_id": index,
                "status": "completed",
                "configuration_dir": str(configuration_dir),
                "hyperparameters": candidate,
                "selection_level": config.hyperparameter_selection_level,
                "selection_metric": config.selection_metric,
                "selection_score": selection_score,
                "dataset_ablation": data_summary,
                "zero_shot_all_trials_mean_scores": overall.get(
                    "zero_shot_all_trials_mean_scores", {}
                ),
                "zero_shot_all_trials_std_scores": overall.get(
                    "zero_shot_all_trials_std_scores", {}
                ),
                "paired_zero_shot_mean_scores": overall.get(
                    "paired_zero_shot_mean_scores", {}
                ),
                "calibrated_mean_scores": overall.get(
                    "calibrated_mean_scores", {}
                ),
                "calibrated_std_scores": overall.get(
                    "calibrated_std_scores", {}
                ),
                "calibration_level_metrics": overall.get(
                    "calibration_levels", {}
                ),
                "delta_mean_scores": overall.get("delta_mean_scores", {}),
                "delta_std_scores": overall.get("delta_std_scores", {}),
            }
            completed_summaries.append(summary)
            logger.info(
                "Completed configuration %d/%d | zero-shot %s=%s | "
                "calibration %s=%s | selected score=%s",
                index,
                len(grid_configurations),
                config.selection_metric,
                summary["zero_shot_all_trials_mean_scores"].get(
                    config.selection_metric
                ),
                config.selection_metric,
                summary["calibrated_mean_scores"].get(config.selection_metric),
                selection_score,
            )
        except Exception as exc:
            failed = {
                "configuration_id": index,
                "status": "failed",
                "configuration_dir": str(configuration_dir),
                "hyperparameters": candidate,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failed_configurations.append(failed)
            _write_json(configuration_dir / "failure.json", failed)
            _write_json(
                progress_path,
                {
                    "completed": completed_summaries,
                    "failed": failed_configurations,
                    "remaining_configuration_ids": list(
                        range(index + 1, len(grid_configurations) + 1)
                    ),
                },
            )
            logger.exception("Configuration %d failed; progress was saved.", index)
            raise
        finally:
            tf.keras.backend.clear_session()

        _write_json(
            progress_path,
            {
                "completed": completed_summaries,
                "failed": failed_configurations,
                "remaining_configuration_ids": list(
                    range(index + 1, len(grid_configurations) + 1)
                ),
            },
        )

    maximize_metric = _metric_mode(config.selection_metric) == "max"
    ranked_summaries = sorted(
        completed_summaries,
        key=lambda row: (
            -float(row["selection_score"])
            if maximize_metric
            else float(row["selection_score"]),
            row["configuration_id"],
        ),
    )
    for rank, summary in enumerate(ranked_summaries, start=1):
        summary["rank"] = rank
    best = ranked_summaries[0]
    search_results = {
        "search_type": "full_cartesian_product_with_nested_subject_calibration",
        "hyperparameter_selection_level": config.hyperparameter_selection_level,
        "selection_metric": config.selection_metric,
        "maximize_metric": maximize_metric,
        "selection_score_source": _selection_score_source(config),
        "calibration_always_runs": True,
        "calibration_plan": [
            {"shots": shots, "folds": folds}
            for shots, folds in _calibration_plan(config)
        ],
        "calibration_selection_shots": _selected_calibration_shots(config),
        "data_hyperparameters": sorted(SIC_DATA_HYPERPARAMETERS),
        "calibration_aggregation": {
            "within_subject": "mean of each shot level's calibration-fold metrics",
            "across_subjects": "mean of subject-level aggregates",
        },
        "n_configurations": len(grid_configurations),
        "best_configuration_id": best["configuration_id"],
        "best_score": best["selection_score"],
        "best_hyperparameters": best["hyperparameters"],
        "ranked_configurations": ranked_summaries,
    }
    _write_json(run_dir / "hyperparameter_search_results.json", search_results)
    _write_json(
        run_dir / "best_hyperparameters.json",
        {
            "configuration_id": best["configuration_id"],
            "selection_level": config.hyperparameter_selection_level,
            "selection_metric": config.selection_metric,
            "selection_score": best["selection_score"],
            "hyperparameters": best["hyperparameters"],
            "configuration_dir": best["configuration_dir"],
        },
    )
    _write_csv(
        run_dir / "hyperparameter_search_summary.csv",
        [_configuration_summary_row(row) for row in ranked_summaries],
    )
    logger.info(
        "Best configuration: id=%d %s/%s=%s hyperparameters=%s",
        best["configuration_id"],
        config.hyperparameter_selection_level,
        config.selection_metric,
        best["selection_score"],
        _json_fingerprint(best["hyperparameters"]),
    )
    logger.info("Saved nested LOSO/calibration grid artifacts to %s", run_dir)
    return {"run_dir": str(run_dir), "results": search_results}


def train_sic(
    feature_array,
    label_array,
    subject_id_array,
    trial_id_array,
    config: SICTrainingConfig,
    original_rating_array=None,
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
    original_ratings = (
        None
        if original_rating_array is None
        else np.asarray(original_rating_array, dtype=np.float64).reshape(-1)
    )
    expected_rank = 4 if config.classification_level == "trial" else 3
    if X.ndim != expected_rank:
        raise ValueError(
            f"SIC {config.classification_level} mode expects rank {expected_rank}; "
            f"got {X.shape}."
        )
    if X.shape[-1] != config.n_channels * config.n_bands:
        raise ValueError(
            f"features={X.shape[-1]} but {config.n_channels}*{config.n_bands}="
            f"{config.n_channels * config.n_bands}."
        )

    hyperparameters = dict(config.model_config)
    hyperparameters.setdefault("classification_level", config.classification_level)
    hyperparameters.setdefault("n_classes", _n_classes(y))
    hyperparameters.setdefault("n_channels", config.n_channels)
    hyperparameters.setdefault("n_bands", config.n_bands)
    if (
        "training_method" not in hyperparameters
        and "use_vrex" not in hyperparameters
    ):
        hyperparameters["training_method"] = "erm"

    fixed_configurations, grid_dimensions = expand_cartesian_grid(hyperparameters)
    if grid_dimensions:
        raise ValueError(
            "subject_calibration accepts one fixed configuration. Use "
            "--training-protocol loso_validation for the nested LOSO/calibration "
            "Cartesian search; grid axes were "
            f"{sorted(grid_dimensions)}."
        )
    hyperparameters = fixed_configurations[0]
    model_config, data_config = _split_data_hyperparameters(hyperparameters)
    X, y, subjects, trials, data_summary = _apply_median_label_ablation(
        X,
        y,
        subjects,
        trials,
        original_ratings,
        remove_median_label=data_config["remove_median_label"],
        median_label=config.median_label,
    )

    _write_json(run_dir / "training_config.json", asdict(config))
    _write_json(run_dir / "model_config.json", hyperparameters)
    _write_json(run_dir / "dataset_ablation.json", data_summary)

    logger.info("Model: SIC (Subject Invariant Calibrator)")
    logger.info("SIC builder API version: %s", SIC_BUILDER_API_VERSION)
    logger.info("Input shape: %s", X.shape)
    if config.classification_level == "trial":
        logger.info(
            "Trial classifier: type=%s units=%s layers=%s dropout=%s; "
            "ordered window embeddings are recurrently summarized without "
            "cross-window averaging",
            model_config.get("classifier_rnn_type", "bigru"),
            model_config.get("classifier_rnn_units", 128),
            model_config.get("n_classifier_rnn_layers", "inferred"),
            model_config.get("classifier_rnn_dropout", 0.20),
        )
    logger.info(
        "Decoder: enabled=%s independent GCN-GRU/BiLSTM plus convex joint "
        "reconstruction when both are active; reconstruction_loss_weight=%s "
        "auxiliary_weight=%s "
        "initial_alpha=%s dropout=%s; frozen during calibration",
        model_config.get("use_decoder", True),
        model_config.get("reconstruction_loss_weight", 0.10),
        model_config.get("joint_reconstruction_auxiliary_weight", 0.25),
        model_config.get("joint_reconstruction_initial_alpha", 0.5),
        model_config.get("decoder_dropout", 0.10),
    )
    logger.info(
        "Dataset: remove_median_label=%s median_label=%s removed_trials=%d "
        "removed_samples=%d retained_trials=%d retained_samples=%d",
        data_summary["remove_median_label"],
        config.median_label,
        data_summary["removed_trials"],
        data_summary["removed_samples"],
        data_summary["retained_trials"],
        data_summary.get("retained_samples", len(X)),
    )
    _log_trial_distribution(logger, data_summary, context="Run dataset")
    logger.info(
        "Protocol: strict LOSO followed by calibration plan %s",
        _calibration_plan(config),
    )
    logger.info(
        "Calibration head depth: %s",
        model_config.get("calibration_unfreeze_layers", 1),
    )
    logger.info(
        "Source optimization: method=%s; V-REx penalty_weight=%s; "
        "MLDG A=%s B=%s trials_per_subject=%s inner_lr=%s beta=%s; "
        "subject_adversarial=%s",
        _configuration_training_method(model_config),
        model_config.get("vrex_penalty_weight", 1.0),
        model_config.get("mldg_meta_train_subjects", "all remaining"),
        model_config.get("mldg_meta_test_subjects", 4),
        model_config.get("mldg_trials_per_subject", 1),
        model_config.get("mldg_inner_learning_rate", 1e-4),
        model_config.get("mldg_meta_test_weight", 1.0),
        model_config.get("use_subject_adversarial", True),
    )

    builder = partial(build_sic_model, input_shape=tuple(X.shape[1:]))

    results = _run_subject_calibration_configuration(
        builder=builder,
        X=X,
        y=y,
        subjects=subjects,
        trials=trials,
        model_config=model_config,
        config=config,
        results_dir=run_dir,
    )

    _save_calibration_artifacts(
        run_dir,
        model_config=hyperparameters,
        results=results,
    )

    overall = results.get("overall", {})
    zero_accuracy = overall.get("zero_shot_all_trials_mean_scores", {}).get(
        "accuracy"
    )
    logger.info("Final 0-shot mean accuracy: %s", zero_accuracy)
    for shots, level in overall.get("calibration_levels", {}).items():
        logger.info(
            "Final %s-shot mean accuracy: %s",
            shots,
            level.get("calibrated_mean_scores", {}).get("accuracy"),
        )
    logger.info("Saved SIC artifacts to %s", run_dir)
    return {"run_dir": str(run_dir), "results": results}


def main(argv=None):
    args, model_config = parse_cli(argv)
    calibration_levels = calibration_levels_from_args(args)
    calibration_selection_shots = (
        max(shots for shots, _ in calibration_levels)
        if args.calibration_selection_shots is None
        else int(args.calibration_selection_shots)
    )

    if args.print_grid_only:
        configurations, dimensions = expand_cartesian_grid(model_config)
        print(
            json.dumps(
                {
                    "search_type": "full_cartesian_product",
                    "hyperparameter_selection_level": (
                        args.hyperparameter_selection_level
                    ),
                    "evaluation_level": args.classification_level,
                    "selection_metric": args.selection_metric,
                    "maximize_metric": _metric_mode(args.selection_metric) == "max",
                    "selection_score_source": (
                        "overall.zero_shot_all_trials_mean_scores"
                        if args.hyperparameter_selection_level == "losocv"
                        else (
                            "overall.calibration_levels."
                            f"{calibration_selection_shots}.calibrated_mean_scores"
                        )
                    ),
                    "calibration_always_runs": True,
                    "calibration_plan": [
                        {"shots": shots, "folds": folds}
                        for shots, folds in calibration_levels
                    ],
                    "calibration_selection_shots": calibration_selection_shots,
                    "data_hyperparameters": sorted(SIC_DATA_HYPERPARAMETERS),
                    "dimensions": dimensions,
                    "n_configurations": len(configurations),
                    "configurations": [
                        {"configuration_id": index, "hyperparameters": candidate}
                        for index, candidate in enumerate(configurations, start=1)
                    ],
                },
                indent=2,
                default=_json_default,
            )
        )
        return 0

    dataset_config = get_dataset_config(args.dataset)
    eeg_path = args.raw_eeg_npy or dataset_config.eeg_path
    labels_path = args.raw_labels_npy or dataset_config.labels_path

    loaded = load_sic_training_data(
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
        # Raw ratings are always loaded so every run can report trial counts
        # for ratings 1--5 and majority-class baselines before training.
        return_original_ratings=True,
    )
    X, y, subjects, trials, original_ratings = loaded

    if args.classification_level == "trial":
        if original_ratings is not None:
            original_ratings = _group_consistent_trial_values(
                original_ratings,
                subjects,
                trials,
                value_name="original target rating",
            )
        X, y, subjects, trials = _group_windows_into_trials(X, y, subjects, trials)

    config = training_config_from_args(
        args,
        model_config,
        calibration_levels,
    )
    if args.training_protocol == "loso_validation":
        train_sic_loso_validation(
            X,
            y,
            subjects,
            trials,
            config,
            original_rating_array=original_ratings,
        )
    else:
        train_sic(
            X,
            y,
            subjects,
            trials,
            config,
            original_rating_array=original_ratings,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
