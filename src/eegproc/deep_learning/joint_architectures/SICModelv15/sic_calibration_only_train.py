"""Calibration-only SIC search; shared runner, version-specific checkpoint type."""
from eegproc.deep_learning.joint_architectures.sic_common import calibration as _runner
from eegproc.deep_learning.joint_architectures.SICModelv15 import sic_model as _sic_model_registration
from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_args import build_parser as build_sic_parser
from eegproc.deep_learning.joint_architectures.sic_common.calibration import (
    _positive_int,
    _default_calibration_configuration,
    _calibration_grid_from_args,
    _validate_calibration_configuration,
    _read_models_config,
    _validate_args,
    _apply_model_calibration_configuration,
    _validate_loaded_model,
    _load_data,
    _validate_model_coverage,
    _select_model_specs,
    _mean_std_metric_rows,
    _subject_record,
    _aggregate_subject_records,
    _log_user_metrics,
    _run_one_user_calibration,
    _calibration_only_process_main,
    _resolve_outer_workers,
    _RUNTIME_CALIBRATION_KEYS,
    _MODEL_CALIBRATION_KEYS,
    _CALIBRATION_KEYS,
    _TARGET_FROM_FILENAME,
    subject_calibration_cv,
    _cross_val,
    _apply_median_label_ablation,
    _configure_logger,
    _configuration_summary_row,
    _ensure_dir,
    _group_consistent_trial_values,
    _group_windows_into_trials,
    _json_default,
    _json_fingerprint,
    _log_trial_distribution,
    _save_calibration_artifacts,
    _selection_score_from_calibration_results,
    _write_csv,
    _write_json,
    load_sic_training_data,
)


def build_parser():
    return _runner.build_parser(build_sic_parser())


def run_calibration_only_search(args):
    args._sic_model_module = _sic_model_registration.__name__
    return _runner.run_calibration_only_search(args)


def main(argv=None):
    return _runner.main(argv, parser=build_parser(), model_module=_sic_model_registration.__name__)


if __name__ == "__main__":
    raise SystemExit(main())
