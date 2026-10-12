"""Integration checks for the research models restored onto the v2 library."""
import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_imported_launchers_parse_and_calibration_help_works():
    base = ROOT / "src/eegproc/deep_learning/joint_architectures"
    for launcher in base.rglob("*.sh"):
        subprocess.run(["bash", "-n", str(launcher)], check=True, capture_output=True)
    for variant in ("SICModelv11", "SICModelv15"):
        launcher = base / variant / "SLURM_scripts/run_sic_calibration_only.sh"
        result = subprocess.run(["bash", str(launcher), "--help"], check=True, capture_output=True, text=True)
        assert "--target-subjects" in result.stdout


def test_upstream_valence_grid_has_a_valid_calibration_selection():
    pytest.importorskip("tensorflow")
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_args import parse_cli, expand_cartesian_grid
    path = ROOT / "src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/full_run_v15_valence.sh"
    text = path.read_text()
    fragment = text.split('MODEL_CONFIG="$(python - <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
    result = subprocess.run([sys.executable, "-c", fragment], check=True, capture_output=True,
                            text=True, env={**os.environ, "TRAINING_SEED": "42"})
    assert "--calibration-selection-shots 3" in text
    args, config = parse_cli(["--training-protocol", "loso_validation", "--hyperparameters-json", result.stdout,
                             "--calibration-level", "3", "6", "--calibration-selection-shots", "3",
                             "--validation-subjects", "0", "--no-early-stopping",
                             "--n-jobs", "2", "--gpus-per-fold", "2", "--gpu-ids", "0", "1", "2", "3"])
    expanded, dimensions = expand_cartesian_grid(config)
    assert len(expanded) == 9
    assert args.calibration_selection_shots == 3
    assert json.loads(result.stdout)["mldg_seed"] == 42


@pytest.mark.parametrize("variant", ["SICModel", "SICModelv11", "SICModelv15"])
def test_source_training_calibration_and_checkpoint(variant, tmp_path):
    pytest.importorskip("tensorflow")
    # Each historical model version registers its own Keras objects. Exercise
    # them in independent interpreters, just like the cluster entry points.
    driver = tmp_path / "smoke.py"
    driver.write_text('''
import importlib
import inspect
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

def main():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    variant, output = sys.argv[1], Path(sys.argv[2])
    module = importlib.import_module("eegproc.deep_learning.joint_architectures." + variant + ".sic_model_train")
    args_module = importlib.import_module("eegproc.deep_learning.joint_architectures." + variant + ".sic_model_args")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(16, 1, 4, 6)).astype(np.float32)
    y = np.tile([0, 1, 0, 1], 4)
    subjects = np.repeat(np.arange(4), 4)
    trials = np.tile(np.arange(4), 4)
    settings = dict(classification_level="trial", n_channels=3, n_bands=2,
        n_classes=2, gcn_units=[3], spectral_gru_units=4, bilstm_units=2,
        classifier_rnn_units=[3], n_classifier_rnn_layers=1,
        classification_hidden_units=[3], gcn_dropout=0.0,
        spectral_gru_dropout=0.0, bilstm_dropout=0.0,
        classifier_rnn_dropout=0.0, subject_hidden_units=3,
        mi_max_observations=24, training_method="erm")
    allowed = set(inspect.signature(module.build_sic_model).parameters)
    settings = {key: value for key, value in settings.items() if key in allowed}
    config = args_module.SICTrainingConfig(output_dir=output, run_name=variant,
        n_channels=3, n_bands=2, classification_level="trial",
        model_config=settings, source_epochs=1, source_batch_size=2,
        validation_subjects=0, early_stopping_patience=None,
        calibration_epochs=1, calibration_batch_size=2,
        calibration_trials=2, calibration_folds=1, calibration_levels=((2, 1),),
        max_subjects=3 if variant == "SICModelv15" else 1,
        n_jobs=2 if variant == "SICModelv15" else 1, cpus_per_worker=1,
        verbose=0)
    result = module.train_sic(x, y, subjects, trials, config)
    directory = Path(result["run_dir"])
    expected = 3 if variant == "SICModelv15" else 1
    assert result["results"]["n_source_model_fits"] == expected
    assert result["results"]["n_calibration_fits"] == expected
    assert (directory / "sic_overall_metrics.json").is_file()
    assert (directory / "sic_calibration_results.json").is_file()
    models = list((directory / "loso_zero_shot_models").glob("*.keras"))
    assert len(models) == expected
    restored = tf.keras.models.load_model(models[0], compile=False)
    probabilities = np.asarray(restored(x[:1], training=False))
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    if variant != "SICModel":
        calibration = importlib.import_module("eegproc.deep_learning.joint_architectures." + variant + ".sic_calibration_only_train")
        assert calibration.subject_calibration_cv is not None
        assert callable(calibration._cross_val._run_spawned_fold_pool)
    print("SIC_IMPORT_OK", variant)

if __name__ == "__main__":
    main()
''')
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "-1",
           "TF_CPP_MIN_LOG_LEVEL": "3", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = subprocess.run([sys.executable, str(driver), variant, str(tmp_path / "output")],
                            env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SIC_IMPORT_OK " + variant in result.stdout
