"""Regression checks for version isolation and the shared SIC calibration runner."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _environment(tmp_path):
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"),
            "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3",
            "MPLCONFIGDIR": str(tmp_path / "mpl")}


@pytest.mark.parametrize("first", ["meta_learning", "sic_model"])
def test_standalone_model_and_meta_learning_import_order(first, tmp_path):
    pytest.importorskip("tensorflow")
    modules = {
        "meta_learning": "eegproc.deep_learning.domain_generalization.meta_learning",
        "sic_model": "eegproc.deep_learning.joint_architectures.SICModelv15.sic_model",
    }
    order = [modules[first], modules["sic_model" if first == "meta_learning" else "meta_learning"]]
    script = f'''
import importlib, sys
for name in {order!r}:
    importlib.import_module(name)
assert not any('.SICModelv11' in name or '.SICModel.' in name for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script], env=_environment(tmp_path),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("variant", ["SICModelv11", "SICModelv15"])
def test_calibration_only_loads_version_in_fresh_workers(variant, tmp_path):
    pytest.importorskip("tensorflow")
    driver = tmp_path / "calibrate.py"
    driver.write_text('''
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf


def main():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    variant, output = sys.argv[1], Path(sys.argv[2])
    output.mkdir(parents=True)
    package = "eegproc.deep_learning.joint_architectures." + variant
    model_module = importlib.import_module(package + ".sic_model")
    calibration = importlib.import_module(package + ".sic_calibration_only_train")
    model = model_module.build_sic_model(
        (1, 4, 6), adjacency=np.eye(2), n_channels=2, n_bands=3,
        gcn_units=(3,), spectral_gru_units=3, bilstm_units=2,
        classifier_rnn_units=3, gcn_dropout=0., spectral_gru_dropout=0.,
        bilstm_dropout=0., classifier_rnn_dropout=0., decoder_dropout=0.,
        use_subject_adversarial=True, n_subject_classes=2, subject_hidden_units=3,
    )
    if variant == "SICModelv15":
        model.joint_reconstruction_fusion.mix_logit.assign(1.25)
    paths = {}
    for subject in (0, 1, 2):
        path = output / f"target_{subject}.keras"
        model.save(path)
        paths[str(subject)] = str(path)
        restored = tf.keras.models.load_model(path, compile=False)
        assert len(restored.weights) == len(model.weights)
        for saved, loaded in zip(model.get_weights(), restored.get_weights()):
            np.testing.assert_array_equal(saved, loaded)
        if variant == "SICModelv15":
            np.testing.assert_equal(restored.joint_reconstruction_fusion.mix_logit.numpy(), 1.25)
            probe = np.ones((1, 1, 4, 6), dtype="float32")
            np.testing.assert_allclose(restored.reconstruct_joint(probe), model.reconstruct_joint(probe))
    models_config = output / "models.json"
    models_config.write_text(json.dumps({"models": paths}))
    raw = np.random.default_rng(24).normal(size=(3, 4, 6, 4)).astype("float32")
    ratings = np.tile(np.array([1., 5., 1., 5.])[None, :, None], (3, 1, 3))
    eeg_path, labels_path = output / "eeg.npy", output / "labels.npy"
    np.save(eeg_path, raw)
    np.save(labels_path, ratings)
    args = calibration.build_parser().parse_args([
        "--models-config", str(models_config), "--raw-eeg-npy", str(eeg_path),
        "--raw-labels-npy", str(labels_path), "--fs", "4", "--window-sec", "1",
        "--n-channels", "2", "--n-bands", "3", "--calibration-epochs", "1",
        "--calibration-batch-size", "2", "--calibration-level", "2", "1",
        "--n-jobs", "2", "--cpus-per-worker", "1", "--out-dir", str(output),
        "--run-name", "shared_calibration", "--verbose", "0",
        "--target-subjects", "0", "1",
    ])
    result = calibration.run_calibration_only_search(args)
    search = json.loads((Path(result["run_dir"]) / "calibration_search_results.json").read_text())
    assert search["target_subjects"] == [0, 1]
    assert search["n_completed"] == 1 and search["n_failed"] == 0
    assert len(search["best_subject_metrics"]) == 2
    print("SHARED_CALIBRATION_OK", variant)


if __name__ == "__main__":
    main()
''')
    result = subprocess.run([sys.executable, str(driver), variant, str(tmp_path / "output")],
                            env=_environment(tmp_path), capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SHARED_CALIBRATION_OK " + variant in result.stdout
