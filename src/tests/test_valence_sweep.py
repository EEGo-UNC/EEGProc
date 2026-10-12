"""Oracle selection, fixed task manifests, timeout safety and real checkpoint I/O."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/sic_v15_valence_sweep.py"
spec = importlib.util.spec_from_file_location("valence_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def row(subject, epoch, score):
    return {"target_subject": subject, "epoch": epoch, "balanced_accuracy": score,
            "decision_threshold": .5, "checkpoint": f"subject_{subject:02d}/epoch_{epoch:03d}.keras"}


def test_common_epoch_not_mean_of_individual_peaks():
    rows = [row(0, 1, .9), row(1, 1, .4), row(0, 2, .5), row(1, 2, .9)]
    result = sweep.select_oracle(rows, [0, 1], 2)
    assert result["best_epoch"] == 2
    assert result["selection_score"] == pytest.approx(.7)
    assert set(result["selected_models"].values()) == {rows[2]["checkpoint"], rows[3]["checkpoint"]}


def test_equal_subject_weight_and_earliest_tie():
    rows = [row(0, 1, 1), row(1, 1, 0), row(0, 2, 0), row(1, 2, 1)]
    rows[0]["n_trials"] = 1000
    assert sweep.select_oracle(rows, [0, 1], 2)["best_epoch"] == 1


def test_incomplete_task_cannot_win():
    result = sweep.select_oracle([row(0, 1, 1), row(1, 1, 1), row(0, 2, 1)], [0, 1], 2)
    assert result["status"] == "incomplete"
    assert result["missing_subject_epochs"] == [[1, 2]]
    assert "selection_score" not in result
    assert "selected_models" not in result


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 1.01])
def test_reject_bad_metrics(bad):
    with pytest.raises(ValueError):
        sweep.select_oracle([row(0, 1, bad)], [0], 1)


def test_reject_duplicates_and_threshold_search():
    with pytest.raises(ValueError, match="Duplicate"):
        sweep.select_oracle([row(0, 1, .5)] * 2, [0], 1)
    bad = row(0, 1, .5)
    bad["decision_threshold"] = .6
    with pytest.raises(ValueError, match="Invalid"):
        sweep.select_oracle([bad], [0], 1)


def test_manifest_is_fixed_and_complete():
    configs = sweep.configurations()
    assert len(configs) == 16
    assert configs[0]["name"] == "baseline"
    assert configs[0]["hyperparameters"]["gcn_units"] == [128, 64]
    for record in configs:
        assert record["hyperparameters"]["mldg_steps_per_epoch"] == 10
        assert not any(isinstance(v, dict) for v in record["hyperparameters"].values())


def test_preprocessing_matches_full_run_windows(tmp_path):
    rng = np.random.default_rng(2)
    eeg = rng.normal(size=(2, 2, 42, 270)).astype(np.float64)
    ratings = np.array([[[2, 4], [3, 1]], [[4, 1], [1, 4]]])
    np.save(tmp_path / "eeg.npy", eeg)
    np.save(tmp_path / "ratings.npy", ratings)
    sweep.prepare_data(tmp_path / "eeg.npy", tmp_path / "ratings.npy", tmp_path / "cache", 2)
    x = np.load(tmp_path / "cache/features.npy")
    reference = []
    for s in range(2):
        for t in range(2):
            windows = np.stack([eeg[s, t, :, i:i+128].T for i in (0, 128)]).astype(np.float32)
            rms = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=(1, 2), keepdims=True))
            reference.append((windows.astype(np.float64) / np.maximum(rms, 1e-6)).astype(np.float32))
    np.testing.assert_array_equal(x, reference)
    np.testing.assert_array_equal(np.load(tmp_path / "cache/labels.npy"), [0, 1, 1, 0])
    np.testing.assert_array_equal(np.load(tmp_path / "cache/subjects.npy"), [0, 0, 1, 1])
    np.testing.assert_array_equal(np.load(tmp_path / "cache/trials.npy"), [0, 1, 0, 1])


def test_dry_run_submits_nothing():
    result = subprocess.run(["bash", str(ROOT / "scripts/full_run_v15_valence_sweep.sh")],
                            env={**os.environ, "DRY_RUN": "1", "SUBMIT_PYTHON": sys.executable},
                            capture_output=True, text=True, check=True)
    assert "--array=0-15%2" in result.stdout
    assert "06:00:00 per task" in result.stdout


def test_rank_separates_budgets_and_excludes_incomplete(tmp_path):
    for index, (epochs, status, score) in enumerate([(6, "completed", .7), (3, "completed", .8), (6, "incomplete", 1)]):
        sweep.write_json(tmp_path / str(index) / "oracle_summary.json",
                         {"status": status, "source_epochs": epochs, "target_subjects": [0, 1],
                          "configuration": {"name": str(index)}, "best_epoch": 1, "selection_score": score})
    result = sweep.rank(tmp_path)
    assert len(result["leaderboards"]) == 2
    assert len(result["excluded_incomplete_or_failed"]) == 1


def test_expired_budget_never_starts_a_fold(tmp_path, monkeypatch):
    from argparse import Namespace
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setattr(sweep, "prepare_data", lambda *args: [0, 1])
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **kw: pytest.fail("Expired task started training"))
    ticks = iter([0, 100, 100, 100, 100, 100, 100, 100])
    monkeypatch.setattr(sweep.time, "time", lambda: next(ticks))
    output = tmp_path / "run"
    args = Namespace(configs=sweep.DEFAULT_CONFIGS, config_index=0, source_epochs=1,
                     time_budget_seconds=1, output_dir=output, eeg=tmp_path / "eeg.npy",
                     labels=tmp_path / "labels.npy", target_subjects=None)
    assert sweep.run(args) == 2
    assert json.loads((output / "oracle_summary.json").read_text())["status"] == "incomplete"


def test_worker_pairs_preserve_slurm_uuid_masks(tmp_path, monkeypatch):
    from argparse import Namespace
    masks = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setattr(sweep, "prepare_data", lambda *args: [0, 1])
    class FakeProcess:
        def __init__(self, command, *, env, stdout, stderr):
            masks.append(env["CUDA_VISIBLE_DEVICES"])
            subject = int(command[-1])
            directory = Path(command[command.index("--spec") + 1]).parent
            record = row(subject, 1, .75)
            path = directory / record["checkpoint"]
            path.parent.mkdir()
            path.touch()
            sweep.write_json(path.with_suffix(".json"), record)
        def poll(self):
            return 0
    monkeypatch.setattr(sweep.subprocess, "Popen", FakeProcess)
    args = Namespace(configs=sweep.DEFAULT_CONFIGS, config_index=0, source_epochs=1,
                     time_budget_seconds=100, output_dir=tmp_path / "run", eeg=tmp_path / "eeg.npy",
                     labels=tmp_path / "labels.npy", target_subjects=None)
    assert sweep.run(args) == 0
    assert masks == ["GPU-a,GPU-b", "GPU-c,GPU-d"]


def test_real_cpu_worker_saves_reloadable_oracle_model(tmp_path):
    pytest.importorskip("tensorflow")
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(4)
    features = rng.normal(size=(8, 1, 4, 6)).astype(np.float32)
    for name, value in {"features": features, "labels": np.tile([0, 1], 4),
                        "subjects": np.repeat(np.arange(4), 2), "trials": np.tile([0, 1], 4)}.items():
        np.save(data / f"{name}.npy", value)
    configuration = sweep.configurations()[0]
    configuration["hyperparameters"].update(n_channels=3, n_bands=2, gcn_units=[3], spectral_gru_units=4,
        bilstm_units=2, classifier_rnn_units=[3], n_classifier_rnn_layers=1,
        mldg_meta_train_subjects=2, mldg_meta_test_subjects=1, mldg_trials_per_subject=2,
        mldg_steps_per_epoch=1, mi_max_observations=24, subject_hidden_units=3)
    sweep.write_json(tmp_path / "task.json", {"configuration": configuration, "target_subjects": [0],
        "source_epochs": 2, "seed": 42, "gpus_per_fold": 0, "deadline": time.time() + 180})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3",
           "MPLCONFIGDIR": str(tmp_path / "mpl"), "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, str(SCRIPT), "worker", "--spec", str(tmp_path / "task.json"),
                             "--subject", "0"], env=env, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    result = sweep.summarize(tmp_path)
    assert result["status"] == "completed"
    import tensorflow as tf
    from eegproc.deep_learning.joint_architectures.SICModelv15 import sic_model  # registers Keras classes
    from eegproc.deep_learning.cross_validation.probabilities import _predict_probabilities
    selected = tmp_path / result["selected_models"]["0"]
    restored = tf.keras.models.load_model(selected, compile=False)
    saved = json.loads(selected.with_suffix(".json").read_text())
    probabilities = _predict_probabilities(restored, features[:2], batch_size=1)
    np.testing.assert_allclose(probabilities, saved["probabilities"], rtol=1e-5, atol=1e-6)
