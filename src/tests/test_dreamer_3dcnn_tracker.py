import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "dreamer_3dcnn"


def _load_tracker():
    import importlib.util

    spec = importlib.util.spec_from_file_location("dreamer_tracker", EXPERIMENT / "tracker.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_trainer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("dreamer_trainer", EXPERIMENT / "train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tracker():
    return _load_tracker()


@pytest.fixture
def campaign(tracker):
    return tracker.load_json(EXPERIMENT / "campaign.json")


def test_first_suggestion_is_a_cheap_smoke(tracker, campaign):
    stage, config, reason = tracker.suggest(campaign, [], "valence")
    assert stage == "smoke"
    assert config["target"] == "valence"
    assert config["max_folds"] == 2
    assert config["epochs"] == 2
    assert "cheaply" in reason


def test_completed_smoke_advances_to_full_baseline(tracker, campaign):
    _, smoke, _ = tracker.suggest(campaign, [], "arousal")
    rows = [{"target": "arousal", "stage": "smoke", "status": "completed", "config": smoke}]
    stage, config, _ = tracker.suggest(campaign, rows, "arousal")
    assert stage == "full"
    assert config["max_folds"] is None
    assert config["epochs"] == campaign["defaults"]["epochs"]


def test_planned_smoke_blocks_duplicate_spend(tracker, campaign):
    _, smoke, _ = tracker.suggest(campaign, [], "valence")
    rows = [{"target": "valence", "stage": "smoke", "status": "planned", "config": smoke}]
    with pytest.raises(RuntimeError, match="already planned or running"):
        tracker.suggest(campaign, rows, "valence")


def test_sync_rejects_tampered_config(tracker, campaign, tmp_path):
    source = tmp_path / "download" / "run-1"
    source.mkdir(parents=True)
    config = dict(campaign["defaults"])
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps({
            "run_id": "run-1",
            "campaign": campaign["campaign"],
            "config_fingerprint": "wrong",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        tracker.sync_bundles(source, tmp_path / "runs", campaign)


def test_completed_full_run_changes_one_setting(tracker, campaign):
    defaults = dict(campaign["defaults"])
    smoke = {**defaults, **campaign["smoke_overrides"]}
    rows = [
        {"run_id": "smoke", "target": "valence", "stage": "smoke", "status": "completed", "config": smoke, "mean": 0.5, "std": 0.1},
        {"run_id": "base", "target": "valence", "stage": "full", "status": "completed", "config": defaults, "mean": 0.7, "std": 0.1},
    ]
    stage, candidate, reason = tracker.suggest(campaign, rows, "valence")
    changed = [key for key in defaults if candidate[key] != defaults[key]]
    assert stage == "full"
    assert changed == [campaign["search_space"][0]["name"]]
    assert "base" in reason


def test_window_assembly_preserves_trial_boundaries_and_target_labels():
    import numpy as np

    trainer = _load_trainer()
    eeg = np.arange(2 * 2 * 42 * 8, dtype=np.float32).reshape(2, 2, 42, 8)
    ratings = np.array([[[2, 5], [3, 1]], [[4, 2], [1, 4]]], dtype=np.float32)
    features, labels, subjects, trials = trainer.make_windows(
        eeg,
        ratings,
        target="valence",
        threshold=3,
        window_samples=4,
        overlap=0.5,
        normalization="none",
    )
    assert features.shape == (12, 4, 42)
    assert labels.tolist() == [0] * 3 + [1] * 6 + [0] * 3
    assert subjects.tolist() == [0] * 6 + [1] * 6
    assert trials.tolist() == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
    np.testing.assert_array_equal(features[0], eeg[0, 0, :, :4].T)
