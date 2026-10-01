"""Exercise the complete tutorial with synthetic DREAMER-shaped recordings."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_dreamer_example_uses_the_matching_held_out_model(tmp_path, monkeypatch):
    tf = pytest.importorskip("tensorflow", reason="example needs eegproc[deep-learning]")
    path = Path(__file__).resolve().parents[2] / "examples" / "dreamer_bilstm_counterfactual.py"
    spec = importlib.util.spec_from_file_location("dreamer_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    # Nonconsecutive IDs, shuffled rows, baseline samples, and irrelevant ECG
    # columns exercise the same boundaries as a converted DREAMER recording.
    rng = np.random.default_rng(8)
    parts = []
    for subject in (1, 7, 23):
        for trial in (2, 5):
            n = 16 * 128
            block = pd.DataFrame(rng.normal(size=(n, 14)), columns=example.CHANNELS)
            block = block.assign(subject_id=subject, trial_id=trial, segment="stimulus",
                                 sample_idx=np.arange(1, n + 1),
                                 valence=1 if trial == 2 else 5, ECG1=1e8, ECG2=-1e8)
            baseline = block.iloc[:128].copy()
            baseline[example.CHANNELS] = 1e6
            baseline["segment"] = "baseline"
            baseline["valence"] = np.nan  # Must be discarded before label validation.
            parts.extend([baseline, block])
    csv_path = tmp_path / "dreamer.csv.gz"
    pd.concat(parts).sample(frac=1, random_state=8).to_csv(csv_path, index=False)

    # Observe the actual training partitions without replacing training or CV.
    trained = []
    original_build = example.BiLSTMClassifier.build

    def recording_build(self, training_subject_ids):
        model = original_build(self)
        trained.append((set(training_subject_ids), model))
        return model

    monkeypatch.setattr(example.BiLSTMClassifier, "build", recording_build)
    output = tmp_path / "output"
    results, counterfactual = example.main(csv_path, output, epochs=1)
    arrays = results["windowed_arrays"]
    assert arrays.features.shape == (12, 4, 84)
    assert set(arrays.labels) == {0, 1}
    assert len(trained) == len(results["fold_results"]) == 3
    assert {row["subject_id"] for row in results["user_metrics"]} == {1, 7, 23}

    # Each reported subject really was absent from that model's training set.
    for (train_codes, _), fold in zip(trained, results["fold_results"]):
        train_subjects = {results["subject_lookup"][code] for code in train_codes}
        assert train_subjects == {1, 7, 23} - set(fold["left_out_subjects"])

    summary = json.loads((output / "counterfactual.json").read_text())
    assert summary["subject_id"] == results["fold_results"][-1]["left_out_subjects"][0]
    assert summary["trial_id"] == 2
    assert summary["label"] == 0
    assert summary["target_class"] == 1 - summary["original"]["predicted_class"]
    index = np.flatnonzero(arrays.subject_ids == 2)[0]
    np.testing.assert_array_equal(counterfactual["arrays"]["x"], arrays.features[index:index + 1])

    # The checkpoint and explanation must refer to the final trained classifier.
    saved = tf.keras.models.load_model(output / "last_fold.keras", compile=False)
    for actual, expected in zip(saved.get_weights(), trained[-1][1].get_weights()):
        np.testing.assert_array_equal(actual, expected)
    probabilities = saved(counterfactual["arrays"]["x"], training=False).numpy()[0]
    np.testing.assert_allclose(summary["original"]["probabilities"], probabilities, atol=1e-6)
    with np.load(output / "counterfactual.npz", allow_pickle=False) as archive:
        assert archive["x_prime_input"].shape == (1, 4, 84)
        assert np.isfinite(archive["x_prime_input"]).all()
        assert len(archive["feature_names"]) == 84
        assert not any("valence" in name or "ECG" in name for name in archive["feature_names"])
    assert len(pd.read_csv(output / "loso_metrics.csv")) == 3
    assert not pd.read_csv(output / "history.csv").empty
