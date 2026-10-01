"""Adapter-based counterfactual optimization, its runner, and topographies."""

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

tf = pytest.importorskip("tensorflow", reason="counterfactuals require eegproc[deep-learning]")

from eegproc.model_explainability.model_agnostic import (  # noqa: E402
    KerasInputAdapter,
    ModelAgnosticCounterfactualOptimizer,
)
from eegproc.model_explainability.model_agnostic.adapter import (  # noqa: E402
    CounterfactualAdapter,
    TrialDataset,
    validate_trial_dataset,
)
from eegproc.model_explainability.model_agnostic.topography import (  # noqa: E402
    restore_source_units,
)


class TinyInputAdapter(CounterfactualAdapter):
    name = "tiny-input"
    optimization_space = "input"
    default_state_weight = 0.1
    default_signal_weight = 0.0

    def __init__(self):
        self.constraint_calls = 0

    def initial_state(self, inputs):
        return inputs

    def logits_from_state(self, state):
        score = tf.reduce_mean(state, axis=tuple(range(1, state.shape.rank)))
        return tf.stack([-score, score], axis=-1)

    def reconstruct(self, state, reference_input):
        del reference_input
        return {"input": state}

    def constraint(self, name, signal):
        if name != "magnitude":
            return super().constraint(name, signal)
        self.constraint_calls += 1
        return tf.reduce_mean(tf.abs(signal))


def _tiny_keras_model(output_activation=None):
    tf.keras.utils.set_random_seed(0)
    inputs = tf.keras.Input((8, 6))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(2, activation=output_activation)(x)
    return tf.keras.Model(inputs, outputs)


def _trial():
    return np.random.default_rng(0).standard_normal((8, 6)).astype("float32")


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------

def test_input_adapter_improves_target_without_a_decoder():
    adapter = TinyInputAdapter()
    result = ModelAgnosticCounterfactualOptimizer(
        adapter,
        target_probability=0.55,
        learning_rate=0.1,
        max_steps=2,
    ).optimize(tf.zeros((1, 2, 3), dtype=tf.float32), target_class=1)

    assert result["summary"]["adapter"]["optimization_space"] == "input"
    assert result["summary"]["counterfactual"]["target_probability"] > 0.5
    assert set(result["summary"]["reconstructed_outputs"]) == {"input"}
    assert set(result["arrays"]) == {
        "x",
        "state",
        "state_prime",
        "x_reconstructed_input",
        "x_prime_input",
    }


def test_report_only_constraint_is_not_computed_during_steps():
    adapter = TinyInputAdapter()
    result = ModelAgnosticCounterfactualOptimizer(
        adapter,
        max_steps=2,
        report_constraints=("magnitude",),
    ).optimize(tf.zeros((1, 2, 3), dtype=tf.float32), target_class=1)

    assert adapter.constraint_calls == 2
    assert not any("constraint_magnitude" in row for row in result["history"])
    metrics = result["summary"]["reconstructed_outputs"]["input"]["constraints"]
    assert metrics["magnitude"]["weight"] == 0.0


def test_adapter_signal_loss_excludes_original_reconstruction_error():
    class BiasedDecoderAdapter(TinyInputAdapter):
        def __init__(self):
            super().__init__()
            self.decode_calls = 0

        def reconstruct(self, state, reference_input):
            self.decode_calls += 1
            return {"input": 2 * state + 7}

    adapter = BiasedDecoderAdapter()
    result = ModelAgnosticCounterfactualOptimizer(
        adapter, signal_weight=0.1, max_steps=2,
    ).optimize(tf.zeros((1, 2, 3), dtype=tf.float32), target_class=1)
    summary = result["summary"]
    decoded = summary["reconstructed_outputs"]["input"]
    assert result["history"][0]["signal"] == pytest.approx(0.0)
    assert decoded["original_reconstruction_mse"] == pytest.approx(49.0)
    assert "counterfactual" not in decoded
    assert "original_reconstruction" not in decoded
    assert summary["selected_losses"]["signal"] == pytest.approx(decoded["decoded_change_mse"])
    assert summary["signal_distance_reference"] == "original_reconstruction"
    # Once for the reference, once per evaluated step, once for the endpoint.
    assert adapter.decode_calls == len(result["history"]) + 2


# --------------------------------------------------------------------------
# KerasInputAdapter — the documented entry point
# --------------------------------------------------------------------------

def test_readme_example_runs_with_default_output_key():
    """The model_agnostic README example, verbatim apart from the toy model."""
    model = _tiny_keras_model()
    trial = _trial()

    adapter = KerasInputAdapter(model, output_kind="logits")
    optimizer = ModelAgnosticCounterfactualOptimizer(adapter, target_probability=0.8)
    result = optimizer.optimize(trial[None, ...], target_class=1)

    assert set(result) == {"summary", "history", "arrays"}
    assert result["summary"]["required_target_probability"] == pytest.approx(0.8)
    assert result["summary"]["counterfactual"]["target_probability"] >= 0.8
    assert result["arrays"]["x"].shape == (1, 8, 6)


def test_keras_input_adapter_accepts_probability_outputs():
    model = _tiny_keras_model(output_activation="softmax")
    adapter = KerasInputAdapter(model, output_kind="probabilities")
    result = ModelAgnosticCounterfactualOptimizer(
        adapter, target_probability=0.8,
    ).optimize(_trial()[None, ...], target_class=1)

    assert result["summary"]["counterfactual"]["target_probability"] >= 0.8


# --------------------------------------------------------------------------
# Datasets and unit restoration
# --------------------------------------------------------------------------

def test_normalization_transform_restores_original_source_values():
    original = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 7
    offset = original.mean(axis=(1, 2), keepdims=True)
    scale = np.sqrt(np.mean((original - offset) ** 2, axis=(1, 2), keepdims=True))
    normalized = (original - offset) / scale

    restored = restore_source_units(normalized, offset, scale)

    np.testing.assert_allclose(restored, original, rtol=1e-6, atol=1e-6)


def test_trial_dataset_accepts_broadcastable_window_transforms():
    dataset = TrialDataset(
        features=np.zeros((2, 3, 4, 5), dtype=np.float32),
        subject_ids=np.array([0, 0]),
        trial_ids=np.array([0, 1]),
        normalization_offset=np.zeros((2, 3, 1, 1), dtype=np.float32),
        normalization_scale=np.ones((2, 3, 1, 1), dtype=np.float32),
    )

    assert validate_trial_dataset(dataset).features.shape == (2, 3, 4, 5)


def test_trial_dataset_rejects_nonbroadcastable_transforms():
    dataset = TrialDataset(
        features=np.zeros((1, 3, 4, 5), dtype=np.float32),
        subject_ids=np.array([0]),
        trial_ids=np.array([0]),
        normalization_scale=np.ones((1, 2, 1), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="cannot broadcast"):
        validate_trial_dataset(dataset)


# --------------------------------------------------------------------------
# Command-line tools
# --------------------------------------------------------------------------

def test_runner_reports_results_without_npz(tmp_path, monkeypatch):
    from eegproc.model_explainability.model_agnostic import runner

    trials = tmp_path / "trials.npz"
    np.savez(trials, features=np.zeros((1, 2, 3), dtype=np.float32),
             subject_ids=[0], trial_ids=[0], labels=[0])
    monkeypatch.setattr(runner, "create_adapter", lambda *args, **kwargs: TinyInputAdapter())

    def forbidden(*args, **kwargs):
        raise AssertionError("Counterfactual runs must not save NPZ files")

    monkeypatch.setattr(np, "savez", forbidden)
    monkeypatch.setattr(np, "savez_compressed", forbidden)
    out = tmp_path / "out"
    args = runner.parse_args(["--model", "unused.keras", "--adapter", "unused:adapter",
                              "--trials-npz", str(trials), "--subject-id", "0",
                              "--out-dir", str(out), "--max-steps", "1", "--log-every", "0"])
    aggregate = runner.run(args)
    saved = json.loads((out / "subject_0_trial_0/result.json").read_text())
    assert aggregate["n_trials"] == 1
    assert aggregate["counterfactual_success_rate"] == float(saved["counterfactual"]["success"])
    assert (out / "subject_0_trial_0/history.csv").is_file()
    assert not list(out.rglob("*.npz"))


def test_runner_end_to_end_with_a_saved_keras_model(tmp_path):
    """The README command line: built-in adapter factory, real .keras file."""
    from eegproc.model_explainability.model_agnostic import runner

    model_path = tmp_path / "model.keras"
    _tiny_keras_model().save(model_path)
    trials = tmp_path / "trials.npz"
    rng = np.random.default_rng(0)
    np.savez(trials, features=rng.standard_normal((2, 8, 6)).astype(np.float32),
             subject_ids=np.array([0, 0]), trial_ids=np.array([0, 1]), labels=np.array([0, 1]))
    out = tmp_path / "results"

    runner.main([
        "--model", str(model_path),
        "--adapter", "eegproc.model_explainability.model_agnostic.adapter:create_keras_input_adapter",
        "--adapter-config", '{"output_kind": "logits"}',
        "--trials-npz", str(trials),
        "--subject-id", "0", "--trial-id", "0",
        "--max-steps", "20", "--log-every", "0",
        "--out-dir", str(out),
    ])

    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_trials"] == 1
    assert (out / "subject_0_trial_0" / "result.json").is_file()
    assert (out / "subject_0_trial_0" / "history.csv").is_file()


def test_topography_cli_writes_a_figure(tmp_path):
    from eegproc.model_explainability.model_agnostic import topography

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 16, 12)).astype(np.float32)   # 4 channels x 3 bands
    archive = tmp_path / "counterfactual.npz"
    np.savez_compressed(
        archive,
        x=x, x_reconstructed_input=x, x_prime_input=x + 0.1,
        channel_positions=rng.uniform(-0.8, 0.8, (4, 2)),
        channel_names=np.array(["AF3", "F3", "P7", "O1"]),
        band_names=np.array(["theta", "alpha", "beta"]),
        feature_order=np.array("channel-major"),
    )
    output = tmp_path / "topography.png"

    topography.main([str(archive), "--branch", "input", "--no-show", "--output", str(output)])

    assert output.is_file() and output.stat().st_size > 0
