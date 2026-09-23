"""Tests for the additive adapter-based counterfactual API."""

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from eegproc.model_explainability.model_agnostic.adapter import (  # noqa: E402
    CounterfactualAdapter,
    TrialDataset,
    validate_trial_dataset,
)
from eegproc.model_explainability.model_agnostic.optimizer import (  # noqa: E402
    ModelAgnosticCounterfactualOptimizer,
)
from eegproc.model_explainability.model_agnostic.topography import (  # noqa: E402
    restore_source_units,
)
from eegproc.model_explainability.model_agnostic.sic_adapter import (  # noqa: E402
    _normalize_windows,
    create_sic_adapter,
)
from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import (  # noqa: E402
    build_sic_model,
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


def test_runner_reports_results_without_npz(tmp_path, monkeypatch):
    import json
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


def test_normalization_transform_restores_original_source_values():
    original = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 7
    normalized, offset, scale = _normalize_windows(original, "global_rms")

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


def test_sic_adapter_restores_lazy_joint_fusion_weight(tmp_path):
    model = build_sic_model(
        input_shape=(2, 4, 42),
        adjacency=np.eye(14, dtype=np.float32),
        classification_level="trial",
        n_classes=2,
        n_channels=14,
        n_bands=3,
        gcn_units=(4,),
        spectral_gru_units=4,
        bilstm_units=2,
        classifier_rnn_units=4,
        use_gcn_gru_branch=True,
        use_bilstm_branch=True,
        use_decoder=True,
        use_subject_adversarial=False,
        decoder_dropout=0.0,
        joint_reconstruction_initial_alpha=0.5,
    )
    expected_alpha = 0.3
    model.joint_reconstruction_fusion.mix_logit.assign(
        np.log(expected_alpha / (1.0 - expected_alpha))
    )
    path = tmp_path / "tiny_sic.keras"
    model.save(path)

    adapter = create_sic_adapter(
        model_path=path,
        config={"decoder_mode": "joint"},
        sample_input=np.zeros((2, 4, 42), dtype=np.float32),
    )

    assert adapter.metadata()["joint_reconstruction_alpha"] == pytest.approx(
        expected_alpha
    )
