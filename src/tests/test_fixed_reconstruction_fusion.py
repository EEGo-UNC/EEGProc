"""Check fixed valence mixing through both consumers and frozen checkpoints."""

from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from eegproc.model_explainability.counterfactuals.fusion import FrozenJointFusion
from eegproc.model_explainability.counterfactuals.optimizer import CounterfactualOptimizer
from eegproc.model_explainability.model_agnostic.sic_adapter import SICCounterfactualAdapter, create_sic_adapter
from eegproc.model_explainability.typicality.runner import parse_args


class IndependentDecoders:
    classification_level = "trial"
    use_decoder = True
    use_gcn_gru_branch = use_bilstm_branch = True
    gcn_gru_decoder = bilstm_decoder = True
    gcn_gru_feature_dim = bilstm_feature_dim = 1

    def decode_branch_feature_sequence(self, name, latent):
        return latent * (2.0 if name == "gcn_gru" else 3.0)


def test_fixed_mixing_matches_both_consumers_and_both_latent_gradients():
    model = IndependentDecoders()
    state = tf.Variable(tf.reshape(tf.range(12, dtype=tf.float32), (1, 2, 3, 2)))
    reference = tf.zeros((1, 2, 3, 1))
    alpha = 0.49751
    adapter = SICCounterfactualAdapter(model, decoder_mode="joint", fixed_joint_alpha=alpha)
    optimizer = CounterfactualOptimizer(model, decoder_mode="joint", fixed_joint_alpha=alpha)
    expected = alpha * 2 * state[..., :1] + (1 - alpha) * 3 * state[..., 1:]
    for decode in (adapter.reconstruct, optimizer._decode):
        with tf.GradientTape() as tape:
            actual = decode(state, reference)["joint"]
            total = tf.reduce_sum(actual)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        gradient = tape.gradient(total, state).numpy()
        np.testing.assert_allclose(gradient[..., 0], 2 * alpha, rtol=1e-6)
        np.testing.assert_allclose(gradient[..., 1], 3 * (1 - alpha), rtol=1e-6)
    assert not hasattr(model, "joint_reconstruction_fusion")
    assert adapter.metadata()["joint_reconstruction_weights"] == pytest.approx(
        {"gcn_gru": 0.49751, "bilstm": 0.50249})


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -0.1, 1.1])
def test_fixed_mixing_rejects_invalid_weights(alpha):
    for consumer in (SICCounterfactualAdapter, CounterfactualOptimizer):
        with pytest.raises(ValueError, match="finite and in"):
            consumer(IndependentDecoders(), decoder_mode="joint", fixed_joint_alpha=alpha)


def test_fixed_mixing_requires_explicit_joint_mode_and_both_decoders():
    for consumer in (SICCounterfactualAdapter, CounterfactualOptimizer):
        with pytest.raises(ValueError, match="requires joint"):
            consumer(IndependentDecoders(), decoder_mode="branches", fixed_joint_alpha=0.49751)
        with pytest.raises(ValueError, match="no joint reconstruction"):
            consumer(IndependentDecoders(), decoder_mode="joint")
        single = IndependentDecoders()
        single.use_bilstm_branch = False
        with pytest.raises(ValueError, match="both SIC branches"):
            consumer(single, decoder_mode="joint", fixed_joint_alpha=0.49751)


def test_fixed_override_does_not_change_learned_mixer():
    class LearnedFusion:
        alpha = tf.Variable(0.3)

        def __call__(self, branches):
            return self.alpha * branches[0] + (1 - self.alpha) * branches[1]

    model = SimpleNamespace(use_joint_reconstruction=True, joint_reconstruction_fusion=LearnedFusion())
    branches = {"gcn_gru": tf.constant([2.0]), "bilstm": tf.constant([5.0])}
    fixed = FrozenJointFusion(model, branches, fixed_alpha=0.49751)
    learned = FrozenJointFusion(model, branches)
    np.testing.assert_allclose(fixed(branches), [3.50747], rtol=1e-6)
    np.testing.assert_allclose(learned(branches), [4.1], rtol=1e-6)
    assert float(model.joint_reconstruction_fusion.alpha.numpy()) == pytest.approx(0.3)


@pytest.mark.parametrize("extra", [["--fixed-joint-alpha", "nan"], ["--fixed-joint-alpha", "1.1"],
                                  ["--fixed-joint-alpha", "0.49751", "--decoder-mode", "branches", "--report-output", "gcn_gru"]])
def test_runner_rejects_invalid_fixed_mixing_before_loading(extra):
    with pytest.raises(SystemExit):
        parse_args(["--models-json", "models.json", "--task", "valence", "--trials-npz", "trials.npz",
                    "--out-dir", "unused", *extra])


def test_saved_v11_checkpoint_fixed_mix_keeps_weights_and_predictions(tmp_path):
    from eegproc.deep_learning.joint_architectures.SICModelv11 import sic_model as v11
    from eegproc.deep_learning.joint_architectures.SICModelv15 import sic_model as v15

    tf.keras.utils.set_random_seed(7)
    model = v11.build_sic_model(
        input_shape=(2, 4, 42), adjacency=np.eye(14, dtype=np.float32),
        classification_level="trial", n_channels=14, n_bands=3, gcn_units=(4,),
        spectral_gru_units=4, bilstm_units=2, classifier_rnn_units=4,
        use_decoder=True, use_subject_adversarial=False, decoder_dropout=0)
    x = tf.random.normal((1, 2, 4, 42))
    expected = model.get_encoder_features(x)["probabilities"].numpy()
    saved_weights = [w.numpy().copy() for w in model.weights]
    path = tmp_path / "v11.keras"
    model.save(path)
    before = path.read_bytes()
    # Mimic a long-lived process in which v15 was imported most recently.
    with tf.keras.utils.custom_object_scope({"EEGProc>SICModel": v15.SICModel}):
        adapter = create_sic_adapter(model_path=path, sample_input=x.numpy(), config={
            "model_module": v11.__name__, "decoder_mode": "joint", "fixed_joint_alpha": 0.49751})
    assert isinstance(adapter.model, v11.SICModel)
    state = adapter.initial_state(x)
    optimizer = CounterfactualOptimizer(adapter.model, decoder_mode="joint", fixed_joint_alpha=0.49751)
    np.testing.assert_allclose(adapter.reconstruct(state, x)["joint"], optimizer._decode(state, x)["joint"])
    np.testing.assert_allclose(adapter.model.get_encoder_features(x)["probabilities"], expected, atol=1e-6)
    assert len(saved_weights) == len(adapter.model.weights)
    for expected_weight, actual_weight in zip(saved_weights, adapter.model.weights):
        np.testing.assert_array_equal(actual_weight.numpy(), expected_weight)
    assert path.read_bytes() == before
