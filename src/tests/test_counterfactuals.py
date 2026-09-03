"""Focused integration coverage for SIC joint counterfactual decoding."""

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import (  # noqa: E402
    build_sic_model,
)
from eegproc.model_explainability.counterfactual_args import (  # noqa: E402
    build_parser,
)
from eegproc.model_explainability.counterfactual_optimizer import (  # noqa: E402
    CounterfactualOptimizer,
)
from eegproc.model_explainability.counterfactual_plotting import (  # noqa: E402
    load_counterfactual_trial,
)


@pytest.fixture(scope="module")
def tiny_joint_model():
    return build_sic_model(
        input_shape=(2, 4, 6),
        adjacency=np.eye(2, dtype=np.float32),
        classification_level="trial",
        n_classes=2,
        n_channels=2,
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
        joint_reconstruction_initial_alpha=0.3,
    )


def test_joint_decoder_mode_uses_only_fused_reconstruction(tiny_joint_model):
    inputs = tf.random.normal((1, 2, 4, 6), seed=7)
    weights_before = [value.numpy().copy() for value in tiny_joint_model.weights]
    optimizer = CounterfactualOptimizer(
        tiny_joint_model,
        max_steps=1,
        decoder_mode="joint",
    )

    result = optimizer.optimize(inputs)

    assert optimizer.decoded_names == ("joint",)
    assert result["summary"]["decoder_mode"] == "joint"
    assert result["summary"]["joint_reconstruction_alpha"] == pytest.approx(0.3)
    assert set(result["summary"]["decoded_trials"]) == {"joint"}
    assert "decoded_joint" in result["history"][0]
    assert "decoded_gcn_gru" not in result["history"][0]
    assert "decoded_bilstm" not in result["history"][0]
    assert set(result["arrays"]) == {
        "x",
        "z",
        "z_prime",
        "x_reconstructed_joint",
        "x_prime_joint",
    }
    np.testing.assert_allclose(
        result["arrays"]["x_reconstructed_joint"],
        tiny_joint_model.reconstruct_joint(inputs).numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert all(
        np.array_equal(before, after.numpy())
        for before, after in zip(weights_before, tiny_joint_model.weights)
    )


def test_branch_decoder_mode_remains_backward_compatible(tiny_joint_model):
    inputs = tf.zeros((1, 2, 4, 6), dtype=tf.float32)
    result = CounterfactualOptimizer(
        tiny_joint_model,
        max_steps=0,
        decoder_mode="branches",
    ).optimize(inputs)

    assert result["summary"]["decoder_mode"] == "branches"
    assert result["summary"]["joint_reconstruction_alpha"] is None
    assert set(result["summary"]["decoded_trials"]) == {"gcn_gru", "bilstm"}
    assert "x_prime_joint" not in result["arrays"]


def test_joint_decoder_mode_is_exposed_by_cli():
    action = next(
        action for action in build_parser()._actions if action.dest == "decoder_mode"
    )
    assert action.default == "branches"
    assert tuple(action.choices) == ("branches", "joint")


def test_plot_loader_accepts_joint_reconstruction(tmp_path):
    original = np.zeros((1, 2, 4, 6), dtype=np.float32)
    counterfactual = np.ones_like(original)
    path = tmp_path / "counterfactual.npz"
    np.savez_compressed(
        path,
        x=original,
        x_reconstructed_joint=original,
        x_prime_joint=counterfactual,
    )

    reference, loaded, branch, names = load_counterfactual_trial(
        path,
        branch="joint",
        reference="reconstruction",
    )

    assert branch == "joint"
    assert names is None
    np.testing.assert_array_equal(reference, original[0])
    np.testing.assert_array_equal(loaded, counterfactual[0])
