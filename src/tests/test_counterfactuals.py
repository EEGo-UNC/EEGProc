"""Focused integration coverage for SIC joint counterfactual decoding."""

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import (  # noqa: E402
    build_sic_model,
)
from eegproc.model_explainability.counterfactuals.arguments import (  # noqa: E402
    build_parser,
)
from eegproc.model_explainability.counterfactuals.loss import (  # noqa: E402
    CounterfactualLoss,
)
from eegproc.model_explainability.counterfactuals.optimizer import (  # noqa: E402
    CounterfactualOptimizer,
)
from eegproc.model_explainability.counterfactuals.plotting import (  # noqa: E402
    load_counterfactual_trial,
)
from eegproc.model_explainability.counterfactuals.runner import (  # noqa: E402
    format_optimization_diagnostics,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:VCSC calibration was measured.*:RuntimeWarning"
)


@pytest.fixture(scope="module")
def tiny_joint_model():
    return build_sic_model(
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
        joint_reconstruction_initial_alpha=0.3,
    )


def test_joint_decoder_mode_uses_only_fused_reconstruction(tiny_joint_model):
    inputs = tf.random.normal((1, 2, 4, 42), seed=7)
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
    assert np.isfinite(result["history"][0]["physiological"])
    assert result["history"][0]["weighted_physiological"] == pytest.approx(0.0)
    assert "decoded_gcn_gru" not in result["history"][0]
    assert "decoded_bilstm" not in result["history"][0]
    assert set(result["arrays"]) == {
        "x",
        "z",
        "z_prime",
        "x_reconstructed_joint",
        "x_prime_joint",
    }
    decoded = result["summary"]["decoded_trials"]["joint"]
    assert "counterfactual" not in decoded
    assert "original_reconstruction" not in decoded
    assert result["history"][0]["decoded"] == pytest.approx(0.0)
    assert decoded["original_reconstruction_mse"] > 0
    assert result["summary"]["selected_losses"]["decoded"] == pytest.approx(
        decoded["decoded_change_mse"]
    )
    assert result["summary"]["decoded_distance_reference"] == "original_reconstruction"
    assert np.isfinite(decoded["vcsc_original_reconstruction"])
    assert np.isfinite(decoded["vcsc_counterfactual"])
    assert decoded["vcsc_delta"] == pytest.approx(
        decoded["vcsc_counterfactual"]
        - decoded["vcsc_original_reconstruction"]
    )
    assert np.isfinite(result["summary"]["physiological_validity"])
    assert result["summary"]["physiological_constraint_enforced"] is False
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
    inputs = tf.zeros((1, 2, 4, 42), dtype=tf.float32)
    result = CounterfactualOptimizer(
        tiny_joint_model,
        max_steps=0,
        decoder_mode="branches",
    ).optimize(inputs)

    assert result["summary"]["decoder_mode"] == "branches"
    assert result["summary"]["joint_reconstruction_alpha"] is None
    assert set(result["summary"]["decoded_trials"]) == {"gcn_gru", "bilstm"}
    assert "x_prime_joint" not in result["arrays"]
    assert result["history"][0]["decoded"] == pytest.approx(0.0)
    assert all(result["history"][0][f"decoded_{branch}"] == pytest.approx(0.0)
               for branch in ("gcn_gru", "bilstm"))


def test_decoded_distance_uses_fixed_reconstruction_and_candidate_gradients():
    original = tf.Variable([1.0, 3.0])
    candidate = tf.Variable([2.0, 5.0])
    decoder = lambda z: {"gcn_gru": 2 * z + 10, "bilstm": -3 * z + 4}
    with tf.GradientTape(persistent=True) as tape:
        distance, _, paths = CounterfactualLoss().decoded_distance(candidate, original, decoder)
    # The original decoder bias does not enter the displacement. Branch MSEs
    # are 4*mean([1,4]) and 9*mean([1,4]), then averaged.
    assert float(paths["gcn_gru"]) == pytest.approx(10.0)
    assert float(paths["bilstm"]) == pytest.approx(22.5)
    assert float(distance) == pytest.approx(16.25)
    np.testing.assert_allclose(tape.gradient(distance, candidate), [6.5, 13.0])
    assert tape.gradient(distance, original) is None


def test_decoded_distance_reuses_cached_original_reconstruction():
    calls = []
    def decoder(z):
        calls.append(z)
        return {"joint": 2 * z + 10}
    original = tf.constant([1.0, 3.0])
    reference = decoder(original)
    distance, _, _ = CounterfactualLoss().decoded_distance(
        tf.constant([2.0, 5.0]), original, decoder, reference_reconstructions=reference,
    )
    assert len(calls) == 2
    assert float(distance) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="paths must match"):
        CounterfactualLoss().decoded_distance(original, original, decoder,
                                              reference_reconstructions={"wrong_path": original})


def test_joint_decoder_mode_is_exposed_by_cli():
    action = next(
        action for action in build_parser()._actions if action.dest == "decoder_mode"
    )
    assert action.default == "branches"
    assert tuple(action.choices) == ("branches", "joint")


def test_learning_rate_decay_is_exposed_by_cli():
    action = next(
        action
        for action in build_parser()._actions
        if action.dest == "learning_rate_decay"
    )
    assert action.default == 1.0


def test_target_loss_component_is_exposed_by_cli():
    action = next(
        action
        for action in build_parser()._actions
        if action.dest == "target_loss_component"
    )
    assert action.default == "confidence"
    assert tuple(action.choices) == ("confidence", "focal", "vc", "focal_vc")


def test_target_loss_is_decomposed_before_gradient(tiny_joint_model):
    inputs = tf.random.normal((1, 2, 4, 42), seed=13)
    optimizer = CounterfactualOptimizer(
        tiny_joint_model,
        target_loss_component="focal_vc",
        max_steps=0,
        decoder_mode="joint",
    )
    latent = tiny_joint_model.get_encoder_features(inputs)["window_features"]
    embedding, logits = optimizer._classification_state(latent)
    selected, components = optimizer._target_components(embedding, logits, 1)

    assert float(selected.numpy()) == pytest.approx(
        float(
            (
                components["target_focal_component"]
                + components["target_vc_component"]
            ).numpy()
        )
    )
    result = optimizer.optimize(inputs, target_class=1)
    row = result["history"][0]
    assert row["target_loss_component"] == "focal_vc"
    assert row["target"] == pytest.approx(
        row["target_focal_component"] + row["target_vc_component"]
    )


def test_learning_rate_decay_is_recorded_per_step(tiny_joint_model):
    inputs = tf.random.normal((1, 2, 4, 42), seed=9)
    result = CounterfactualOptimizer(
        tiny_joint_model,
        learning_rate=0.2,
        learning_rate_decay=0.5,
        max_steps=2,
        decoder_mode="joint",
    ).optimize(inputs)

    assert [row["learning_rate"] for row in result["history"]] == pytest.approx(
        [0.2, 0.1, 0.05]
    )


def test_consecutive_low_gradients_stop_optimization(tiny_joint_model):
    inputs = tf.random.normal((1, 2, 4, 42), seed=29)
    result = CounterfactualOptimizer(
        tiny_joint_model,
        max_steps=20,
        decoder_mode="joint",
        min_gradient_norm=1e9,
        low_gradient_patience=2,
    ).optimize(inputs)

    assert result["summary"]["stop_reason"] == "low_gradient"
    assert result["summary"]["steps_completed"] == 1
    assert [row["low_gradient_steps"] for row in result["history"]] == [1, 2]
    stopping = result["summary"]["stopping"]
    assert stopping["stop_on_success"] is False
    assert stopping["min_gradient_norm"] == 1e9
    assert stopping["low_gradient_patience"] == 2


def test_vcsc_settings_are_exposed_by_cli():
    actions = {action.dest: action for action in build_parser()._actions}

    assert actions["physiological_weight"].default == 0.0
    assert actions["vcsc_distance_cm"].default == 12.0
    assert actions["vcsc_tau_cm"].default == 4.0
    assert actions["vcsc_z0"].default == 2.0
    assert actions["vcsc_z_max"].default == 20.0


def test_vcsc_weight_contributes_to_total_objective():
    x = tf.random.normal((1, 2, 4, 42), seed=11)
    z = tf.zeros((1, 1, 1, 1), dtype=tf.float32)
    loss = CounterfactualLoss(
        latent_weight=0.0,
        decoded_weight=0.0,
        physiological_weight=0.25,
    )

    terms, _ = loss.central_loss(
        logits=tf.constant([[0.0, 0.0]]),
        target_class=1,
        z_prime=z,
        z=z,
        x=x,
        decoder=lambda _: {"joint": x},
    )

    assert float(terms["physiological"].numpy()) > 0.0
    assert float(terms["weighted_physiological"].numpy()) == pytest.approx(
        0.25 * float(terms["physiological"].numpy())
    )
    assert float(terms["total"].numpy()) == pytest.approx(
        float(terms["weighted_target"].numpy())
        + float(terms["weighted_physiological"].numpy())
    )


def test_step_diagnostics_show_all_objective_contributions():
    row = {
        "step": 3,
        "total": 1.0,
        "target": 2.0,
        "latent": 3.0,
        "decoded": 4.0,
        "physiological": 0.0,
        "weighted_target": 0.5,
        "weighted_latent": 0.2,
        "weighted_decoded": 0.3,
        "weighted_physiological": 0.0,
        "target_probability": 0.75,
        "predicted_class": 1,
        "gradient_norm": 0.125,
        "success": False,
    }

    diagnostics = format_optimization_diagnostics(row)

    assert "RAW[target=2 latent=3 decoded=4 physiological=0]" in diagnostics
    assert "WEIGHTED[target=0.5 latent=0.2 decoded=0.3 physiological=0]" in diagnostics
    assert "SHARE[target=50.0% latent=20.0% decoded=30.0% physiological=0.0%]" in diagnostics


def test_cli_prints_every_optimization_step_by_default():
    action = next(
        action for action in build_parser()._actions if action.dest == "log_every"
    )
    assert action.default == 1


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
