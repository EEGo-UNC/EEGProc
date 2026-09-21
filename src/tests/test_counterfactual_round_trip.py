"""Latent-only CFO plus read compatibility for historical round-trip reports."""

import json

import numpy as np
import pytest

from eegproc.model_explainability.typicality.results import build_report, population_summary


def test_counterfactuals_never_reencode_reconstructions_or_candidates():
    tf = pytest.importorskip("tensorflow")
    from eegproc.model_explainability.counterfactuals.loss import CounterfactualLoss
    from eegproc.model_explainability.counterfactuals.optimizer import CounterfactualOptimizer
    from eegproc.model_explainability.typicality.core import TypicalityRegion

    class Loss(CounterfactualLoss):
        def physiological_validity(self, signal):
            return tf.reduce_sum(signal) * 0

    class Head:
        def __call__(self, embedding, training=False):
            assert training is False
            return tf.concat([-embedding, embedding], axis=1)

        def vc_loss_components(self, **kwargs):
            zero = tf.reduce_sum(kwargs["mh"]) * 0
            return {key: zero for key in ("weighted_focal_loss", "weighted_latent_posterior_kl",
                                          "weighted_discriminator_kl", "weighted_class_prior_kl")}

    class Model:
        classification_level = "trial"
        use_decoder = True
        use_gcn_gru_branch = True
        use_bilstm_branch = False
        gcn_gru_decoder = True
        gcn_gru_feature_dim = 1
        vc_target = Head()

        def __init__(self, decoder_sign):
            self.decoder_sign = decoder_sign
            self.encoder_inputs = []

        def trial_recurrent_classifier(self, sequence, training=False):
            assert training is False
            weights = tf.cast(tf.range(1, tf.shape(sequence)[1] + 1), tf.float32)[None, :, None]
            return tf.reduce_sum(sequence * weights, axis=1) / tf.reduce_sum(weights)

        def get_encoder_features(self, signal):
            if self.encoder_inputs:
                raise AssertionError("CFO must encode only the original trial")
            self.encoder_inputs.append(signal.numpy().copy())
            embedding = self.trial_recurrent_classifier(tf.reshape(signal, [1, -1, 1]))
            return {"window_features": signal, "classification_embedding": embedding,
                    "probabilities": tf.nn.softmax(self.vc_target(embedding))}

        def decode_branch_feature_sequence(self, name, latent):
            return self.decoder_sign * latent

    x = tf.reshape(tf.linspace(-2.0, -0.5, 12), (1, 3, 4, 1))
    region = TypicalityRegion(np.array([0.8]), np.array([0.25]), 1.0, {"target_class": 1})
    prior_before = region.prior_mean.copy(), region.prior_variance.copy(), region.tau
    results = []
    for sign in (1, -1):
        model = Model(sign)
        result = CounterfactualOptimizer(
            model, loss=Loss(target_probability=0.75, latent_weight=0, decoded_weight=0),
            max_steps=80, learning_rate=0.1, stop_on_success=True, typicality=region,
        ).optimize(x, target_class=1)
        results.append(result)
        summary, arrays = result["summary"], result["arrays"]
        decoded = summary["decoded_trials"]["gcn_gru"]
        assert summary["latent_counterfactual"]["success"]
        assert summary["typicality"]["typicality_success"]
        assert summary["round_trip_evaluation"] == "latent_only"
        assert summary["counterfactual_validity_prediction_space"] == "latent"
        assert len(model.encoder_inputs) == 1
        np.testing.assert_array_equal(model.encoder_inputs[0], arrays["x"])
        assert model.encoder_inputs[0].shape == (1, 3, 4, 1)
        assert "counterfactual" not in decoded
        assert "original_reconstruction" not in decoded
        assert not any("reencoded" in key or "embedding_reconstructed" in key for key in arrays)
        assert np.isfinite(decoded["original_reconstruction_mse"])
        assert np.isfinite(decoded["decoded_change_mse"])
        np.testing.assert_array_equal(arrays["x_prime_gcn_gru"], sign * arrays["z_prime"])
        json.dumps(summary, allow_nan=False)
        np.testing.assert_array_equal(region.prior_mean, prior_before[0])
        np.testing.assert_array_equal(region.prior_variance, prior_before[1])
        assert region.tau == prior_before[2]
    np.testing.assert_array_equal(results[0]["arrays"]["z_prime"], results[1]["arrays"]["z_prime"])
    assert results[0]["summary"]["selected_step"] == results[1]["summary"]["selected_step"]


def test_round_trip_rates_keep_failed_reconstructions_and_all_attempts():
    rows = [dict(status="completed", round_trip_expected=True, round_trip_evaluated=True,
                 latent_target_success=True, typical=True, typicality_success=True,
                 decoded_target_success=False, decoded_typical=False, decoded_typicality_success=False,
                 reconstruction_preserves_prediction=False),
            dict(status="completed", round_trip_expected=True, round_trip_evaluated=True,
                 decoded_target_success=True, decoded_typical=True, decoded_typicality_success=True,
                 reconstruction_preserves_prediction=True),
            dict(status="error", round_trip_expected=True), dict(status="pending", round_trip_expected=True)]
    result = population_summary(rows)
    assert result["n_eligible"] == 4
    assert result["n_round_trip_evaluated"] == 2
    assert result["decoded_target_success_percent"] == 25
    assert result["decoded_typicality_success_percent"] == 25
    assert result["reconstruction_preserves_prediction_percent"] == 25
    assert result["provisional"]
    assert population_summary([dict(status="error", round_trip_expected=True)])["decoded_target_success_percent"] == 0
    assert population_summary([dict(status="completed")])["decoded_target_success_percent"] is None
    assert population_summary(rows + [dict(status="completed")])["decoded_target_success_percent"] is None


def test_reports_reject_mixing_latent_only_and_round_trip_protocols(tmp_path):
    roots = [tmp_path / name for name in ("old", "new")]
    for root in roots:
        root.mkdir()
    (roots[0] / "study.json").write_text(json.dumps({"task": "valence"}))
    (roots[1] / "study.json").write_text(json.dumps({"task": "valence", "round_trip_evaluation": "full_trial_decoder_encoder_v1"}))
    with pytest.raises(ValueError, match="evaluation protocols"):
        build_report(roots, tmp_path / "report")


def test_report_selects_a_decoded_success_and_retains_reconstruction_failures(tmp_path, monkeypatch):
    from eegproc.model_explainability.typicality import results

    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({"round_trip_evaluation": "full_trial_decoder_encoder_v1"}))
    rows = [dict(task="valence", subject_id=0, trial_id=trial, objective=objective, status="completed",
                 round_trip_expected=True, round_trip_evaluated=True,
                 typical=True, latent_target_success=True, typicality_success=True,
                 decoded_target_success=trial == 1, decoded_typical=True, decoded_typicality_success=trial == 1,
                 reconstruction_preserves_prediction=trial == 0, d_z=trial + 1, delta_dec=0.1, e_rec=0.2)
            for trial in (0, 1) for objective in ("base", "typicality")]
    folds = [dict(task="valence", subject_id=0, status="completed")]
    monkeypatch.setattr(results, "collect_study", lambda _: (rows, [], [], folds))
    report = results.build_report([root], tmp_path / "report")
    assert report["examples"][0]["trial_id"] == 1
    assert "decoded-target-and-reencoded-typical" in report["examples"][0]["selection_rule"]
    assert all(row["n_eligible"] == 2 for row in report["population"])
    assert all(row["latent_target_success_percent"] == 100 for row in report["population"])
    assert all(row["decoded_target_success_percent"] == 50 for row in report["population"])
    assert all(row["decoded_typicality_success_percent"] == 50 for row in report["population"])
    assert all(row["reconstruction_preserves_prediction_percent"] == 50 for row in report["population"])
    assert report["subject_typicality"][0]["typicality_space"] == "decoded_then_reencoded"
