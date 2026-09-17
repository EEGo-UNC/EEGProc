"""Formula, leakage, paired accounting, frozen SIC mapping, and archive tests."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from eegproc.model_explainability.typicality import TypicalityRegion, diagonal_gaussian_kl, trial_representation
from eegproc.model_explainability.typicality.artifacts import TrialRecorder, completed_attempt
from eegproc.model_explainability.typicality.results import population_summary, recognition_metrics, build_report


def test_eq7_matches_analytic_mean_and_variance_terms():
    # Per-coordinate KLs: 0.5*(4 + 1 - 1 - log(4)); 0.5*(1 + 4 - 1).
    actual = diagonal_gaussian_kl([[1, 2]], [[4, 1]], [0, 0], [1, 1])
    expected = (0.5 * (4 - np.log(4)) + 2) / 2
    assert actual[0] == pytest.approx(expected)
    assert diagonal_gaussian_kl([0, 1], [2, 3], [0, 1], [2, 3]) == pytest.approx(0)
    assert np.isfinite(diagonal_gaussian_kl([1], [0], [0], [0]))
    with pytest.raises(ValueError, match="nonnegative"):
        diagonal_gaussian_kl([0], [-1], [0], [1])


def calibrated_region():
    return TypicalityRegion.calibrate(
        np.array([[0, 1], [1, 1], [100, 1]], dtype=float),
        prior_mean=[0], prior_variance=[1], subject_ids=[1, 2, 2],
        trial_ids=[0, 0, 1], labels=[1, 1, 0], held_out_subject=0,
    )


def test_calibration_keeps_learned_prior_and_excludes_class_zero(tmp_path):
    region = calibrated_region()
    np.testing.assert_array_equal(region.prior_mean, [0])
    np.testing.assert_array_equal(region.prior_variance, [1])
    assert region.tau == pytest.approx(0.5)
    assert region.metadata["calibration_trial_keys"] == [[1, 0], [2, 0]]
    region.save(tmp_path)
    loaded = TypicalityRegion.load(tmp_path)
    assert loaded.score([[1, 1]])[0] == pytest.approx(region.tau)
    assert loaded.metadata == region.metadata


def test_heldout_subject_is_rejected_even_for_class_zero():
    with pytest.raises(ValueError, match="Held-out"):
        TypicalityRegion.calibrate([[0, 1], [1, 1], [10, 1]], prior_mean=[0], prior_variance=[1],
            subject_ids=[1, 2, 0], trial_ids=[0, 0, 0], labels=[1, 1, 0], held_out_subject=0)


def test_trial_moments_use_complete_sequence_ddof_zero():
    z = np.asarray([[[[0, 2], [2, 4]], [[4, 6], [6, 8]]]])
    np.testing.assert_allclose(trial_representation(z), [[3, 5, 5, 5]])


def test_failed_and_pending_optimizations_remain_in_denominator():
    rows = [dict(status="completed", latent_target_success=True, typical=True, typicality_success=True,
                 d_z=1, delta_dec=2, e_rec=3, physiological_passed=None),
            dict(status="error"), dict(status="pending")]
    result = population_summary(rows)
    assert result["typicality_success_percent"] == pytest.approx(100 / 3)
    assert result["n_eligible"] == 3
    assert result["d_z_median"] == 1
    assert result["d_z_n"] == 1
    assert result["physiological_pass_percent"] is None
    assert result["provisional"]
    assert population_summary([])["typicality_success_percent"] is None


def test_recognition_ece_and_absent_class():
    result = recognition_metrics([0, 1], [[0.8, 0.2], [0.6, 0.4]], ece_bins=10)
    assert result["balanced_accuracy"] == 0.5
    assert result["auroc"] == 1
    assert result["ece"] == pytest.approx(0.4)
    result = recognition_metrics([0], [[1, 0]])
    assert result["balanced_accuracy"] is None
    assert result["auroc"] is None
    assert result["ece"] == 0


def test_trace_survives_error_and_complete_marker_verifies_arrays(tmp_path):
    arm = tmp_path / "arm"
    recorder = TrialRecorder(arm / "attempt_0001")
    recorder.record({"step": 0, "loss": 3.0}, {"z": np.zeros((1, 2))})
    assert (recorder.directory / "history.jsonl").read_text().strip()
    recorder.fail(RuntimeError("injected failure"), {"trial_id": 0})
    assert completed_attempt(arm) is None
    recorder = TrialRecorder(arm / "attempt_0002")
    for step in range(3):
        recorder.record(
            {
                "step": step,
                "target_probability": 0.25 + step / 10,
                "target": 1.0 - step / 10,
                "latent": step / 10,
                "decoded": step / 20,
                "physiological": 0.0,
                "typicality": step / 30,
                "weighted_target": 1.0 - step / 10,
                "weighted_latent": step / 100,
                "weighted_decoded": step / 200,
                "weighted_physiological": 0.0,
                "weighted_typicality": step / 300,
                "total": 1.0,
            },
            {"z": np.ones((1, 2)) * (step + 1)},
        )
    recorder.finish({"summary": {}, "arrays": {"z": np.ones((1, 2))}}, {})
    snapshots = sorted(path.name for path in (recorder.directory / "trajectory").glob("*.npz"))
    assert snapshots == ["step_000000.npz", "step_000002.npz"]
    rows = [json.loads(line) for line in (recorder.directory / "history.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert all("target_probability" in row and "total" in row for row in rows)
    assert completed_attempt(arm) == recorder.directory
    (recorder.directory / "counterfactual.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        completed_attempt(arm)


def test_tf_eq7_gradient_matches_finite_difference():
    tf = pytest.importorskip("tensorflow")
    z = tf.Variable([[[[-0.5], [2.0]], [[1.0], [3.0]]]], dtype=tf.float32)
    region = calibrated_region()
    with tf.GradientTape() as tape:
        d = region.discrepancy(z)
    gradient = tape.gradient(d, z).numpy()
    expected = region.score(trial_representation(z.numpy()))[0]
    assert float(d) == pytest.approx(expected, rel=1e-6)
    delta = 1e-3
    before, after = z.numpy(), z.numpy()
    before[0, 0, 0, 0] -= delta
    after[0, 0, 0, 0] += delta
    numerical = (region.score(trial_representation(after)) - region.score(trial_representation(before))) / (2 * delta)
    assert gradient[0, 0, 0, 0] == pytest.approx(float(numerical[0]), rel=1e-3)


def test_exact_hinge_and_typicality_aware_stopping():
    tf = pytest.importorskip("tensorflow")
    from eegproc.model_explainability.counterfactuals.loss import CounterfactualLoss
    from eegproc.model_explainability.counterfactuals.optimizer import CounterfactualOptimizer

    class Loss(CounterfactualLoss):
        def physiological_validity(self, x):
            return tf.reduce_sum(x) * 0

    class Head:
        def __call__(self, embedding, training=False):
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

        def trial_recurrent_classifier(self, sequence, training=False):
            return tf.reduce_mean(sequence, axis=1)

        def get_encoder_features(self, x):
            return {"window_features": x, "probabilities": tf.nn.softmax(self(x))}

        def __call__(self, x, training=False):
            return self.vc_target(tf.reduce_mean(x, axis=(1, 2)))

        def decode_branch_feature_sequence(self, name, z):
            return z

    region = TypicalityRegion(np.array([2.0]), np.array([0.25]), 0.05, {"target_class": 1})
    x = np.array([[[[-1.0], [0.0]]]], dtype=np.float32)
    loss = Loss(target_probability=0.6, latent_weight=0, decoded_weight=0)
    common = dict(loss=loss, typicality=region, max_steps=80, learning_rate=0.1, stop_on_success=True)
    base = CounterfactualOptimizer(Model(), typicality_weight=0, **common).optimize(x, target_class=1)
    constrained = CounterfactualOptimizer(Model(), typicality_weight=1, **common).optimize(x, target_class=1)
    first = constrained["history"][0]
    assert first["weighted_typicality"] == pytest.approx(max(0, first["discrepancy"] - region.tau) ** 2)
    assert base["summary"]["latent_counterfactual"]["success"]
    assert not base["summary"]["typicality"]["typical"]
    assert constrained["summary"]["typicality"]["typical"]
    assert constrained["summary"]["selected_step"] > base["summary"]["selected_step"]


@pytest.mark.parametrize("mode", ["vc_window_embeddings", "vc_hidden_sequence"])
def test_sic_mapping_has_vc_width_and_frozen_weights(mode):
    tf = pytest.importorskip("tensorflow")
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
    from eegproc.model_explainability.typicality.sic_sequence import SICVCSequence
    model = build_sic_model(input_shape=(3, 8, 42), adjacency=np.eye(14, dtype=np.float32),
                            classification_level="trial", n_channels=14, n_bands=3, gcn_units=(4,),
                            spectral_gru_units=5, bilstm_units=2, classifier_rnn_units=3,
                            use_decoder=True, use_subject_adversarial=False, decoder_dropout=0)
    x = tf.random.normal((1, 3, 8, 42), seed=41)
    z = tf.Variable(model.get_encoder_features(x)["window_features"])
    before = [w.numpy().copy() for w in model.weights]
    mapping = SICVCSequence(model, mode=mode)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(z)
        output = mapping(z)
        value = tf.reduce_sum(output ** 2)
    assert output.shape == (1, 3 if mode == "vc_window_embeddings" else 24, 6)
    assert np.isfinite(tape.gradient(value, z).numpy()).all()
    assert len(before) == len(model.weights)
    for left, right in zip(before, model.weights):
        np.testing.assert_array_equal(left, right.numpy())


def test_band_filtered_physiology_never_claims_complete_pass():
    from eegproc.model_explainability.typicality.physiology import signal_diagnostics, PhysiologicalReference
    rng = np.random.default_rng(14)
    diagnostics = [signal_diagnostics(rng.normal(size=(4, 32, 6)), fs=128, n_channels=2) for _ in range(3)]
    reference = PhysiologicalReference.fit(diagnostics)
    result = reference.assess(diagnostics[1])
    assert result["required_count"] == 5
    assert result["available_count"] == 4
    assert result["all_required_passed"] is None
    assert result["checks"]["aperiodic_exponent"]["reason"] == "not_estimable_from_band_filtered_decoder"


def test_vcsc_reference_decodes_each_initial_state():
    tf = pytest.importorskip("tensorflow")
    from eegproc.model_explainability.typicality.runner import _initial_reconstructions

    class Adapter:
        def __init__(self):
            self.initial_inputs = []

        def initial_state(self, inputs):
            self.initial_inputs.append(inputs.numpy())
            return inputs + 2

        def reconstruct(self, state, reference_input):
            return {"joint": state * 3, "branch": reference_input - 1}

    features = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    adapter = Adapter()
    actual = np.stack(list(_initial_reconstructions(adapter, features, "joint")))
    np.testing.assert_array_equal(actual, (features + 2) * 3)
    assert len(adapter.initial_inputs) == len(features)


def test_typicality_runner_enables_stopping_defaults():
    from eegproc.model_explainability.typicality.runner import build_parser

    actions = {action.dest: action for action in build_parser()._actions}
    assert actions["stop_on_success"].default is True
    assert actions["min_gradient_norm"].default == pytest.approx(1e-6)
    assert actions["low_gradient_patience"].default == 5


def test_end_to_end_saved_study_and_resume_without_model(tmp_path, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
    from eegproc.model_explainability.typicality import runner as study
    tf.keras.utils.set_random_seed(17)
    model = build_sic_model(input_shape=(3, 32, 42), adjacency=np.eye(14, dtype=np.float32),
                            classification_level="trial", n_channels=14, n_bands=3, gcn_units=(4,),
                            spectral_gru_units=4, bilstm_units=2, classifier_rnn_units=4,
                            use_decoder=True, use_subject_adversarial=False, decoder_dropout=0)
    model.vc_target.prior_mu.assign(np.stack([np.zeros(8), np.ones(8) * 2]))
    model_path = tmp_path / "model.keras"
    model.save(model_path)
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"models": [{"path": str(model_path), "target_subject": 0,
                                                "stage": "zero_shot_source_model"}]}))
    data_path = tmp_path / "trials.npz"
    x = np.random.default_rng(19).normal(size=(6, 3, 32, 42)).astype(np.float32)
    np.savez_compressed(data_path, features=x, subject_ids=[0, 0, 1, 1, 2, 2],
                        trial_ids=[0, 1, 0, 1, 0, 1], labels=[0, 0, 1, 1, 1, 1])
    out = tmp_path / "study"
    args = study.parse_args(["--models-json", str(manifest), "--task", "valence", "--trials-npz", str(data_path),
                            "--typicality-sequence", "vc_window_embeddings", "--out-dir", str(out),
                            "--trial-ids", "0", "--max-steps", "1", "--log-every", "0"])
    result = study.run(args)
    assert result["complete"]
    assert len(result["population"]) == 2
    assert all(row["n_eligible"] == row["n_completed"] == 1 for row in result["population"])
    with np.load(out / "subject_0/calibration/vcsc.npz", allow_pickle=False) as data:
        assert data["reference"].item() == "held_out_subject_initial_reconstruction"
        assert data["decoder_output"].item() == "joint"
        np.testing.assert_array_equal(data["subject_ids"], [0, 0])
        np.testing.assert_array_equal(data["trial_ids"], [0, 1])
        assert "labels" not in data.files
        assert data["reference_coherence"].shape[0] == 2
    for objective in ("base", "typicality"):
        attempt = completed_attempt(out / "subject_0/trial_0" / objective)
        assert attempt is not None
        with np.load(attempt / "counterfactual.npz", allow_pickle=False) as data:
            assert "typicality_sequence_prime" in data.files
            assert "x_prime_joint" in data.files
        assert (attempt / "trajectory/step_000000.npz").exists()
        assert (attempt / "physiology_counterfactual.npz").exists()
    # Rebuilding results is independent of both the checkpoint and TensorFlow calls.
    rebuilt = build_report([out], tmp_path / "rebuilt")
    assert rebuilt == result
    from eegproc.model_explainability.typicality.plotting import plot_report
    figure_directory = tmp_path / "figures"
    plot_report(tmp_path / "rebuilt", figure_directory)
    assert (figure_directory / "valence_discrepancy.png").stat().st_size > 1000
    assert (figure_directory / "valence_base_population_power.npz").exists()
    # Exercise the single-example plotting path without claiming this failed
    # synthetic endpoint is a successful empirical counterfactual.
    example = {"task": "valence", "subject_id": 0, "trial_id": 0,
               "artifact_directory": str(completed_attempt(out / "subject_0/trial_0/typicality")),
               "threshold": json.loads((out / "subject_0/fold.json").read_text())["threshold"]}
    visual_fixture = {**rebuilt, "examples": [example]}
    (tmp_path / "rebuilt/results.json").write_text(json.dumps(visual_fixture))
    plot_report(tmp_path / "rebuilt", figure_directory)
    assert (figure_directory / "valence_single_trial_trajectory.pdf").stat().st_size > 1000
    def forbidden_model_load(**kwargs):
        raise AssertionError("Resume loaded a completed fold's model")
    monkeypatch.setattr(study, "create_sic_adapter", forbidden_model_load)
    args.resume = True
    assert study.run(args) == result
    args.typicality_weight = 2
    with pytest.raises(ValueError, match="protocol changed"):
        study.run(args)


def test_subject_probe_uses_disjoint_matched_trials_and_records_fitted_model(tmp_path):
    from eegproc.model_explainability.typicality.subject_probe import fit_subject_probe
    keys = np.array([[s, t] for s in range(3) for t in range(4)])
    original = np.column_stack((keys[:, 0] * 3.0, keys[:, 1] * 0.01))
    values = {"original": original, "base": original + 0.1, "typicality": original - 0.1}
    result = fit_subject_probe(keys, values, tmp_path, coordinate_policy="shared",
                               coordinate_spaces={0: "same", 1: "same", 2: "same"})
    assert all(row["n_subjects"] == 3 for row in result["rows"])
    assert all(row["chance_percent"] == pytest.approx(100 / 3) for row in result["rows"])
    with np.load(tmp_path / "probe_inputs.npz", allow_pickle=False) as data:
        assert not set(data["train_indices"]).intersection(data["test_indices"])
        assert len(data["train_indices"]) + len(data["test_indices"]) == len(keys)
    for name in values:
        with np.load(tmp_path / f"probe_{name}.npz", allow_pickle=False) as data:
            assert "coefficients" in data.files
            assert data["confusion_matrix"].shape == (3, 3)
    with pytest.raises(ValueError, match="not a shared"):
        fit_subject_probe(keys, values, tmp_path, coordinate_policy="shared",
                          coordinate_spaces={0: "a", 1: "b", 2: "c"})


def test_pending_fold_prevents_final_population_claim(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({"task": "valence", "folds": [{"subject_id": 0}]}))
    result = build_report([root], tmp_path / "report")
    assert not result["complete"]
    assert all(row["provisional"] for row in result["population"])
    assert all(row["n_pending_folds"] == 1 for row in result["population"])
