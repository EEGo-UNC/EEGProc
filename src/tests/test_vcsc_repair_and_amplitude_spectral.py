"""Repaired baseline, real-trial VCSC tolerance, and amplitude/spectral term."""

import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from eegproc.model_explainability.counterfactuals.amplitude_spectral import (  # noqa: E402
    AmplitudeSpectralReference, amplitude_and_power)
from eegproc.model_explainability.typicality.physiology import (  # noqa: E402
    DEFAULT_BANDS, PhysiologicalReference, signal_diagnostics)

FS = 128.0


def _trial(seed, windows=4, samples=128):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(windows, samples, 42)).astype(np.float32)


def _per_window_scale(seed, windows=4):
    # global_rms normalization: one positive scale per window, zero offset.
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.5, 3.0, size=(windows, 1, 1)).astype(np.float32)
    return scale, np.zeros_like(scale)


def test_tf_descriptors_reproduce_signal_diagnostics():
    for seed in range(3):
        x = _trial(seed)
        scale, offset = _per_window_scale(seed + 10)
        amplitude, power = amplitude_and_power(x, fs=FS, n_channels=14, band_edges=DEFAULT_BANDS,
                                               scale=scale, offset=offset)
        expected = signal_diagnostics(x * scale + offset, fs=FS, n_channels=14, band_edges=DEFAULT_BANDS)
        np.testing.assert_allclose(amplitude.numpy(), expected["amplitude"], rtol=1e-5)
        np.testing.assert_allclose(power.numpy(), expected["spectral_power"], rtol=1e-4)


def _reference(seed=0, n=40):
    diagnostics = [signal_diagnostics(_trial(seed * 1000 + i), fs=FS, n_channels=14, band_edges=DEFAULT_BANDS)
                   for i in range(n)]
    return PhysiologicalReference.fit(diagnostics, quantile=0.95, required_fraction=0.95)


def test_penalty_fractions_match_reported_check_and_penalty_is_graded():
    physiological = _reference()
    reference = AmplitudeSpectralReference.from_physiological_reference(physiological)
    for factor in (1.0, 1.15, 2.0):
        x = _trial(999) * factor
        amplitude, power = amplitude_and_power(x, fs=FS, n_channels=14, band_edges=DEFAULT_BANDS)
        penalty, diagnostics = reference.penalty({"amplitude": amplitude, "spectral_power": power})
        assessed = physiological.assess(signal_diagnostics(x, fs=FS, n_channels=14, band_edges=DEFAULT_BANDS))
        for family in ("amplitude", "spectral_power"):
            assert float(diagnostics[f"{family}_in_range"]) == pytest.approx(
                assessed["checks"][family]["fraction_in_range"])
        assert reference.feasible({k: float(v) for k, v in diagnostics.items()}) == all(
            assessed["checks"][f]["passed"] for f in ("amplitude", "spectral_power"))
        if factor == 2.0:
            assert float(penalty) > 0
            assert not reference.feasible({k: float(v) for k, v in diagnostics.items()})
    # A trial built exactly at every interval midpoint has zero penalty.
    mid = {f: (physiological.lower[f] + physiological.upper[f]) / 2 for f in ("amplitude", "spectral_power")}
    penalty, diagnostics = reference.penalty({f: tf.constant(v, tf.float32) for f, v in mid.items()})
    assert float(penalty) == 0.0
    assert float(diagnostics["amplitude_in_range"]) == 1.0


def test_penalty_gradient_reaches_signal():
    reference = AmplitudeSpectralReference.from_physiological_reference(_reference())
    x = tf.Variable(_trial(5) * 3.0)
    with tf.GradientTape() as tape:
        amplitude, power = amplitude_and_power(x, fs=FS, n_channels=14, band_edges=DEFAULT_BANDS)
        penalty, _ = reference.penalty({"amplitude": amplitude, "spectral_power": power})
    gradient = tape.gradient(penalty, x)
    assert float(penalty) > 0
    assert np.isfinite(gradient.numpy()).all()
    assert float(tf.reduce_sum(tf.abs(gradient))) > 0


def test_loss_without_reference_keeps_historical_terms():
    from eegproc.model_explainability.counterfactuals.loss import CounterfactualLoss
    from eegproc.model_explainability.typicality.physiology import make_vcsc_loss
    assert CounterfactualLoss().amplitude_spectral_reference() is None
    with pytest.raises(ValueError, match="requires an amplitude/spectral reference"):
        make_vcsc_loss({"c_hat": 0, "w_hat": 0, "sigma_raw": 1, "sigma_spec": 1},
                       amplitude_spectral_weight=1.0)


def _small_study(tmp_path):
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
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
    return manifest, data_path


def test_runner_repair_baseline_tolerance_quantile_and_amplitude_spectral(tmp_path, monkeypatch):
    from eegproc.model_explainability.typicality import runner as study
    from eegproc.model_explainability.typicality import plotting
    from eegproc.model_explainability.typicality.artifacts import completed_attempt
    # Unrelated, pre-existing signature mismatch in the report figures.
    monkeypatch.setattr(plotting, "plot_report", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(plotting, "_power_maps", lambda *a, **k: None)
    manifest, data_path = _small_study(tmp_path)
    out = tmp_path / "study"
    args = study.parse_args([
        "--models-json", str(manifest), "--task", "valence", "--trials-npz", str(data_path),
        "--out-dir", str(out), "--trial-ids", "0", "--max-steps", "2", "--log-every", "0",
        "--include-target-latent", "--include-typicality-no-physiology",
        "--include-repair-baseline", "--physiological-tolerance-quantile", "0.5",
        "--amplitude-spectral-weight", "1.0", "--amplitude-spectral-margin", "0.1"])
    calls = []
    optimize = study.CounterfactualOptimizer.optimize

    def capture(self, *a, **k):
        calls.append((self.repair_only, k.get("signal_scale") is not None))
        return optimize(self, *a, **k)
    monkeypatch.setattr(study.CounterfactualOptimizer, "optimize", capture)
    study.run(args)

    # The repair arm runs first and every arm receives the trial's normalization.
    assert calls[0][0] is True and sum(c[0] for c in calls) == 1
    study_json = json.loads((out / "study.json").read_text())
    assert study_json["repair_baseline"] is True
    assert "repair" not in study_json["objectives"]
    assert "quantile 0.5" in study_json["physiological_tolerance_rule"]

    tolerance = json.loads((out / "subject_0/calibration/vcsc_tolerance.json").read_text())
    fold = json.loads((out / "subject_0/fold.json").read_text())
    assert fold["physiological_tolerance"] == tolerance["tolerance"] == pytest.approx(
        tolerance["source_vcsc_quantiles"]["0.5"])
    assert tolerance["n_source_trials"] == 4
    assert (out / "subject_0/calibration/amplitude_spectral.json").is_file()
    # Ablation arms drop BOTH physiological constraints.
    for arm in ("target_latent", "typicality_no_physiology"):
        assert fold["objective_losses"][arm]["physiological_weight"] == 0
        assert fold["objective_losses"][arm]["amplitude_spectral_weight"] == 0
    assert fold["objective_losses"]["base"]["amplitude_spectral_weight"] == 1.0

    repair_dir = completed_attempt(out / "subject_0/trial_0/repair")
    repair = json.loads((repair_dir / "result.json").read_text())
    assert repair["repair_only"] is True
    assert repair["selected_losses"]["weighted_target"] == 0.0
    assert "repair_preserves_prediction" in repair
    with np.load(repair_dir / "repair_reconstruction.npz") as saved:
        assert saved.files == ["x_repair"]
    assert len(list(out.rglob("*.npz"))) == 1

    for arm in ("base", "typicality", "target_latent", "typicality_no_physiology"):
        result = json.loads((completed_attempt(out / f"subject_0/trial_0/{arm}") / "result.json").read_text())
        baseline = result["repair_baseline"]
        assert baseline["available"] and baseline["delta_dec_repair"] == pytest.approx(repair["delta_dec_repair"])
        assert baseline["delta_dec_explanation"] >= 0
        assert "amplitude_spectral" in result["selected_losses"]
        assert "amplitude_in_range" in result["selected_losses"]
        assert result["physiological_tolerance"] == fold["physiological_tolerance"]
        power = json.loads((completed_attempt(out / f"subject_0/trial_0/{arm}") / "band_power.json").read_text())
        assert "power_change_vs_repair" in power

    metrics = (out / "report/trial_metrics.csv").read_text()
    assert "delta_dec_explanation" in metrics and "delta_dec_repair" in metrics

    # Resume reuses the archived repair instead of re-optimizing it.
    calls.clear()
    args.resume = True
    study.run(args)
    assert not any(c[0] for c in calls)


def test_repair_only_requires_an_active_constraint():
    from eegproc.model_explainability.counterfactuals.optimizer import CounterfactualOptimizer
    from eegproc.model_explainability.counterfactuals.loss import CounterfactualLoss

    class Model:
        classification_level, use_decoder = "trial", True
        use_gcn_gru_branch, gcn_gru_decoder, gcn_gru_feature_dim = True, object(), 4
    with pytest.raises(ValueError, match="repair_only requires"):
        CounterfactualOptimizer(Model(), loss=CounterfactualLoss(physiological_weight=0.0), repair_only=True)
