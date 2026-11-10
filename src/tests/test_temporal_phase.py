"""Temporal smoothing of the latent edit, phase-invariant distance, seams."""

import csv
import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from scipy.signal import hilbert  # noqa: E402

from eegproc.model_explainability.counterfactuals.loss import CounterfactualLoss  # noqa: E402
from eegproc.model_explainability.counterfactuals.temporal_phase import (  # noqa: E402
    edit_structure, envelope_distance_np, envelope_distance_tf, envelope_np, envelope_tf,
    latent_scale, resolve_pool, temporal_penalties)
from eegproc.model_explainability.typicality.physiology import (  # noqa: E402
    make_vcsc_loss, seam_diagnostics)

FS = 128.0


def _sines(windows=4, samples=128, phase=0.0, amplitude=1.0, frequency=10.0):
    t = np.arange(windows * samples) / FS
    wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return np.tile(wave[:, None], (1, 42)).reshape(windows, samples, 42)


# ------------------------------------------------------------- envelope

def test_envelope_tf_matches_numpy_and_scipy_hilbert():
    rng = np.random.default_rng(0)
    for samples in (32, 30, 128):
        x = rng.normal(size=(1, 3, samples, 42)).astype(np.float32)
        pool = resolve_pool(samples)
        np.testing.assert_allclose(envelope_tf(x).numpy(), envelope_np(x), rtol=1e-4, atol=1e-5)
        tapered = x[0] * np.hanning(samples)[None, :, None]
        reference = np.abs(hilbert(tapered, axis=1))
        reference = reference.reshape(3, samples // pool, pool, 42).mean(axis=2)
        np.testing.assert_allclose(envelope_np(x), reference, rtol=1e-10, atol=1e-12)
    y = rng.normal(size=(1, 3, 128, 42)).astype(np.float32)
    assert float(envelope_distance_tf(x, y)) == pytest.approx(envelope_distance_np(x, y), rel=1e-4)
    assert resolve_pool(128) == 32 and resolve_pool(100) == 25 and resolve_pool(7) == 7
    with pytest.raises(ValueError, match="must divide"):
        resolve_pool(128, 30)


def test_envelope_distance_ignores_phase_but_sees_power():
    reference = _sines()
    shifted = _sines(phase=np.pi / 2)
    louder = _sines(amplitude=2.0)
    mse_shift = float(np.mean((shifted - reference) ** 2))
    assert mse_shift > 0.9  # a quarter-period shift costs as much as the signal power
    assert envelope_distance_np(shifted, reference) < 0.05 * mse_shift
    assert envelope_distance_np(louder, reference) == pytest.approx(np.log(2.0) ** 2, rel=0.02)


def test_envelope_distance_gradient_is_finite_on_silent_input():
    x = tf.Variable(np.zeros((1, 2, 32, 42), np.float32))
    reference = np.random.default_rng(1).normal(size=(1, 2, 32, 42)).astype(np.float32)
    with tf.GradientTape() as tape:
        value = envelope_distance_tf(x, reference)
    gradient = tape.gradient(value, x)
    assert np.isfinite(float(value)) and np.isfinite(gradient.numpy()).all()


# ------------------------------------------------------------ smoothing

def _smooth_abs(x, eps=1e-3):
    return np.sqrt(x ** 2 + eps ** 2) - eps


def test_temporal_penalties_match_numpy_and_vanish_for_no_edit():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(1, 3, 8, 5)).astype(np.float32)
    z_prime = z + rng.normal(size=z.shape).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, size=5).astype(np.float32)
    sparsity, tv = temporal_penalties(z_prime, z, scale)
    delta = ((z_prime - z) / scale).reshape(-1, 5).astype(np.float64)
    assert float(sparsity) == pytest.approx(_smooth_abs(delta).mean(), rel=1e-5)
    assert float(tv) == pytest.approx(_smooth_abs(np.diff(delta, axis=0)).mean(), rel=1e-5)
    # The flattening runs over windows then timesteps: window boundaries are
    # adjacent in time, so a constant edit has zero TV everywhere.
    constant = z + np.float32(0.7)
    _, tv_constant = temporal_penalties(constant, z)
    assert float(tv_constant) == pytest.approx(0.0, abs=1e-6)

    variable = tf.Variable(z)
    with tf.GradientTape() as tape:
        sparsity, tv = temporal_penalties(variable, z, scale)
        total = sparsity + tv
    gradient = tape.gradient(total, variable).numpy()
    assert float(total) == 0.0 and np.isfinite(gradient).all() and np.abs(gradient).max() == 0.0


def test_edit_structure_counts_episodes():
    z = np.zeros((1, 2, 10, 3))
    edit = z.copy()
    edit[0, 0, 2:5] = 1.0          # first episode
    edit[0, 1, 1:3] = -2.0         # second episode (time steps 11-12)
    stats = edit_structure(z, edit, scale=np.array([1.0, 1.0, 2.0]))
    assert stats["n_episodes"] == 2
    assert stats["active_fraction"] == pytest.approx(5 / 20)
    assert stats["standardized"] is True
    assert stats["l1_mean"] == pytest.approx(np.mean(np.abs(edit.reshape(-1, 3) / [1.0, 1.0, 2.0])))
    empty = edit_structure(z, z)
    assert empty["n_episodes"] == 0 and empty["active_fraction"] == 0.0 and empty["l1_mean"] == 0.0


def test_latent_scale_matches_pooled_std_and_floors_dead_features():
    rng = np.random.default_rng(3)
    blocks = [rng.normal(loc=[i, -i, 2 * i, 5.0], scale=[1.0, 3.0, 0.5, 0.0], size=(1, 2, 7, 4)) for i in range(5)]
    scale = latent_scale(iter(blocks))
    pooled = np.concatenate([b.reshape(-1, 4) for b in blocks])
    expected = pooled.std(axis=0, ddof=1)
    np.testing.assert_allclose(scale[:3], expected[:3], rtol=1e-10)
    assert scale[3] == pytest.approx(1e-3 * np.median(expected[:3]))
    with pytest.raises(ValueError):
        latent_scale(iter([np.zeros((1, 4))]))


# ---------------------------------------------------------------- loss

def _identity_decoder(z):
    return {"joint": z}


def _central(loss, z_prime, z):
    return loss.central_loss(logits=tf.constant([[0.0, 1.0]]), target_class=1, z_prime=z_prime, z=z,
                             x=z, decoder=_identity_decoder)[0]


def test_default_loss_keeps_historical_terms_and_options_are_validated():
    rng = np.random.default_rng(4)
    z = tf.constant(rng.normal(size=(1, 2, 32, 42)).astype(np.float32))
    z_prime = z + 0.1
    terms = _central(CounterfactualLoss(), z_prime, z)
    assert "temporal_sparsity" not in terms and "temporal_tv" not in terms
    with pytest.raises(ValueError, match="decoded_metric"):
        CounterfactualLoss(decoded_metric="l1")
    with pytest.raises(ValueError, match="temporal_tv_weight"):
        CounterfactualLoss(temporal_tv_weight=-1.0)
    calibration = {"c_hat": 0, "w_hat": 0, "sigma_raw": 1, "sigma_spec": 1}
    for bad in (np.zeros(3), np.array([1.0, np.nan]), np.ones((2, 2))):
        with pytest.raises(ValueError, match="temporal_scale"):
            make_vcsc_loss(calibration, temporal_scale=bad)


def test_temporal_terms_and_envelope_metric_enter_the_objective():
    rng = np.random.default_rng(5)
    z = tf.constant(rng.normal(size=(1, 2, 32, 42)).astype(np.float32))
    z_prime = z + tf.constant(rng.normal(scale=0.3, size=(1, 2, 32, 42)).astype(np.float32))
    scale = rng.uniform(0.5, 2.0, size=42)
    calibration = {"c_hat": 0, "w_hat": 0, "sigma_raw": 1, "sigma_spec": 1}
    loss = make_vcsc_loss(calibration, temporal_scale=scale, temporal_sparsity_weight=0.2,
                          temporal_tv_weight=0.3, decoded_metric="envelope", physiological_weight=0.0)
    terms = _central(loss, z_prime, z)
    sparsity, tv = temporal_penalties(z_prime, z, scale)
    assert float(terms["temporal_sparsity"]) == pytest.approx(float(sparsity), rel=1e-6)
    assert float(terms["weighted_temporal_tv"]) == pytest.approx(0.3 * float(tv), rel=1e-6)
    assert float(terms["decoded"]) == pytest.approx(envelope_distance_np(z_prime.numpy(), z.numpy()), rel=1e-3)
    expected_total = sum(float(terms[f"weighted_{name}"]) for name in
                         ("target", "latent", "decoded", "physiological", "temporal_sparsity", "temporal_tv"))
    assert float(terms["total"]) == pytest.approx(expected_total, rel=1e-5)
    # Only the sparsity weight: both terms are still logged, TV contributes 0.
    only_sparsity = make_vcsc_loss(calibration, temporal_scale=scale, temporal_sparsity_weight=0.2)
    terms = _central(only_sparsity, z_prime, z)
    assert float(terms["weighted_temporal_tv"]) == 0.0 and float(terms["temporal_tv"]) > 0


# ---------------------------------------------------------------- seams

def test_seam_diagnostics_separate_continuous_from_seamed_signals():
    continuous = _sines()
    # Each window restarts the oscillation with a phase 1 rad ahead of where a
    # continuous signal would be: a seam of exactly 1 rad at every boundary.
    seamed = np.stack([_sines(windows=1, phase=p)[0] for p in (0.0, 1.0, 2.0, 3.0)])
    smooth = seam_diagnostics(continuous, n_channels=14)
    broken = seam_diagnostics(seamed, n_channels=14)
    assert smooth["n_boundaries"] == broken["n_boundaries"] == 3
    assert np.all(smooth["phase_residual_median_rad"] < 0.01)
    np.testing.assert_allclose(broken["phase_residual_median_rad"], 1.0, atol=0.25)
    assert np.all(broken["amplitude_jump_ratio_median"] > smooth["amplitude_jump_ratio_median"])
    # Band-major input is reordered, not misread.
    band_major = continuous.reshape(4, 128, 14, 3).transpose(0, 1, 3, 2).reshape(4, 128, 42)
    reordered = seam_diagnostics(band_major, n_channels=14, feature_order="band-major")
    np.testing.assert_allclose(reordered["phase_residual_median_rad"], smooth["phase_residual_median_rad"])
    single = seam_diagnostics(continuous[:1], n_channels=14)
    assert single["n_boundaries"] == 0 and single["phase_residual_median_rad"] is None
    flat = seam_diagnostics(np.ones_like(continuous), n_channels=14)
    assert np.all(np.isnan(flat["amplitude_jump_ratio_median"]))
    with pytest.raises(ValueError):
        seam_diagnostics(continuous[..., :40], n_channels=14)


def test_runner_seams_are_json_safe():
    from types import SimpleNamespace
    from eegproc.model_explainability.typicality.runner import _seams
    dataset = SimpleNamespace(normalization_scale=None, normalization_offset=None, feature_order="channel-major")
    flat = _seams(dataset, 0, np.ones((1, 3, 32, 42), np.float32))
    assert flat["amplitude_jump_ratio_median"] == [None, None, None]
    json.dumps(flat, allow_nan=False)
    scaled = SimpleNamespace(normalization_scale=np.full((1, 3, 1, 1), 2.0), normalization_offset=np.zeros((1, 3, 1, 1)),
                             feature_order="channel-major")
    values = _sines(windows=3, samples=32)
    assert _seams(scaled, 0, values) == _seams(dataset, 0, 2.0 * values)


# ------------------------------------------------------------ end to end

def test_runner_with_temporal_smoothing_and_envelope_distance(tmp_path, monkeypatch):
    from eegproc.model_explainability.typicality import runner as study
    from eegproc.model_explainability.typicality.artifacts import completed_attempt
    from tests.test_vcsc_repair_and_amplitude_spectral import _small_study
    manifest, data_path = _small_study(tmp_path)
    out = tmp_path / "study"
    args = study.parse_args([
        "--models-json", str(manifest), "--task", "valence", "--trials-npz", str(data_path),
        "--out-dir", str(out), "--directions", "0_to_1", "--subjects", "0", "--trial-ids", "0", "--max-steps", "2", "--log-every", "0",
        "--include-target-latent", "--include-typicality-no-physiology", "--include-repair-baseline",
        "--temporal-sparsity-weight", "0.1", "--temporal-tv-weight", "0.2", "--decoded-distance", "envelope"])
    study.run(args)

    study_json = json.loads((out / "0_to_1/study.json").read_text())
    assert "fused lasso" in study_json["temporal_smoothing_objective"]
    assert "envelope" in study_json["decoded_distance"]
    assert any(name.endswith("temporal_phase.py") for name in study_json["source_sha256"])
    scale = json.loads((out / "0_to_1/subject_0/calibration/latent_scale.json").read_text())
    assert scale["n_source_trials"] == 4 and all(v > 0 for v in scale["latent_std"])

    losses = json.loads((out / "0_to_1/subject_0/fold.json").read_text())["objective_losses"]
    assert losses["target_latent"]["temporal_sparsity_weight"] == 0
    assert losses["target_latent"]["temporal_tv_weight"] == 0
    for arm in ("base", "typicality", "typicality_no_physiology"):
        assert losses[arm]["temporal_sparsity_weight"] == 0.1
        assert losses[arm]["temporal_tv_weight"] == 0.2
        assert losses[arm]["decoded_metric"] == "envelope"

    for arm in ("target_latent", "base", "typicality", "typicality_no_physiology"):
        attempt = completed_attempt(out / f"0_to_1/subject_0/trial_0/{arm}")
        result = json.loads((attempt / "result.json").read_text())
        assert result["latent_edit"]["standardized"] is True
        assert ("temporal_sparsity" in result["selected_losses"]) == (arm != "target_latent")
        decoded = result["decoded_trials"]["joint"]
        assert decoded["decoded_change_env"] >= 0 and decoded["original_reconstruction_env"] >= 0
        assert result["repair_baseline"]["explanation_change_env"] >= 0
        assert result["repair_baseline"]["repair_change_env"] >= 0
        # Every arm's delivered EEG is re-encoded, with the reconstruction
        # and the repair scored as controls.
        for key in ("decoded_counterfactual", "decoded_reconstruction", "decoded_repair"):
            scored = result[key]
            assert scored["target_class"] == 1 and len(scored["probabilities"]) == 2
            assert scored["target_probability"] == scored["probabilities"][1]
            assert scored["success"] == (scored["predicted_class"] == 1
                                         and scored["target_probability"] >= args.target_probability)
            assert scored["typical"] == (scored["discrepancy"] <= scored["source_threshold"])
        assert result["decoded_counterfactual"]["input"] == "decoded_counterfactual_eeg_reencoded_by_frozen_classifier"
        seams = json.loads((attempt / "seams.json").read_text())
        assert set(seams) == {"band_names", "original", "reconstruction", "counterfactual", "repair"}
        assert seams["counterfactual"]["n_boundaries"] == 2
        assert len(seams["counterfactual"]["phase_residual_median_rad"]) == 3
        physiology = json.loads((attempt / "physiology.json").read_text())
        assert set(physiology) == {"original", "reconstruction", "counterfactual", "repair"}
        assert physiology["counterfactual"]["graded_count"] >= 1
        assert physiology["counterfactual"]["checks"]["amplitude"]["deviation"]["unit"] == "half_interval_widths"
    assert len(list(out.rglob("*.npz"))) == 1

    with open(out / "report/trial_metrics.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for column in ("reencoded_target_success", "reencoded_typical", "reencoded_reconstruction_target_success",
                   "reencoded_repair_typical", "physiological_worst_exceedance", "physiological_exceedance_amplitude", "delta_env", "e_rec_env", "delta_env_explanation", "delta_env_repair",
                   "latent_edit_tv", "latent_edit_episodes"):
        assert all(row[column] != "" for row in rows if row["status"] == "completed"), column
    with open(out / "report/population_counterfactuals.csv", newline="") as handle:
        populations = list(csv.DictReader(handle))
    assert populations and all(row["reencoded_target_success_percent"] != "" for row in populations)
    assert all(row["reencoded_repair_typical_percent"] != "" for row in populations)
