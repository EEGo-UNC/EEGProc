"""Verify which archived and decoded EEG signals enter the comparison."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eegproc.model_explainability.report_iclr_waveform_subject_invariance import (
    _archived_decoded_eeg, _decoded_source_reference, _fit_waveform_region,
    _waveform_discrepancy,
)


def test_counterfactual_reads_paired_start_and_final_waveforms(tmp_path):
    attempt = tmp_path / "attempt_0001"
    attempt.mkdir()
    result = attempt / "result.json"
    original = np.zeros((2, 3, 4), dtype=np.float32)
    archive = attempt / "counterfactual.npz"
    np.savez_compressed(
        archive, x=original[None], x_prime_joint=np.ones((1, *original.shape)),
        x_reconstructed_joint=np.full((1, *original.shape), 99),
    )
    starting, decoded, actual_path = _archived_decoded_eeg(
        result, original_x=original, report_output="joint",
    )
    assert actual_path == archive
    np.testing.assert_array_equal(starting, 99)
    np.testing.assert_array_equal(decoded, 1)
    with pytest.raises(ValueError, match="original trial identity"):
        _archived_decoded_eeg(
            result, original_x=np.full_like(original, 2), report_output="joint",
        )


def test_source_reference_is_decoded_from_selected_typical_trials(monkeypatch):
    import tensorflow as tf
    from eegproc.model_explainability.model_agnostic import sic_adapter

    class Adapter:
        def initial_state(self, raw):
            return raw + tf.constant(3, dtype=tf.float32)

        def reconstruct(self, state, raw):
            return {"joint": state * tf.constant(2, dtype=tf.float32)}

    monkeypatch.setattr(sic_adapter, "create_sic_adapter", lambda **kwargs: Adapter())
    features = np.arange(36, dtype=np.float32).reshape(3, 2, 3, 2)
    dataset = SimpleNamespace(features=features)
    study = {"arguments": {
        "model_module": "unused", "decoder_mode": "joint",
        "fixed_joint_alpha": None, "report_output": "joint",
    }}
    reference = _decoded_source_reference(
        study, Path("unused.keras"), dataset, [0, 2], batch_size=1,
    )
    np.testing.assert_array_equal(reference, (features[[0, 2]] + 3) * 2)
    assert not np.array_equal(reference, features[[0, 2]])


def test_discrepancy_uses_typicality_formula_in_decoded_waveform_space():
    reference = np.asarray([
        [[[0.0, 2.0]]], [[[2.0, 4.0]]],
    ])
    mean, variance, n = _fit_waveform_region(reference, variance_floor=1e-6)
    assert n == 2
    np.testing.assert_array_equal(mean, [[[1.0, 3.0]]])
    np.testing.assert_array_equal(variance, [[[1.0, 1.0]]])
    assert _waveform_discrepancy(np.asarray([[[3.0, 4.0]]]), mean, variance) == 2.5
