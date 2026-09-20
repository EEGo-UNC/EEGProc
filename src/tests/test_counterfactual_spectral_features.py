import csv
import json

import numpy as np
import pytest

from eegproc.model_explainability.typicality.spectral_features import (
    analyze_counterfactual_artifact,
    write_spectral_features,
)


def _band_trial(frequencies, *, windows=4, samples=128, fs=128.0):
    time = np.arange(samples) / fs
    values = np.empty((1, windows, samples, 6), dtype=np.float32)
    for window in range(windows):
        columns = []
        for channel_scale in (1.0, 0.7):
            columns.extend(
                channel_scale * np.sin(2 * np.pi * frequency * time)
                for frequency in frequencies
            )
        values[0, window] = np.stack(columns, axis=-1)
    return values


def test_saved_counterfactual_produces_paired_spectral_entropy_rows(tmp_path):
    artifact = tmp_path / "counterfactual.npz"
    np.savez_compressed(
        artifact,
        x_reconstructed_joint=_band_trial((5.0, 10.0, 16.0)),
        x_prime_joint=_band_trial((7.0, 12.0, 25.0)),
        channel_names=np.asarray(["AF3", "AF4"]),
        band_names=np.asarray(["theta", "alpha", "beta"]),
        feature_order=np.asarray("channel-major"),
    )
    (tmp_path / "result.json").write_text(json.dumps({
        "task": "valence", "subject_id": 2, "trial_id": 7,
        "objective": "typicality", "report_output": "joint",
    }))

    rows = analyze_counterfactual_artifact(artifact, fs=128.0)

    assert len(rows) == 6
    af3_theta = next(row for row in rows if row["channel"] == "AF3" and row["band"] == "theta")
    assert af3_theta["reference_peak_frequency_hz"] == pytest.approx(5.0)
    assert af3_theta["counterfactual_peak_frequency_hz"] == pytest.approx(7.0)
    assert af3_theta["delta_peak_frequency_hz"] == pytest.approx(2.0)
    assert af3_theta["reference_spectral_centroid_hz"] == pytest.approx(5.0, abs=0.2)
    assert af3_theta["counterfactual_spectral_centroid_hz"] == pytest.approx(7.0, abs=0.2)
    assert 0 <= af3_theta["reference_spectral_entropy_median"] <= 1
    assert af3_theta["reference"] == "decoded_original_reconstruction"
    assert af3_theta["n_windows"] == 4

    output = tmp_path / "spectral_entropy_features.csv"
    written = write_spectral_features([tmp_path], output, fs=128.0)
    with output.open(newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(written) == len(saved) == 6
    assert "delta_spectral_entropy_median" in saved[0]


def test_requires_decoded_original_reconstruction(tmp_path):
    artifact = tmp_path / "counterfactual.npz"
    np.savez_compressed(
        artifact,
        x=np.zeros((1, 2, 128, 6)),
        x_prime_joint=np.ones((1, 2, 128, 6)),
    )
    with pytest.raises(KeyError, match="x_reconstructed_joint"):
        analyze_counterfactual_artifact(artifact, fs=128.0)
