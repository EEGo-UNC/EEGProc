"""Helpers of the reconstruction comparison script (no checkpoints needed)."""
import json

import numpy as np
import pytest
import tensorflow as tf

from eegproc.deep_learning.joint_architectures.sic_common.model import (
    temporal_reconstruction_losses,
)
from eegproc.model_explainability.typicality.compare_reconstructions import (
    load_manifest,
    markdown_table,
    summarize,
    training_penalties,
)


def test_training_penalties_match_the_training_loss():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(6, 16, 42))
    r = x + rng.normal(scale=0.4, size=x.shape)
    seam, derivative = training_penalties(x, r)
    tf_seam, tf_derivative = temporal_reconstruction_losses(
        tf.constant(x, tf.float64), tf.constant(r, tf.float64), 6
    )
    np.testing.assert_allclose(seam, tf_seam, rtol=1e-10)
    np.testing.assert_allclose(derivative, tf_derivative, rtol=1e-10)
    assert training_penalties(x[:1], r[:1])[0] == 0.0


def _manifest(directory, entries):
    path = directory / "loso_zero_shot_models.json"
    path.write_text(json.dumps({"saved": True, "models": entries}))
    return path


def _entry(subject, name, path="/elsewhere/{}"):
    return {"stage": "zero_shot_source_model", "target_subject": subject,
            "filename": name, "path": path.format(name)}


def test_manifest_prefers_checkpoints_next_to_it(tmp_path):
    models = tmp_path / "loso_zero_shot_models"
    models.mkdir()
    for subject in (0, 13):
        (models / f"fold_{subject}.keras").write_bytes(b"")
    path = _manifest(tmp_path, [_entry(0, "fold_0.keras"), _entry(13, "fold_13.keras")])
    assert load_manifest(path) == {0: models / "fold_0.keras", 13: models / "fold_13.keras"}


def test_manifest_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(_manifest(tmp_path, [_entry(0, "missing.keras")]))
    entry = {**_entry(0, "a.keras"), "stage": "calibrated"}
    with pytest.raises(ValueError, match="zero_shot_source_model"):
        load_manifest(_manifest(tmp_path, [entry]))
    (tmp_path / "a.keras").write_bytes(b"")
    duplicate = [_entry(0, "a.keras", str(tmp_path / "{}"))] * 2
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(_manifest(tmp_path, duplicate))


def test_summary_and_table():
    rows = [
        {"subject": 0, "trial": t, "label": t % 2,
         "probabilities": [0.2, 0.8] if t % 2 else [0.7, 0.3],
         "mse": 0.5, "derivative": 0.1, "seam": 0.2 + t,
         "amplitude_jump": 2.0, "phase_residual": 1.1,
         "worst_family_exceedance": 0.1 * t, "all_pass": t == 0}
        for t in range(4)
    ]
    real = [{k: v for k, v in row.items()
             if k not in ("probabilities", "mse", "derivative", "seam")} for row in rows]
    summaries = {"real EEG": summarize(real), "run": summarize(rows)}
    assert summaries["run"]["balanced_accuracy"] == 1.0
    assert summaries["run"]["seam"][0] == pytest.approx(1.7)
    assert summaries["run"]["all_pass_percent"] == 25.0
    assert "balanced_accuracy" not in summaries["real EEG"]
    table = markdown_table(summaries, [0])
    assert "| real EEG | 4 | n/a | n/a | n/a | n/a | 2.00 |" in table
    assert "| run | 4 | 100.0% | 0.500 |" in table
