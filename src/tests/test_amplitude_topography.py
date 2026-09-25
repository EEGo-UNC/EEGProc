"""The optional map uses saved physiology amplitudes rather than waveform means."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegproc.model_explainability.counterfactuals import topography


def _attempt(tmp_path):
    directory = tmp_path / "attempt_0001"
    directory.mkdir()
    names = np.asarray(list(topography.DREAMER_POSITIONS))
    before = np.arange(42, dtype=float).reshape(14, 3) + 1
    after = before + np.arange(42, dtype=float).reshape(14, 3) / 10
    original = before + 2
    np.savez_compressed(directory / "counterfactual.npz",
                        x=np.zeros((1, 1, 8, 42)),
                        x_reconstructed_joint=np.zeros((1, 1, 8, 42)),
                        x_prime_joint=np.ones((1, 1, 8, 42)),
                        channel_names=names)
    for label, values in (("counterfactual", after), ("reconstruction", before),
                          ("original", original)):
        np.savez_compressed(directory / f"physiology_{label}.npz", amplitude=values)
    return directory, before, after, original


def test_amplitude_map_subtracts_saved_arrays(tmp_path, monkeypatch):
    directory, before, after, original = _attempt(tmp_path)
    captured = []

    def capture(values, **kwargs):
        captured.append((values.copy(), kwargs))
        return plt.figure(), {}

    monkeypatch.setattr(topography, "plot_band_topographies", capture)
    assert topography.main([str(directory / "counterfactual.npz"), "--branch", "joint",
                            "--measure", "amplitude", "--no-show"]) == 0
    np.testing.assert_allclose(captured[-1][0], (after - before).T)
    assert captured[-1][1]["signed"] is True
    assert "title" not in captured[-1][1]
    assert (directory / "counterfactual_joint_difference_amplitude_topography.png").exists()

    assert topography.main([str(directory / "counterfactual.npz"), "--branch", "joint",
                            "--reference", "input", "--measure", "amplitude",
                            "--no-show"]) == 0
    np.testing.assert_allclose(captured[-1][0], (after - original).T)


def test_amplitude_map_requires_saved_diagnostics(tmp_path):
    directory, _, _, _ = _attempt(tmp_path)
    (directory / "physiology_counterfactual.npz").unlink()
    with pytest.raises(FileNotFoundError, match="physiology_counterfactual.npz"):
        topography.main([str(directory / "counterfactual.npz"), "--branch", "joint",
                         "--measure", "amplitude", "--no-show"])
