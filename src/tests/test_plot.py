import os
import re
import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt

from eegproc.plotting.plots import plot_eeg_features


def make_df():
    n = 10
    t = np.linspace(0, 1, n)
    return pd.DataFrame(
        {
            "AF3_wentropy": np.sin(2 * np.pi * 1 * t),
            "F7_wentropy": np.cos(2 * np.pi * 1 * t),
            "AF3_alpha_entropy": 0.5 * np.sin(2 * np.pi * 2 * t),
            "AF3_theta_entropy": 0.4 * np.cos(2 * np.pi * 3 * t),
            "F7_alpha_entropy": 0.6 * np.sin(2 * np.pi * 4 * t),
        }
    )


def test_plot_raises_on_empty_and_invalid_slice(tmp_path):
    df = make_df()
    # Empty input
    with pytest.raises(ValueError, match="input_data is empty"):
        plot_eeg_features(pd.DataFrame(), save_path=tmp_path / "out.png")
    with pytest.raises(ValueError, match="Empty window"):
        plot_eeg_features(df, start_row=5, end_row=5, save_path=tmp_path / "out.png")


def test_plot_filters_channels_and_bands_and_saves(tmp_path, monkeypatch):
    """
    Checks if only chosen columns were selected
    """
    df = make_df()
    called = {"nrows": None}

    orig_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        # Capture nrows regardless of positional/keyword usage
        if "nrows" in kwargs:
            called["nrows"] = kwargs["nrows"]
        elif len(args) >= 1:
            called["nrows"] = args[0]
        return orig_subplots(*args, **kwargs)

    monkeypatch.setattr(
        plt, "subplots", spy_subplots
    )  # This checks on subplot attributes

    out_path = tmp_path / "filtered.png"
    plot_eeg_features(
        df,
        title="Test",
        seconds=4.0,
        start_row=0,
        end_row=len(df),
        save_path=str(out_path),
        channels=["AF3"],
        frequency_bands=["alpha", "theta"],
    )

    # Only two band columns should be plotted
    assert called["nrows"] == 2
    assert out_path.exists() and out_path.stat().st_size > 0
