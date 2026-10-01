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
    with pytest.raises(ValueError, match="empty row range"):
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


# --------------------------------------------------------------------------
# Regressions for the v2 plot fixes. None of these were covered before.
# --------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")


def _band_frame(rows: int = 20) -> pd.DataFrame:
    t = np.arange(rows)
    return pd.DataFrame({
        "AF3_alpha": np.sin(0.1 * t),
        "AF3_betaL": np.cos(0.1 * t),
        "F3_alpha": np.sin(0.2 * t),
        "F7_alpha": np.cos(0.2 * t),
    })


def test_channel_filter_does_not_match_by_substring():
    """`F3` must not also select `AF3` — the montage contains both."""
    fig, axes = plot_eeg_features(_band_frame(), channels=["F3"])
    labels = [ax.get_ylabel() for ax in axes]
    plt.close(fig)
    assert labels == ["F3_alpha"], f"substring over-selection returned {labels}"


def test_regex_metacharacters_in_names_do_not_raise():
    with pytest.raises(ValueError, match="unknown channel"):
        plot_eeg_features(_band_frame(), channels=["AF3("])


def test_unknown_band_is_reported_not_silently_empty():
    with pytest.raises(ValueError, match="unknown band"):
        plot_eeg_features(_band_frame(), frequency_bands=["gamma"])


def test_band_filter_on_bandless_columns_explains_itself():
    """shannons_entropy emits {channel}_entropy; a band filter cannot match."""
    df = pd.DataFrame({"AF3_entropy": np.arange(10.0), "F7_entropy": np.arange(10.0)})
    with pytest.raises(ValueError, match="none of these columns carry a band"):
        plot_eeg_features(df, frequency_bands=["alpha"])


def test_no_figure_is_leaked_when_selection_is_empty():
    before = set(plt.get_fignums())
    with pytest.raises(ValueError):
        plot_eeg_features(_band_frame(), channels=["NOPE"])
    assert set(plt.get_fignums()) == before, "a figure leaked on the error path"


def test_caller_can_close_the_returned_figure():
    before = set(plt.get_fignums())
    fig, axes = plot_eeg_features(_band_frame(), channels=["AF3"])
    assert len(axes) == 2
    plt.close(fig)
    assert set(plt.get_fignums()) == before


def test_time_axis_accounts_for_window_overlap():
    """At 50% overlap consecutive windows advance by half a window, not a whole one."""
    fig, axes = plot_eeg_features(_band_frame(rows=5), channels=["F7"], seconds=4.0, overlap=0.5)
    x = axes[0].lines[0].get_xdata()
    plt.close(fig)
    np.testing.assert_allclose(x, [0.0, 2.0, 4.0, 6.0, 8.0])


def test_default_plots_every_row():
    """end_row previously defaulted to 1, rendering a single invisible point."""
    fig, axes = plot_eeg_features(_band_frame(rows=25), channels=["F7"])
    n = len(axes[0].lines[0].get_xdata())
    plt.close(fig)
    assert n == 25


def test_non_numeric_columns_are_skipped():
    df = _band_frame()
    df["source_file"] = "subject01.csv"
    fig, axes = plot_eeg_features(df)
    labels = [ax.get_ylabel() for ax in axes]
    plt.close(fig)
    assert "source_file" not in labels


def test_too_many_subplots_is_refused_with_guidance():
    wide = pd.DataFrame({f"C{i}_alpha": np.arange(5.0) for i in range(50)})
    with pytest.raises(ValueError, match="max_subplots"):
        plot_eeg_features(wide)
