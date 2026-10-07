import numpy as np
import pandas as pd
import pytest
from eegproc.preprocessing import (
    apply_detrend,
    numeric_interp,
    sosfiltfilt_safe,
    apply_notch_once,
    bandpass_filter,
    FREQUENCY_BANDS,
)

def trim_edges(arr, fs, seconds=1):
    n = int(fs * seconds)
    return arr[n:-n]

# --------
# helpers
# --------
def test_apply_detrend_invalid_raises():
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        apply_detrend("invalid", df)


def test_apply_detrend_constant_and_linear_paths():
    df = pd.DataFrame({"A": np.linspace(0, 1, 256), "B": np.linspace(1, 0, 256)})
    out_const = apply_detrend("constant", df)
    out_lin = apply_detrend("linear", df)
    # Constant detrend returns 0 mean
    assert np.allclose(out_const.mean(axis=0).to_numpy(), 0.0, atol=1e-8)
    # Detrend removes linear trend so peak to peak is smaller than original
    assert np.max(out_lin["A"]) - np.min(out_lin["A"]) < np.max(df["A"]) - np.min(
        df["A"]
    )


def test_numeric_interp_only_numeric_and_interpolates():
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "C": ["x", "y", "z"]})
    out = numeric_interp(df)
    assert list(out.columns) == ["A"]
    assert np.allclose(out["A"].to_numpy(), [1.0, 2.0, 3.0])


def test_sosfiltfilt_safe_interpolates_interior_nans():
    y = np.array([0.0, np.nan, 1.0, np.nan, 2.0, 3.0] + [0.0] * 10, dtype=float)
    sos = np.array([[1, 0, 0, 1, 0, 0]])
    out = sosfiltfilt_safe(sos, y)
    assert np.isfinite(out).all()


def test_apply_notch_once():
    # uses 50Hz (instead of 60Hz to be far from nyquist at 64Hz)
    t = np.arange(0, 2 * FS) / FS
    x = (
        np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 25 * t)
    )
    df = pd.DataFrame({"A1": x})
    out = apply_notch_once(df, notch_hz=[25, 50], notch_q=30.0, nyq=FS / 2.0)
    out = trim_edges(out, FS, seconds=0.5)

    # Values in out should be close to 0
    assert np.isclose(out.to_numpy(), 0.0, rtol=0.5, atol=0.25).all()


"""Bandpass Filter tests"""
ORDERED_BANDS = [
    ("delta", FREQUENCY_BANDS["delta"]),
    ("theta", FREQUENCY_BANDS["theta"]),
    ("alpha", FREQUENCY_BANDS["alpha"]),
    ("betaL", FREQUENCY_BANDS["betaL"]),
    ("betaH", FREQUENCY_BANDS["betaH"]),
    ("gamma", FREQUENCY_BANDS["gamma"]),
]

PASS_CASES = [
    (2.0, "delta"),
    (6.0, "theta"),
    (10.0, "alpha"),
    (15.0, "betaL"),
    (25.0, "betaH"),
    (35.0, "gamma"),
]

STOP_CASES = [
    (2.0, "betaH"),
    (6.0, "gamma"),
    (10.0, "delta"),
    (15.0, "delta"),
    (25.0, "theta"),
    (35.0, "alpha"),
]

FS = 128


@pytest.mark.parametrize("sine_df", [2.0], indirect=True)
@pytest.mark.filterwarnings("ignore:reref=True ignored:RuntimeWarning")
def test_bandpass_filter_column_output(sine_df):
    out = bandpass_filter(
        sine_df, FS, bands=FREQUENCY_BANDS, reref=False, detrend=False
    )

    in_cols = list(sine_df.columns)
    for c in in_cols:
        for band_name in FREQUENCY_BANDS.keys():
            expected_col = f"{c}_{band_name}"
            assert expected_col in out.columns, f"Missing column {expected_col}"


@pytest.mark.parametrize(
    "sine_df,freq,band", [(f, f, b) for f, b in PASS_CASES], indirect=["sine_df"]
)
@pytest.mark.filterwarnings("ignore:reref=True ignored:RuntimeWarning")
def test_bandpass_filter_passband_gain(sine_df, freq, band):
    out = bandpass_filter(
        sine_df, FS, bands=FREQUENCY_BANDS, reref=False, detrend=False
    )

    x = trim_edges(sine_df["A1"].to_numpy(), FS, seconds=1)
    y = trim_edges(out[f"A1_{band}"].to_numpy(), FS, seconds=1)

    in_std = x.std()
    out_std = y.std()
    gain = out_std / (in_std + 1e-12)

    # Allow moderate ripple
    assert gain > 0.80, f"{band} should pass {freq} Hz (gain={gain:.3f})"


@pytest.mark.parametrize(
    "sine_df,freq,band", [(f, f, b) for f, b in STOP_CASES], indirect=["sine_df"]
)
@pytest.mark.filterwarnings("ignore:reref=True ignored:RuntimeWarning")
def test_bandpass_filter_stopband_attenuation(sine_df, freq, band):
    out = bandpass_filter(
        sine_df, FS, bands=FREQUENCY_BANDS, reref=False, detrend=False
    )

    x = trim_edges(sine_df["A1"].to_numpy(), FS, seconds=1)
    y = trim_edges(out[f"A1_{band}"].to_numpy(), FS, seconds=1)

    in_std = x.std()
    out_std = y.std()
    gain = out_std / (in_std + 1e-12)

    # -10 dB ≈ 0.316x
    assert gain < 0.316, f"{band} should attenuate {freq} Hz (gain={gain:.3f})"


@pytest.mark.parametrize(
    "sine_df,freq,band", [(f, f, b) for f, b in PASS_CASES], indirect=["sine_df"]
)
@pytest.mark.filterwarnings("ignore:reref=True ignored:RuntimeWarning")
def test_bandpass_optional_flags_do_not_distort_gain(sine_df, freq, band):
    """
    Sanity-check that enabling reref/detrend doesn't collapse the signal.
    A single signal alone should not allow reref.
    """
    out = bandpass_filter(sine_df, FS, bands=FREQUENCY_BANDS, reref=True, detrend=True)

    x = trim_edges(sine_df["A1"].to_numpy(), FS, seconds=1)
    y = trim_edges(out[f"A1_{band}"].to_numpy(), FS, seconds=1)

    gain = y.std() / (x.std() + 1e-12)
    assert (
        gain > 0.75
    ), f"{band} with reref/detrend should still pass {freq} Hz (gain={gain:.3f})"


def test_bandpass_low_high_path_valid_and_invalid():
    df = pd.DataFrame({"A": np.random.randn(256), "B": np.random.randn(256)})
    # invalid when bands=None but cuts missing/invalid
    with pytest.raises(ValueError):
        bandpass_filter(df, FS, bands=None, low=None, high=None, reref=False, detrend=False)
    with pytest.raises(ValueError):
        bandpass_filter(df, FS, bands=None, low=40.0, high=10.0, reref=False, detrend=False)
    with pytest.raises(ValueError):
        bandpass_filter(df, FS, bands=None, low=0.0, high=FS/2, reref=False, detrend=False)

    # should be valid
    out = bandpass_filter(df, FS, bands=None, low=8.0, high=12.0, reref=False, detrend=True)
    assert list(out.columns) == list(df.columns)
    # detrend=True in low/high path approx 0 mean
    assert np.allclose(out.mean().to_numpy(), 0.0, atol=1e-6)
