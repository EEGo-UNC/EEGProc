"""Wavelet Transform Tests"""

import numpy as np
import pandas as pd
import pywt
import math
import pytest

from .utils import *
from eegproc.featurization import (
    choose_dwt_level,
    dwt_subband_ranges,
    wavelet_band_energy,
    wavelet_entropy,
)

FS = 128


@pytest.mark.parametrize(
    "n_sec,wavelet,min_freq",
    [
        (8.0, "db4", 16.0),
        (8.0, "db4", 8.0),
        (8.0, "db4", 4.0),
        (8.0, "db4", 2.0),
        (8.0, "db4", 1.0),
        (8.0, "sym5", 6.0),
        (12.0, "db2", 3.0),
    ],
)
def test_choose_dwt_level_matches_formula(n_sec: float, wavelet, min_freq: float):
    n_samples = int(n_sec * FS)
    dec_len = pywt.Wavelet(wavelet).dec_len
    max_lvl = pywt.dwt_max_level(n_samples, dec_len)

    # target from implementation
    target = max(1, math.floor(math.log2(FS / max(min_freq, 1e-6)) - 1))
    expected = max(1, min(max_lvl, target))

    lvl = choose_dwt_level(n_samples, FS, wavelet, min_freq)

    assert 1 <= lvl <= max_lvl
    assert (
        lvl == expected
    ), f"expected {expected}, got {lvl} (max={max_lvl}, target={target})"


def test_choose_dwt_level_caps_at_max_for_tiny_minfreq():
    """Extremely small min_freq makes target very high; function must cap at max level."""
    n_samples = int(8.0 * FS)
    wavelet = "db4"
    max_lvl = pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet).dec_len)

    lvl = choose_dwt_level(n_samples, FS, wavelet, min_freq=1e-9)
    assert lvl == max_lvl


def test_choose_dwt_level_floors_at_one_when_minfreq_is_high():
    """min_freq near Nyquist should push target below 1; function must floor at 1."""
    n_samples = int(8.0 * FS)
    wavelet = "db4"

    lvl = choose_dwt_level(n_samples, FS, wavelet, min_freq=FS / 2)
    assert lvl == 1


def test_dwt_subband_ranges_exact_powers_of_two_level3():
    level = 3
    bands = dwt_subband_ranges(FS, level)

    expected = {
        "D1": (FS / 2**2, FS / 2**1),  # (32, 64)
        "D2": (FS / 2**3, FS / 2**2),  # (16, 32)
        "D3": (FS / 2**4, FS / 2**3),  # (8, 16)
        "A3": (0.0, FS / 2**4),  # (0, 8)
    }

    assert set(bands.keys()) == set(expected.keys())
    for k, (flo, fhi) in expected.items():
        blo, bhi = bands[k]
        assert np.isclose(blo, flo), f"{k} lo {blo} != {flo}"
        assert np.isclose(bhi, fhi), f"{k} hi {bhi} != {fhi}"


@pytest.mark.parametrize("level", [1, 2, 3, 5])
def test_dwt_subband_ranges_cover_and_partition(level):
    bands = dwt_subband_ranges(FS, level)

    for j in range(1, level + 1):
        assert f"D{j}" in bands
    assert f"A{level}" in bands

    lo_A, hi_A = bands[f"A{level}"]
    assert np.isclose(lo_A, 0.0)
    assert np.isclose(hi_A, FS / (2 ** (level + 1)))

    lo_D_last, _ = bands[f"D{level}"]
    _, hi_D1 = bands["D1"]
    assert np.isclose(lo_D_last, hi_A)  # A's high equals D{L}'s low
    assert np.isclose(hi_D1, FS / 2)  # D1's high hits Nyquist freq

    # Contiguity and non-overlap across D-bands
    for j in range(1, level):
        lo_next, hi_next = bands[f"D{j+1}"]
        lo_cur, hi_cur = bands[f"D{j}"]

        assert np.isclose(hi_next, lo_cur)
        assert lo_next < hi_next and lo_cur < hi_cur

    # Bandwidth halving for details: width(Dj+1) == 0.5 * width(Dj)
    if level >= 2:
        widths = []
        for j in range(1, level + 1):
            lo, hi = bands[f"D{j}"]
            widths.append(hi - lo)
        for j in range(len(widths) - 1):
            assert np.isclose(widths[j + 1], widths[j] / 2.0)


def test_wenergy_columns_and_rows(df_all_bands):
    bands = {
        "low": (0.0, 8.0),
        "mid": (8.0, 20.0),
        "high": (20.0, FS / 2),
    }
    window_sec = 4.0
    overlap = 0.5

    n = len(df_all_bands)
    nperseg = int(round(window_sec * FS))
    hop = int(round(nperseg * (1.0 - overlap)))
    expected_rows = 1 + (n - nperseg) // hop

    out = wavelet_band_energy(
        df_all_bands,
        FS,
        bands,
        wavelet="db4",
        mode="periodization",
        window_sec=window_sec,
        overlap=overlap,
    )

    expected_cols = [
        f"{ch}_{b}_wenergy" for ch in df_all_bands.columns for b in bands.keys()
    ]
    assert list(out.columns) == expected_cols
    assert len(out) == expected_rows


def test_wenergy_equals_wavelet_energy_when_bands_match_subbands():
    """
    Sum over bands should equal the total wavelet energy (sum of squares of all coeffs)
    for each window and channel (up to numerical tolerances and boundary effects).
    """
    x = make_sine(3.0, amp=1.2) + make_sine(10.0, amp=0.8) + make_sine(28.0, amp=0.6)
    df = pd.DataFrame({"A1": x})

    window_sec = 4.0
    overlap = 0.5
    nperseg = int(round(window_sec * FS))

    L = choose_dwt_level(n_samples=nperseg, fs=FS, wavelet="db4", min_freq=0.0)
    sub_bands = dwt_subband_ranges(FS, L)

    out = wavelet_band_energy(
        df,
        FS,
        sub_bands,
        wavelet="db4",
        mode="periodization",
        window_sec=window_sec,
        overlap=overlap,
    )

    # Compute total wavelet energy per window using pywt
    totals = []
    for start in range(0, len(df) - nperseg + 1, int(round(nperseg * (1.0 - overlap)))):
        y = df["A1"].iloc[start : start + nperseg].to_numpy()
        coeffs = pywt.wavedec(y, wavelet="db4", level=L, mode="periodization")
        approx = coeffs[0]
        details = coeffs[1:]
        total_energy = float(
            np.sum(approx.astype(float) ** 2)
            + sum(np.sum(d.astype(float) ** 2) for d in details)
        )
        totals.append(total_energy)

    # Compare total energy row-wise sum across all sub-bands
    summed = out.sum(axis=1).to_numpy()
    assert len(summed) == len(totals)
    for i, (we_sum, ref) in enumerate(zip(summed, totals)):
        # 10% tolerance
        assert np.isclose(
            we_sum, ref, rtol=0.10, atol=1e-6
        ), f"row {i}: {we_sum:.6f} vs {ref:.6f}"


def test_wentropy_columns_and_rows(df_all_bands):
    """
    Columns should be <channel>_wentropy for each input column present in the
    energy dataframe; rows should match the sliding-window count.
    """
    bands = {
        "low": (0.0, 8.0),
        "mid": (8.0, 20.0),
        "high": (20.0, FS / 2),
    }
    window_sec = 4.0
    overlap = 0.5

    n = len(df_all_bands)
    nperseg = int(round(window_sec * FS))
    hop = int(round(nperseg * (1.0 - overlap)))
    expected_rows = 1 + (n - nperseg) // hop

    wenergy = wavelet_band_energy(
        df_all_bands,
        FS,
        bands,
        wavelet="db4",
        mode="periodization",
        window_sec=window_sec,
        overlap=overlap,
    )
    went = wavelet_entropy(wenergy, bands, normalize=True)

    expected_cols = [f"{ch}_wentropy" for ch in df_all_bands.columns]
    assert list(went.columns) == expected_cols
    assert len(went) == expected_rows


def test_wentropy_extremes_uniform_vs_spiky():
    """
    If band energies are equal across K bands then normalized entropy is closer to 1.
    If all energy is in one band then normalized entropy is closer to 0.
    """
    bands = {"low": (0.0, 8.0), "mid": (8.0, 20.0), "high": (20.0, FS / 2)}
    K = len(bands)
    T = 5  # number of windows/rows

    uniform_low = np.full(T, 1.0)
    uniform_mid = np.full(T, 1.0)
    uniform_high = np.full(T, 1.0)

    # Channel S: spiky energy (all in 'mid')
    spiky_low = np.zeros(T)
    spiky_mid = np.full(T, 3.0)  # same total as U across 3 bands
    spiky_high = np.zeros(T)

    wv_energy_df = pd.DataFrame(
        {
            "uniform_low_wenergy": uniform_low,
            "uniform_mid_wenergy": uniform_mid,
            "uniform_high_wenergy": uniform_high,
            "spiky_low_wenergy": spiky_low,
            "spiky_mid_wenergy": spiky_mid,
            "spiky_high_wenergy": spiky_high,
        }
    )

    went = wavelet_entropy(wv_energy_df, bands, normalize=True)

    # Normalized entropy should be in [0,1]
    assert (went["uniform_wentropy"].between(0.0, 1.0)).all()
    assert (went["spiky_wentropy"].between(0.0, 1.0)).all()

    # Uniform energy should be ~1.0; Spiky should be ~0.0 (allow small error)
    uniform_med = went["uniform_wentropy"].median()
    spiky_med = went["spiky_wentropy"].median()

    assert (
        uniform_med > 0.97
    ), f"Uniform energy should yield ~1 entropy, got {uniform_med:.3f}"
    assert (
        spiky_med < 0.03
    ), f"Spiky energy should yield ~0 entropy, got {spiky_med:.3f}"
