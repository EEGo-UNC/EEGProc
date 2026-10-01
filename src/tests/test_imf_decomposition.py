import numpy as np
import pandas as pd
import pytest

from eegproc.featurization import imf_band_energy, imf_entropy, FREQUENCY_BANDS
from .utils import FS, make_sine, window_rows


def test_imf_energy_columns_and_rows(df_all_bands):
    """Real EMD: column names align with imf_to_band; row count equals sliding-window math."""
    window_sec, overlap = 4.0, 0.5
    out = imf_band_energy(
        df_all_bands,
        FS,
        window_sec=window_sec,
        overlap=overlap,
        EMD_kwargs={},
    )
    expected_cols = [
        f"{ch}_{band}_imfenergy" for ch in df_all_bands.columns for band in list(FREQUENCY_BANDS.keys())[::-1]
    ]
    assert list(out.columns) == expected_cols

    # row count = window count with overlap
    erows = window_rows(
        len(df_all_bands), fs=FS, window_sec=window_sec, overlap=overlap
    )
    assert len(out) == erows

    # Energies are non-negative
    assert (out.fillna(0.0) >= 0).all().all()


@pytest.mark.parametrize(
    "overlap, ok", [(-0.1, False), (0.0, True), (0.5, True), (0.99, True), (1.0, False)]
)
def test_imf_energy_overlap_bounds(df_simple_sinusoid, overlap, ok):
    if ok:
        _ = imf_band_energy(df_simple_sinusoid, FS, window_sec=4.0, overlap=overlap)
    else:
        with pytest.raises(ValueError):
            imf_band_energy(df_simple_sinusoid, FS, window_sec=4.0, overlap=overlap)


def test_imf_energy_amplitude_scaling_property():
    """
    EMD scaling input by k should scale IMFs by k,
    so energy should scale by k^2.
    """
    k = 0.5
    df = pd.DataFrame(
        {
            "A": make_sine(10.0, dur_sec=8.0, amp=1.0),
            "B": make_sine(10.0, dur_sec=8.0, amp=k),
        }
    )

    window_sec, overlap = 4.0, 0.5
    out = imf_band_energy(df, FS, window_sec=window_sec, overlap=overlap)

    # Sum energy across bands per channel and compare ratio per-window
    A_cols = [c for c in out.columns if c.startswith("A_")]
    B_cols = [c for c in out.columns if c.startswith("B_")]

    A_total = out[A_cols].sum(axis=1).to_numpy()
    B_total = out[B_cols].sum(axis=1).to_numpy()

    ratio = B_total / A_total
    target = k**2
    print(target, ratio)
    assert np.allclose(
        ratio, target, rtol=0.05, atol=1e-12
    ), f"expected ~{target}, got {ratio}"


def test_imf_entropy_columns_and_rows(df_all_bands):
    T = 6
    cols = []
    data = {}
    for ch in df_all_bands.columns:
        for b in FREQUENCY_BANDS.keys():
            col = f"{ch}_{b}_imfenergy"
            cols.append(col)
            data[col] = np.abs(np.random.randn(T)) + 0.5

    df_energy = pd.DataFrame(data, columns=cols)
    out = imf_entropy(df_energy, normalize=True)

    expected_cols = [f"{ch}_imfentropy" for ch in df_all_bands.columns]
    assert list(out.columns) == expected_cols
    assert len(out) == T

    # Normalized entropy is between 0 and 1 inclusive
    for c in out.columns:
        assert (out[c].between(0.0, 1.0)).all()


def test_imf_entropy_uniform_vs_spiky():
    """
    Equal energy across K bands → entropy ~ 1.
    All mass in one band     → entropy ~ 0.
    """
    T = 5
    data = {}
    # Uniform channel
    for b in FREQUENCY_BANDS.keys():
        data[f"U_{b}_imfenergy"] = np.full(T, 1.0)
    
    # First channel is spiky
    for i, b in enumerate(FREQUENCY_BANDS.keys()):
        data[f"S_{b}_imfenergy"] = np.full(T, 1.0 if i == 0 else 0.0)

    df_energy = pd.DataFrame(data)
    went = imf_entropy(df_energy, normalize=True)

    assert set(went.columns) == {"U_imfentropy", "S_imfentropy"}
    u_med = went["U_imfentropy"].median()
    s_med = went["S_imfentropy"].median()

    assert u_med > 0.97, f"uniform should be ~1, got {u_med:.3f}"
    assert s_med < 0.03, f"spiky should be ~0, got {s_med:.3f}"
