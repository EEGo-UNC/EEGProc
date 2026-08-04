'''Tests for PSD, Shannon's Entropy, and Hjorth Params'''
import numpy as np
import pandas as pd
import pywt
import math
import pytest
from .utils import *

from eegproc.featurization import psd_bandpowers, shannons_entropy, hjorth_params, feature_grouped_by_metadata
from eegproc.preprocessing import FREQUENCY_BANDS


FS = 128


'''Test PSD'''
def test_psd_raises_when_no_band_suffixed_columns():
    df = pd.DataFrame({"A1": make_sine(10.0)})  # no "_alpha" column name
    with pytest.raises(ValueError, match="No columns named like"):
        psd_bandpowers(df, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)

@pytest.mark.parametrize("window_sec", [0.01, 0.05])
def test_psd_window_too_small_raises(window_sec):
    df = pd.DataFrame({"A1_alpha": make_sine(10.0)})
    with pytest.raises(ValueError, match="window_sec too small"):
        psd_bandpowers(df, FS, FREQUENCY_BANDS, window_sec=window_sec, overlap=0.5)

@pytest.mark.parametrize("overlap", [-0.1, 1.0])
def test_psd_overlap_bounds_raises(overlap):
    df = pd.DataFrame({"A1_alpha": make_sine(10.0)})
    with pytest.raises(ValueError, match="overlap must be in"):
        psd_bandpowers(df, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=overlap)

def test_psd_overlap_hop_zero_raises():
    df = pd.DataFrame({"A1_alpha": make_sine(10.0)})
    # Large overlap forces hop less than 0
    with pytest.raises(ValueError, match="hop size must be"):
        psd_bandpowers(df, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.999999)


def test_psd_bandpowers_columns_and_rows(df_all_bands: pd.DataFrame):
    """
    Ensures the output has the same columns and the correct number of rows
    based on window_sec and overlap.
    """
    window_sec = 4.0
    overlap = 0.5
    n = len(df_all_bands)
    nperseg = int(round(window_sec * FS))
    hop = int(round(nperseg * (1.0 - overlap)))
    expected_rows = 1 + (n - nperseg) // hop

    out = psd_bandpowers(
        df_all_bands,
        FS,
        FREQUENCY_BANDS,
        window_sec=window_sec,
        overlap=overlap,
    )

    assert list(out.columns) == list(df_all_bands.columns)
    assert len(out) == expected_rows


def test_band_energy_concentrates_in_matching_band(df_single_alpha: pd.DataFrame):
    """
    A 10 Hz sine named A1_alpha should produce non-trivial power in alpha.
    """
    out_alpha = psd_bandpowers(
        df_single_alpha, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5
    )
    # Power should be positive and stable across windows
    assert (out_alpha["A1_alpha"] > 0).all()


def test_band_energy_is_near_zero_if_suffix_band_does_not_cover_freq(
    df_mismatch_alpha_as_theta: pd.DataFrame,
):
    """
    If we name the column A1_theta but put a 10 Hz sine into it, the theta
    integration (4–8 Hz) should capture ~0 power.
    """
    out = psd_bandpowers(
        df_mismatch_alpha_as_theta, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5
    )
    assert np.all(out["A1_theta"].to_numpy() < 1e-6)


def test_amplitude_scaling_is_quadratic_in_amp():
    """
    Power should scale by amplitude squared. Compare amp=2 vs amp=1 on the same 10 Hz alpha column.
    """
    df1 = pd.DataFrame({"A1_alpha": make_sine(10.0, amp=1.0)})
    df2 = pd.DataFrame({"A1_alpha": make_sine(10.0, amp=2.0)})

    p1 = psd_bandpowers(df1, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)[
        "A1_alpha"
    ].median()
    p2 = psd_bandpowers(df2, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)[
        "A1_alpha"
    ].median()

    ratio = p2 / (p1 + 1e-12)  # 1e-12 avoids divide by 0 but doesn't affect calculation
    # Hann/Welch/windowing should tolerate ~10–15% deviation
    assert 3.4 < ratio < 4.6, f"Expected ~4× power, got {ratio:.2f}×"



'''Test Shannon's'''
def test_entropy_columns_and_rows(df_all_bands: pd.DataFrame):
    """
    Output columns should be <input>_entropy; number of rows should match
    sliding-window count implied by window_sec and overlap.
    """
    window_sec = 4.0
    overlap = 0.5
    n = len(df_all_bands)
    nperseg = int(round(window_sec * FS))
    hop = int(round(nperseg * (1.0 - overlap)))
    expected_rows = 1 + (n - nperseg) // hop

    out = shannons_entropy(
        df_all_bands, FS, FREQUENCY_BANDS, window_sec=window_sec, overlap=overlap
    )

    expected_cols = [f"{c}_entropy" for c in df_all_bands.columns]
    assert list(out.columns) == expected_cols
    assert len(out) == expected_rows


def test_entropy_bounds_on_valid_data(df_theta_noise: pd.DataFrame):
    """
    Shannon's entropy (normalized) is in [0, 1] whenever it's defined.
    """
    out = shannons_entropy(df_theta_noise, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)
    vals = out["A1_theta_entropy"].to_numpy()
    assert np.isfinite(vals).all()
    assert (vals >= -1e-9).all() and (vals <= 1.0 + 1e-9).all()


def test_pure_tone_has_low_entropy(df_single_alpha: pd.DataFrame):
    """
    A narrow-line 10 Hz tone inside alpha should yield low normalized entropy
    (power concentrated in a few bins).
    """
    out = shannons_entropy(df_single_alpha, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)
    # Expect small values
    assert out["A1_alpha_entropy"].median() < 0.30


def test_white_noise_has_high_entropy(df_theta_noise: pd.DataFrame):
    """
    Band-limited white noise should distribute power more uniformly across bins → high entropy.
    """
    out = shannons_entropy(df_theta_noise, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)
    # Depending on band width and nperseg, expect values near 1; tolerate some variance.
    assert out["A1_theta_entropy"].median() > 0.80


def test_entropy_is_amplitude_invariant_for_tone():
    """
    Entropy is based on normalized PSD (probabilities), so scaling amplitude shouldn't change it.
    """
    df1 = pd.DataFrame({"A1_alpha": make_sine(10.0, amp=0.5)})
    df2 = pd.DataFrame({"A1_alpha": make_sine(10.0, amp=2.0)})

    H1 = shannons_entropy(df1, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)["A1_alpha_entropy"].median()
    H2 = shannons_entropy(df2, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)["A1_alpha_entropy"].median()

    assert abs(H1 - H2) < 0.05, f"Amplitude invariance violated: {H1:.3f} vs {H2:.3f}"


def test_entropy_is_amplitude_invariant_for_noise():
    df1 = pd.DataFrame({"A1_theta": make_white_noise(amp=0.5, seed=123)})
    df2 = pd.DataFrame({"A1_theta": make_white_noise(amp=2.0, seed=123)})

    H1 = shannons_entropy(df1, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)["A1_theta_entropy"].median()
    H2 = shannons_entropy(df2, FS, FREQUENCY_BANDS, window_sec=4.0, overlap=0.5)["A1_theta_entropy"].median()

    assert abs(H1 - H2) < 0.05


'''Test Hjorth Params'''
def test_rows_from_window_and_overlap(df_simple_sinusoid: pd.DataFrame):
    win_sec = 4.0
    ov = 0.5
    n = len(df_simple_sinusoid)
    expected = window_rows(n, FS, win_sec, ov)

    out = hjorth_params(df_simple_sinusoid, FS, window_sec=win_sec, overlap=ov, detrend="constant")
    assert len(out) == expected

    # Columns should be triplets per input column
    for base in ["A1"]:
        for suffix in ["activity", "mobility", "complexity"]:
            assert f"{base}_{suffix}" in out.columns


@pytest.mark.parametrize("freq", [2.0, 6.0, 10.0, 15.0, 25.0, 35.0])
def test_hjorth_on_pure_sine_matches_theory(freq: float):
    """
    For a pure sine with zero mean:
      activity ≈ A^2/2
      mobility ≈ 2*sin(pi*f/fs)
      complexity ≈ 1
    """
    amp = 1.6
    df = pd.DataFrame({"X": make_sine(freq, amp=amp)})
    out = hjorth_params(df, FS, window_sec=4.0, overlap=0.5, detrend="constant")

    act = out["X_activity"].median()
    mob = out["X_mobility"].median()
    comp = out["X_complexity"].median()

    # activity ~ A^2/2
    expected_act = (amp ** 2) / 2.0
    assert np.isclose(act, expected_act, rtol=0.08), f"act {act:.4f} vs {expected_act:.4f}"

    # mobility ~ 2*sin(pi*f/fs)
    expected_mob = expected_mobility(freq, FS)
    assert np.isclose(mob, expected_mob, rtol=0.08, atol=0.01), f"mob {mob:.4f} vs {expected_mob:.4f}"

    # complexity ~ 1
    assert 0.9 <= comp <= 1.1, f"comp {comp:.3f} not ~1"


def test_mobility_increases_with_frequency(df_multi_freq_sines_same_amp: pd.DataFrame):
    """
    Mobility should be monotone increasing with frequency for same amplitude sines.
    """
    out = hjorth_params(df_multi_freq_sines_same_amp, FS, window_sec=4.0, overlap=0.5, detrend="constant")

    m1 = out["A1_mobility"].median()  # 2 Hz
    m2 = out["A2_mobility"].median()  # 10 Hz
    m3 = out["A3_mobility"].median()  # 35 Hz

    assert m1 < m2 < m3, f"Expected monotone mobility, got {m1:.3f}, {m2:.3f}, {m3:.3f}"


def test_activity_scales_with_amplitude_mobility_complexity_invariant():
    """
    Double amplitude should result in 4x activity; mobility and complexity should be unchanged for a fixed freq.
    """
    f = 10.0
    df1 = pd.DataFrame({"X": make_sine(f, amp=1.0)})
    df2 = pd.DataFrame({"X": make_sine(f, amp=2.0)})

    h1 = hjorth_params(df1, FS, window_sec=4.0, overlap=0.5, detrend="constant")
    h2 = hjorth_params(df2, FS, window_sec=4.0, overlap=0.5, detrend="constant")

    a1, a2 = h1["X_activity"].median(), h2["X_activity"].median()
    m1, m2 = h1["X_mobility"].median(), h2["X_mobility"].median()
    c1, c2 = h1["X_complexity"].median(), h2["X_complexity"].median()

    # Activity = amplitude^2
    ratio = a2 / (a1 + 1e-12)
    assert 3.6 < ratio < 4.4, f"Activity ratio ~4 expected, got {ratio:.2f}"

    # Mobility and complexity amplitude-invariant
    assert abs(m1 - m2) < 0.05, f"Mobility changed with amplitude: {m1:.3f} vs {m2:.3f}"
    assert abs(c1 - c2) < 0.05, f"Complexity changed with amplitude: {c1:.3f} vs {c2:.3f}"



FS = 128

def make_series(n=256, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n))

def spy_target_factory(capture):
    """
    Returns a dummy target_function that:
      - records the incoming df.columns into `capture["seen"].append([...])`
      - returns a one-row DataFrame with the same columns (so shapes stay consistent).
    """
    def _spy(df: pd.DataFrame, fs: int = 0, bands=None, **kwargs) -> pd.DataFrame:
        capture.setdefault("seen", []).append(list(df.columns))
        # Return 1-row with the same columns to make downstream logic easy to check
        return pd.DataFrame([{c: 1.0 for c in df.columns}])
    return _spy


def test_filters_to_requested_channel_band_columns_and_preserves_meta_order():
    # DataFrame has meta + a handful of channel_band columns (some missing on purpose)
    df = pd.DataFrame({
        "patient_index": [1]*10 + [2]*10,
        "video_index":   [7]*10 + [8]*10,
        "AF3_alpha":     make_series(20, seed=1),
        "AF3_beta":      make_series(20, seed=2),
        "F7_alpha":      make_series(20, seed=3),
        "Pz_gamma":      make_series(20, seed=4),  # should be filtered out by `channels`
    })

    channels = ["AF3", "F7"]
    # must be a dict for the real features; keys are used for suffix matching
    bands = {"alpha": (8, 12), "beta": (12, 30)}

    capture = {}
    spy = spy_target_factory(capture)

    out = feature_grouped_by_metadata(
        eeg_df=df,
        target_function=spy,
        fs=FS,
        bands=bands,
        channels=channels,
        group_by_metadata_columns=["patient_index", "video_index"],
        drop_metadata_for_fn=True,
    )

    # The iterator should see only AF3_alpha, AF3_beta, F7_alpha (F7_beta absent; Pz_* excluded)
    expected_seen = set(["AF3_alpha", "AF3_beta", "F7_alpha"])
    assert len(capture["seen"]) == 2  # two groups
    for seen_cols in capture["seen"]:
        assert set(seen_cols) == expected_seen

    # Output should prepend metadata and keep them leading
    lead = ["patient_index", "video_index"]
    assert list(out.columns[:2]) == lead
    assert set(out.columns[2:]) == expected_seen
    # Two groups, spy returns one row per group → 2 rows total
    assert len(out) == 2


def test_drop_metadata_for_fn_true_means_meta_not_in_block():
    df = pd.DataFrame({
        "patient_index": [1]*8 + [1]*8,
        "video_index":   [5]*8 + [6]*8,
        "AF3_alpha":     make_series(16, seed=10),
        "F7_beta":       make_series(16, seed=11),
    })

    channels = ["AF3", "F7"]
    bands = {"alpha": (8, 12), "beta": (12, 30)}

    capture = {}
    spy = spy_target_factory(capture)

    _ = feature_grouped_by_metadata(
        eeg_df=df,
        target_function=spy,
        fs=FS,
        bands=bands,
        channels=channels,
        group_by_metadata_columns=["patient_index", "video_index"],
        drop_metadata_for_fn=True,
    )

    for seen_cols in capture["seen"]:
        assert "patient_index" not in seen_cols
        assert "video_index" not in seen_cols
        # Only the requested intersections should appear
        assert set(seen_cols).issubset({"AF3_alpha", "F7_beta"})


def test_no_grouping_processes_once_and_returns_no_meta_columns():
    df = pd.DataFrame({
        "AF3_alpha": make_series(16, seed=21),
        "F7_alpha":  make_series(16, seed=22),
    })
    channels = ["AF3", "F7"]
    bands = {"alpha": (8, 12)}

    capture = {}
    spy = spy_target_factory(capture)

    out = feature_grouped_by_metadata(
        eeg_df=df,
        target_function=spy,
        fs=FS,
        bands=bands,
        channels=channels,
        group_by_metadata_columns=None,   # <- no meta grouping
        drop_metadata_for_fn=True,
    )

    # Called exactly once
    assert len(capture["seen"]) == 1
    assert set(capture["seen"][0]) == {"AF3_alpha", "F7_alpha"}
    # No meta → output should also have only EEG columns
    assert set(out.columns) == {"AF3_alpha", "F7_alpha"}
    assert len(out) == 1


def test_returns_empty_when_no_channel_band_matches_with_meta():
    df = pd.DataFrame({
        "patient_index": [1, 1, 2, 2],
        "video_index":   [5, 6, 5, 6],
        "Pz_gamma":      make_series(4, seed=33),  # does not match requested channels
    })

    channels = ["AF3", "F7"]                  # no AF3_* or F7_* present
    bands = {"alpha": (8, 12), "beta": (12, 30)}

    # Expect an empty frame but with the meta columns present
    out = feature_grouped_by_metadata(
        eeg_df=df,
        target_function=lambda d, **k: pd.DataFrame(),
        fs=FS,
        bands=bands,
        channels=channels,
        group_by_metadata_columns=["patient_index", "video_index"],
        drop_metadata_for_fn=True,
    )

    assert out.empty
    assert list(out.columns) == ["patient_index", "video_index"]


def test_returns_empty_when_no_channel_band_matches_and_no_meta():
    df = pd.DataFrame({
        "Pz_gamma": make_series(4, seed=44)
    })

    channels = ["AF3"]
    bands = {"alpha": (8, 12)}

    out = feature_grouped_by_metadata(
        eeg_df=df,
        target_function=lambda d, **k: pd.DataFrame(),
        fs=FS,
        bands=bands,
        channels=channels,
        group_by_metadata_columns=None,
        drop_metadata_for_fn=True,
    )

    # Should be empty; columns ideally [].
    # If your current impl returns columns=None, this test will fail (and should—fix to []).
    assert out.empty
    assert list(out.columns) == []
