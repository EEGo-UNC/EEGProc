import numpy as np
import pandas as pd
import pywt
from math import log2, floor
from scipy.signal import welch
from .preprocessing import bandpass_filter, apply_detrend, FREQUENCY_BANDS
from PyEMD import EMD


# ---------------------------
# SPECTRAL ENERGY and ENTROPY
# ---------------------------
def psd_bandpowers(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]] = FREQUENCY_BANDS,
    window_sec: float = 4.0,
    overlap: float = 0.5,
    detrend: str | None = "constant",
) -> pd.DataFrame:
    """Compute Welch PSD band powers per channel-band column over selected window size.

    Expects columns named ``{channel}_{band}`` where each ``band`` is a key in ``bands``
    (bandpass_filter) may be used to achieve the expected table.
    Each row of the expected df is a wave amplitude reading of the channel-band combination.
    For each window, integrates the Welch PSD within each band using the trapezoid rule.

    Parameters
    ----------
    df : pandas.DataFrame
        Bandpass Filtered EEG dataframe. Numeric columns must be named like
        ``{channel}_{band}`` (e.g., ``AF3_alpha``).
    fs : float
        Sampling rate in Hz.
    bands : dict[str, tuple[float, float]], default=FREQUENCY_BANDS
        Mapping of band name to inclusive frequency bounds (Hz) ``(lo, hi)``.
    window_sec : float, default=4.0
        Window length in seconds used for Welch's method (``nperseg = round(fs*window_sec)``).
    overlap : float, default=0.5
        Fractional overlap in ``[0, 1)`` between consecutive windows.
    detrend : {"constant", "linear", None}, default="constant"
        Detrending applied before PSD computation via ``apply_detrend``.

    Returns
    -------
    pandas.DataFrame
        One row per window, columns match the input band columns (e.g., ``AF3_alpha``)
        and contain power spectral density-integrated band powers.

    Raises
    ------
    ValueError
        If no valid ``{channel}_{band}`` columns are found, if the window is too small,
        or if ``overlap`` is outside ``[0, 1)``.

    Notes
    -----
    - Uses ``scipy.signal.welch`` with a Hann window and no overlap inside the Welch call
      (windowing/overlap are controlled at the outer sliding-window level).

    Examples
    --------
    Minimal example with synthetic data (two channels, one band):

    >>> import numpy as np, pandas as pd
    >>> fs = 128.0
    >>> t = np.arange(int(8*fs)) / fs   # 8 seconds
    >>> # Two synthetic signals with an ~10 Hz component (alpha band)
    >>> af3_alpha = 0.8*np.sin(2*np.pi*10*t) + 0.1*np.random.randn(t.size)
    >>> f7_alpha  = 0.6*np.sin(2*np.pi*10*t + 0.7) + 0.1*np.random.randn(t.size)
    >>> df = pd.DataFrame({
    ...     "AF3_alpha": af3_alpha,
    ...     "F7_alpha":  f7_alpha,
    ... })
    >>> bands = {"alpha": (8.0, 12.0)}
    >>> out = psd_bandpowers(df, fs=fs, bands=bands, window_sec=2.0, overlap=0.5)
    >>> out.head()  # doctest: +ELLIPSIS
           AF3_alpha   F7_alpha
    0       ...         ...
    1       ...         ...
    2       ...         ...
    """

    df = apply_detrend(detrend, df)

    band_keys = set(bands.keys())
    col_band, col_chan = {}, {}
    for col in df.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in band_keys:
            col_band[col] = parts[1]
            col_chan[col] = parts[0]
    if not col_band:
        raise ValueError(
            "No columns named like '{channel}_{band}' with band in FREQUENCY_BANDS."
        )
    df = df[list(col_band.keys())]

    data = df.to_numpy(dtype=float, copy=False)
    n_samples, n_cols = data.shape
    nperseg = int(round(window_sec * fs))
    if nperseg <= 8:
        raise ValueError("window_sec too small for given fs; increase window_sec.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    hop = int(round(nperseg * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")
    if nperseg > n_samples:
        return pd.DataFrame(columns=list(df.columns))

    band_to_idx = {}
    for i, col in enumerate(df.columns):
        band_to_idx.setdefault(col_band[col], []).append(i)

    rows = []
    for start in range(0, n_samples - nperseg + 1, hop):
        seg = data[start : start + nperseg, :]

        f, psd = welch(
            seg,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=0,
            detrend=False,
            scaling="density",
            return_onesided=True,
            axis=0,
        )

        row = {}
        for band, idxs in band_to_idx.items():
            lo, hi = bands[band]
            m = (f >= lo) & (f <= hi)
            if not m.any():
                for j in idxs:
                    row[df.columns[j]] = 0.0
                continue

            band_power = np.trapezoid(psd[m][:, idxs], f[m], axis=0)
            for k, j in enumerate(idxs):
                row[df.columns[j]] = float(band_power[k])

        rows.append(row)

    return pd.DataFrame(rows, columns=list(df.columns))


def shannons_entropy(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]] = FREQUENCY_BANDS,
    window_sec: float = 4.0,
    overlap: float = 0.5,
    eps: float = 1e-300,
    detrend: str | None = "constant",
) -> pd.DataFrame:
    """Compute normalized Shannon spectral entropy per channel-band over windows.

    Expects columns named ``{channel}_{band}`` where each ``band`` is a key in ``bands``
    (bandpass_filter) may be used to achieve the expected table.
    For each ``{channel}_{band}`` column, computes a Welch PSD in the band's frequency range,
    then it converts each windowed row (bin) of energy to probability.
    Returns ``-Σplog2p/log2(#bins)`` in ``[0, 1]`` (NaN if insufficient bins or invalid totals).

    Parameters
    ----------
    df : pandas.DataFrame
        Bandpass Filtered EEG dataframe. Numeric columns must be named like
        ``{channel}_{band}`` (e.g., ``AF3_alpha``).
    fs : float
        Sampling rate in Hz.
    bands : dict[str, tuple[float, float]], default=FREQUENCY_BANDS
        Mapping from band name to inclusive frequency bounds (Hz).
    window_sec : float, default=4.0
        Window length in seconds for Welch.
    overlap : float, default=0.5
        Fractional overlap in ``[0, 1)`` between windows.
    eps : float, default=1e-300
        Numerical guard to avoid log(0) and zero division.
    detrend : {"constant", "linear", None}, default="constant"
        Detrending applied before PSD and Shannon.

    Returns
    -------
    pandas.DataFrame
        One row per window. Columns are ``{channel}_{band}_entropy`` for each input band column.

    Raises
    ------
    ValueError
        If no band-annotated columns are found, window is too small, or overlap invalid.

    Notes
    -----
    - Entropy is normalized by ``log2(count_of_band_bins)`` to yield values in ``[0, 1]``.
    - Outputs NaN when a band's PSD has < 2 valid bins in a window.
    """

    df = apply_detrend(detrend, df)
    band_keys = set(bands.keys())
    col_band = {}
    for col in df.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in band_keys:
            col_band[col] = parts[1]
    if not col_band:
        raise ValueError(
            "No columns named like '{channel}_{band}' with band in FREQUENCY_BANDS."
        )
    df = df[list(col_band.keys())]

    data = df.to_numpy(dtype=float, copy=False)
    n_samples, n_cols = data.shape
    nperseg = int(round(window_sec * fs))
    if nperseg <= 8:
        raise ValueError("window_sec too small for given fs; increase window_sec.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    hop = int(round(nperseg * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")
    if nperseg > n_samples:
        return pd.DataFrame(columns=[f"{c}_entropy" for c in df.columns])

    band_to_idx = {}
    for i, col in enumerate(df.columns):
        band_to_idx.setdefault(col_band[col], []).append(i)

    rows = []
    for start in range(0, n_samples - nperseg + 1, hop):
        seg = data[start : start + nperseg, :]

        f, psd = welch(
            seg,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=0,
            detrend=False,
            scaling="density",
            return_onesided=True,
            axis=0,
        )

        row = {}
        for band, idxs in band_to_idx.items():
            lo, hi = bands[band]
            m = (f >= lo) & (f <= hi)
            count = int(np.count_nonzero(m))
            if count < 2:
                for j in idxs:
                    row[f"{df.columns[j]}_entropy"] = np.nan
                continue

            band_power = psd[m][:, idxs]
            totals = np.sum(band_power, axis=0)
            valid = np.isfinite(totals) & (totals > 0)

            p = np.empty_like(band_power)
            p[:, valid] = band_power[:, valid] / totals[valid]
            p[:, ~valid] = np.nan
            p = np.clip(p, eps, 1.0)

            H = -np.nansum(p * np.log2(p), axis=0)
            H /= np.log2(count)

            for k, j in enumerate(idxs):
                row[f"{df.columns[j]}_entropy"] = (
                    float(H[k]) if np.isfinite(H[k]) else np.nan
                )

        rows.append(row)

    return pd.DataFrame(rows, columns=[f"{c}_entropy" for c in df.columns])


# ----------------------
# Hjorth Parametrization
# ----------------------
def hjorth_params(
    df: pd.DataFrame,
    fs: float,
    window_sec: float = 4.0,
    overlap: float = 0.5,
    detrend: str | None = "constant",
    eps: float = 1e-300,
) -> pd.DataFrame:
    """Compute Hjorth parameters (activity, mobility, complexity) per channel over windows.

    Expects columns named ``{channel}_{band}`` where each ``band`` is a key in ``bands``
    (bandpass_filter) may be used to achieve the expected table.
    For each numeric column, the function computes:
    - **Activity**: variance of the signal
    - **Mobility**: sqrt(var(Δx) / var(x))
    - **Complexity**: mobility(Δx) / mobility(x)

    Parameters
    ----------
    df : pandas.DataFrame
        Bandpass Filtered EEG dataframe. Numeric columns must be named like
        ``{channel}_{band}`` (e.g., ``AF3_alpha``).
    fs : float
        Sampling rate in Hz.
    window_sec : float, default=4.0
        Window length in seconds.
    overlap : float, default=0.5
        Fractional overlap in ``[0, 1)`` between windows.
    detrend : {"constant", "linear", None}, default="constant"
        Detrending applied per column prior to differencing.
    eps : float, default=1e-300
        Numerical guard to prevent division by zero.

    Returns
    -------
    pandas.DataFrame
        One row per window, with columns:
        ``{channel}_activity``, ``{channel}_mobility``, ``{channel}_complexity``.

    Raises
    ------
    ValueError
        If window is too small (needs >= 3 samples) or overlap is invalid.
    """

    df = apply_detrend(detrend, df)

    cols = list(df.columns)
    data = df.to_numpy(dtype=float)
    n_samples, n_cols = data.shape

    win = int(round(window_sec * fs))
    if win < 3:
        raise ValueError(
            "window_sec too small (need >= 3 samples for second differences)."
        )
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")

    hop = int(round(win * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")

    rows = []
    starts = range(0, n_samples - win + 1, hop)
    for i0 in starts:
        i1 = i0 + win
        seg = data[i0:i1, :]
        if seg.shape[0] < 3:
            continue

        act = np.nanvar(seg, axis=0, ddof=0)

        dx = np.diff(seg, n=1, axis=0)
        ddx = np.diff(seg, n=2, axis=0)

        var_dx = np.nanvar(dx, axis=0, ddof=0)
        var_ddx = np.nanvar(ddx, axis=0, ddof=0)

        mob = np.sqrt((var_dx + eps) / (act + eps))
        mob_dx = np.sqrt((var_ddx + eps) / (var_dx + eps))
        comp = mob_dx / (mob + eps)

        row = {}

        for k, c in enumerate(cols):
            row[f"{c}_activity"] = float(act[k]) if np.isfinite(act[k]) else np.nan
            row[f"{c}_mobility"] = float(mob[k]) if np.isfinite(mob[k]) else np.nan
            row[f"{c}_complexity"] = float(comp[k]) if np.isfinite(comp[k]) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------
# WAVELET FEATURES
# ----------------
def choose_dwt_level(n_samples: int, fs: float, wavelet: str, min_freq: float) -> int:
    """Choose a discrete wavelet transform (DWT) level given data length and a target minimum frequency.

    The selected level is bounded by PyWavelets' maximum permissible level for the
    given signal length and wavelet filter length, and ensures that the *detail*
    subband captures energy at or above ``min_freq``.

    Args:
        n_samples (int): Number of time-domain samples in the signal.
        fs (float): Sampling rate in Hz. Must be positive.
        wavelet (str): Name of a PyWavelets wavelet (e.g., ``"db4"``, ``"sym5"``).
        min_freq (float): Target minimum frequency (Hz) you still wish to capture
            in a detail subband. If non-positive, a tiny floor (``1e-6``) is used
            to avoid taking ``log2(0)`` and to return a conservative level.

    Returns:
        int: The chosen DWT level (>= 1) that balances the maximum allowable level
        with the goal of preserving content at or above ``min_freq``.

    See Also:
        pywt.dwt_max_level: Maximum level allowed by signal and filter length.
        pywt.Wavelet: Wavelet objects carrying filter lengths and properties.
    """
    max_lvl = pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet).dec_len)
    target = max(1, floor(log2(fs / max(min_freq, 1e-6)) - 1))
    return max(1, min(max_lvl, target))


def dwt_subband_ranges(fs: float, level: int) -> dict[str, tuple[float, float]]:
    """Return nominal frequency ranges (Hz) for DWT subbands up to a level.

    Uses paired ranges for detail bands ``D{j}`` and the approximation band ``A{level}``.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    level : int
        Decomposition level (>= 1).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping like ``{"D1": (f_lo, f_hi), ..., "A{level}": (0, f_c)}``.
    """

    bands: dict[str, tuple[float, float]] = {}
    for j in range(1, level + 1):
        f_hi = fs / (2**j)
        f_lo = fs / (2 ** (j + 1))
        bands[f"D{j}"] = (f_lo, f_hi)
    bands[f"A{level}"] = (0.0, fs / (2 ** (level + 1)))
    return bands


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


def wavelet_band_energy(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]],
    wavelet: str = "db4",
    mode: str = "periodization",
    window_sec: float = 4.0,
    overlap: float = 0.5,
) -> pd.DataFrame:
    """Compute band energies by distributing DWT subband energies into target bands.

    Expects columns named ``{channel}`` where each row is a reading of raw EEG data.
    For each window and channel, performs a multilevel DWT, computes energy in each
    detail/approximation subband, then proportionally assigns subband energy to user
    bands by frequency overlap.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data EEG dataframe. Numeric columns.
    fs : float
        Sampling rate in Hz.
    bands : dict[str, tuple[float, float]]
        Target bands as ``{name: (lo, hi)}`` in Hz.
    wavelet : str, default="db4"
        PyWavelets wavelet name for ``pywt.wavedec``.
    mode : str, default="periodization"
        Signal extension mode for DWT.
    window_sec : float, default=4.0
        Window length in seconds.
    overlap : float, default=0.5
        Fractional overlap in ``[0, 1)``.

    Returns
    -------
    pandas.DataFrame
        One row per window; columns: ``{channel}_{band}_wenergy`` for each
        ``channel`` in ``df`` and each band in ``bands``.

    Raises
    ------
    ValueError
        If window too small, ``overlap`` invalid, or window longer than available samples.

    Notes
    -----
    - The DWT level is chosen automatically via :func:`choose_dwt_level` to respect
      both data length and minimum target band frequency.
    - Energy is computed as the sum of squared coefficients per subband.
    - Subband energy is apportioned to target bands by fractional frequency overlap.
    """
    df = df.select_dtypes(include=[np.number])

    n_samples = len(df)
    nperseg = int(round(window_sec * fs))
    if nperseg <= 8:
        raise ValueError("window_sec too small for given fs; increase window_sec.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    hop = int(round(nperseg * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")
    if nperseg > n_samples:
        return pd.DataFrame(
            columns=[f"{ch}_{b}_wenergy" for ch in df.columns for b in bands]
        )

    min_band_lo = min(lo for lo, _ in bands.values())
    L = choose_dwt_level(
        n_samples=nperseg, fs=fs, wavelet=wavelet, min_freq=min_band_lo
    )
    sub_ranges = dwt_subband_ranges(fs, L)

    cols = [f"{ch}_{b}_wenergy" for ch in df.columns for b in bands]
    rows = []

    for start in range(0, n_samples - nperseg + 1, hop):
        win = df.iloc[start : start + nperseg]
        row: dict[str, float] = {}

        for ch in df.columns:
            y = win[ch].to_numpy(dtype=float, copy=False)

            coeffs = pywt.wavedec(y, wavelet=wavelet, level=L, mode=mode)
            wv_coeff_approx = coeffs[0]
            wv_coeff_details = coeffs[1:]

            levels: dict[str, float] = {}
            for idx, c in enumerate(wv_coeff_details):
                j = L - idx
                levels[f"D{j}"] = float(np.sum(c.astype(float) ** 2))
            levels[f"A{L}"] = float(np.sum(wv_coeff_approx.astype(float) ** 2))

            band_energy = {name: 0.0 for name in bands}
            for sub_name, energy_sub in levels.items():
                f_lo, f_hi = sub_ranges[sub_name]
                width = (f_hi - f_lo) or 1.0
                if width <= 0:
                    continue
                for band_name, (band_lo, band_hi) in bands.items():
                    olap = _overlap((f_lo, f_hi), (band_lo, band_hi))
                    if olap > 0:
                        band_energy[band_name] += energy_sub * (olap / width)

            for band_name, e in band_energy.items():
                row[f"{ch}_{band_name}_wenergy"] = float(e)

        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


def wavelet_entropy(
    wv_band_energy_df: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    normalize: bool = True,
    eps: float = 1e-300,
) -> pd.DataFrame:
    """Compute (optionally normalized) Shannon entropy of wavelet band-energy distributions per channel.

    Expects columns ``{channel}_{band}_wenergy`` where each row is a windowed wavelet band energy.
    For each window and channel, treats the vector of ``_wenergy`` values across
    ``bands`` as a distribution, then computes ``-Σplogp``.
    Optionally normalizes by ``log(K)`` where ``K = len(bands)``.

    Parameters
    ----------
    wv_band_energy_df : pandas.DataFrame
        Output of :func:`wavelet_band_energy`; columns like ``{ch}_{band}_wenergy``.
    bands : dict[str, tuple[float, float]]
        Same band dictionary used for energy computation (order defines component order).
    normalize : bool, default=True
        If True, divide entropy by ``log(K)`` to obtain values in ``[0, 1]``.
    eps : float, default=1e-300
        Numerical guard for zero energies.

    Returns
    -------
    pandas.DataFrame
        One row per window, columns ``{ch}_wentropy``.

    Raises
    ------
    ValueError
        If no matching ``{channel}_{band}_wenergy`` columns are found.
    """
    df = wv_band_energy_df.select_dtypes(include=[np.number]).copy()

    band_list = list(bands.keys())
    K = len(band_list)
    norm = np.log(K) if (normalize and K > 1) else 1.0

    channel_to_cols: dict[str, list[str]] = {}
    for col in df.columns:
        if not col.endswith("_wenergy"):
            continue
        core = col[:-8]
        if "_" not in core:
            continue
        ch, b = core.rsplit("_", 1)
        if b in bands:
            channel_to_cols.setdefault(ch, [None] * K)

    if not channel_to_cols:
        raise ValueError(
            "No columns with pattern '{channel}_{band}_wenergy' matching provided bands."
        )

    for col in df.columns:
        if not col.endswith("_wenergy"):
            continue
        core = col[:-8]
        if "_" not in core:
            continue
        ch, b = core.rsplit("_", 1)
        if ch in channel_to_cols and b in bands:
            idx = band_list.index(b)
            channel_to_cols[ch][idx] = col

    out_cols = [f"{ch}_wentropy" for ch in channel_to_cols.keys()]
    rows = []

    for i in range(len(df)):
        row_out = {}
        for ch, cols_in_order in channel_to_cols.items():
            vals = []
            for c in cols_in_order:
                if c is None:
                    vals.append(0.0)
                else:
                    v = df.iat[i, df.columns.get_loc(c)]
                    vals.append(float(v) if np.isfinite(v) else 0.0)

            total = float(np.nansum(vals))
            total = total if (np.isfinite(total) and total > 0) else eps

            p = np.asarray(vals, dtype=float) / total
            p = np.clip(p, eps, 1.0)
            p /= p.sum()

            H = -np.sum(p * np.log(p))
            row_out[f"{ch}_wentropy"] = float(H / (norm or 1.0))
        rows.append(row_out)

    return pd.DataFrame(rows, columns=out_cols)


"""IMF FEATURES"""


def imf_band_energy(
    df: pd.DataFrame,
    fs: float,
    imf_to_band=["gamma", "betaH", "betaL", "alpha", "theta", "delta"],
    window_sec: float = 4.0,
    overlap: float = 0.5,
    EMD_kwargs: dict = {},
) -> pd.DataFrame:
    """Compute Intrinsic Mode Function (IMF) band-energy distributions per channel using
    Empirical Mode Decomposition (EMD).

    Expects columns named ``{channel}`` where each row is a reading of raw EEG data.
    For each column, applies EMD decomposition to find mean m(t) of envelopes (cubic splines
    fit to extremas) and takes difference between raw data point and m(t) [d(t) = x(t) - m(t)].
    For each window and channel, the IMF band-energy is based on the cumulative sums of the IMFs
    returned by the EMD function.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data EEG dataframe. Numeric columns.
    fs : float
        Sampling rate in Hz (used only for window sizing).
    imf_to_band : list[str], default=["gamma", "betaH", "betaL", "alpha", "theta", "delta"]
        Labels to assign to IMFs 1..K (K = number of labels).
    window_sec : float, default=4.0
        Window length in seconds.
    overlap : float, default=0.5
        Fractional overlap in ``[0, 1)``.
    EMD_kwargs : dict, default={}
        Extra keyword arguments passed to ``PyEMD.EMD`` (e.g., ``max_imf`` is internally set).

    Returns
    -------
    pandas.DataFrame
        One row per window; columns: ``{channel}_{band}_imfenergy`` for each band label.

    Raises
    ------
    ValueError
        If window too small, overlap invalid, or window longer than available samples.

    Notes
    -----
    - IMF indices are assigned in order to the names in ``imf_to_band``; if fewer IMFs
      are present than labels, missing energies are filled with 0.0.
    - The EMD is computed once per full signal and reused for all windows via cumulative sums.
    """

    df = df.select_dtypes(include=[np.number])

    n_samples = len(df)
    nperseg = int(round(window_sec * fs))

    if nperseg <= 8:
        raise ValueError("window_sec too small for given fs; increase window_sec.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    hop = int(round(nperseg * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("overlap too large; hop size must be >= 1 sample.")
    if nperseg > n_samples:
        cols = [f"{ch}_{band}_imfenergy" for ch in df.columns for band in imf_to_band]
        return pd.DataFrame(columns=cols)

    max_imf_needed = int(len(imf_to_band))
    cols = [f"{ch}_{band}_imfenergy" for ch in df.columns for band in imf_to_band]

    emd = EMD(**(EMD_kwargs))

    rows = []
    row: dict[str, float] = {}

    emd._imf_cumsums = {}

    for ch in df.columns:
        y_full = df[ch].to_numpy(dtype=float, copy=False).astype(float, copy=False)
        imfs_full = emd.emd(y_full, max_imf=max_imf_needed)
        sq = imfs_full**2
        cum_sq = np.hstack(
            [np.zeros((sq.shape[0], 1), dtype=sq.dtype), np.cumsum(sq, axis=1)]
        )

        emd._imf_cumsums[ch] = (imfs_full, cum_sq)

    for start in range(0, n_samples - nperseg + 1, hop):
        end = start + nperseg
        row: dict[str, float] = {}

        for ch in df.columns:
            e_win = emd._imf_cumsums[ch][1][:, end] - emd._imf_cumsums[ch][1][:, start]

            for imf_idx in range(len(imf_to_band)):
                e = float(e_win[imf_idx]) if imf_idx < e_win.shape[0] else 0.0
                row[f"{ch}_{imf_to_band[imf_idx]}_imfenergy"] = e

        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


def imf_entropy(
    imf_energy_df: pd.DataFrame,
    bands=["gamma", "betaH", "betaL", "alpha", "theta", "delta"],
    normalize: bool = True,
    eps: float = 1e-300,
) -> pd.DataFrame:
    """Compute (normalized) Shannon entropy of IMF energy distributions per channel.

    For each window and channel, uses the vector of ``_imfenergy`` values across
    ``bands`` as a distribution and computes ``-Σplogp``. Optionally normalizes
    by ``log(K)`` where ``K = len(bands)``.

    Parameters
    ----------
    imf_energy_df : pandas.DataFrame
        Output from :func:`imf_band_energy` with columns ``{ch}_{band}_imfenergy``.
    bands : list[str], default=["gamma", "betaH", "betaL", "alpha", "theta", "delta"]
        Band labels (order defines component order).
    normalize : bool, default=True
        If True, divide entropy by ``log(K)`` to obtain values in ``[0, 1]``.
    eps : float, default=1e-300
        Numerical guard for zero energies.

    Returns
    -------
    pandas.DataFrame
        One row per window; columns ``{ch}_imfentropy``.
    """

    df = imf_energy_df.select_dtypes(include=[np.number])

    k = len(bands)
    norm = np.log(k) if (normalize and k > 1) else 1.0

    channel_to_cols: dict[str, list[str]] = {}
    suffix = "_imfenergy"

    for col in df.columns:
        if not col.endswith(suffix):
            continue
        ch_band = col[: -len(suffix)]
        if "_" not in ch_band:
            continue
        ch, band = ch_band.rsplit("_", 1)
        if band in bands:
            channel_to_cols.setdefault(ch, [None] * k)

    for col in df.columns:
        if not col.endswith(suffix):
            continue
        core = col[: -len(suffix)]
        if "_" not in core:
            continue
        ch, band = core.rsplit("_", 1)
        if ch in channel_to_cols and band in bands:
            channel_to_cols[ch][bands.index(band)] = col

    out_cols = [f"{ch}_imfentropy" for ch in channel_to_cols.keys()]
    rows: list[dict[str, float]] = []

    for i in range(len(df)):
        row_out: dict[str, float] = {}
        for ch, cols_in_order in channel_to_cols.items():
            vals = []
            for c in cols_in_order:
                v = df.iat[i, df.columns.get_loc(c)]
                vals.append(float(v) if np.isfinite(v) else 0.0)

            total = float(np.nansum(vals))
            if not (np.isfinite(total) and total > 0):
                row_out[f"{ch}_imfentropy"] = np.nan
                continue

            p = np.asarray(vals, dtype=float) / total
            p = np.clip(p, eps, 1.0)
            p /= p.sum()
            H = -np.sum(p * np.log(p))
            row_out[f"{ch}_imfentropy"] = float(H / (norm or 1.0))

        rows.append(row_out)

    return pd.DataFrame(rows, columns=out_cols)


def generate_all_features(
    raw_eeg_df: pd.DataFrame,
    fs: int,
    bands: list[str] | None = None,
    channels: list[str] | None = None,
    index: bool = True,
):
    
    if bands is None:
        bands = FREQUENCY_BANDS
    if channels is not None:
        keep_columns = [ch for ch in channels if ch in raw_eeg_df.columns]

    clean = bandpass_filter(
        eeg_df, fs, bands=bands, low=0.5, high=45.0, notch_hz=60
    )
    psd = psd_bandpowers(clean, fs, bands=bands)
    shannons = shannons_entropy(clean, fs, bands=bands)
    hj = hjorth_params(clean, fs)
    wt_energy = wavelet_band_energy(raw_eeg_df, fs, bands=bands)
    wt_entropy = wavelet_entropy(wt_energy, bands=bands)
    imf_energy = imf_band_energy(raw_eeg_df, fs)
    imf_entr = imf_entropy(imf_energy)

    res_df = pd.concat(
        [psd, shannons, hj, wt_energy, wt_entropy, imf_energy, imf_entr],
        axis=1,
        index=index,
    )

    return res_df


if __name__ == "__main__":
    FS = 128
    csv_path = "DREAMER.csv"
    dreamer_df = pd.read_csv(csv_path)

    for patient_id in dreamer_df["patient_index"].unique():
        for video_id in dreamer_df["video_index"].unique():
            mask = (dreamer_df["patient_index"] == patient_id) & (
                dreamer_df["video_index"] == video_id
            )
            eeg_df = dreamer_df.loc[mask, :]
            del eeg_df["patient_index"]
            del eeg_df["video_index"]

            # clean = bandpass_filter(
            #     eeg_df, FS, bands=FREQUENCY_BANDS, low=0.5, high=45.0, notch_hz=60
            # )
            # hj = hjorth_params(clean, FS)
            # psd_df = psd_bandpowers(clean, FS, bands=FREQUENCY_BANDS)
            # shannons_df = shannons_entropy(clean, FS, bands=FREQUENCY_BANDS)
            # wt_df = wavelet_band_energy(eeg_df, FS, bands=FREQUENCY_BANDS)
            # print("Energy", wt_df)
            # wt_df = wavelet_entropy(wt_df, bands=FREQUENCY_BANDS)
            # print("Entropy", wt_df)

            imf_df = imf_band_energy(eeg_df, FS)
            print(imf_df)
            imf_df = imf_entropy(imf_df)
            print(imf_df)
            exit()
