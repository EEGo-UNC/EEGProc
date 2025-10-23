import numpy as np
import pandas as pd
import warnings
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.signal import detrend as scipy_detrend

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "betaL": (13.0, 20.0),
    "betaH": (20.0, 30.0),
    "gamma": (30.0, 45.0),
}


import numpy as np
import pandas as pd
from scipy.signal import detrend as scipy_detrend


def apply_detrend(detrend: str | None, df: pd.DataFrame) -> pd.DataFrame:
    """Apply interpolation and (optionally) remove a trend from each numeric column.

    This is a small convenience wrapper that first ensures numeric columns are
    float and linearly interpolated (forward/backward) via :func:`numeric_interp`,
    then, depending on ``detrend``, optionally removes a trend from each column
    using :func:`detrend_df`.

    Args:
        detrend: Which trend to remove. Accepted values are:
            - ``"constant"``: remove the mean from each series (demean).
            - ``"linear"``: remove a best-fit linear trend from each series.
            - ``None``: do not detrend; only interpolate numeric columns.
        df: Input dataframe. Non-numeric columns are ignored for interpolation
            and detrending (they are effectively dropped by the pipeline).

    Returns:
        pd.DataFrame: A dataframe with only numeric columns, interpolated, and
        optionally detrended according to ``detrend``. Index and column names
        are preserved for the remaining numeric columns.

    Raises:
        ValueError: If ``detrend`` is not one of ``{"constant", "linear", None}``.

    See Also:
        numeric_interp: Coerces numeric columns to float and bi-directionally interpolates.
        detrend_df: Applies SciPy detrending (constant or linear) column-wise.

    Examples:
        >>> import pandas as pd, numpy as np
        >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 11.0, np.nan], "c": ["x", "y", "z"]})
        >>> apply_detrend("constant", df)  # demean numeric columns after interpolation
                 a     b
        0 -1.000000 -0.5
        1  0.000000  0.5
        2  1.000000  0.0

        >>> apply_detrend(None, df)  # only numeric interpolation
             a     b
        0  1.0  10.0
        1  2.0  11.0
        2  3.0  11.0
    """
    if detrend in {"constant", "linear"}:
        df = _detrend_df(df, kind=detrend)
    elif detrend is None:
        df = numeric_interp(df)
    else:
        raise ValueError("detrend must be 'constant', 'linear', or None")
    return df


def numeric_interp(df: pd.DataFrame) -> pd.DataFrame:
    """Select numeric columns, cast to float, and interpolate missing values.

    The function:
      1) keeps only numeric dtypes,
      2) casts them to ``float`` (to allow NaNs and interpolation),
      3) fills missing values with linear interpolation along the index,
         using ``limit_direction='both'`` to also fill leading/trailing gaps.

    Non-numeric columns are dropped.

    Args:
        df: Input dataframe containing numeric and/or non-numeric columns.

    Returns:
        pd.DataFrame: A *copy* containing only the numeric columns as floats,
        with missing values linearly interpolated forward and backward.

    Notes:
        - Interpolation is performed independently per column.
        - If a column is entirely NaN, it will remain NaN.

    Examples:
        >>> import pandas as pd, numpy as np
        >>> df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": ["a", "b", "c"]})
        >>> numeric_interp(df)
             x
        0  1.0
        1  2.0
        2  3.0
    """
    df = df.select_dtypes(include=[np.number]).astype(float).copy()
    return df.apply(lambda s: s.interpolate(limit_direction="both"))


def _detrend_df(df: pd.DataFrame, kind: str = "linear") -> pd.DataFrame:
    if kind not in {"linear", "constant"}:
        raise ValueError("kind must be 'linear' or 'constant'")

    df = numeric_interp(df)
    arr = df.to_numpy(copy=False)
    arr = scipy_detrend(arr, type=kind, axis=0)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def sosfiltfilt_safe(sos, y):
    """Zero-phase Second-order Section (SOS) filter,
    handles NaNs and short signals.

    Applies ``scipy.signal.sosfiltfilt`` to a 1-D array.
    Linearly interpolates internal NaNs (or fills with 0.0 if
    too few finite samples to interpolate).

    Parameters
    ----------
    sos : array-like
        Second-order-sections filter coefficients as produced by
        ``scipy.signal.butter(..., output="sos")``.
    y : array-like of shape (n_samples,)
        Input signal (1-D). May contain NaNs.

    Returns
    -------
    numpy.ndarray
        Filtered signal. If the input is all-NaN or too short, returns a
        copy of the input (with NaNs interpolated when possible).

    Notes
    -----
    - NaNs at the edges are handled by linear interpolation using the nearest
      finite samples when available; otherwise they are set to 0.0.
    - The minimum-length guard (15) prevents filtfilt endpoint artifacts
      and poor padding when the window is too small.
    """
    if np.all(np.isnan(y)):
        return y
    y = y.copy()
    nans = np.isnan(y)
    if nans.any():
        iddf = np.where(~nans)[0]
        if iddf.size >= 2:
            y[nans] = np.interp(np.flatnonzero(nans), iddf, y[iddf])
        else:
            y[nans] = 0.0
    if y.size < 15:
        return y
    return sosfiltfilt(sos, y)


def apply_notch_once(
    dfin: pd.DataFrame,
    notch_hz: float | int | list | tuple | None,
    notch_q: float,
    nyq: float,
) -> pd.DataFrame:
    """Apply single-pass IIR notch filtering (and optional 2x harmonic) to each column
    to remove narrow amplitude readings over selected frequencies.

    Builds one or more IIR notch filters at the requested fundamental frequencies,
    optionally adding a 2x harmonic notch when it is safely below Nyquist, and applies
    zero-phase filtering (``scipy.signal.filtfilt``, ``method="gust"``) column-wise.

    Parameters
    ----------
    dfin : pandas.DataFrame
        Raw data EEG dataframe. Numeric columns.
    notch_hz : float | int | list | tuple | None
        Fundamental notch frequency (e.g., 50 or 60), or a list/tuple of such
        frequencies. If ``None``, no notch filtering is applied.
    notch_q : float
        Quality factor for ``scipy.signal.iirnotch`` (higher = narrower notch).
    nyq : float
        Nyquist frequency (``fs/2``), used to normalize the notch frequency.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with the notch (and eligible 2x harmonic) applied
        to all columns with sufficient length.

    Notes
    -----
    - A 2x harmonic is added for each fundamental ``f0`` when ``2*f0 < nyq - 1.0``.
    - Columns shorter than ``max(15, 3 * max(len(a), len(b)))`` samples are skipped
      for stability.
    - Frequencies are normalized as ``w0 = f0 / nyq`` for ``iirnotch``.
    """
    if notch_hz is None:
        return dfin
    freqs = notch_hz if isinstance(notch_hz, (list, tuple)) else [notch_hz]
    edfpanded = []
    for f0 in freqs:
        edfpanded.append(float(f0))
        if 2 * f0 < nyq - 1.0:
            edfpanded.append(float(2 * f0))
    out = dfin.copy()
    for f0 in edfpanded:
        w0 = f0 / nyq
        if 0 < w0 < 1:
            b, a = iirnotch(w0, notch_q)
            for c in dfin.columns:
                y = out[c].to_numpy()
                if y.size >= max(15, 3 * max(len(a), len(b))):
                    out[c] = filtfilt(b, a, y, method="gust")
    return out


def bandpass_filter(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]] = FREQUENCY_BANDS,
    low: float | None = None,
    high: float | None = None,
    *,
    order: int = 4,
    notch_hz: float | int | list | tuple | None = [60, 120],
    notch_q: float = 30.0,
    reref: bool = True,
    detrend: bool = True,
) -> pd.DataFrame:
    """Applies band-pass filtering over raw EEG data on each channel.

    Pipeline
    --------
    1) Coerce to numeric (with interpolation).

    2) Optional common average reference (CAR) across channels (``reref=True``).

    3) Optional notch filtering at ``notch_hz`` (and 2x harmonics when safe).

    4a) If ``bands`` is a dict: apply a per-band Butterworth band-pass (SOS) and
        return columns named ``{channel}_{band}``.

    4b) Else: require ``(low, high)`` and apply a single band-pass to each channel,
        returning the original channel names.

    5) Optional constant detrending per output column.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data EEG dataframe. Numeric columns; NaNs are
        interpolated internally by ``numeric_interp`` before filtering.
    fs : float
        Sampling rate in Hz.
    bands : dict[str, tuple[float, float]] or None, default=FREQUENCY_BANDS
        If provided, a mapping from band name to (low, high) in Hz. When given,
        one output column per ``{channel}_{band}`` is produced. If ``None``,
        ``low`` and ``high`` must be provided to define a single passband.
    low : float or None, default=None
        Low cutoff in Hz for the single band-pass path (ignored if ``bands`` provided).
    high : float or None, default=None
        High cutoff in Hz for the single band-pass path (ignored if ``bands`` provided).
    order : int, default=4
        Butterworth filter order for band-pass design.
    notch_hz : float | int | list | tuple | None, default=[60, 120]
        Fundamental notch frequency/frequencies. ``None`` disables notch filtering.
    notch_q : float, default=30.0
        Quality factor for the notch (higher = narrower).
    reref : bool, default=True
        If True and there are ≥2 channels, apply common average reference (CAR).
    detrend : bool, default=True
        If True, remove the mean (constant detrend) after filtering.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe indexed like the input. If ``bands`` is provided,
        columns are ``{channel}_{band}``; otherwise, the original channel names
        are preserved.

    Raises
    ------
    ValueError
        - When ``bands`` is ``None`` and either ``low`` or ``high`` is missing.
        - When any cutoff does not satisfy ``0 < low < high < fs/2``.
        - When a band in ``bands`` violates Nyquist constraints.

    Warnings
    --------
    RuntimeWarning
        Emitted when ``reref=True`` but only one channel is present (CAR requires ≥2).

    Notes
    -----
    - Band-pass filters are designed with ``scipy.signal.butter(..., output="sos")`` and
      applied with a NaN-robust zero-phase filter via :func:`_sosfiltfilt_safe`.
    - Notch filtering is applied once before band-pass filtering.
    - Constant detrend uses ``scipy.signal.detrend(..., type="constant")``.
    - For stability, very short columns may be skipped inside helper routines.
    """

    df = numeric_interp(df).apply(pd.to_numeric, errors="coerce")
    df = df.astype("float64")
    nyq = fs / 2.0
    cols = list(df.columns)

    if reref and len(cols) > 1:
        car = df.mean(axis=1)
        for c in cols:
            df[c] = df[c] - car
    elif len(cols) <= 1:
        warnings.warn(
            "reref=True ignored: only one channel present; CAR requires >= 2 channels.",
            RuntimeWarning,
        )

    df = apply_notch_once(df, notch_hz, notch_q, nyq)

    if bands is None:
        if low is None or high is None:
            raise ValueError("Provide (low, high) or a bands dict.")
        if not (0 < low < high < nyq):
            raise ValueError(f"Cutoffs must satisfy 0 < low < high < fs/2={nyq:.3f}.")
        sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
        Y = pd.DataFrame(index=df.index)
        for c in cols:
            Y[c] = sosfiltfilt_safe(sos, df[c].to_numpy())
        if detrend:
            for c in cols:
                Y[c] = scipy_detrend(Y[c].to_numpy(), type="constant")
        return Y

    out = {}
    band_sos = {}
    for name, (lo, hi) in bands.items():
        if not (0 < lo < hi < nyq):
            raise ValueError(f"Bad band {name}: {lo}-{hi} vs fs/2={nyq:.3f}")
        band_sos[name] = butter(
            order, [lo / nyq, hi / nyq], btype="bandpass", output="sos"
        )

    for c in cols:
        y0 = df[c].to_numpy()
        for name, sos in band_sos.items():
            yb = sosfiltfilt_safe(sos, y0)
            if detrend:
                yb = scipy_detrend(yb, type="constant")
            out[f"{c}_{name}"] = yb

    return pd.DataFrame(out, index=df.index)
