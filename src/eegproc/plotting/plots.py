"""Plotting for EEG feature tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..data.schema import parse_feature_column



def plot_eeg_features(
    input_data: pd.DataFrame,
    title: str = "EEG Features",
    xlabel: str = "Time (s)",
    seconds: float = 4.0,
    overlap: float = 0.0,
    start_row: int = 0,
    end_row: int | None = None,
    save_path: str | None = None,
    max_width: float = 12.0,
    max_height_per_channel: float = 0.6,
    max_subplots: int | None = 40,
    channels: list[str] | None = None,
    frequency_bands: list[str] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot stacked EEG feature traces per channel and/or band.

    This function creates a vertically stacked line plot showing channel-level features 
    (e.g., entropy or bandpower) across time windows. Each subplot corresponds to one 
    feature column, with time on the x-axis (derived from the row index multiplied by the
    window duration ``seconds``).

    It supports filtering by subsets of channels and/or frequency bands, and automatically
    arranges figure size and subplot layout.

    Parameters
    ----------
    input_data : pandas.DataFrame
        DataFrame containing per-window EEG features (e.g., from
        :func:`shannons_entropy`, :func:`wavelet_entropy`, etc.).
    title : str, default="Entropy Plot"
        Figure title.
    xlabel : str, default="Time"
        X-axis label (typically "Time").
    seconds : float, default=4.0
        Duration represented by each row, in seconds. Used to scale the time axis.
    start_row : int, default=0
        Inclusive start row index to plot.
    end_row : int, default=1
        Exclusive end row index (like ``df.iloc[start:end]``). If ``None``, plots until
        the end of the DataFrame.
    save_path : str or None, optional
        If provided, saves the figure to this path (e.g., ``"entropy_plot.png"``).
        Otherwise, displays it interactively via ``plt.show()``.
    max_width : int or None, optional
        Maximum width (in inches) of the entire figure. If ``None``, width is auto-scaled.
    max_height_per_channel : int or None, optional
        Maximum height (in inches) allocated per channel subplot. If ``None``, auto-scaled.
    channels : list[str] or None, optional
        Subset of channel names to plot (e.g., ``["AF3", "F7"]``). If ``None``, includes all.
    frequency_bands : list[str] or None, optional
        Subset of frequency bands to include when aggregating (e.g., ``["alpha", "theta"]``).
        If ``None``, includes all bands found in column names.

    Raises
    ------
    ValueError
        If ``input_data`` is empty or the specified start/end rows yields an empty range.

    Notes
    -----
    - Each row of ``input_data`` corresponds to one analysis window (e.g., 4 seconds).
    - Columns are expected to follow patterns like:
      ``AF3_wentropy``, ``F7_wentropy``, ``AF3_alpha_entropy`` etc.
    - The function automatically infers which columns to plot based on substring matches
      for the requested ``channels`` and ``frequency_bands``.

    Examples
    --------
    Basic synthetic example:

    >>> import numpy as np, pandas as pd
    >>> from eegproc.plotting import plot_eeg_features
    >>>
    >>> # Per-window band powers for two channels, as psd_bandpowers emits them
    >>> t = np.arange(100)
    >>> df = pd.DataFrame({
    ...     "AF3_alpha": np.sin(0.1 * t),
    ...     "AF3_betaL": np.cos(0.1 * t),
    ...     "AF3_theta": np.cos(0.2 * t),
    ...     "F7_alpha": np.sin(0.1 * t + 1.0),
    ... })
    >>>
    >>> # Plot only AF3's alpha and betaL traces, for the first 50 windows
    >>> plot_eeg_features(
    ...     df,
    ...     title="AF3 Band Power (Synthetic Example)",
    ...     seconds=4,
    ...     start_row=0,
    ...     end_row=50,
    ...     channels=["AF3"],
    ...     frequency_bands=["alpha", "betaL"],
    ... )   # doctest: +SKIP

    Note that band filtering only applies to features whose column names carry a
    band token. ``shannons_entropy``, ``wavelet_entropy`` and ``imf_entropy``
    emit one column per channel (``{channel}_entropy``) with no band, so pass
    ``channels=`` alone for those.

    """

    if input_data is None or len(input_data) == 0:
        raise ValueError("input_data is empty.")

    n_rows = len(input_data)
    start = max(0, int(start_row))
    stop = n_rows if end_row is None else min(n_rows, int(end_row))
    if start >= stop:
        raise ValueError(
            f"empty row range [{start}:{stop}) for a frame with {n_rows} rows."
        )
    window = input_data.iloc[start:stop]

    parsed = {c: parse_feature_column(str(c)) for c in window.columns}

    # Match parsed structure, never a substring: `channels=["F3"]` must not also
    # select AF3, and the 10-20 montage contains both AF3/F3 and AF4/F4.
    selected = []
    for column, info in parsed.items():
        if channels is not None and info.channel not in channels:
            continue
        if frequency_bands is not None and info.band not in frequency_bands:
            continue
        if not pd.api.types.is_numeric_dtype(window[column]):
            continue
        selected.append(column)

    if not selected:
        _raise_empty_selection(window, parsed, channels, frequency_bands)

    if max_subplots is not None and len(selected) > max_subplots:
        raise ValueError(
            f"{len(selected)} columns matched, which would produce a "
            f"{max_height_per_channel * len(selected):.0f}-inch figure. Narrow the "
            "selection with channels=/frequency_bands=, or raise max_subplots."
        )

    # Consecutive windows advance by the hop, not the full window length.
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}.")
    hop_seconds = seconds * (1.0 - overlap)
    x = start * hop_seconds + np.arange(stop - start) * hop_seconds

    fig, axes = plt.subplots(
        nrows=len(selected),
        ncols=1,
        figsize=(max_width, max_height_per_channel * (len(selected) + 1)),
        sharex=True,
        squeeze=False,
    )
    axes = list(axes[:, 0])

    try:
        for ax, column in zip(axes, selected):
            ax.plot(x, window[column].to_numpy(dtype=float))
            ax.set_ylabel(column, rotation=0, ha="right", va="center", labelpad=20)
            ax.grid(True, linewidth=0.5, alpha=0.5)

        axes[-1].set_xlabel(xlabel)
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        if save_path:
            fig.savefig(save_path, dpi=150)
    except Exception:
        plt.close(fig)
        raise

    return fig, axes


def _raise_empty_selection(window, parsed, channels, frequency_bands) -> None:
    """Explain precisely why nothing matched, instead of failing in matplotlib."""
    available_channels = sorted({i.channel for i in parsed.values()})
    available_bands = sorted({i.band for i in parsed.values() if i.band})

    problems = []
    if channels is not None:
        unknown = [c for c in channels if c not in available_channels]
        if unknown:
            problems.append(
                f"unknown channel(s) {unknown}; available: {available_channels}"
            )
    if frequency_bands is not None:
        if not available_bands:
            problems.append(
                "none of these columns carry a band, so frequency_bands selects "
                "nothing. shannons_entropy, wavelet_entropy and imf_entropy emit "
                "one column per channel with no band — filter by channels only"
            )
        else:
            unknown = [b for b in frequency_bands if b not in available_bands]
            if unknown:
                problems.append(
                    f"unknown band(s) {unknown}; available: {available_bands}"
                )
    if not problems:
        problems.append(
            "the requested channel and band combination matched no column, or "
            "every matching column is non-numeric"
        )
    raise ValueError("no columns to plot: " + "; ".join(problems) + ".")
