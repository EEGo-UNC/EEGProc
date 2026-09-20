import re
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_eeg_features(
    input_data: pd.DataFrame,
    title: str = "Entropy Plot",
    xlabel: str = "Time",
    seconds: float = 4.0,
    start_row: int = 0,
    end_row: int = 1,
    save_path: Optional[str] = None,
    max_width: Optional[int] = None,
    max_height_per_channel: Optional[int] = None,
    channels: Optional[list[str]] = None,
    frequency_bands: Optional[list[str]] = None,
) -> None:
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
    >>> from matplotlib import pyplot as plt
    >>> from eegproc.plotting import plot_per_channel
    >>>
    >>> # Simulate 3 channels and 2 bands over 100 windows
    >>> t = np.arange(100)
    >>> df = pd.DataFrame({
    ...     "AF3_alpha_entropy": np.sin(0.1 * t) + 0.1*np.random.randn(100),
    ...     "AF3_beta_entropy":  np.cos(0.1 * t) + 0.1*np.random.randn(100),
    ...     "AF3_theta_entropy":  np.cos(0.1 * t) + 0.1*np.random.randn(100),
    ...     "F7_alpha_entropy":  np.sin(0.1 * t + 1.0),
    ... })
    >>>
    >>> # Plot only AF3 alpha and beta band entropies, for the first 50 windows
    >>> plot_per_channel(
    ...     df,
    ...     title="AF3 Entropy (Synthetic Example)",
    ...     seconds=4,
    ...     start_row=0,
    ...     end_row=50,
    ...     channels=["AF3"],
    ...     frequency_bands=["alpha", "beta"]
    ... )
    >>>
    >>> # To save instead of showing:
    >>> # plot_per_channel(df, save_path="entropy_AF3.png", channels=["AF3"])

    """

    if input_data is None or len(input_data) == 0:
        raise ValueError("input_data is empty.")

    n = len(input_data)
    s = max(0, int(start_row))
    e = n if end_row is None else min(n, end_row)
    if s >= e:
        raise ValueError(f"Empty window [{s}:{e}) for n={n}.")

    df = input_data.iloc[s:e].copy()


    columns = []
    include = True

    for key in df.keys():
        include = True

        if channels is not None:
            include = any(re.search(rf"{ch}", key) for ch in channels)

        if frequency_bands is not None and include:
            include = any(re.search(rf"{band}", key, re.IGNORECASE) for band in frequency_bands)

        if include:
            columns.append(key)


    # X-axis in seconds
    x = (np.arange(s, e) - s) * seconds

    # Plot
    max_width = 12 if max_width is None else max_width
    max_height_per_channel = 0.6 if max_height_per_channel is None else max_height_per_channel
    fig, axes = plt.subplots(
        nrows=len(columns),
        ncols=1,
        figsize=(max_width, max_height_per_channel * (len(columns) + 1)),
        sharex=True,
    )

    if len(columns) == 1:
        axes = [axes]

    for ax, ch in zip(axes, columns):
        y = df[ch]
        ax.plot(x, y)
        ax.set_ylabel(ch, rotation=0, ha="right", va="center", labelpad=20) 
        ax.grid(True, linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    from .. import wavelet_band_energy, wavelet_entropy, FREQUENCY_BANDS
    from .. import bandpass_filter, shannons_entropy

    FS = 128
    csv_path = "DREAMER.csv"
    chunk_iter = pd.read_csv(csv_path, chunksize=1)
    first_chunk = next(chunk_iter)
    sensor_columns = [col for col in first_chunk.columns if col[len(col) - 1].isdigit()]
    print(f"Detected sensor columns: {sensor_columns}")

    dreamer_df = []

    for chunk in pd.read_csv(csv_path, chunksize=10000):
        sensor_df = chunk[sensor_columns]
        dreamer_df.append(sensor_df)

    dreamer_df = pd.concat(dreamer_df, ignore_index=True)

    clean = bandpass_filter(dreamer_df, FS, bands=FREQUENCY_BANDS, low=0.5, high=45.0, notch_hz=60)

    # hj = hjorth_params(clean, FS)
    # print("Hjorth Parameters\n", hj)

    # psd_df = psd_bandpowers(clean, FS, bands=FREQUENCY_BANDS)
    # print("PSD\n", psd_df)

    shannons_df = shannons_entropy(clean, FS, bands=FREQUENCY_BANDS)
    print("Shannons\n", shannons_df)

    plot_per_channel(
        shannons_df,
        title="Shannons Entropy per Channel",
        seconds=4,
        start_row=0,
        end_row=500,
        max_height_per_channel=0.8,
        save_path="shannons_entropy_plot",
        channels=['AF3'],
        frequency_bands=['delta'],
    )



    # wt_df = wavelet_band_energy(dreamer_df, FS, bands=FREQUENCY_BANDS)
    # print("WT Energy\n", wt_df)

    # wt_df = wavelet_entropy(wt_df, bands=FREQUENCY_BANDS)
    # print("WT Entropy\n", wt_df)

    # plot_per_channel(
    #     wt_df,
    #     title="Wavelet Entropy per Channel",
    #     seconds=4,
    #     start_row=0,
    #     end_row=500,
    # )
