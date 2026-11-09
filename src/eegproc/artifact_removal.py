"""Remove artifacts such as eye blinks from EEG tables with independent component analysis."""

from __future__ import annotations

import importlib.util
import warnings
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, welch

from .preprocessing import bandpass_filter, numeric_interp, sosfiltfilt_safe

# Bands used to compare power before and after removal.
_CHECK_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
_MIN_SAMPLES_PER_WEIGHT = 20  # common rule of thumb: k * n_components ** 2 samples
_BLINK_BAND_HZ = (1.0, 8.0)

#: Component types scored by ICLabel, in the order of its output columns.
ICLABEL_LABELS = (
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
)
_ICLABEL_TOP_HZ = 100.0  # ICLabel was trained on 1-100 Hz data


def _require_mne():
    try:
        import mne
    except ImportError as error:
        raise ImportError(
            "ICA artifact removal requires MNE-Python. "
            "Install it with: pip install 'eegproc[ica]'"
        ) from error
    return mne


def _require_iclabel():
    """The ICLabel scoring function, or an ImportError that says how to install it."""
    try:
        from mne_icalabel.iclabel import iclabel_label_components
    except ImportError as error:
        raise ImportError(
            "remove_artifacts_with_iclabel requires mne-icalabel. "
            "Install it with: pip install 'eegproc[iclabel]'"
        ) from error
    if not any(importlib.util.find_spec(name) for name in ("onnxruntime", "torch")):
        raise ImportError(
            "ICLabel needs onnxruntime or torch to run its network. "
            "Install it with: pip install 'eegproc[iclabel]'"
        )
    return iclabel_label_components


def _default_montage(mne) -> str:
    """The 10-20 montage under this MNE version's name for it."""
    return "colin27_1020" if "colin27_1020" in mne.channels.get_builtin_montages() else "standard_1020"


def _segment_positions(df: pd.DataFrame, segment_columns: Sequence[str]) -> list[np.ndarray]:
    """Row positions of each continuous recording, in table order."""
    if not segment_columns:
        return [np.arange(len(df))]
    missing = [column for column in segment_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"segment_columns {missing} are not in the table. Columns named in "
            "group_by_metadata_columns are removed before this function runs."
        )
    groups = df.groupby(list(segment_columns), sort=False, dropna=False).indices
    return sorted(groups.values(), key=lambda positions: positions[0])


def _eeg_signal(df: pd.DataFrame, channels: Sequence[str]) -> np.ndarray:
    """Numeric samples shaped ``(n_samples, n_channels)`` with gaps interpolated."""
    channels = list(channels)
    if len(channels) < 2 or len(set(channels)) != len(channels):
        raise ValueError("eeg_channels must name at least two different columns.")
    missing = [channel for channel in channels if channel not in df.columns]
    if missing:
        raise ValueError(f"eeg_channels {missing} are not in the table.")
    frame = numeric_interp(df[channels])
    if list(frame.columns) != channels:
        dropped = sorted(set(channels) - set(frame.columns))
        raise ValueError(f"eeg_channels must be numeric; got {dropped}.")
    signal = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError("eeg_channels contain a column with no finite samples.")
    return signal


def _filtered_copy(signal, segments, channels, fs, highpass, lowpass) -> np.ndarray:
    """Join each segment's filtered samples, shaped ``(n_channels, n_samples)``.

    ``lowpass=None`` applies only the high-pass, keeping everything up to Nyquist.
    """
    if lowpass is None and not 0 < highpass < fs / 2:
        raise ValueError(f"highpass must satisfy 0 < highpass < fs/2={fs / 2:.3f}.")
    high_only = butter(4, highpass / (fs / 2), btype="highpass", output="sos") if lowpass is None else None
    parts = []
    for positions in segments:
        if len(positions) < fs:  # too short to filter at the high-pass cutoff
            continue
        if high_only is not None:
            filtered = np.stack([sosfiltfilt_safe(high_only, column) for column in signal[positions].T])
            parts.append(filtered - filtered.mean(axis=1, keepdims=True))
            continue
        frame = pd.DataFrame(signal[positions], columns=channels)
        filtered = bandpass_filter(
            frame, fs, bands=None, low=highpass, high=lowpass,
            notch_hz=None, reref=False, detrend=True,
        )
        parts.append(filtered.to_numpy().T)
    if not parts:
        return np.empty((len(channels), 0))
    return np.concatenate(parts, axis=1)


def _drop_transients(data: np.ndarray, fs: float, factor: float) -> tuple[np.ndarray, float]:
    """Drop one-second windows far larger than this recording's typical window.

    The limit is relative because typical amplitude differs widely between
    recordings. It sits well above blink amplitude so that blinks stay in the
    data ICA learns from.
    """
    size = int(round(fs))
    n_windows = data.shape[1] // size
    if n_windows == 0:
        return data[:, :0], 0.0
    windows = data[:, : n_windows * size].reshape(data.shape[0], n_windows, size)
    peak_to_peak = (windows.max(axis=2) - windows.min(axis=2)).max(axis=0)
    keep = peak_to_peak <= factor * np.median(peak_to_peak)
    kept = windows[:, keep, :].reshape(data.shape[0], -1)
    return kept, float(1.0 - keep.mean())


def _numerical_rank(data: np.ndarray) -> int:
    """Independent dimensions in ``data``; one fewer than channels after CAR."""
    eigenvalues = np.linalg.eigvalsh(np.cov(data))
    return int((eigenvalues > eigenvalues.max() * 1e-10).sum())


def _head_regions(mne, channels, montage) -> tuple[dict[str, list[int]], list[str]]:
    """Channel indices per head region, and the two frontmost channel names."""
    positions = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    missing = [channel for channel in channels if channel not in positions]
    if missing:
        raise ValueError(f"Channels {missing} are not in the {montage!r} montage.")
    order = np.argsort([-positions[channel][1] for channel in channels])
    edge = max(2, int(round(0.3 * len(channels))))
    front = order[:edge].tolist()
    regions = {
        "front": front,
        "back": order[-edge:].tolist(),
        "front_left": [i for i in front if positions[channels[i]][0] < 0],
        "front_right": [i for i in front if positions[channels[i]][0] > 0],
    }
    return regions, [channels[position] for position in order[:2]]


def _left_right_balance(maps: np.ndarray, left: list[int], right: list[int]) -> np.ndarray:
    """Per component, 1 for equal same-sign frontal weight and 0 for one-sided."""
    if not left or not right:  # a montage without both sides cannot show this
        return np.ones(maps.shape[1])
    left_weight, right_weight = maps[left].mean(axis=0), maps[right].mean(axis=0)
    smaller = np.minimum(np.abs(left_weight), np.abs(right_weight))
    larger = np.maximum(np.maximum(np.abs(left_weight), np.abs(right_weight)), 1e-12)
    return np.where(left_weight * right_weight > 0, smaller / larger, 0.0)


def _find_blink_components(
    ica, raw, regions, blink_channels, fs, reref,
    min_correlation, min_front_back_ratio, min_left_right_balance, min_low_frequency_share,
) -> list[int]:
    """Components that track frontal channels and look like a blink.

    EEG tables rarely include an EOG channel, so the frontmost channels stand in
    for one. Correlation with them is not enough on its own: large non-eye
    components can also correlate. A blink must also sit at the front of the
    head, reach both sides equally, and be slow.
    """
    front, back = regions["front"], regions["back"]
    _, scores = ica.find_bads_eog(
        raw, ch_name=blink_channels, threshold=min_correlation,
        measure="correlation", verbose=False,
    )
    correlation = np.abs(np.atleast_2d(np.asarray(scores))).max(axis=0)

    maps = ica.get_components()
    if reref:
        # An average reference shifts a frontal map by a constant, which would
        # otherwise read as weight at the back of the head.
        maps = maps - maps[back].mean(axis=0)
    balance = _left_right_balance(maps, regions["front_left"], regions["front_right"])
    maps = np.abs(maps)
    ratio = maps[front].mean(axis=0) / np.maximum(maps[back].mean(axis=0), 1e-12)

    sources = ica.get_sources(raw).get_data()
    frequencies, power = welch(sources, fs, nperseg=int(4 * fs), axis=-1)
    in_blink_band = (frequencies >= _BLINK_BAND_HZ[0]) & (frequencies < _BLINK_BAND_HZ[1])
    in_check_range = (frequencies >= 1.0) & (frequencies < 45.0)
    low_share = power[:, in_blink_band].sum(axis=1) / power[:, in_check_range].sum(axis=1)

    passed = (
        (correlation >= min_correlation)
        & (ratio >= min_front_back_ratio)
        & (balance >= min_left_right_balance)
        & (low_share >= min_low_frequency_share)
    )
    return [int(component) for component in np.flatnonzero(passed)]


def _band_powers(data: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Median-averaged Welch power, so rare large windows do not set the result."""
    frequencies, power = welch(data, fs, nperseg=int(2 * fs), axis=-1, average="median")
    return {
        band: power[:, (frequencies >= low) & (frequencies < high)].sum(axis=1)
        for band, (low, high) in _CHECK_BANDS.items()
    }


def _power_change(before, after, front, back) -> dict[str, dict[str, float]]:
    """Fractional power change per band at each end of the head."""
    return {
        group: {
            band: float(after[band][rows].mean() / before[band][rows].mean() - 1.0)
            for band in before
        }
        for group, rows in (("front", front), ("back", back))
    }


def _passes_power_check(change, limit: float, limit_back_falls: bool) -> bool:
    """Removing an artifact should never add power, and blink removal should
    leave the back of the head alone.

    No band may rise by more than ``limit`` anywhere. With ``limit_back_falls``,
    bands above delta may not fall by more than ``limit`` at the back of the
    head either. Delta is exempt because slow blink activity does reach those
    channels. Callers turn the fall limit off under an average reference, which
    copies part of each blink onto all channels, and when removing artifacts
    such as muscle that do sit at the back of the head.
    """
    rises = [*change["front"].values(), *change["back"].values()]
    if max(rises) > limit:
        return False
    if not limit_back_falls:
        return True
    return min(value for band, value in change["back"].items() if band != "delta") >= -limit


def remove_eye_artifacts(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    *,
    eeg_channels: Sequence[str],
    segment_columns: Sequence[str] = (),
    method: str = "infomax",
    reref: bool = False,
    highpass: float = 1.0,
    lowpass: float = 45.0,
    transient_factor: float = 3.0,
    min_correlation: float = 0.7,
    min_front_back_ratio: float = 3.0,
    min_left_right_balance: float = 0.6,
    min_low_frequency_share: float = 0.7,
    max_power_change: float = 0.10,
    component_selector: Callable | None = None,
    montage: str | None = None,
    random_state: int = 42,
    report: list | None = None,
) -> pd.DataFrame:
    """Remove eye-blink components from one person's EEG with ICA.

    ICA needs several minutes of data from one wearer, so run this per subject
    rather than per trial, before :func:`~eegproc.preprocessing.bandpass_filter`.
    Requires the optional dependency: ``pip install 'eegproc[ica]'``.

    Pipeline
    --------
    1) Filter a copy of each continuous segment and join them.

    2) Drop rare, very large transients from that copy. They would otherwise
       dominate the decomposition.

    3) Fit ICA on the copy, then pick blink components.

    4) Compare power on the copy with and without those components. If the
       change looks wrong, keep the data unchanged and warn.

    5) Remove the components from the original samples, keeping each channel's
       offset and everything below ``highpass``.

    Parameters
    ----------
    df : pandas.DataFrame
        One person's samples, one row per sample. Rows of a segment must be in
        time order. Columns other than ``eeg_channels`` pass through unchanged.
    fs : float
        Sampling rate in Hz.
    bands : dict, optional
        Ignored. Accepted so this runs through
        :func:`~eegproc.featurization.feature_grouped_by_metadata`.
    eeg_channels : sequence of str
        EEG columns, named as in ``montage``. Named separately from the
        ``channels`` argument of ``feature_grouped_by_metadata``, which selects
        ``{channel}_{band}`` columns.
    segment_columns : sequence of str, default=()
        Columns that together identify a continuous recording, such as
        ``("trial_id", "segment")``. Filtering does not cross these boundaries.
        Empty treats the table as one continuous recording.
    method : {"infomax", "picard", "fastica"}, default="infomax"
        ICA algorithm. Infomax runs in its extended form, which is what ICLabel
        expects. ``"picard"`` needs ``python-picard``; ``"fastica"`` needs
        scikit-learn and is much faster.
    reref : bool, default=False
        Apply a common average reference before ICA. The returned EEG then keeps
        that reference, and one fewer component is available. ICLabel expects it.
    highpass, lowpass : float, default=1.0, 45.0
        Passband in Hz of the copy used for fitting and for the power check.
    transient_factor : float, default=3.0
        Drop one-second windows whose largest peak-to-peak amplitude exceeds
        this multiple of the median window. Applies to the fitting copy only.
    min_correlation : float, default=0.7
        Smallest absolute correlation with the frontmost channels.
    min_front_back_ratio : float, default=3.0
        Smallest ratio of front-of-head to back-of-head component weight.
    min_left_right_balance : float, default=0.6
        Smallest ratio of the weaker to the stronger frontal side. Blinks reach
        both sides equally; one-sided components are usually a noisy electrode
        or a sideways eye movement.
    min_low_frequency_share : float, default=0.7
        Smallest share of component power in 1-8 Hz, relative to 1-45 Hz.
    max_power_change : float, default=0.10
        Largest allowed fractional rise in any band, and largest allowed fall
        above delta at the back of the head. Falls are not limited with
        ``reref=True``, because the reference spreads blinks to every channel.
    component_selector : callable, optional
        Replaces the built-in blink search. Called as
        ``component_selector(ica, raw)`` with the fitted ``mne.preprocessing.ICA``
        and the ``mne.io.Raw`` it was fitted on, and returns component indices.
        The power check still applies.
    montage : str, optional
        Name of the MNE built-in montage that locates the channels. Defaults to
        the 10-20 montage.
    random_state : int, default=42
        Seed for the ICA fit.
    report : list, optional
        Receives one dict per call, in group order when run per subject, with
        ``status``, ``components``, ``n_components``, ``rejected_fraction`` and
        ``power_change``. ``status`` is ``"cleaned"``, ``"no_blink_component"``,
        ``"failed_power_check"`` or ``"not_enough_data"``.

    Returns
    -------
    pandas.DataFrame
        Same columns and row order as ``df``, with a default integer index.
        Missing EEG samples are interpolated. When no components are removed,
        the EEG is unchanged apart from that and ``reref``.

    Raises
    ------
    ImportError
        If MNE-Python is not installed.
    ValueError
        If ``eeg_channels`` or ``segment_columns`` are missing, non-numeric,
        empty, or not in ``montage``.

    Warnings
    --------
    RuntimeWarning
        Emitted whenever the EEG is returned without any component removed.

    Notes
    -----
    Run this only on data that has not already had eye artifacts removed. The
    published DEAP and AMIGOS preprocessed files have.

    Without an EOG channel the search relies on frontal channels, which also
    carry frontal brain activity. Removal can lower frontal alpha and beta power
    along with blinks, so inspect ``report`` and compare results with and
    without this step.

    Examples
    --------
    >>> from eegproc import feature_grouped_by_metadata, remove_eye_artifacts
    >>> report = []
    >>> clean = feature_grouped_by_metadata(  # doctest: +SKIP
    ...     raw, target_function=remove_eye_artifacts, fs=128,
    ...     group_by_metadata_columns=["subject_id"],
    ...     eeg_channels=CHANNELS, segment_columns=["trial_id", "segment"],
    ...     report=report,
    ... )
    """
    return _remove_ica_components(
        df, fs, eeg_channels=eeg_channels, segment_columns=segment_columns,
        method=method, reref=reref, highpass=highpass, lowpass=lowpass,
        transient_factor=transient_factor, max_power_change=max_power_change,
        limit_back_falls=not reref, unit_in_volts=None, montage=montage,
        random_state=random_state, report=report,
        select=_rule_selector(
            component_selector, min_correlation, min_front_back_ratio,
            min_left_right_balance, min_low_frequency_share,
        ),
        no_match_status="no_blink_component",
        no_match_message="no component looks like a blink.",
    )


def _rule_selector(component_selector, min_correlation, min_front_back_ratio,
                   min_left_right_balance, min_low_frequency_share):
    """The blink rules, or a caller's selector, in the shared pipeline's form."""
    def select(ica, raw, context, summary):
        if component_selector is not None:
            return component_selector(ica, raw)
        return _find_blink_components(
            ica, raw, context["regions"], context["blink_channels"], context["fs"],
            context["reref"], min_correlation, min_front_back_ratio,
            min_left_right_balance, min_low_frequency_share,
        )
    return select


def remove_artifacts_with_iclabel(
    df: pd.DataFrame,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    *,
    eeg_channels: Sequence[str],
    segment_columns: Sequence[str] = (),
    labels: Sequence[str] = ("eye blink",),
    threshold: float = 0.9,
    reref: bool = True,
    highpass: float = 1.0,
    transient_factor: float = 3.0,
    max_power_change: float = 0.10,
    unit_in_volts: float = 1e-6,
    method: str = "infomax",
    montage: str | None = None,
    random_state: int = 42,
    report: list | None = None,
) -> pd.DataFrame:
    """Remove ICA components that ICLabel classifies as artifacts.

    Works like :func:`remove_eye_artifacts`, but a trained classifier decides
    which components to remove instead of hand-written blink rules. ICLabel
    (Pion-Tonachini et al., 2019) scores every component as brain, muscle
    artifact, eye blink, heart beat, line noise, channel noise or other. Run it
    per subject, before :func:`~eegproc.preprocessing.bandpass_filter`.
    Requires the optional dependency: ``pip install 'eegproc[iclabel]'``.

    Pipeline
    --------
    1) Optionally apply a common average reference, which ICLabel expects.

    2) High-pass a copy of each continuous segment at ``highpass`` and join
       them. ICLabel was trained on 1-100 Hz data, so the copy keeps everything
       up to 100 Hz, or up to Nyquist when the sampling rate is lower.

    3) Drop rare, very large transients from that copy, fit extended infomax
       ICA, and score each component with ICLabel.

    4) Select components whose probability for any of ``labels`` is at least
       ``threshold``.

    5) Compare power with and without those components. If any band rises, keep
       the data unchanged and warn. Otherwise remove the components from the
       original samples, keeping each channel's offset.

    Parameters
    ----------
    df : pandas.DataFrame
        One person's samples, one row per sample. Rows of a segment must be in
        time order. Columns other than ``eeg_channels`` pass through unchanged.
    fs : float
        Sampling rate in Hz.
    bands : dict, optional
        Ignored. Accepted so this runs through
        :func:`~eegproc.featurization.feature_grouped_by_metadata`.
    eeg_channels : sequence of str
        EEG columns, named as in ``montage``.
    segment_columns : sequence of str, default=()
        Columns that together identify a continuous recording, such as
        ``("trial_id", "segment")``. Filtering does not cross these boundaries.
    labels : sequence of str, default=("eye blink",)
        Component types to remove, from :data:`ICLABEL_LABELS`. Add
        ``"muscle artifact"`` or ``"heart beat"`` to remove those too.
    threshold : float, default=0.9
        Smallest ICLabel probability, between 0 and 1, for removal. 0.9 is the
        cutoff EEGLAB commonly uses for eye and muscle components; lower values
        also catch one-sided frontal components that are often electrode noise.
    reref : bool, default=True
        Apply a common average reference first, as ICLabel expects. The returned
        EEG then keeps that reference, and one fewer component is available.
        If later steps do not use an average reference, compare with
        ``reref=False``: ICLabel then warns, but the reference stays unchanged.
    highpass : float, default=1.0
        High-pass cutoff in Hz of the copy used for fitting and scoring.
    transient_factor : float, default=3.0
        Drop one-second windows whose largest peak-to-peak amplitude exceeds
        this multiple of the median window. Applies to the fitting copy only.
    max_power_change : float, default=0.10
        Largest allowed fractional rise in any band. When removing eye blinks
        only and ``reref=False``, also the largest allowed fall above delta at
        the back of the head.
    unit_in_volts : float, default=1e-6
        Volts per unit of the EEG columns; the default means microvolts.
        ICLabel's spectrum feature depends on absolute amplitude, so this must
        be right.
    method : {"infomax", "picard", "fastica"}, default="infomax"
        ICA algorithm. ICLabel was trained on extended infomax, which
        ``"infomax"`` runs, and warns about the others.
    montage : str, optional
        Name of the MNE built-in montage that locates the channels. Defaults to
        the 10-20 montage.
    random_state : int, default=42
        Seed for the ICA fit.
    report : list, optional
        Receives one dict per call with ``status``, ``components``,
        ``n_components``, ``rejected_fraction``, ``power_change`` and
        ``iclabel``: one dict of label probabilities per component. ``status``
        is ``"cleaned"``, ``"no_matching_component"``,
        ``"failed_power_check"`` or ``"not_enough_data"``.

    Returns
    -------
    pandas.DataFrame
        Same columns and row order as ``df``, with a default integer index.
        Missing EEG samples are interpolated. When no components are removed,
        the EEG is unchanged apart from that and ``reref``.

    Raises
    ------
    ImportError
        If MNE-Python, mne-icalabel, or both onnxruntime and torch are missing.
    ValueError
        If ``labels`` or ``threshold`` are invalid, or if ``eeg_channels`` or
        ``segment_columns`` are missing, non-numeric, empty, or not in
        ``montage``.

    Warnings
    --------
    RuntimeWarning
        Emitted whenever the EEG is returned without any component removed.

    Notes
    -----
    ICLabel was trained on 1-100 Hz recordings, mostly from research caps with
    many more channels than consumer headsets. Below 200 Hz sampling it fills
    the spectrum above Nyquist with the last value, as the original EEGLAB code
    does. Its scores on low-density or low-rate data have not been validated,
    so inspect ``report`` and compare results with and without this step.

    Examples
    --------
    >>> from eegproc import feature_grouped_by_metadata, remove_artifacts_with_iclabel
    >>> report = []
    >>> clean = feature_grouped_by_metadata(  # doctest: +SKIP
    ...     raw, target_function=remove_artifacts_with_iclabel, fs=128,
    ...     group_by_metadata_columns=["subject_id"],
    ...     eeg_channels=CHANNELS, segment_columns=["trial_id", "segment"],
    ...     report=report,
    ... )
    """
    labels = tuple(labels)
    unknown = [label for label in labels if label not in ICLABEL_LABELS]
    if not labels or unknown:
        raise ValueError(f"labels must be chosen from {ICLABEL_LABELS}; got {list(labels)}.")
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1]; got {threshold}.")
    if not unit_in_volts > 0:
        raise ValueError(f"unit_in_volts must be positive; got {unit_in_volts}.")
    label_components = _require_iclabel()

    def select(ica, raw, context, summary):
        scores = _iclabel_scores(label_components, raw, ica, context["reref"], highpass)
        summary["iclabel"] = [
            {label: float(p) for label, p in zip(ICLABEL_LABELS, row)} for row in scores
        ]
        columns = [ICLABEL_LABELS.index(label) for label in labels]
        return np.flatnonzero(scores[:, columns].max(axis=1) >= threshold).tolist()

    return _remove_ica_components(
        df, fs, eeg_channels=eeg_channels, segment_columns=segment_columns,
        method=method, reref=reref, highpass=highpass,
        lowpass=_ICLABEL_TOP_HZ if fs / 2 > _ICLABEL_TOP_HZ else None,
        transient_factor=transient_factor, max_power_change=max_power_change,
        limit_back_falls=not reref and set(labels) == {"eye blink"},
        unit_in_volts=unit_in_volts, montage=montage, random_state=random_state,
        report=report, select=select, extra_summary={"iclabel": []},
        no_match_status="no_matching_component",
        no_match_message=f"no component is labeled {' or '.join(labels)} "
                         f"with probability >= {threshold}.",
    )


def _iclabel_scores(label_components, raw, ica, reref: bool, highpass: float) -> np.ndarray:
    """ICLabel probabilities, shaped ``(n_components, 7)``."""
    scored = raw.copy()
    if reref:
        # The data is already average-referenced; this records that in its info.
        scored.set_eeg_reference("average", projection=False, verbose=False)
    with warnings.catch_warnings():
        if highpass == 1.0:
            # The copy is filtered from 1 Hz to 100 Hz, or to Nyquist when that is
            # lower, which is as close to ICLabel's training data as the rate allows.
            warnings.filterwarnings("ignore", message=".*not filtered between 1 and 100 Hz.*")
        return np.asarray(label_components(scored, ica, inplace=False))


def _remove_ica_components(
    df, fs, *, eeg_channels, segment_columns, method, reref, highpass, lowpass,
    transient_factor, max_power_change, limit_back_falls, unit_in_volts, montage,
    random_state, report, select, no_match_status, no_match_message, extra_summary=None,
) -> pd.DataFrame:
    """Fit ICA on a filtered copy, remove the selected components, and check power.

    ``select(ica, raw, context, summary)`` returns component indices; it may add
    entries to ``summary``. ``unit_in_volts=None`` rescales the data to a fixed
    size, which suits anything that ignores absolute amplitude.
    """
    mne = _require_mne()
    channels = list(eeg_channels)
    signal = _eeg_signal(df, channels)
    segments = _segment_positions(df, segment_columns)
    montage = montage or _default_montage(mne)
    regions, blink_channels = _head_regions(mne, channels, montage)
    front, back = regions["front"], regions["back"]
    if reref:
        signal = signal - signal.mean(axis=1, keepdims=True)

    summary = {
        "status": "not_enough_data", "components": [], "n_components": 0,
        "rejected_fraction": 0.0, "power_change": {}, **(extra_summary or {}),
    }

    def finish(data: np.ndarray, message: str | None = None) -> pd.DataFrame:
        if report is not None:
            report.append(summary)
        if message is not None:
            warnings.warn(f"EEG returned without removal: {message}", RuntimeWarning, stacklevel=4)
        out = df.reset_index(drop=True)
        out[channels] = data
        return out

    copy = _filtered_copy(signal, segments, channels, fs, highpass, lowpass)
    kept, summary["rejected_fraction"] = _drop_transients(copy, fs, transient_factor)
    if kept.shape[1] < _MIN_SAMPLES_PER_WEIGHT * len(channels) ** 2:
        return finish(signal, f"only {kept.shape[1]} usable samples for ICA.")

    n_components = _numerical_rank(kept)
    summary["n_components"] = n_components
    # MNE works in volts. The ICA itself ignores scale, so without known units
    # the data is brought to a typical EEG size.
    scale = unit_in_volts if unit_in_volts is not None else 1e-5 / kept.std()
    info = mne.create_info(channels, fs, "eeg")
    raw = mne.io.RawArray(kept * scale, info, verbose=False)
    raw.set_montage(montage, verbose=False)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        fit_params={"extended": True} if method == "infomax" else None,
        random_state=random_state,
        max_iter="auto",
    )
    with warnings.catch_warnings():
        # Filtering happened outside MNE, so its info does not record the high-pass.
        warnings.filterwarnings("ignore", message=".*high-pass filtered.*")
        ica.fit(raw, verbose=False)

    context = {"regions": regions, "blink_channels": blink_channels, "fs": fs, "reref": reref}
    components = sorted({int(component) for component in select(ica, raw, context, summary)})
    if any(not 0 <= component < n_components for component in components):
        raise ValueError(f"component_selector returned {components}; "
                         f"valid indices stop at {n_components - 1}.")
    summary["components"] = components
    if not components:
        summary["status"] = no_match_status
        return finish(signal, no_match_message)

    ica.exclude = components
    cleaned_copy = ica.apply(raw.copy(), verbose=False).get_data() / scale
    summary["power_change"] = _power_change(
        _band_powers(kept, fs), _band_powers(cleaned_copy, fs), front, back,
    )
    if not _passes_power_check(summary["power_change"], max_power_change, limit_back_falls):
        summary["status"] = "failed_power_check"
        return finish(signal, f"removing components {components} failed the power check.")

    # Center each segment so channel offsets do not leak through the spatial filter.
    offsets = np.zeros_like(signal)
    for positions in segments:
        offsets[positions] = signal[positions].mean(axis=0)
    centered = mne.io.RawArray((signal - offsets).T * scale, info, verbose=False)
    cleaned = ica.apply(centered, verbose=False).get_data().T / scale + offsets
    summary["status"] = "cleaned"
    return finish(cleaned)
