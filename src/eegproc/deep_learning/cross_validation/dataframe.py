"""Run cross-validation directly from a tidy DataFrame.

The array-level entry points (:func:`~.loso.loso_cv` and friends) stay the
low-level API for callers who already hold tensors. This is the documented
default: it takes the table featurization produces, derives the subject and trial
arrays from columns, and maps the integer codes in the results back to the
original identifiers so per-subject rows read ``"P07"`` rather than ``0``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Literal

import pandas as pd

from ...data.schema import EEGFrame
from ...data.windowing import WindowedArrays, to_supervised_arrays
from .calibration import subject_calibration_cv
from .loso import fixed_loso_cv, loso_cv
from .nested import nested_lnso_cv

__all__ = ["cross_validate_dataframe"]

_STRATEGIES: dict[str, Callable[..., dict]] = {
    "loso": loso_cv,
    "fixed_loso": fixed_loso_cv,
    "subject_calibration": subject_calibration_cv,
    "nested_lnso": nested_lnso_cv,
}


def _remap(value, lookup: dict[int, tuple]):
    """Turn an integer code back into its original key (unwrapping 1-tuples)."""
    try:
        key = lookup[int(value)]
    except (KeyError, TypeError, ValueError):
        return value
    return key[0] if len(key) == 1 else key


def _remap_results(results: dict, windowed: WindowedArrays) -> dict:
    """Relabel subject codes in the result structure with the original ids."""
    out = dict(results)
    if isinstance(out.get("user_metrics"), list):
        out["user_metrics"] = [
            {**row, "subject_id": _remap(row.get("subject_id"), windowed.subject_lookup)}
            if isinstance(row, dict) else row
            for row in out["user_metrics"]
        ]
    if isinstance(out.get("fold_results"), list):
        folds = []
        for fold in out["fold_results"]:
            if not isinstance(fold, dict):
                folds.append(fold)
                continue
            fold = dict(fold)
            for key in ("left_out_subjects", "validation_subjects", "subject_set_a", "subject_set_b"):
                value = fold.get(key)
                if isinstance(value, (list, tuple)):
                    fold[key] = [_remap(v, windowed.subject_lookup) for v in value]
                elif value is not None:
                    fold[key] = _remap(value, windowed.subject_lookup)
            folds.append(fold)
        out["fold_results"] = folds
    # expose the same unwrapped identifiers used in the remapped rows
    out["subject_lookup"] = {
        code: (key[0] if len(key) == 1 else key)
        for code, key in windowed.subject_lookup.items()
    }
    return out


def cross_validate_dataframe(
    frame: EEGFrame | pd.DataFrame,
    model_builder_function: Callable[..., Any],
    *,
    strategy: Literal["loso", "fixed_loso", "subject_calibration", "nested_lnso"] = "loso",
    fs: float | None = None,
    kind: Literal["signal", "features"] = "features",
    label_column: str = "label",
    subject_columns: tuple[str, ...] = ("subject",),
    trial_columns: tuple[str, ...] = ("trial",),
    time_column: str | None = None,
    feature_columns: tuple[str, ...] | None = None,
    window_sec: float | None = None,
    window_samples: int | None = None,
    window_rows: int = 1,
    overlap: float = 0.0,
    normalize: str | None = None,
    label_transform: Callable | None = None,
    on_short_trial: str = "error",
    return_arrays: bool = False,
    **strategy_kwargs,
) -> dict:
    """Cross-validate a model over a tidy EEG table.

    Parameters
    ----------
    frame : EEGFrame or pandas.DataFrame
        A plain DataFrame is wrapped using the column-role keywords below.
    model_builder_function : callable
        Passed through unchanged; see :func:`~.loso.loso_cv`.
    strategy : {"loso", "fixed_loso", "subject_calibration", "nested_lnso"}
        Which cross-validation scheme to run.
    kind : {"signal", "features"}, default="features"
        ``"features"`` treats each row as one already-computed window, which is
        what :func:`~eegproc.featurization.feature_grouped_by_metadata` emits.
        ``"signal"`` windows raw samples and needs ``time_column`` and a window length.
    **strategy_kwargs
        Forwarded to the chosen entry point, validated against its signature so a
        typo raises instead of being silently ignored.

    Returns
    -------
    dict
        The chosen strategy's results, with subject codes mapped back to the
        original identifiers and a ``subject_lookup`` entry added.

    Examples
    --------
    >>> feats = feature_grouped_by_metadata(          # doctest: +SKIP
    ...     eeg_df, target_function=psd_bandpowers, fs=128,
    ...     group_by_metadata_columns=["subject", "trial"],
    ... )
    >>> feats["label"] = ...                          # doctest: +SKIP
    >>> results = cross_validate_dataframe(           # doctest: +SKIP
    ...     feats, build_model, strategy="loso", fs=128,
    ... )
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose one of {sorted(_STRATEGIES)}."
        )
    runner = _STRATEGIES[strategy]

    if isinstance(frame, EEGFrame):
        eeg_frame = frame
    else:
        if fs is None:
            raise ValueError("fs is required when passing a plain DataFrame.")
        label_columns = tuple(
            c for c in (label_column,) if c in getattr(frame, "columns", ())
        )
        if not label_columns:
            raise ValueError(
                f"label column {label_column!r} is not in the frame; "
                f"available columns: {list(frame.columns)[:12]}..."
            )
        eeg_frame = EEGFrame(
            data=frame,
            fs=fs,
            kind=kind,
            subject_columns=tuple(subject_columns),
            trial_columns=tuple(trial_columns),
            time_column=time_column,
            label_columns=label_columns,
            feature_columns=feature_columns,
        )

    windowed = to_supervised_arrays(
        eeg_frame,
        window_sec=window_sec,
        window_samples=window_samples,
        overlap=overlap,
        label_column=label_column,
        label_transform=label_transform,
        normalize=normalize,
        window_rows=window_rows,
        on_short_trial=on_short_trial,
    )

    accepted = set(inspect.signature(runner).parameters)
    unknown = sorted(set(strategy_kwargs) - accepted)
    if unknown:
        raise TypeError(
            f"{runner.__name__}() got unexpected keyword argument(s) {unknown}. "
            f"Accepted: {sorted(a for a in accepted if not a.startswith('_'))}"
        )

    results = runner(
        model_builder_function,
        windowed.features,
        windowed.labels,
        windowed.subject_ids,
        windowed.trial_ids,
        **strategy_kwargs,
    )
    out = _remap_results(results, windowed)
    if return_arrays:
        out["windowed_arrays"] = windowed
    return out
