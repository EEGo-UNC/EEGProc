"""Convert Cowen/Keltner mean ratings into the existing 27-emotion CSV mapping."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._csv_common import load_mat

CATEGORY_NAMES = [
    "admiration", "adoration", "aesthetic_appreciation", "amusement", "anger", "anxiety",
    "awe", "awkwardness", "boredom", "calmness", "confusion", "contempt", "craving",
    "disappointment", "disgust", "empathic_pain", "entrancement", "envy", "excitement",
    "fear", "guilt", "horror", "interest", "joy", "nostalgia", "pride", "relief",
    "romance", "sadness", "satisfaction", "sexual_desire", "surprise", "sympathy", "triumph",
]
EXCLUDED = {"contempt", "disappointment", "envy", "guilt", "pride", "sympathy", "triumph"}
# EEGEmotions-27 numbers its emotions 1..27 in this (alphabetical) order. The
# mapping CSV is sorted by quadrant for reading, so consumers must match rows by
# emotion name rather than by position.
EMOTION_ID_ORDER = tuple(name for name in CATEGORY_NAMES if name not in EXCLUDED)


def _vector(path, size):
    candidates = [array.astype(float).ravel() for name, value in load_mat(path).items()
                  if not name.startswith("__")
                  and np.issubdtype((array := np.asarray(value)).dtype, np.number)
                  and array.size == size]
    if len(candidates) != 1:
        raise ValueError(f"{path}: expected exactly one numeric vector of length {size}")
    return candidates[0]


def _weighted_stats(values, weights):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan, np.nan
    mean = np.average(values[valid], weights=weights[valid])
    sd = np.sqrt(np.average((values[valid] - mean) ** 2, weights=weights[valid]))
    return float(mean), float(sd)


def cowen27_frames(root: Path):
    """Yield the legacy mapping, with median-based quadrants (not official labels)."""
    candidates = [root, root / "data/features/amt/mean_score_concat",
                  root / "features/amt/mean_score_concat", root / "amt/mean_score_concat"]
    base = next((p for p in candidates if (p / "category").is_dir() and (p / "dimension").is_dir()), None)
    if base is None:
        raise FileNotFoundError(f"Cannot find mean_score_concat/category and dimension under {root}")
    files = sorted((base / "category").glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No category MAT files in {base}")
    categories = np.vstack([_vector(path, 34) for path in files])
    dimensions = np.vstack([_vector(base / "dimension" / path.name, 14) for path in files])
    if np.any(categories < 0):
        raise ValueError("Category weights must be nonnegative; use unstandardized mean scores")
    valence, arousal = dimensions[:, 13], dimensions[:, 1]
    if not np.isfinite(valence).any() or not np.isfinite(arousal).any():
        raise ValueError("Cowen dimensions have no finite valence/arousal ratings")
    v_cutoff, a_cutoff = np.nanmedian(valence), np.nanmedian(arousal)
    rows = []
    for i, emotion in enumerate(CATEGORY_NAMES):
        if emotion in EXCLUDED:
            continue
        weights = categories[:, i]
        v, v_sd = _weighted_stats(valence, weights)
        a, a_sd = _weighted_stats(arousal, weights)
        quadrant = None
        if np.isfinite(v) and np.isfinite(a):
            quadrant = ("positive" if v >= v_cutoff else "negative") + "_" + (
                "high_arousal" if a >= a_cutoff else "low_arousal")
        rows.append(dict(emotion=emotion, valence=v, arousal=a, valence_sd=v_sd, arousal_sd=a_sd,
                         rating_weight=float(np.nansum(weights)),
                         videos_with_rating=int(np.sum(np.isfinite(weights) & (weights > 0))),
                         quadrant=quadrant))
    yield pd.DataFrame(rows).sort_values(["quadrant", "valence", "arousal"],
                                        ascending=[True, False, False]).reset_index(drop=True)
