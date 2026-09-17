"""Post-hoc spectral descriptors for saved EEG counterfactual artifacts.

The analysis compares each decoded counterfactual with the decoded original
reconstruction, so decoder reconstruction error is not mistaken for an
intervention effect. Periodograms are evaluated within the model's original
windows; windows are never concatenated across decoder seams.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ...featurization import shannons_entropy
from ..counterfactuals.plotting import (
    DEFAULT_BAND_NAMES,
    load_counterfactual_trial,
    resolve_channel_names,
)
from .artifacts import write_csv
from .physiology import DEFAULT_BANDS, signal_diagnostics


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_result(path: Path) -> dict:
    result_path = path.with_name("result.json")
    if not result_path.is_file():
        return {}
    payload = json.loads(result_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{result_path} must contain a JSON object")
    return payload


def _study_metadata(path: Path) -> dict:
    for parent in path.parents:
        candidate = parent / "study.json"
        if candidate.is_file():
            payload = json.loads(candidate.read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"{candidate} must contain a JSON object")
            return payload
    return {}


def _artifact_metadata(path: Path, n_features: int) -> tuple[list[str], list[str], str]:
    with np.load(path, allow_pickle=False) as data:
        band_names = (
            [_text(value).lower() for value in np.asarray(data["band_names"]).reshape(-1)]
            if "band_names" in data.files
            else [name.lower() for name in DEFAULT_BAND_NAMES]
        )
        feature_order = (
            _text(np.asarray(data["feature_order"]).reshape(-1)[0])
            if "feature_order" in data.files
            else "channel-major"
        )
        saved_channels = (
            [_text(value) for value in np.asarray(data["channel_names"]).reshape(-1)]
            if "channel_names" in data.files
            else None
        )
    if feature_order not in {"channel-major", "band-major"}:
        raise ValueError(f"Unsupported feature order {feature_order!r} in {path}")
    if not band_names or n_features % len(band_names):
        raise ValueError(
            f"{path} has {n_features} features, incompatible with bands {band_names}"
        )
    n_channels = n_features // len(band_names)
    channels = resolve_channel_names(
        n_channels,
        saved=saved_channels if saved_channels and len(saved_channels) == n_channels else None,
    )
    return channels, band_names, feature_order


def _column_names(channels: list[str], bands: list[str], feature_order: str) -> list[str]:
    if feature_order == "channel-major":
        return [f"{channel}_{band}" for channel in channels for band in bands]
    return [f"{channel}_{band}" for band in bands for channel in channels]


def _entropy_by_window(
    trial: np.ndarray,
    *,
    fs: float,
    channels: list[str],
    bands: list[str],
    band_edges: tuple[tuple[float, float], ...],
    feature_order: str,
) -> np.ndarray:
    windows, samples, n_features = trial.shape
    columns = _column_names(channels, bands, feature_order)
    if len(columns) != n_features:
        raise ValueError("Channel/band metadata does not match the saved feature axis")
    band_mapping = dict(zip(bands, band_edges))
    table = shannons_entropy(
        pd.DataFrame(trial.reshape(windows * samples, n_features), columns=columns),
        fs,
        bands=band_mapping,
        window_sec=samples / fs,
        overlap=0.0,
        detrend="constant",
    )
    if len(table) != windows:
        raise ValueError(
            f"Expected one entropy estimate per decoder window; got {len(table)} for {windows}"
        )
    values = np.empty((windows, len(channels), len(bands)), dtype=float)
    for channel_index, channel in enumerate(channels):
        for band_index, band in enumerate(bands):
            values[:, channel_index, band_index] = table[
                f"{channel}_{band}_entropy"
            ].to_numpy()
    return values


def _load_or_compute_psd(
    artifact: Path,
    label: str,
    trial: np.ndarray,
    *,
    fs: float,
    n_channels: int,
    band_edges: tuple[tuple[float, float], ...],
    feature_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    saved = artifact.with_name(f"physiology_{label}.npz")
    if saved.is_file():
        with np.load(saved, allow_pickle=False) as data:
            frequencies = np.asarray(data["frequencies_hz"], dtype=float)
            psd = np.asarray(data["psd"], dtype=float)
    else:
        diagnostics = signal_diagnostics(
            trial,
            fs=fs,
            n_channels=n_channels,
            band_edges=band_edges,
            feature_order=feature_order,
        )
        frequencies, psd = diagnostics["frequencies_hz"], diagnostics["psd"]
    expected = (n_channels, len(band_edges), len(frequencies))
    if psd.shape != expected or not np.isfinite(frequencies).all() or not np.isfinite(psd).all():
        raise ValueError(f"Invalid {label} PSD in {artifact}: expected {expected}, got {psd.shape}")
    return frequencies, psd


def _peak_and_centroid(
    frequencies: np.ndarray,
    psd: np.ndarray,
    band_edges: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_channels, n_bands, _ = psd.shape
    peaks = np.full((n_channels, n_bands), np.nan)
    centroids = np.full_like(peaks, np.nan)
    edge_peaks = np.zeros((n_channels, n_bands), dtype=bool)
    for band_index, (low, high) in enumerate(band_edges):
        mask = (frequencies >= low) & (frequencies <= high)
        selected_frequencies = frequencies[mask]
        if len(selected_frequencies) < 2:
            raise ValueError(f"Band {low}-{high} Hz has fewer than two spectral bins")
        for channel_index in range(n_channels):
            power = np.clip(psd[channel_index, band_index, mask], 0.0, np.inf)
            total = float(power.sum())
            if not np.isfinite(total) or total <= 0:
                continue
            peak_index = int(np.argmax(power))
            peaks[channel_index, band_index] = selected_frequencies[peak_index]
            centroids[channel_index, band_index] = float(
                np.sum(selected_frequencies * power) / total
            )
            edge_peaks[channel_index, band_index] = peak_index in {0, len(power) - 1}
    return peaks, centroids, edge_peaks


def _quantiles(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None, None, None
    q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
    return float(median), float(q25), float(q75)


def analyze_counterfactual_artifact(
    artifact: Path,
    *,
    fs: float | None = None,
    branch: str | None = None,
    band_edges: Iterable[Iterable[float]] | None = None,
) -> list[dict]:
    """Return one paired trial-level row per channel and band."""
    artifact = Path(artifact)
    if artifact.name != "counterfactual.npz" or not artifact.is_file():
        raise ValueError(f"Expected an existing counterfactual.npz file: {artifact}")
    result = _read_result(artifact)
    study = _study_metadata(artifact)
    selected_branch = branch or result.get("report_output")
    reference, counterfactual, selected_branch, _ = load_counterfactual_trial(
        artifact, branch=selected_branch, reference="reconstruction"
    )
    channels, bands, feature_order = _artifact_metadata(artifact, reference.shape[-1])

    study_arguments = study.get("arguments", {}) if isinstance(study.get("arguments", {}), dict) else {}
    fs = float(fs if fs is not None else study_arguments.get("fs", 128.0))
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if band_edges is None:
        band_edges = study.get("band_edges_hz", DEFAULT_BANDS)
    band_edges = tuple(tuple(float(value) for value in edge) for edge in band_edges)
    if len(band_edges) != len(bands) or any(len(edge) != 2 for edge in band_edges):
        raise ValueError("Provide exactly one (low, high) edge pair per saved EEG band")

    reference_f, reference_psd = _load_or_compute_psd(
        artifact,
        "reconstruction",
        reference,
        fs=fs,
        n_channels=len(channels),
        band_edges=band_edges,
        feature_order=feature_order,
    )
    counterfactual_f, counterfactual_psd = _load_or_compute_psd(
        artifact,
        "counterfactual",
        counterfactual,
        fs=fs,
        n_channels=len(channels),
        band_edges=band_edges,
        feature_order=feature_order,
    )
    if not np.array_equal(reference_f, counterfactual_f):
        raise ValueError("Reference and counterfactual PSD frequency grids differ")
    reference_peak, reference_centroid, reference_edge = _peak_and_centroid(
        reference_f, reference_psd, band_edges
    )
    cf_peak, cf_centroid, cf_edge = _peak_and_centroid(
        counterfactual_f, counterfactual_psd, band_edges
    )
    reference_entropy = _entropy_by_window(
        reference,
        fs=fs,
        channels=channels,
        bands=bands,
        band_edges=band_edges,
        feature_order=feature_order,
    )
    cf_entropy = _entropy_by_window(
        counterfactual,
        fs=fs,
        channels=channels,
        bands=bands,
        band_edges=band_edges,
        feature_order=feature_order,
    )

    task = result.get("task", study.get("task"))
    objective = result.get("objective")
    if objective is None:
        objective = next((parent.name for parent in artifact.parents if parent.name in {"base", "typicality"}), None)
    rows = []
    for channel_index, channel in enumerate(channels):
        for band_index, band in enumerate(bands):
            ref_entropy_stats = _quantiles(reference_entropy[:, channel_index, band_index])
            cf_entropy_stats = _quantiles(cf_entropy[:, channel_index, band_index])
            delta_entropy_stats = _quantiles(
                cf_entropy[:, channel_index, band_index]
                - reference_entropy[:, channel_index, band_index]
            )
            ref_peak = reference_peak[channel_index, band_index]
            after_peak = cf_peak[channel_index, band_index]
            ref_centroid = reference_centroid[channel_index, band_index]
            after_centroid = cf_centroid[channel_index, band_index]
            rows.append({
                "schema_version": 1,
                "source_artifact": str(artifact.resolve()),
                "task": task,
                "subject_id": result.get("subject_id"),
                "trial_id": result.get("trial_id"),
                "objective": objective,
                "decoder_branch": selected_branch,
                "reference": "decoded_original_reconstruction",
                "fs_hz": fs,
                "n_windows": int(reference.shape[0]),
                "window_samples": int(reference.shape[1]),
                "frequency_resolution_hz": float(fs / reference.shape[1]),
                "feature_order": feature_order,
                "channel_index": channel_index,
                "channel": channel,
                "band_index": band_index,
                "band": band,
                "band_low_hz": band_edges[band_index][0],
                "band_high_hz": band_edges[band_index][1],
                "reference_peak_frequency_hz": float(ref_peak) if np.isfinite(ref_peak) else None,
                "counterfactual_peak_frequency_hz": float(after_peak) if np.isfinite(after_peak) else None,
                "delta_peak_frequency_hz": float(after_peak - ref_peak) if np.isfinite(ref_peak + after_peak) else None,
                "reference_peak_at_band_edge": bool(reference_edge[channel_index, band_index]),
                "counterfactual_peak_at_band_edge": bool(cf_edge[channel_index, band_index]),
                "reference_spectral_centroid_hz": float(ref_centroid) if np.isfinite(ref_centroid) else None,
                "counterfactual_spectral_centroid_hz": float(after_centroid) if np.isfinite(after_centroid) else None,
                "delta_spectral_centroid_hz": float(after_centroid - ref_centroid) if np.isfinite(ref_centroid + after_centroid) else None,
                "reference_spectral_entropy_median": ref_entropy_stats[0],
                "reference_spectral_entropy_q25": ref_entropy_stats[1],
                "reference_spectral_entropy_q75": ref_entropy_stats[2],
                "counterfactual_spectral_entropy_median": cf_entropy_stats[0],
                "counterfactual_spectral_entropy_q25": cf_entropy_stats[1],
                "counterfactual_spectral_entropy_q75": cf_entropy_stats[2],
                "delta_spectral_entropy_median": delta_entropy_stats[0],
                "delta_spectral_entropy_q25": delta_entropy_stats[1],
                "delta_spectral_entropy_q75": delta_entropy_stats[2],
            })
    return rows


def discover_counterfactual_artifacts(inputs: Iterable[Path]) -> list[Path]:
    artifacts = set()
    for supplied in inputs:
        path = Path(supplied)
        if path.is_file():
            if path.name != "counterfactual.npz":
                raise ValueError(f"Input file must be named counterfactual.npz: {path}")
            artifacts.add(path.resolve())
        elif path.is_dir():
            artifacts.update(candidate.resolve() for candidate in path.rglob("counterfactual.npz"))
        else:
            raise ValueError(f"Input does not exist: {path}")
    if not artifacts:
        raise ValueError("No counterfactual.npz artifacts found")
    return sorted(artifacts)


def write_spectral_features(
    inputs: Iterable[Path],
    output: Path,
    *,
    fs: float | None = None,
    branch: str | None = None,
) -> list[dict]:
    rows = [
        row
        for artifact in discover_counterfactual_artifacts(inputs)
        for row in analyze_counterfactual_artifact(artifact, fs=fs, branch=branch)
    ]
    write_csv(Path(output), rows)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Study directories or counterfactual.npz files")
    parser.add_argument("--output", required=True, type=Path, help="Long-form CSV output")
    parser.add_argument("--fs", type=float, help="Override saved sampling rate (default: saved value or 128 Hz)")
    parser.add_argument("--branch", help="Decoder branch; inferred when only one is present")
    args = parser.parse_args(argv)
    rows = write_spectral_features(args.inputs, args.output, fs=args.fs, branch=args.branch)
    print(f"Saved {len(rows)} channel-band rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
