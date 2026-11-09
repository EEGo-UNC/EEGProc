#!/usr/bin/env python3
"""Compare decoder reconstructions R(Z) across SIC training runs.

For each run (a ``loso_zero_shot_models.json`` manifest) and each held-out
subject, every held-out trial is encoded and decoded with that fold's
checkpoint, with no counterfactual optimization. Reported per run, next to
the real EEG of the same trials:

  balanced accuracy   zero-shot, pooled over the run's held-out trials
  MSE                 mean (x - r)^2 in normalized units
  derivative          mean ((r[t+1]-r[t]) - (x[t+1]-x[t]))^2 within windows
  seam                mean ((r[w+1,0]-r[w,T-1]) - (x[w+1,0]-x[w,T-1]))^2
  amp-jump, phase     seam_diagnostics on physical units (per trial: mean
                      over bands of the per-band median; then median over
                      trials); a continuous recording gives ~1 and ~0
  graded physiology   worst-family exceedance against the source-subject
                      real-EEG reference of that fold, median [Q1, Q3]
  all checks pass     share of trials passing every available binary check

The derivative and seam columns are the two training penalties of
``temporal_reconstruction_losses``, evaluated on held-out trials.

Usage
-----
    python -m eegproc.model_explainability.typicality.compare_reconstructions \\
        --run base=<.../loso_zero_shot_models.json> \\
        --run seam=<.../loso_zero_shot_models.json> \\
        --raw-eeg-npy datasets/dreamer_eeg.npy \\
        --raw-labels-npy datasets/dreamer_labels.npy \\
        --label-dimension arousal --out compare_reconstructions
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np

from .physiology import DEFAULT_BANDS, PhysiologicalReference, seam_diagnostics, signal_diagnostics
from .results import recognition_metrics

N_CHANNELS = 14
COLUMNS = (
    ("mse", "MSE"), ("derivative", "derivative"), ("seam", "seam"),
    ("amplitude_jump", "amp-jump"), ("phase_residual", "phase (rad)"),
)


def training_penalties(x, r):
    """NumPy mirror of ``temporal_reconstruction_losses`` for one trial (W,T,F)."""
    x, r = np.asarray(x, np.float64), np.asarray(r, np.float64)
    derivative = float(np.mean((np.diff(r, axis=1) - np.diff(x, axis=1)) ** 2))
    if x.shape[0] < 2:
        return 0.0, derivative
    seam = float(np.mean(((r[1:, 0] - r[:-1, -1]) - (x[1:, 0] - x[:-1, -1])) ** 2))
    return seam, derivative


def _seam_summary(signal, feature_order):
    seams = seam_diagnostics(signal, n_channels=N_CHANNELS, band_edges=DEFAULT_BANDS,
                             feature_order=feature_order)
    return (float(np.nanmean(seams["amplitude_jump_ratio_median"])),
            float(np.mean(seams["phase_residual_median_rad"])))


def _physical(dataset, index, values):
    if dataset.normalization_scale is None:
        return np.asarray(values)
    return np.asarray(values) * dataset.normalization_scale[index] + dataset.normalization_offset[index]


def _signal_record(dataset, index, values, reference, fs):
    """Seam and physiology measures of one trial-shaped signal."""
    order = dataset.feature_order or "channel-major"
    physical = _physical(dataset, index, values)
    jump, phase = _seam_summary(physical, order)
    assessed = reference.assess(signal_diagnostics(
        physical, fs=fs, n_channels=N_CHANNELS, band_edges=DEFAULT_BANDS, feature_order=order))
    return {"amplitude_jump": jump, "phase_residual": phase,
            "worst_family_exceedance": assessed["worst_family_exceedance"],
            "all_pass": bool(assessed["available_count"] and
                             assessed["passed_count"] == assessed["available_count"])}


def load_manifest(path, model_dir=None):
    """Return {held-out subject: checkpoint path} for one training run."""
    path = Path(path)
    payload = json.loads(path.read_text())
    entries = payload["models"] if isinstance(payload, dict) else payload
    folds = {}
    for entry in entries:
        if entry.get("stage") != "zero_shot_source_model":
            raise ValueError(f"{path}: only zero_shot_source_model checkpoints are accepted")
        subject = int(entry["target_subject"])
        if subject in folds:
            raise ValueError(f"{path}: duplicate checkpoint for subject {subject}")
        name = Path(entry.get("filename", entry["path"])).name
        # Manifests record the absolute path of the machine that trained them;
        # prefer the checkpoint folder that sits next to this manifest.
        candidates = [Path(model_dir) / name] if model_dir else [
            path.parent / "loso_zero_shot_models" / name, Path(entry["path"])]
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"{path}: checkpoint {name} not found in {[str(c) for c in candidates]}")
        folds[subject] = found
    if not folds:
        raise ValueError(f"{path}: no checkpoints listed")
    return folds


@contextlib.contextmanager
def _without_keras_object_sharing():
    """Load SIC checkpoints without Keras' shared-object cache.

    ``keras.saving.load_model`` keys a cache of serialized objects on
    ``id(obj)``. Every ``Bidirectional`` built while the model is
    reconstructed serializes a temporary GRU; once Python reuses a freed
    GRU's id, the cache hands back an unrelated GRU and Keras raises
    "Forward layer and backward layer should have different `go_backwards`
    value". Which load fails depends on memory reuse, so loading several
    checkpoints in one process fails at random. SIC checkpoints are
    subclassed models rebuilt from hyperparameters and share no objects in
    their saved config, so turning the cache off changes nothing else.
    """
    from keras.src.saving import serialization_lib

    original = serialization_lib.record_object_after_serialization

    def record_without_sharing(obj, config):
        if config["module"] == "__main__":
            config["module"] = None

    serialization_lib.record_object_after_serialization = record_without_sharing
    try:
        yield
    finally:
        serialization_lib.record_object_after_serialization = original


def load_adapter(path, model_module, sample_input):
    from ..model_agnostic.sic_adapter import create_sic_adapter

    with _without_keras_object_sharing():
        return create_sic_adapter(
            model_path=path, config={"model_module": model_module, "decoder_mode": "joint"},
            sample_input=sample_input)


def evaluate_fold(adapter, dataset, held, reference, fs):
    """Encode and decode each held-out trial; return per-trial records."""
    records = []
    for index in held:
        x = dataset.features[index:index + 1]
        features = adapter.model.get_encoder_features(x)
        outputs = adapter.reconstruct(features["window_features"], x)
        if "joint" not in outputs:
            raise ValueError("compare_reconstructions needs the joint decoder output")
        r = np.asarray(outputs["joint"])[0]
        if r.shape != x[0].shape or not np.isfinite(r).all():
            raise ValueError("Decoder output must be finite and match the trial shape")
        seam, derivative = training_penalties(x[0], r)
        records.append({
            "subject": int(dataset.subject_ids[index]), "trial": int(dataset.trial_ids[index]),
            "label": int(dataset.labels[index]),
            "probabilities": np.asarray(features["probabilities"])[0].astype(float).tolist(),
            "mse": float(np.mean((np.asarray(x[0], np.float64) - r) ** 2)),
            "derivative": derivative, "seam": seam,
            **_signal_record(dataset, index, r, reference, fs),
        })
    return records


def _quartiles(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return None
    return [float(np.median(values)), float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))]


def summarize(records):
    summary = {"n_trials": len(records),
               "n_subjects": len({r["subject"] for r in records}),
               "all_pass_percent": 100.0 * float(np.mean([r["all_pass"] for r in records])),
               "worst_family_exceedance": _quartiles([r["worst_family_exceedance"] for r in records])}
    for key, _ in COLUMNS:
        if key in records[0]:
            summary[key] = _quartiles([r[key] for r in records])
    if "probabilities" in records[0]:
        labels = np.asarray([r["label"] for r in records])
        probabilities = np.asarray([r["probabilities"] for r in records])
        metrics = recognition_metrics(labels, probabilities)
        summary.update(balanced_accuracy=metrics["balanced_accuracy"],
                       recall_0=metrics["recall_0"], recall_1=metrics["recall_1"])
    return summary


def _cell(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return f"{value[0]:.{digits}f} [{value[1]:.{digits}f}, {value[2]:.{digits}f}]"
    return f"{value:.{digits}f}"


def markdown_table(summaries, subjects):
    lines = [
        f"Held-out subjects: {', '.join(map(str, subjects))}. Median [Q1, Q3] over held-out trials.",
        "",
        "| run | trials | bal. acc. | MSE | derivative | seam | amp-jump | phase (rad) | graded e_max | all checks pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        accuracy = s.get("balanced_accuracy")
        lines.append("| " + " | ".join([
            name, str(s["n_trials"]),
            "n/a" if accuracy is None else f"{100 * accuracy:.1f}%",
            *(_cell(s[key][0]) if s.get(key) else "n/a" for key in ("mse", "derivative", "seam")),
            *(_cell(s[key][0], 2) if s.get(key) else "n/a" for key in ("amplitude_jump", "phase_residual")),
            _cell(s["worst_family_exceedance"]), f"{s['all_pass_percent']:.1f}%",
        ]) + " |")
    return "\n".join(lines) + "\n"


def run(args):
    import tensorflow as tf

    from ..model_agnostic.sic_adapter import load_sic_raw_trials

    runs = {}
    for item in args.run:
        name, _, manifest = item.partition("=")
        if not name or not manifest or name in runs or name == "real EEG":
            raise ValueError(f"--run expects unique NAME=MANIFEST entries; got {item!r}")
        runs[name] = load_manifest(manifest, args.model_dir.get(name))
    common = set.intersection(*(set(folds) for folds in runs.values()))
    subjects = sorted(common if args.subjects is None else set(args.subjects))
    missing = {name: sorted(set(subjects) - set(folds)) for name, folds in runs.items()
               if set(subjects) - set(folds)}
    if not subjects or missing:
        raise ValueError(f"Runs do not share the requested held-out subjects; missing: {missing}")

    dataset = load_sic_raw_trials(dict(
        raw_eeg_npy=str(args.raw_eeg_npy), raw_labels_npy=str(args.raw_labels_npy),
        dataset="dreamer", label_dimension=args.label_dimension, fs=args.fs,
        window_sec=args.window_sec, window_overlap=0.0,
        window_normalization=args.window_normalization,
        label_threshold_mode="global", median_label=3.0, model_module=args.model_module))
    order = dataset.feature_order or "channel-major"

    records = {"real EEG": [], **{name: [] for name in runs}}
    for subject in subjects:
        held = np.flatnonzero(dataset.subject_ids == subject)
        source = np.flatnonzero(dataset.subject_ids != subject)
        if not len(held) or not len(source):
            raise ValueError(f"Subject {subject} has no held-out or no source trials")
        print(f"Subject {subject}: fitting the physiology reference on {len(source)} source trials", flush=True)
        reference = PhysiologicalReference.fit(
            [signal_diagnostics(_physical(dataset, i, dataset.features[i]), fs=args.fs,
                                n_channels=N_CHANNELS, band_edges=DEFAULT_BANDS, feature_order=order)
             for i in source],
            quantile=args.physiology_quantile, required_fraction=args.physiology_required_fraction)
        for index in held:
            records["real EEG"].append({
                "subject": int(subject), "trial": int(dataset.trial_ids[index]),
                "label": int(dataset.labels[index]),
                **_signal_record(dataset, index, dataset.features[index], reference, args.fs)})
        for name, folds in runs.items():
            print(f"Subject {subject}: decoding {len(held)} trials with run {name}", flush=True)
            adapter = load_adapter(folds[subject], args.model_module, dataset.features[held[0]])
            records[name].extend(evaluate_fold(adapter, dataset, held, reference, args.fs))
            del adapter
            tf.keras.backend.clear_session()

    summaries = {name: summarize(rows) for name, rows in records.items()}
    table = markdown_table(summaries, subjects)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(table)
    out.with_suffix(".json").write_text(json.dumps({
        "subjects": subjects, "label_dimension": args.label_dimension,
        "runs": {name: {str(k): str(v) for k, v in folds.items()} for name, folds in runs.items()},
        "summaries": summaries, "trials": records}, indent=1))
    print("\n" + table)
    print(f"Wrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return summaries


class _KeyValue(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        name, _, value = values.partition("=")
        if not name or not value:
            parser.error(f"{option_string} expects NAME=PATH")
        getattr(namespace, self.dest)[name] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=MANIFEST",
                        help="A training run's loso_zero_shot_models.json; repeat per run.")
    parser.add_argument("--model-dir", action=_KeyValue, default={}, metavar="NAME=DIR",
                        help="Checkpoint folder for a run, if not next to its manifest.")
    parser.add_argument("--raw-eeg-npy", required=True, type=Path)
    parser.add_argument("--raw-labels-npy", required=True, type=Path)
    parser.add_argument("--label-dimension", required=True, choices=("arousal", "valence"))
    parser.add_argument("--subjects", type=int, nargs="+",
                        help="Held-out subjects; default is every subject all runs share.")
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--window-normalization", default="global_rms")
    parser.add_argument("--physiology-quantile", type=float, default=0.95)
    parser.add_argument("--physiology-required-fraction", type=float, default=0.95)
    parser.add_argument("--model-module",
                        default="eegproc.deep_learning.joint_architectures.SICModelv15.sic_model")
    parser.add_argument("--out", default="compare_reconstructions",
                        help="Output path stem; writes <stem>.md and <stem>.json.")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
