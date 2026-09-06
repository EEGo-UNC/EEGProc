"""Screen trials for counterfactual suitability before spending runs on them.

A counterfactual only explains a decision the model actually made. On a trial
where the model sits at ~50% confidence it has no real opinion, so "flipping"
it is meaningless and no target probability above ~0.55 is reachable. Observed
directly: subject 0 trial 0 starts at target_p=0.4956 and plateaus at 0.5412
after 200 steps, never reaching a 0.8 threshold.

This script runs a forward pass only -- no optimization -- and reports, per
trial: the true valence rating, the model's predicted class, and its
confidence. Use it to pick confident trials with a balance of high and low
valence before running the experiment matrix.

Usage:
    PYTHONPATH=src python -m eegproc.model_explainability.screen_trials \\
        --model-root runs/full/.../loso_zero_shot_models \\
        --raw-eeg-npy datasets/dreamer_eeg.npy \\
        --raw-labels-npy datasets/dreamer_labels.npy \\
        --subjects 0 1 2
"""

import argparse
import csv
import importlib
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_parser():
    parser = argparse.ArgumentParser(
        description="Report per-trial model confidence to guide trial selection."
    )
    parser.add_argument("--model-root", type=Path, required=True,
                        help="Directory holding loso_fold_XXXX_target_Y_zero_shot.keras files.")
    parser.add_argument(
        "--model-module",
        default="eegproc.deep_learning.joint_architectures.SICModelv11.sic_model",
    )
    parser.add_argument("--raw-eeg-npy", type=Path, required=True)
    parser.add_argument("--raw-labels-npy", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--dataset", default="dreamer")
    parser.add_argument("--label-dimension", default="valence")
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--window-overlap", type=float, default=0.0)
    parser.add_argument("--window-normalization", default="global_rms")
    parser.add_argument("--median-label", type=float, default=3.0)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser


def load_trials(args):
    """Group windows into trials the same way the counterfactual runner does."""
    data = importlib.import_module(
        "eegproc.deep_learning.joint_architectures.joint_models_data"
    )
    features, labels, subjects = data.build_dataset(
        dataset=args.dataset,
        eeg_path=args.raw_eeg_npy,
        labels_path=args.raw_labels_npy,
        label_dimension=args.label_dimension,
        window_size_sec=args.window_sec,
        fs=args.fs,
        overlap=args.window_overlap,
        median_label=args.median_label,
        zscore=False,
    )[:3]
    features = np.asarray(features, dtype=np.float32)

    raw = np.load(args.raw_eeg_npy, mmap_mode="r")
    n_subjects, n_trials, _, n_samples = raw.shape
    window = int(round(args.window_sec * args.fs))
    hop = max(1, int(round(window * (1.0 - args.window_overlap))))
    n_windows = 1 + (n_samples - window) // hop

    if args.window_normalization == "global_rms":
        rms = np.sqrt(np.mean(np.square(features, dtype=np.float64),
                              axis=(1, 2), keepdims=True))
        features = (features.astype(np.float64)
                    / np.maximum(rms, 1e-6)).astype(np.float32)
    elif args.window_normalization != "none":
        raise ValueError(f"Unsupported normalization: {args.window_normalization}")

    grouped = features.reshape(n_subjects * n_trials, n_windows,
                               features.shape[1], features.shape[2])
    classes = np.asarray(labels).reshape(n_subjects * n_trials, n_windows)[:, 0]
    ratings = np.load(args.raw_labels_npy, allow_pickle=False)
    column = 0 if args.label_dimension == "valence" else 1
    return grouped, classes, ratings[:, :, column], n_trials


def main(argv=None):
    args = build_parser().parse_args(argv)
    importlib.import_module(args.model_module)
    grouped, classes, ratings, n_trials = load_trials(args)

    rows = []
    for subject in args.subjects:
        # loso_fold_NNNN_target_M holds out subject M, with NNNN = M + 1.
        name = f"loso_fold_{subject + 1:04d}_target_{subject}_zero_shot.keras"
        path = args.model_root / name
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint for subject {subject}: {path}")
        print(f"\n=== subject {subject}  ({name}) ===", flush=True)
        model = tf.keras.models.load_model(path, compile=False, safe_mode=True)

        print(f"{'trial':>5} {'valence':>8} {'class':>6} {'pred':>5} "
              f"{'conf':>7} {'usable':>7}")
        for trial in range(n_trials):
            index = subject * n_trials + trial
            x = tf.constant(grouped[index: index + 1])
            probabilities = tf.nn.softmax(model(x, training=False), axis=-1).numpy()[0]
            predicted = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted])
            rating = float(ratings[subject, trial])
            true_class = int(classes[index])
            # Near-chance trials cannot support a meaningful counterfactual.
            usable = "yes" if confidence >= 0.70 else ("weak" if confidence >= 0.60 else "no")
            print(f"{trial:5d} {rating:8.1f} {true_class:6d} {predicted:5d} "
                  f"{confidence:7.4f} {usable:>7}")
            rows.append({
                "subject": subject, "trial": trial, "valence_rating": rating,
                "true_class": true_class, "predicted_class": predicted,
                "confidence": confidence, "correct": int(predicted == true_class),
                "usable": usable,
            })

    print("\n=== recommended (confident, both classes represented) ===")
    for subject in args.subjects:
        subject_rows = [r for r in rows if r["subject"] == subject and r["usable"] != "no"]
        for target in (0, 1):
            pool = sorted((r for r in subject_rows if r["true_class"] == target),
                          key=lambda r: -r["confidence"])[:2]
            label = "low" if target == 0 else "high"
            ids = ", ".join(f"trial {r['trial']} (conf {r['confidence']:.3f})" for r in pool)
            print(f"  subject {subject} {label:>4} valence: {ids or 'none confident enough'}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
