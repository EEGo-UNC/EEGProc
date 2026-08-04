"""Leave-One-Subject-Out (LOSO) cross-validation for MTLFuseNet.

For each of the DREAMER subjects: train a fresh model on every *other* subject's
trials, then evaluate on the held-out subject. Each held-out trial's window-level
softmax predictions are averaged into a single trial-level prediction, so metrics
are reported per trial (18 trials / subject), matching the reference results file.

Reads the cached per-trial samples produced by ``mtl_preprocess.py`` and writes
``experiment_outputs/dreamer_{task}_results.json`` in the same schema as the
existing reference results.
"""

import glob
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

from eegproc.deep_learning.supervised.mtlfusenet.mtl_model import MTLFuseNet

AUTOTUNE = tf.data.AUTOTUNE
SIGNATURE = (
    (
        tf.TensorSpec(shape=(9, 9, 128), dtype=tf.float32),
        tf.TensorSpec(shape=(3, 14), dtype=tf.float32),
        tf.TensorSpec(shape=(3, 14, 14), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(), dtype=tf.int32),
)


# --------------------------------------------------------------------- data load
def load_trials(processed_dir):
    """Load every cached trial into memory, grouped by subject id.

    Loading once up front avoids re-reading pickles across the 23 folds x epochs.
    Total footprint is ~4.5 GB of float32 grid windows for the full dataset; on a
    low-RAM machine, adapt the generator below to load per-trial from disk instead.
    """
    paths = sorted(glob.glob(os.path.join(processed_dir, "subj*_trial*.pkl")))
    by_subject = {}
    for p in paths:
        with open(p, "rb") as f:
            s = pickle.load(f)
        by_subject.setdefault(s["subject_id"], []).append(s)
    return by_subject


def _window_generator(trials, task):
    """Yield ((X_ST, DE, adj), label) for every window across ``trials``."""
    def gen():
        for tr in trials:
            label = np.int32(tr[task])
            adj = tr["adj"]
            X, DE = tr["X_ST"], tr["DE"]
            for i in range(tr["num_win"]):
                yield (X[i], DE[i], adj), label
    return gen


def make_dataset(trials, task, batch_size, shuffle=True, shuffle_buffer=4096):
    ds = tf.data.Dataset.from_generator(_window_generator(trials, task),
                                        output_signature=SIGNATURE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


# ------------------------------------------------------------------- evaluation
def predict_trials(model, trials, task):
    """Aggregate each trial's window predictions -> one label per trial."""
    y_true, y_pred = [], []
    for tr in trials:
        n = tr["num_win"]
        adj = np.broadcast_to(tr["adj"], (n, 3, 14, 14)).astype(np.float32)
        out = model((tr["X_ST"], tr["DE"], adj), training=False)
        probs = out["y_pred"].numpy().mean(axis=0)   # mean softmax over windows
        y_pred.append(int(np.argmax(probs)))
        y_true.append(int(tr[task]))
    return np.array(y_true), np.array(y_pred)


def _metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ------------------------------------------------------------------------- LOSO
def run_loso(processed_dir="processed_trials", task="valence", out_dir="experiment_outputs",
             epochs=30, batch_size=64, lr=1e-4, subjects=None, verbose=True):
    by_subject = load_trials(processed_dir)
    all_subjects = sorted(by_subject.keys())
    if subjects is not None:
        all_subjects = [s for s in all_subjects if s in set(subjects)]

    fold_metrics, user_metrics, prediction_log = [], [], []
    subject_id_mapping = {}

    for fold, held_out in enumerate(all_subjects):
        train_trials = [tr for s in all_subjects if s != held_out for tr in by_subject[s]]
        test_trials = by_subject[held_out]
        subject_id_mapping[str(fold)] = int(held_out)

        tf.keras.backend.clear_session()
        model = MTLFuseNet()
        model.compile(optimizer=tf.keras.optimizers.Adam(lr))

        train_ds = make_dataset(train_trials, task, batch_size, shuffle=True)
        model.fit(train_ds, epochs=epochs, verbose=2 if verbose else 0)

        # test-set per-window loss, then trial-level classification metrics
        test_ds = make_dataset(test_trials, task, batch_size, shuffle=False)
        eval_res = model.evaluate(test_ds, verbose=0, return_dict=True)
        y_true, y_pred = predict_trials(model, test_trials, task)
        m = _metrics(y_true, y_pred)

        row = {"fold": fold + 1, "n_samples": len(test_trials),
               "loss": float(eval_res["loss"]), **m}
        fold_metrics.append(row)
        user_metrics.append({"fold": fold + 1, "subject_id": int(held_out),
                             "n_samples": len(test_trials), **m})
        prediction_log.append({"fold": fold + 1, "subject_id": int(held_out),
                               "y_true": y_true.tolist(), "y_pred": y_pred.tolist()})
        if verbose:
            print(f"[fold {fold + 1}/{len(all_subjects)}] subject {held_out}: "
                  f"acc={m['accuracy']:.3f} f1={m['f1']:.3f} loss={row['loss']:.3f}")

    keys = ["loss", "accuracy", "f1", "precision", "recall"]
    mean_scores = {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys}
    std_scores = {k: float(np.std([f[k] for f in fold_metrics])) for k in keys}

    results = {
        f"dreamer_{task}": {
            "fold_metrics": fold_metrics,
            "user_metrics": user_metrics,
            "prediction_log": prediction_log,
            "mean_scores": mean_scores,
            "std_scores": std_scores,
            "subject_id_mapping": subject_id_mapping,
            "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "task": task},
        }
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dreamer_{task}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nLOSO done ({task}). mean acc={mean_scores['accuracy']:.3f} "
              f"+/- {std_scores['accuracy']:.3f} -> {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LOSO cross-validation for MTLFuseNet")
    ap.add_argument("--processed", default="processed_trials")
    ap.add_argument("--task", default="valence", choices=["valence", "arousal"])
    ap.add_argument("--out", default="experiment_outputs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    args = ap.parse_args()
    run_loso(args.processed, task=args.task, out_dir=args.out, epochs=args.epochs,
             batch_size=args.batch_size, lr=args.lr, subjects=args.subjects)
