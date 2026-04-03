"""
train_eval.py
=============
Leave-One-Subject-Out Cross-Validation (LOSOCV) training and evaluation
for STSNet on the DEAP and DREAMER datasets.

Matches the experimental protocol in Li et al. (2023):
  - Subject-independent LOSOCV strategy
  - Binary classification of arousal / valence (median split)
  - Evaluation: accuracy, precision, recall, F1, ROC-AUC

Usage
-----
    python train_eval.py --dataset deap --dimension valence

The script expects pre-processed EEG data as NumPy arrays saved to disk:
    {dataset}_eeg.npy    — shape (n_subjects, n_trials, n_channels, n_samples)
    {dataset}_labels.npy — shape (n_subjects, n_trials, n_label_dims)
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data_representation import preprocess_dataset
from stsnet import STSNet


# ---------------------------------------------------------------------------
# Dataset configurations  (Table 1 & 3 in the paper)
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "deap": {
        "fs"              : 128,
        "bands"           : ["theta", "alpha", "beta", "gamma"],
        "n_channels"      : 32,
        "n_windows"       : 15,         # nc
        "window_size_sec" : 4.0,        # Tlen (best from ablation, Fig. 5)
        "median_label"    : 5,          # median for binary split
        "bilstm_units"    : 256,
        "bilstm_dropout"  : 0.3,
        "lr"              : 1e-4,
        "weight_decay"    : 5e-4,
        "epochs"          : 50,
        "batch_size"      : 32,
    },
    "dreamer": {
        "fs"              : 128,
        "bands"           : ["theta", "alpha", "beta"],
        "n_channels"      : 14,
        "n_windows"       : 15,
        "window_size_sec" : 4.0,
        "median_label"    : 3,
        "bilstm_units"    : 256,
        "bilstm_dropout"  : 0.3,
        "lr"              : 1e-4,
        "weight_decay"    : 5e-4,
        "epochs"          : 3, # CHANGE BACK TO 50 FOR REAL TRAINING
        "batch_size"      : 32,
    },
}

LABEL_DIMS = {"valence": 0, "arousal": 1}


# ---------------------------------------------------------------------------
# Label processing (Section: Label processing)
# ---------------------------------------------------------------------------

def binarize_labels(raw_labels: np.ndarray, median: int) -> np.ndarray:
    """Convert continuous scores to binary classes via median split.

    Scores >= median → 1 (high), scores < median → 0 (low).
    (The paper maps to +1/-1; we use 0/1 for SparseCategoricalCrossEntropy.)

    Parameters
    ----------
    raw_labels : ndarray, shape (n_trials,)
    median     : int — dataset-specific median threshold

    Returns
    -------
    ndarray of int32, shape (n_trials,)
    """
    return (raw_labels >= median).astype(np.int32)


# ---------------------------------------------------------------------------
# Single LOSOCV fold
# ---------------------------------------------------------------------------

def run_fold(
    subject_idx: int,
    all_eeg: np.ndarray,
    all_labels: np.ndarray,
    cfg: dict,
    use_variable_windows: bool = True,
    gpu_strategy: tf.distribute.Strategy | None = None,
) -> dict:
    """Train and evaluate on one LOSOCV fold.

    Parameters
    ----------
    subject_idx          : int — index of the held-out test subject
    all_eeg              : ndarray (n_subj, n_trials, n_ch, n_samples)
    all_labels           : ndarray (n_subj, n_trials) — already binarised
    cfg                  : dict — dataset configuration
    use_variable_windows : bool — VW vs FW data representation
    gpu_strategy         : optional tf.distribute.Strategy for multi-GPU

    Returns
    -------
    dict with keys: acc, precision, recall, f1, roc_auc
    """
    n_subjects = all_eeg.shape[0]

    # --- Split train / test ---
    test_eeg  = all_eeg[subject_idx]

    train_idx = [i for i in range(n_subjects) if i != subject_idx]
    train_eeg = np.concatenate([all_eeg[i] for i in train_idx], axis=0)

    # Compute median threshold from training labels only, then binarize both sets
    train_raw    = np.concatenate([all_labels[i] for i in train_idx], axis=0)
    test_raw     = all_labels[subject_idx]
    fold_median  = np.median(train_raw)
    train_labels = (train_raw >= fold_median).astype(np.int32)
    test_labels  = (test_raw  >= fold_median).astype(np.int32)

    print(
        f"  Fold {subject_idx+1}/{n_subjects} — "
        f"train={len(train_labels)}, test={len(test_labels)}"
    )

    # --- Preprocess ---
    xd_tr, bi_tr, y_tr = preprocess_dataset(
        train_eeg, train_labels,
        fs=cfg["fs"], bands=cfg["bands"],
        n_windows=cfg["n_windows"],
        window_size_sec=cfg["window_size_sec"],
        use_variable_windows=use_variable_windows,
    )
    xd_te, bi_te, y_te = preprocess_dataset(
        test_eeg, test_labels,
        fs=cfg["fs"], bands=cfg["bands"],
        n_windows=cfg["n_windows"],
        window_size_sec=cfg["window_size_sec"],
        use_variable_windows=use_variable_windows,
    )
    print(np.bincount(y_tr))
    print(np.bincount(y_te))
    # Convert to tensors
    xd_tr = tf.constant(xd_tr); bi_tr = tf.constant(bi_tr); y_tr = tf.constant(y_tr)
    xd_te = tf.constant(xd_te); bi_te = tf.constant(bi_te)

    # --- Build model (inside strategy scope for multi-GPU) ---
    build_fn = lambda: STSNet(
        n_channels     = cfg["n_channels"],
        n_classes      = 2,
        bilstm_units   = cfg["bilstm_units"],
        bilstm_dropout = cfg["bilstm_dropout"],
        n_fm_iters     = 3, # CHANGE BACK TO 10 FOR ACTUAL TRAINING
    )

    if gpu_strategy is not None:
        with gpu_strategy.scope():
            model = build_fn()
    else:
        model = build_fn()

    # --- Train ---
    model.fit_joint(
        xd_tr, bi_tr, y_tr,
        epochs     = cfg["epochs"],
        batch_size = cfg["batch_size"],
        lr         = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )

    # --- Evaluate ---
    logits    = model((xd_te, bi_te), training=False).numpy()
    probs     = tf.nn.softmax(logits).numpy()[:, 1]   # probability of class 1
    preds     = np.argmax(logits, axis=-1)

    return {
        "acc"      : accuracy_score(y_te,  preds),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall"   : recall_score(y_te,   preds, zero_division=0),
        "f1"       : f1_score(y_te,       preds, zero_division=0),
        "roc_auc"  : roc_auc_score(y_te,  probs),
    }


# ---------------------------------------------------------------------------
# Full LOSOCV experiment
# ---------------------------------------------------------------------------

def run_losocv(
    eeg_path: str,
    label_path: str,
    dataset: str,
    dimension: str,
    use_variable_windows: bool = True,
    results_dir: str = "results",
) -> None:
    """Run the full LOSOCV experiment and print / save a summary.

    Parameters
    ----------
    eeg_path   : str — path to (n_subj, n_trials, C, T) .npy file
    label_path : str — path to (n_subj, n_trials, n_dims) .npy file
    dataset    : str — 'deap' or 'dreamer'
    dimension  : str — 'valence' or 'arousal'
    use_variable_windows : bool
    results_dir: str — where to save per-subject result CSVs
    """
    cfg       = DATASET_CONFIGS[dataset]
    dim_idx   = LABEL_DIMS[dimension]
    median    = cfg["median_label"]

    # --- Detect GPUs and build strategy ---
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy on {len(gpus)} GPUs.")
    elif len(gpus) == 1:
        strategy = None
        print("Using single GPU.")
    else:
        strategy = None
        print("No GPU found; using CPU.")

    # --- Load data ---
    all_eeg    = np.load(eeg_path)    # (n_subj, n_trials, C, T)
    all_labels_raw = np.load(label_path)  # (n_subj, n_trials, n_dims)
    # all_labels = np.stack(
    #     [binarize_labels(all_labels_raw[s, :, dim_idx], median)
    #      for s in range(all_eeg.shape[0])], axis=0
    # )  # (n_subj, n_trials)

    n_subjects = all_eeg.shape[0]
    fold_results = []

    for subj in range(n_subjects):
        metrics = run_fold(
            subj, all_eeg, all_labels_raw[:, :, dim_idx], cfg,
            use_variable_windows=use_variable_windows,
            gpu_strategy=strategy,
        )
        fold_results.append(metrics)
        print(
            f"  Subject {subj+1:02d}:  "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    # --- Aggregate ---
    print(f"\n{'='*60}")
    print(f"Dataset={dataset.upper()}  Dimension={dimension}  "
          f"Windows={'VW' if use_variable_windows else 'FW'}")
    print(f"{'='*60}")
    for metric in ["acc", "precision", "recall", "f1", "roc_auc"]:
        vals = [r[metric] for r in fold_results]
        print(f"  {metric:12s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    # --- Save ---
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(
        results_dir,
        f"{dataset}_{dimension}_{'vw' if use_variable_windows else 'fw'}.csv",
    )
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject"] + list(fold_results[0]))
        writer.writeheader()
        for i, r in enumerate(fold_results):
            writer.writerow({"subject": i + 1, **r})
    print(f"\nPer-subject results saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STSNet LOSOCV experiment runner")
    parser.add_argument("--dataset",   choices=["deap", "dreamer"], default="deap")
    parser.add_argument("--dimension", choices=["valence", "arousal"], default="valence")
    parser.add_argument("--eeg_path",  type=str, default=None,
                        help="Path to EEG .npy file. Defaults to {dataset}_eeg.npy")
    parser.add_argument("--label_path", type=str, default=None,
                        help="Path to labels .npy file. Defaults to {dataset}_labels.npy")
    parser.add_argument("--fixed_windows", action="store_true",
                        help="Use fixed-length windows instead of variable-length (VW)")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    eeg_path   = args.eeg_path   or f"{args.dataset}_eeg.npy"
    label_path = args.label_path or f"{args.dataset}_labels.npy"

    run_losocv(
        eeg_path   = eeg_path,
        label_path = label_path,
        dataset    = args.dataset,
        dimension  = args.dimension,
        use_variable_windows = not args.fixed_windows,
        results_dir= args.results_dir,
    )
