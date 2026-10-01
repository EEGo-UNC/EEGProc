"""DREAMER: preprocessing -> band powers -> BiLSTM LOSOCV -> counterfactual.

From the repository root (see examples/README.md for details):
    pip install -e ".[deep-learning]"
    eegproc-to-csv --dataset dreamer --input /path/to/DREAMER.mat \
        --output /path/to/dreamer.csv.gz
    python examples/dreamer_bilstm_counterfactual.py /path/to/dreamer.csv.gz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from eegproc import bandpass_filter, feature_grouped_by_metadata, psd_bandpowers
from eegproc.deep_learning.cross_validation import cross_validate_dataframe
from eegproc.deep_learning.supervised.rnn_architectures import BiLSTMClassifier
from eegproc.model_explainability.model_agnostic import (
    KerasInputAdapter,
    ModelAgnosticCounterfactualOptimizer,
)

FS = 128  # DREAMER EEG sampling rate in Hz.
SEQUENCE_ROWS = 4  # Four non-overlapping 2-second PSD rows = 8 seconds per input.
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def trial_features(trial, fs, bands):
    """Filter each trial independently; only EEG channels become model inputs."""
    # Common-average reference, 50 Hz notch (DREAMER was collected in Europoe), default six bands, and detrending.
    clean = bandpass_filter(trial[CHANNELS], fs, bands=bands, notch_hz=50)
    return psd_bandpowers(clean, fs, bands=bands, window_sec=2, overlap=0)


def main(csv_path, output_dir, epochs=10):
    """Run the example and save metrics, the last fold's model, and explanation."""
    tf.keras.utils.set_random_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the converted CSV; discard baseline and keep samples in time order.
    raw = pd.read_csv(csv_path, usecols=CHANNELS + [
        "subject_id", "trial_id", "segment", "sample_idx", "valence",
    ])
    raw = raw.loc[raw["segment"] == "stimulus"].sort_values(
        ["subject_id", "trial_id", "sample_idx"]
    )
    if not raw["valence"].between(1, 5).all():
        raise ValueError("DREAMER valence must contain finite ratings from 1 to 5.")
    # Example convention: 1-2 = low (0), 3-5 = high (1), including neutral in high.
    raw["label"] = (raw["valence"] >= 3).astype(int)

    # 2. Preprocess and featurize within each subject/trial, preserving labels.
    metadata = ["subject_id", "trial_id", "label"]
    features = feature_grouped_by_metadata(
        raw, target_function=trial_features, fs=FS,
        group_by_metadata_columns=metadata,
    )
    feature_columns = tuple(c for c in features.columns if c not in metadata)
    builder = BiLSTMClassifier(
        timesteps=SEQUENCE_ROWS, n_features=len(feature_columns), n_classes=2,
        lstm_units=16, n_bilstm_layers=1,
    )

    # 3. Fixed settings: every subject is held out once, with no tuning on it.
    # Sequential folds overwrite this checkpoint, leaving the LAST fold's model.
    model_path = output_dir / "last_fold.keras"
    results = cross_validate_dataframe(
        features, builder.build, strategy="fixed_loso", fs=FS,
        subject_columns=("subject_id",), trial_columns=("trial_id",),
        feature_columns=feature_columns, window_rows=SEQUENCE_ROWS,
        # Offline normalization uses each subject's own unlabeled feature rows,
        # including the held-out subject. Windows never cross trial boundaries.
        normalize="subject_zscore", return_arrays=True,
        fixed_config={}, n_epochs=epochs, batch_size=32, n_jobs=1,
        log_predictions=False,
        extra_fit_kwargs={"callbacks": [tf.keras.callbacks.ModelCheckpoint(
            str(model_path), save_best_only=False,
        )]},
    )
    pd.DataFrame(results["user_metrics"]).to_csv(output_dir / "loso_metrics.csv", index=False)

    # 4. Explain the first 8-second input of the last fold's unseen subject.
    arrays = results["windowed_arrays"]  # Reuse exactly the inputs evaluated by CV.
    subject = results["fold_results"][-1]["left_out_subjects"][0]
    code = next(k for k, v in results["subject_lookup"].items() if v == subject)
    index = np.flatnonzero(arrays.subject_ids == code)[0]
    sample = arrays.features[index:index + 1]
    model = tf.keras.models.load_model(model_path, compile=False)
    optimizer = ModelAgnosticCounterfactualOptimizer(
        KerasInputAdapter(model, output_kind="probabilities"),
        target_probability=0.8, max_steps=200,
    )
    # For two classes, the default target is the opposite predicted class.
    counterfactual = optimizer.optimize(sample)

    # These changes are in standardized PSD feature space, not reconstructed EEG.
    # Success is not guaranteed: inspect counterfactual.success in the JSON.
    summary = dict(counterfactual["summary"], subject_id=int(subject),
                   trial_id=int(arrays.trial_lookup[arrays.trial_ids[index]][-1]),
                   label=int(arrays.labels[index]))
    (output_dir / "counterfactual.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(counterfactual["history"]).to_csv(output_dir / "history.csv", index=False)
    np.savez_compressed(output_dir / "counterfactual.npz",
                        **counterfactual["arrays"], feature_names=feature_columns)
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output_dir.resolve()}")
    return results, counterfactual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="DREAMER CSV or CSV.gz from eegproc-to-csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/dreamer_example"),
                        help="Output directory; existing example outputs are replaced")
    parser.add_argument("--epochs", type=int, default=10, help="Fixed training epochs per fold")
    args = parser.parse_args()
    main(args.csv, args.output, args.epochs)
