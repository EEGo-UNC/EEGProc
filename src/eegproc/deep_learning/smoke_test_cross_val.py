# smoke_test_lstm_nested_lnso.py

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from eegproc.preprocessing import bandpass_filter, FREQUENCY_BANDS
from eegproc.featurization import feature_grouped_by_metadata, psd_bandpowers
from eegproc.deep_learning.supervised.rnn_architectures import LSTMClassifier
from eegproc.deep_learning.cross_val import nested_lnso_cv


FS = 128

CSV_PATH = "datasets/dreamer_joined.csv"

SUBJECT_COL = "patient_index"
SESSION_COL = "video_index"
LABEL_COL = "valence"

EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


def make_psd_lstm_dataset(
    csv_path: str,
    label_col: str = LABEL_COL,
    max_subjects: int | None = 4,
):
    raw_df = pd.read_csv(csv_path)

    session_labels = (
        raw_df[[SUBJECT_COL, SESSION_COL, label_col]]
        .groupby([SUBJECT_COL, SESSION_COL], sort=False)
        .first()
        .reset_index()
    )

    # Binary labels for smoke test.
    # For DREAMER-style ratings, this turns the target into low/high.
    threshold = session_labels[label_col].median()
    session_labels["target"] = (
        session_labels[label_col] >= threshold
    ).astype("int64")

    eeg_raw = raw_df[EEG_CHANNELS]

    clean = bandpass_filter(
        eeg_raw,
        fs=FS,
        bands=FREQUENCY_BANDS,
        low=0.5,
        high=45.0,
        notch_hz=60,
    )

    clean = pd.concat(
        [
            raw_df[[SUBJECT_COL, SESSION_COL]].reset_index(drop=True),
            clean.reset_index(drop=True),
        ],
        axis=1,
    )

    psd_df = feature_grouped_by_metadata(
        eeg_df=clean,
        target_function=psd_bandpowers,
        fs=FS,
        bands=FREQUENCY_BANDS,
        channels=EEG_CHANNELS,
        group_by_metadata_columns=[SUBJECT_COL, SESSION_COL],
        drop_metadata_for_fn=True,
        window_sec=4.0,
        overlap=0.5,
    )

    psd_df = psd_df.merge(
        session_labels[[SUBJECT_COL, SESSION_COL, "target"]],
        on=[SUBJECT_COL, SESSION_COL],
        how="left",
    )

    feature_cols = [
        c for c in psd_df.columns
        if c not in {SUBJECT_COL, SESSION_COL, "target"}
    ]

    groups = list(psd_df.groupby([SUBJECT_COL, SESSION_COL], sort=False))

    max_timesteps = max(len(g) for _, g in groups)
    n_features = len(feature_cols)

    X = np.zeros((len(groups), max_timesteps, n_features), dtype=np.float32)
    y = np.zeros((len(groups),), dtype=np.int64)
    subject_ids = np.zeros((len(groups),), dtype=np.int64)

    for i, ((subject_id, session_id), group) in enumerate(groups):
        seq = group[feature_cols].to_numpy(dtype=np.float32)

        # PSD powers are nonnegative; log transform keeps values smaller.
        seq = np.clip(seq, 0.0, np.inf)
        seq = np.log1p(seq)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        X[i, : len(seq), :] = seq
        y[i] = int(group["target"].iloc[0])
        subject_ids[i] = int(subject_id)

    if max_subjects is not None:
        keep_subjects = np.sort(np.unique(subject_ids))[:max_subjects]
        keep_mask = np.isin(subject_ids, keep_subjects)

        X = X[keep_mask]
        y = y[keep_mask]
        subject_ids = subject_ids[keep_mask]

    return X, y, subject_ids


def make_lstm_builder(
    timesteps: int,
    n_features: int,
    n_classes: int,
):
    def build_lstm_model(
        lstm_units: int = 4,
        n_lstm_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
    ):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        return LSTMClassifier(
            timesteps=timesteps,
            n_features=n_features,
            n_classes=n_classes,
            lstm_units=lstm_units,
            n_lstm_layers=n_lstm_layers,
            dropout=dropout,
            loss="softmax_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        ).build()

    return build_lstm_model


if __name__ == "__main__":
    X, y, subject_ids = make_psd_lstm_dataset(
        csv_path=CSV_PATH,
        label_col=LABEL_COL,
        max_subjects=4,
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("subject_ids shape:", subject_ids.shape)
    print("subjects:", np.unique(subject_ids))
    print("classes:", np.unique(y))

    model_builder = make_lstm_builder(
        timesteps=X.shape[1],
        n_features=X.shape[2],
        n_classes=len(np.unique(y)),
    )

    results = nested_lnso_cv(
        model_builder_function=model_builder,
        feature_array=X,
        label_array=y,
        subject_id_array=subject_ids,
        n_outer_subjects_to_leave_out=1,
        n_inner_subjects_to_leave_out=1,
        n_epochs=1,
        batch_size=2,
        hyperparameters={
            "lstm_units": [4],
            "n_lstm_layers": [1],
            "dropout": [0.0],
            "learning_rate": [1e-3],
            "epochs": [1],
            "batch_size": [2],
        },
        selection_metric="loss",
        maximize_metric=False,
        verbose=0,
    )

    print(results["mean_scores"])