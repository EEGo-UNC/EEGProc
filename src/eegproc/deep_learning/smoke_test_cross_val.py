from __future__ import annotations

import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from ..preprocessing import bandpass_filter, FREQUENCY_BANDS
from ..featurization import feature_grouped_by_metadata, psd_bandpowers
from .supervised.rnn_architectures import LSTMClassifier
from .cross_val import nested_lnso_cv

SEED = 7
FS = 128

CSV_PATH = "datasets/dreamer_joined.csv"
RESULTS_DIR = Path("smoke_test_outputs")


SUBJECT_COL = "subject_id"
SESSION_COL = "trial_id"
LABEL_COL = "valence"

MAX_SUBJECTS = 5
MAX_SESSIONS_PER_SUBJECT = 2

PSD_WINDOW_SEC = 2.0
PSD_OVERLAP = 0.5

EEG_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

CV_METRICS = ["accuracy", "f1", "precision", "recall"]
SELECTION_METRIC = "accuracy"
MAXIMIZE_SELECTION_METRIC = True

N_OUTER_SUBJECTS_TO_LEAVE_OUT = 1
N_INNER_SUBJECTS_TO_LEAVE_OUT = 1
N_EPOCHS = 1
BATCH_SIZE = 2

N_UNCERTAINTY_SAMPLES = 30
CI_LEVEL = 0.95

PRINT_PREDICTION_LOG = True
PRINT_VARIATIONAL_INTERVAL_LOG = True
SAVE_RESULTS_JSON = True

REQUIRED_NESTED_CV_ARGS = {
    "metrics",
    "selection_metric",
    "maximize_metric",
    "log_predictions",
    "log_variational_intervals",
    "n_uncertainty_samples",
    "ci_level",
}


@dataclass(frozen=True)
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    task_ids: np.ndarray
    subject_id_mapping: dict[int, Any]
    feature_cols: list[str]
    label_threshold: float

def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def configure_printing() -> None:
    """Avoid truncated terminal output for numpy/pandas objects."""
    np.set_printoptions(
        threshold=np.inf,
        linewidth=220,
        suppress=True,
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", None)

def limit_subjects_and_sessions(
    raw_df: pd.DataFrame,
    max_subjects: int | None,
    max_sessions_per_subject: int | None,
) -> pd.DataFrame:
    """Keep the smoke test small by limiting subjects and sessions."""
    raw_df = raw_df.copy()

    if max_subjects is not None:
        keep_subjects = np.sort(raw_df[SUBJECT_COL].unique())[:max_subjects]
        raw_df = raw_df[raw_df[SUBJECT_COL].isin(keep_subjects)].copy()

    if max_sessions_per_subject is not None:
        keep_parts = []

        for _subject_id, subject_df in raw_df.groupby(SUBJECT_COL, sort=False):
            keep_sessions = np.sort(subject_df[SESSION_COL].unique())[
                :max_sessions_per_subject
            ]
            keep_parts.append(subject_df[subject_df[SESSION_COL].isin(keep_sessions)])

        raw_df = pd.concat(keep_parts, axis=0, ignore_index=True)

    return raw_df



def prep_dataset(
    csv_path: str,
    label_col: str = LABEL_COL,
    max_subjects: int | None = MAX_SUBJECTS,
    max_sessions_per_subject: int | None = MAX_SESSIONS_PER_SUBJECT,
) -> DatasetBundle:
    raw_df = pd.read_csv(csv_path)

    raw_df = limit_subjects_and_sessions(
        raw_df=raw_df,
        max_subjects=max_subjects,
        max_sessions_per_subject=max_sessions_per_subject,
    )

    session_labels = (
        raw_df[[SUBJECT_COL, SESSION_COL, label_col]]
        .groupby([SUBJECT_COL, SESSION_COL], sort=False)
        .first()
        .reset_index()
    )

    threshold = float(session_labels[label_col].median())
    session_labels["target"] = (session_labels[label_col] >= threshold).astype("int64")

    eeg_raw = raw_df[EEG_CHANNELS]

    clean = bandpass_filter(
        eeg_raw,
        fs=FS,
        bands=FREQUENCY_BANDS,
        low=1.0,
        high=45.0,
        notch_hz=50,
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
        window_sec=PSD_WINDOW_SEC,
        overlap=PSD_OVERLAP,
    )

    psd_df = psd_df.merge(
        session_labels[[SUBJECT_COL, SESSION_COL, "target"]],
        on=[SUBJECT_COL, SESSION_COL],
        how="left",
    )

    feature_cols = [
        col for col in psd_df.columns if col not in {SUBJECT_COL, SESSION_COL, "target"}
    ]

    groups = list(psd_df.groupby([SUBJECT_COL, SESSION_COL], sort=False))
    if not groups:
        raise ValueError("No subject/session groups found after PSD extraction.")

    max_timesteps = max(len(group) for _, group in groups)
    n_features = len(feature_cols)

    X = np.zeros((len(groups), max_timesteps, n_features), dtype=np.float32)
    y = np.zeros((len(groups),), dtype=np.int64)

    original_subject_ids = []
    original_task_ids = []

    for i, ((subject_id, session_id), group) in enumerate(groups):
        seq = group[feature_cols].to_numpy(dtype=np.float32)
        seq = np.clip(seq, 0.0, np.inf)
        seq = np.log1p(seq)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        X[i, : len(seq), :] = seq
        y[i] = int(group["target"].iloc[0])
        original_subject_ids.append(subject_id)
        original_task_ids.append(session_id)

    subject_codes, subject_uniques = pd.factorize(original_subject_ids, sort=True)
    subject_ids = subject_codes.astype(np.int64)
    subject_id_mapping = {
        int(code): original_id
        for code, original_id in enumerate(subject_uniques.tolist())
    }

    # Tasks are the original session/trial ids (shared across subjects), so the
    # raw value is the most meaningful label for per-task reporting.
    task_ids = np.asarray(original_task_ids)

    return DatasetBundle(
        X=X,
        y=y,
        subject_ids=subject_ids,
        task_ids=task_ids,
        subject_id_mapping=subject_id_mapping,
        feature_cols=feature_cols,
        label_threshold=threshold,
    )


def make_lstm_builder(
    timesteps: int,
    n_features: int,
    n_classes: int,
    loss_name: str,
):
    """
    Return a model-builder function compatible with nested_lnso_cv.

    cross_val.py should pass all model hyperparameters except epochs/batch_size
    into this builder, so every grid key below must be accepted as an argument.
    """

    def build_lstm_model(
        lstm_units: int = 4,
        n_lstm_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-2,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 1e-4,
        lambda_: float = 0.0,
    ):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        return LSTMClassifier(
            timesteps=timesteps,
            n_features=n_features,
            n_classes=n_classes,
            lstm_units=lstm_units,
            n_lstm_layers=n_lstm_layers,
            dropout=dropout,
            loss=loss_name,
            optimizer=optimizer,
            metrics=["accuracy"],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_=lambda_,
        ).build()

    return build_lstm_model


def print_dataset_summary(dataset: DatasetBundle) -> None:
    print("\nDataset ready")
    print("=" * 80)
    print(f"X shape:               {dataset.X.shape}")
    print(f"y shape:               {dataset.y.shape}")
    print(f"subject_ids shape:     {dataset.subject_ids.shape}")
    print(f"integer subject IDs:   {np.unique(dataset.subject_ids)}")
    print(f"subject ID mapping:    {dataset.subject_id_mapping}")
    print(f"classes:               {np.unique(dataset.y)}")
    print(f"label column:          {LABEL_COL}")
    print(f"label threshold:       {dataset.label_threshold:.6f}")
    print(f"n feature columns:     {len(dataset.feature_cols)}")


def print_rows(title: str, rows: list[dict], max_rows: int | None = None) -> None:
    print("\n" + title)
    print("=" * 80)

    if not rows:
        print("<empty>")
        return

    rows_to_print = rows if max_rows is None else rows[:max_rows]

    for row_idx, row in enumerate(rows_to_print, start=1):
        print(f"\n[{row_idx}]")
        pprint(row, width=160, sort_dicts=False)

    if max_rows is not None and len(rows) > max_rows:
        print(f"\n... omitted {len(rows) - max_rows} rows")


def print_cv_result_summary(name: str, results: dict) -> None:
    print("\n" + "#" * 80)
    print(f"RESULT SUMMARY: {name}")
    print("#" * 80)

    print("\nMean scores")
    print("=" * 80)
    pprint(results.get("mean_scores", {}), width=160, sort_dicts=False)

    print_rows(
        title="Best configs by outer fold",
        rows=results.get("best_configs", []),
        max_rows=None,
    )
    print_rows(
        title="Fold metrics",
        rows=results.get("fold_metrics", []),
        max_rows=None,
    )
    print_rows(
        title="Per-user metrics",
        rows=results.get("user_metrics", []),
        max_rows=None,
    )
    if PRINT_PREDICTION_LOG:
        print_rows(
            title="Per-classification prediction log",
            rows=results.get("prediction_log", []),
            max_rows=None,
        )
    if PRINT_VARIATIONAL_INTERVAL_LOG:
        print_rows(
            title="Per-classification variational interval log",
            rows=results.get("variational_interval_log", []),
            max_rows=None,
        )


def save_results_json(name: str, results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{name}_results.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return output_path


def smoke_test_hyperparameter_grid() -> dict:
    return {
        "lstm_units": [4],
        "n_lstm_layers": [1],
        "dropout": [0.0],
        "learning_rate": [1e-2, 1e-3],
        "alpha": [1.0],
        "beta": [0.0],
        "gamma": [1e-4],
        "lambda_": [0.0],
        "epochs": [N_EPOCHS],
        "batch_size": [BATCH_SIZE],
    }


def run_nested_lnso_smoke_test(
    name: str,
    loss_name: str,
    dataset: DatasetBundle,
) -> dict:
    """Run one tiny nested LNSO smoke test and print all returned CV logs."""
    print("\n" + "=" * 80)
    print(f"Running smoke test: {name}")
    print("=" * 80)
    print(f"loss_name: {loss_name}")
    print(f"metrics:   {CV_METRICS}")
    print(f"selection: {SELECTION_METRIC}, maximize={MAXIMIZE_SELECTION_METRIC}")

    model_builder = make_lstm_builder(
        timesteps=dataset.X.shape[1],
        n_features=dataset.X.shape[2],
        n_classes=len(np.unique(dataset.y)),
        loss_name=loss_name,
    )

    results = nested_lnso_cv(
        model_builder_function=model_builder,
        feature_array=dataset.X,
        label_array=dataset.y,
        subject_id_array=dataset.subject_ids,
        task_id_array=dataset.task_ids,
        n_outer_subjects_to_leave_out=N_OUTER_SUBJECTS_TO_LEAVE_OUT,
        n_inner_subjects_to_leave_out=N_INNER_SUBJECTS_TO_LEAVE_OUT,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        hyperparameters=smoke_test_hyperparameter_grid(),
        metrics=CV_METRICS,
        selection_metric=SELECTION_METRIC,
        maximize_metric=MAXIMIZE_SELECTION_METRIC,
        log_predictions=True,
        log_variational_intervals=True,
        n_uncertainty_samples=N_UNCERTAINTY_SAMPLES,
        ci_level=CI_LEVEL,
        verbose=0,
    )

    # Persist the integer-code -> original-id mapping so reporting can label
    # axes with the real DREAMER subject ids instead of 0..n-1 codes.
    results["subject_id_mapping"] = dataset.subject_id_mapping

    print_cv_result_summary(name=name, results=results)

    if SAVE_RESULTS_JSON:
        output_path = save_results_json(name=name, results=results)
        print(f"\nSaved JSON results: {output_path}")

    return results


def main() -> None:
    configure_printing()
    set_seed(SEED)

    dataset = prep_dataset(
        csv_path=CSV_PATH,
        label_col=LABEL_COL,
        max_subjects=MAX_SUBJECTS,
        max_sessions_per_subject=MAX_SESSIONS_PER_SUBJECT,
    )

    print_dataset_summary(dataset)

    all_results = {}

    all_results["variational_lstm"] = run_nested_lnso_smoke_test(
        name="variational_lstm",
        loss_name="variational",
        dataset=dataset,
    )

    print("\n" + "#" * 80)
    print("SMOKE TESTS COMPLETE")
    print("#" * 80)

    print("\nFinal mean score comparison")
    print("=" * 80)
    for run_name, run_results in all_results.items():
        print(f"\n{run_name}")
        pprint(run_results.get("mean_scores", {}), width=160, sort_dicts=False)

    if SAVE_RESULTS_JSON:
        output_path = save_results_json(name="all_smoke_tests", results=all_results)
        print(f"\nSaved combined JSON results: {output_path}")


if __name__ == "__main__":
    main()
