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


# ---------------------------------------------------------------------
# Smoke-test config
# ---------------------------------------------------------------------

SEED = 7
FS = 128

CSV_PATH = "datasets/dreamer_joined.csv"
RESULTS_DIR = Path("smoke_test_outputs")

# Canonical names used inside this script.
# Your CSV may use aliases such as subject_id/trial_id; these are renamed below.
SUBJECT_COL = "patient_index"
SESSION_COL = "video_index"
LABEL_COL = "valence"

SUBJECT_COL_ALIASES = (
    "patient_index",
    "subject_id",
    "subject",
    "participant_id",
    "participant",
    "user_id",
    "user",
)

SESSION_COL_ALIASES = (
    "video_index",
    "trial_id",
    "trial_index",
    "trial",
    "video_id",
    "video",
    "session_id",
    "session",
    "stimulus_id",
    "stimulus",
)

MAX_SUBJECTS = 4
MAX_SESSIONS_PER_SUBJECT = 2

PSD_WINDOW_SEC = 4.0
PSD_OVERLAP = 0.5

EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

CV_METRICS = ["accuracy", "f1", "precision", "recall"]
SELECTION_METRIC = "f1"
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

# This script assumes cross_val.py has been updated to accept these args.
# If it has not, this smoke test fails early with a clear message.
REQUIRED_NESTED_CV_ARGS = {
    "metrics",
    "selection_metric",
    "maximize_metric",
    "log_predictions",
    "log_variational_intervals",
    "n_uncertainty_samples",
    "ci_level",
}


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    subject_id_mapping: dict[int, Any]
    feature_cols: list[str]
    label_threshold: float


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Make the smoke test more reproducible."""
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


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """
    Return the first matching column from candidates.

    Matching is exact first, then case-insensitive. The returned value is the
    actual column name present in df, not necessarily the candidate spelling.
    """
    columns = list(df.columns)

    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_to_original = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        match = lower_to_original.get(candidate.lower())
        if match is not None:
            return match

    return None


def standardize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename subject/session ID columns to the canonical names used internally.

    Examples:
        subject_id -> patient_index
        trial_id   -> video_index

    These columns are metadata. They are used to group EEG samples before PSD;
    they are not passed into psd_bandpowers as signal channels.
    """
    df = df.copy()

    subject_source = find_column(df, SUBJECT_COL_ALIASES)
    session_source = find_column(df, SESSION_COL_ALIASES)

    if subject_source is None:
        raise ValueError(
            "Could not find a subject ID column. "
            f"Tried aliases: {list(SUBJECT_COL_ALIASES)}. "
            f"Available columns: {list(df.columns)}"
        )

    if session_source is None:
        raise ValueError(
            "Could not find a session/video/trial ID column. "
            f"Tried aliases: {list(SESSION_COL_ALIASES)}. "
            f"Available columns: {list(df.columns)}"
        )

    rename_map = {}
    if subject_source != SUBJECT_COL:
        rename_map[subject_source] = SUBJECT_COL
    if session_source != SESSION_COL:
        rename_map[session_source] = SESSION_COL

    if rename_map:
        print("Renaming ID columns:")
        pprint(rename_map, width=120, sort_dicts=False)
        df = df.rename(columns=rename_map)

    return df


def validate_required_columns(df: pd.DataFrame, label_col: str) -> None:
    """Validate that the standardized dataframe has IDs, label, and EEG channels."""
    required_cols = {SUBJECT_COL, SESSION_COL, label_col, *EEG_CHANNELS}
    missing_cols = sorted(required_cols - set(df.columns))

    if missing_cols:
        raise ValueError(
            "Missing required columns after standardizing ID columns: "
            f"{missing_cols}. Available columns: {list(df.columns)}"
        )


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
            keep_sessions = (
                np.sort(subject_df[SESSION_COL].unique())[:max_sessions_per_subject]
            )
            keep_parts.append(subject_df[subject_df[SESSION_COL].isin(keep_sessions)])

        raw_df = pd.concat(keep_parts, axis=0, ignore_index=True)

    return raw_df


def assert_nested_lnso_cv_supports_required_args() -> None:
    """
    Fail early if cross_val.py has not been updated for metric/prediction logging.
    """
    signature = inspect.signature(nested_lnso_cv)
    params = signature.parameters
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in params.values()
    )

    if accepts_kwargs:
        return

    missing = sorted(REQUIRED_NESTED_CV_ARGS - set(params))
    if missing:
        raise TypeError(
            "This smoke test expects eegproc.deep_learning.cross_val.nested_lnso_cv "
            "to support the new metric/prediction logging API. Missing args: "
            f"{missing}. Update cross_val.py before running this smoke test."
        )


def make_json_safe(value: Any) -> Any:
    """Convert numpy/tensorflow-heavy result objects into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tf.Tensor):
        return value.numpy().tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Last-resort fallback for objects like History or custom classes.
    return str(value)


# ----------------------------------------j-----------------------------
# Dataset construction
# ---------------------------------------------------------------------

def make_psd_lstm_dataset(
    csv_path: str,
    label_col: str = LABEL_COL,
    max_subjects: int | None = MAX_SUBJECTS,
    max_sessions_per_subject: int | None = MAX_SESSIONS_PER_SUBJECT,
) -> DatasetBundle:
    """
    Load DREAMER-style EEG rows, compute PSD features by subject/session,
    and pack each session as one LSTM sequence.

    Accepted input ID column names:
        subject: patient_index, subject_id, subject, participant_id, ...
        session: video_index, trial_id, trial, session_id, video_id, ...

    Output shapes:
        X           : (n_sessions, timesteps, n_features)
        y           : (n_sessions,)
        subject_ids : (n_sessions,), integer-coded subject IDs for CV splitting
    """
    raw_df = pd.read_csv(csv_path)
    raw_df = standardize_id_columns(raw_df)
    validate_required_columns(raw_df, label_col=label_col)

    raw_df = limit_subjects_and_sessions(
        raw_df=raw_df,
        max_subjects=max_subjects,
        max_sessions_per_subject=max_sessions_per_subject,
    )

    if raw_df.empty:
        raise ValueError(
            "No rows left after applying MAX_SUBJECTS and "
            "MAX_SESSIONS_PER_SUBJECT. Increase those limits."
        )

    session_labels = (
        raw_df[[SUBJECT_COL, SESSION_COL, label_col]]
        .groupby([SUBJECT_COL, SESSION_COL], sort=False)
        .first()
        .reset_index()
    )

    threshold = float(session_labels[label_col].median())
    session_labels["target"] = (
        session_labels[label_col] >= threshold
    ).astype("int64")

    # Only EEG channels go into the filter/PSD. IDs stay outside as metadata.
    eeg_raw = raw_df[EEG_CHANNELS]

    clean = bandpass_filter(
        eeg_raw,
        fs=FS,
        bands=FREQUENCY_BANDS,
        low=0.5,
        high=45.0,
        notch_hz=60,
    )

    # Add IDs back only for grouping windows by subject/session.
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

    if psd_df.empty:
        raise ValueError(
            "PSD dataframe is empty. Your sessions may be shorter than "
            "PSD_WINDOW_SEC * FS. Try a smaller PSD_WINDOW_SEC."
        )

    psd_df = psd_df.merge(
        session_labels[[SUBJECT_COL, SESSION_COL, "target"]],
        on=[SUBJECT_COL, SESSION_COL],
        how="left",
    )

    if psd_df["target"].isna().any():
        raise ValueError(
            "Some PSD rows did not receive labels after merging. Check that "
            f"{SUBJECT_COL} and {SESSION_COL} are stable across preprocessing."
        )

    feature_cols = [
        col for col in psd_df.columns
        if col not in {SUBJECT_COL, SESSION_COL, "target"}
    ]

    groups = list(psd_df.groupby([SUBJECT_COL, SESSION_COL], sort=False))
    if not groups:
        raise ValueError("No subject/session groups found after PSD extraction.")

    max_timesteps = max(len(group) for _, group in groups)
    n_features = len(feature_cols)

    X = np.zeros((len(groups), max_timesteps, n_features), dtype=np.float32)
    y = np.zeros((len(groups),), dtype=np.int64)

    original_subject_ids = []

    for i, ((subject_id, _session_id), group) in enumerate(groups):
        seq = group[feature_cols].to_numpy(dtype=np.float32)

        # PSD powers are nonnegative; log compression keeps values smaller.
        seq = np.clip(seq, 0.0, np.inf)
        seq = np.log1p(seq)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        X[i, : len(seq), :] = seq
        y[i] = int(group["target"].iloc[0])
        original_subject_ids.append(subject_id)

    subject_codes, subject_uniques = pd.factorize(original_subject_ids, sort=True)
    subject_ids = subject_codes.astype(np.int64)
    subject_id_mapping = {
        int(code): original_id
        for code, original_id in enumerate(subject_uniques.tolist())
    }

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"Only one class found: {classes}. Increase MAX_SUBJECTS, "
            "increase MAX_SESSIONS_PER_SUBJECT, or change LABEL_COL."
        )

    return DatasetBundle(
        X=X,
        y=y,
        subject_ids=subject_ids,
        subject_id_mapping=subject_id_mapping,
        feature_cols=feature_cols,
        label_threshold=threshold,
    )


# ---------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------

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
        json.dump(make_json_safe(results), f, indent=2)

    return output_path


# ---------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------

def smoke_test_hyperparameter_grid() -> dict:
    """
    Tiny grid intended only to verify that nested CV plumbing works.

    Keep this deliberately small. This is not meant to find a good model.
    """
    return {
        # Model args.
        "lstm_units": [4],
        "n_lstm_layers": [1],
        "dropout": [0.0],
        "learning_rate": [1e-2, 1e-3],

        # Variational-loss args. These are harmless for softmax only if the
        # model builder/loss resolver accepts them.
        "alpha": [1.0],
        "beta": [0.0],
        "gamma": [1e-4],
        "lambda_": [0.0],

        # Fit args. cross_val.py should split these from model-builder args.
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

    log_variational_intervals = loss_name == "variational"

    results = nested_lnso_cv(
        model_builder_function=model_builder,
        feature_array=dataset.X,
        label_array=dataset.y,
        subject_id_array=dataset.subject_ids,
        n_outer_subjects_to_leave_out=N_OUTER_SUBJECTS_TO_LEAVE_OUT,
        n_inner_subjects_to_leave_out=N_INNER_SUBJECTS_TO_LEAVE_OUT,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        hyperparameters=smoke_test_hyperparameter_grid(),
        metrics=CV_METRICS,
        selection_metric=SELECTION_METRIC,
        maximize_metric=MAXIMIZE_SELECTION_METRIC,
        log_predictions=True,
        log_variational_intervals=log_variational_intervals,
        n_uncertainty_samples=N_UNCERTAINTY_SAMPLES,
        ci_level=CI_LEVEL,
        verbose=0,
    )

    print_cv_result_summary(name=name, results=results)

    if SAVE_RESULTS_JSON:
        output_path = save_results_json(name=name, results=results)
        print(f"\nSaved JSON results: {output_path}")

    return results


def main() -> None:
    configure_printing()
    set_seed(SEED)
    assert_nested_lnso_cv_supports_required_args()

    dataset = make_psd_lstm_dataset(
        csv_path=CSV_PATH,
        label_col=LABEL_COL,
        max_subjects=MAX_SUBJECTS,
        max_sessions_per_subject=MAX_SESSIONS_PER_SUBJECT,
    )

    print_dataset_summary(dataset)

    all_results = {}

    all_results["standard_lstm_softmax"] = run_nested_lnso_smoke_test(
        name="standard_lstm_softmax",
        loss_name="softmax_crossentropy",
        dataset=dataset,
    )

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
