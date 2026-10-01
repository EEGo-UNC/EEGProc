"""End-to-end: a tidy feature table goes straight into cross-validation.

A user goes from raw EEG with subject and trial columns to LOSO results without
building a single array by hand, and the results are reported against the
user's own subject identifiers.
"""

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow", reason="requires eegproc[deep-learning]")

from eegproc import FREQUENCY_BANDS, bandpass_filter, feature_grouped_by_metadata, psd_bandpowers  # noqa: E402
from eegproc.deep_learning.cross_validation import cross_validate_dataframe  # noqa: E402

FS = 128
SUBJECTS = ["P01", "P07", "P12"]        # deliberately not 0, 1, 2
TRIALS = ["t-a", "t-b"]
DURATION_SEC = 12


def _raw_table() -> pd.DataFrame:
    """Raw EEG with subject/trial metadata, the shape a researcher actually has."""
    rng = np.random.RandomState(0)
    t = np.arange(FS * DURATION_SEC) / FS
    blocks = []
    for si, subject in enumerate(SUBJECTS):
        for ti, trial in enumerate(TRIALS):
            label = (si + ti) % 2
            # class 1 carries extra alpha power, so the task is learnable
            alpha = (1.0 + 1.5 * label) * np.sin(2 * np.pi * 10 * t)
            blocks.append(pd.DataFrame({
                "subject": subject,
                "trial": trial,
                "label": label,
                "AF3": alpha + 0.3 * rng.randn(t.size),
                "F7": 0.8 * np.sin(2 * np.pi * 20 * t) + 0.3 * rng.randn(t.size),
            }))
    return pd.concat(blocks, ignore_index=True)


def _feature_table() -> pd.DataFrame:
    raw = _raw_table()
    # feature_grouped_by_metadata only drops the *grouping* columns, so any other
    # metadata column would be filtered as if it were signal. Keep labels out.
    signal_only = raw.drop(columns=["label"])
    feats = feature_grouped_by_metadata(
        eeg_df=signal_only,
        target_function=lambda d, **k: psd_bandpowers(
            bandpass_filter(d, FS, bands=FREQUENCY_BANDS), FS, bands=FREQUENCY_BANDS
        ),
        fs=FS,
        bands=FREQUENCY_BANDS,
        group_by_metadata_columns=["subject", "trial"],
        drop_metadata_for_fn=True,
    )
    labels = raw.drop_duplicates(["subject", "trial"])[["subject", "trial", "label"]]
    return feats.merge(labels, on=["subject", "trial"], how="left")


def _build_model(n_features: int = 12, **_):
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((1, n_features)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="binary_crossentropy")
    return model


@pytest.fixture(scope="module")
def features():
    return _feature_table()


def test_featurization_output_is_a_valid_tidy_frame(features):
    assert {"subject", "trial", "label"} <= set(features.columns)
    assert len(features) == len(SUBJECTS) * len(TRIALS) * 5   # 5 PSD windows per trial
    # metadata columns lead, feature columns follow
    assert list(features.columns[:2]) == ["subject", "trial"]


@pytest.fixture(scope="module")
def results(features):
    return cross_validate_dataframe(
        features,
        _build_model,
        strategy="loso",
        fs=FS,
        kind="features",
        label_column="label",
        n_epochs=2,
        batch_size=4,
        metrics=("accuracy",),
        selection_metric="accuracy",
        log_predictions=False,
        early_stopping_patience=None,
        verbose=0,
        n_jobs=1,
    )


def test_results_report_original_subject_ids(results):
    """The whole point of the lookup: no user should ever see subject '0'."""
    reported = {row["subject_id"] for row in results["user_metrics"]}
    assert reported == set(SUBJECTS), f"expected {SUBJECTS}, got {sorted(reported)}"


def test_one_fold_per_subject(results):
    assert results["n_subjects"] == len(SUBJECTS)
    held_out = [f["left_out_subjects"] for f in results["fold_results"]]
    flat = [s for group in held_out for s in np.atleast_1d(group).tolist()]
    assert sorted(flat) == sorted(SUBJECTS)


def test_subject_lookup_is_returned(results):
    assert set(results["subject_lookup"].values()) == set(SUBJECTS)


def test_unknown_strategy_kwarg_raises_instead_of_being_ignored(features):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cross_validate_dataframe(
            features, _build_model, strategy="loso", fs=FS,
            kind="features", label_column="label",
            validation_n_users=2,     # the real name is validation_subjects_per_fold
        )


def test_unknown_strategy_name_raises(features):
    with pytest.raises(ValueError, match="unknown strategy"):
        cross_validate_dataframe(features, _build_model, strategy="nope", fs=FS)
