"""Characterization tests for the cross-validation entry points.

These tests pin the *shape* of what ``loso_cv`` returns (key sets, types,
invariants) and the *values* it produces for a fixed seed, so a refactor that
accidentally changes behaviour fails loudly.

Golden values in ``data/loso_cv_golden.json`` depend on the TensorFlow, Keras
and NumPy kernels and on the CPU, so they are compared only in the environment
they were recorded in (``GOLDEN_ENVIRONMENT``). Every other environment checks
the invariants the values must satisfy instead. Regenerate the golden file
deliberately, never to make a red test go green.
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="cross-validation requires eegproc[deep-learning]")

from eegproc.deep_learning.cross_val import loso_cv  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "data" / "loso_cv_golden.json"
RTOL = 1e-4

GOLDEN_ENVIRONMENT = {
    "python": "3.13",
    "tensorflow": "2.20",
    "keras": "3.12",
    "numpy": "2.5",
    "platform": "darwin-arm64",
}


def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def _current_environment() -> dict:
    import keras

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "tensorflow": _major_minor(tf.__version__),
        "keras": _major_minor(keras.__version__),
        "numpy": _major_minor(np.__version__),
        "platform": f"{sys.platform}-{platform.machine().lower()}",
    }


requires_golden_environment = pytest.mark.skipif(
    _current_environment() != GOLDEN_ENVIRONMENT,
    reason=(
        f"golden values were recorded under {GOLDEN_ENVIRONMENT}; "
        f"this is {_current_environment()}. The invariant tests still run."
    ),
)

N_SUBJECTS, N_TRIALS, N_WINDOWS = 4, 4, 8
TIMESTEPS, N_FEATURES = 16, 4


def make_loso_data(seed: int = 0):
    """Rank-3 window tensor with a weak, learnable subject-crossing signal."""
    rng = np.random.RandomState(seed)
    rows = N_SUBJECTS * N_TRIALS * N_WINDOWS
    features = rng.randn(rows, TIMESTEPS, N_FEATURES).astype(np.float32)
    subjects = np.repeat(np.arange(N_SUBJECTS), N_TRIALS * N_WINDOWS).astype(np.int64)
    trials = np.tile(
        np.repeat(np.arange(N_TRIALS), N_WINDOWS), N_SUBJECTS
    ).astype(np.int64)
    labels = ((subjects + trials) % 2).astype(np.int64)
    features += labels[:, None, None] * 0.22
    return features, labels, subjects, trials


def build_tiny_model(timesteps=TIMESTEPS, n_features=N_FEATURES, units=4, **_):
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((timesteps, n_features)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="binary_crossentropy")
    return model


def run_loso():
    features, labels, subjects, trials = make_loso_data()
    tf.keras.utils.set_random_seed(0)
    return loso_cv(
        build_tiny_model, features, labels, subjects, trials,
        n_epochs=2, batch_size=16, metrics=("accuracy", "f1"),
        log_predictions=False, early_stopping_patience=None, verbose=0, n_jobs=1,
    )


@pytest.fixture(scope="module")
def result():
    return run_loso()


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.fail(f"golden fixture missing: {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_text())


# --------------------------------------------------------------------------
# Structure — catches accidental API changes during the split
# --------------------------------------------------------------------------

EXPECTED_TOP_LEVEL = {
    "best_config", "best_config_index", "config_results", "cv_strategy",
    "early_stopping_min_delta", "early_stopping_mode", "early_stopping_monitor",
    "early_stopping_patience", "evaluation_level", "fold_results",
    "hyperparameter_search", "latent_sampling_seed", "max_folds", "maximize_metric",
    "mldg_meta_test_subjects", "mldg_meta_train_subjects", "mldg_samples_per_subject",
    "mldg_seed", "n_configs", "n_evaluated_folds_per_config",
    "n_prediction_latent_samples", "n_subjects", "n_total_loso_fits",
    "restore_best_weights", "selection_level", "selection_metric", "selection_score",
    "selection_score_std", "use_mldg", "user_metrics", "validation_seed",
    "validation_subjects_per_fold",
}

EXPECTED_CONFIG_RESULT = {
    "config", "config_index", "fold_metrics", "fold_training", "oracle_epoch_log",
    "selection_score", "selection_score_std", "trial_mean_scores", "trial_std_scores",
    "window_mean_scores", "window_std_scores",
}

EXPECTED_USER_METRIC = {
    "accuracy", "evaluation_level", "f1", "fold", "loss", "n_samples", "n_trials",
    "n_windows", "subject_id", "trial_accuracy", "trial_f1", "trial_loss",
    "window_accuracy", "window_f1", "window_loss",
}


def test_top_level_keys(result):
    assert set(result) == EXPECTED_TOP_LEVEL


def test_config_result_keys(result):
    assert len(result["config_results"]) == 1
    assert set(result["config_results"][0]) == EXPECTED_CONFIG_RESULT


def test_user_metric_keys(result):
    assert len(result["user_metrics"]) == N_SUBJECTS
    for row in result["user_metrics"]:
        assert set(row) == EXPECTED_USER_METRIC


def test_one_fold_per_subject(result):
    assert result["n_subjects"] == N_SUBJECTS
    assert result["n_evaluated_folds_per_config"] == N_SUBJECTS
    assert result["n_total_loso_fits"] == N_SUBJECTS
    assert len(result["fold_results"]) == N_SUBJECTS
    held_out = [f["left_out_subjects"] for f in result["fold_results"]]
    assert sorted(np.concatenate([np.atleast_1d(h) for h in held_out]).tolist()) == list(
        range(N_SUBJECTS)
    )


def test_echoed_configuration(result):
    assert result["cv_strategy"] == "flat_loso_hyperparameter_search"
    assert result["evaluation_level"] == "trial"
    assert result["selection_metric"] == "f1"
    assert result["selection_level"] == "trial"
    assert result["maximize_metric"] is True
    assert result["use_mldg"] is False
    assert result["n_configs"] == 1


# --------------------------------------------------------------------------
# Invariants — hold in every environment
# --------------------------------------------------------------------------

def test_selection_score_is_mean_and_std_of_subject_trial_f1(result):
    trial_f1 = np.array([row["trial_f1"] for row in result["user_metrics"]])
    assert result["selection_score"] == pytest.approx(trial_f1.mean(), rel=1e-12)
    assert result["selection_score_std"] == pytest.approx(trial_f1.std(), rel=1e-12)


@pytest.mark.parametrize("level", ["trial", "window"])
def test_level_means_are_means_over_subjects(result, level):
    means = result["config_results"][0][f"{level}_mean_scores"]
    for metric in ("accuracy", "f1"):
        per_subject = [row[f"{level}_{metric}"] for row in result["user_metrics"]]
        assert means[metric] == pytest.approx(np.mean(per_subject), rel=1e-12), metric


def test_trial_level_is_the_primary_level(result):
    for row in result["user_metrics"]:
        for metric in ("accuracy", "f1", "loss"):
            assert row[metric] == row[f"trial_{metric}"], metric


def test_counts_and_metric_granularity(result):
    """Accuracies are counts of correct trials or windows, so they move in fixed steps."""
    for row in result["user_metrics"]:
        assert row["n_trials"] == N_TRIALS
        assert row["n_windows"] == N_TRIALS * N_WINDOWS
        correct_trials = row["trial_accuracy"] * N_TRIALS
        correct_windows = row["window_accuracy"] * N_TRIALS * N_WINDOWS
        assert correct_trials == pytest.approx(round(correct_trials), abs=1e-9)
        assert correct_windows == pytest.approx(round(correct_windows), abs=1e-9)
        for metric in ("trial_f1", "window_f1"):
            assert 0.0 <= row[metric] <= 1.0, metric
        for metric in ("trial_loss", "window_loss"):
            assert np.isfinite(row[metric]) and 0.0 < row[metric] < 2.0, metric


# --------------------------------------------------------------------------
# Values — catches behavioural drift in the recording environment
# --------------------------------------------------------------------------

@requires_golden_environment
def test_selection_score_matches_golden(result, golden):
    assert result["selection_score"] == pytest.approx(golden["selection_score"], rel=RTOL)
    assert result["selection_score_std"] == pytest.approx(
        golden["selection_score_std"], rel=RTOL
    )


@requires_golden_environment
@pytest.mark.parametrize("level", ["trial", "window"])
def test_mean_scores_match_golden(result, golden, level):
    actual = result["config_results"][0][f"{level}_mean_scores"]
    expected = golden[f"{level}_mean_scores"]
    assert set(actual) == set(expected)
    for metric, value in expected.items():
        assert actual[metric] == pytest.approx(value, rel=RTOL), metric


@requires_golden_environment
def test_per_subject_metrics_match_golden(result, golden):
    actual = {row["subject_id"]: row for row in result["user_metrics"]}
    for expected in golden["user_metrics"]:
        row = actual[expected["subject_id"]]
        for metric, value in expected.items():
            if metric == "subject_id":
                continue
            assert row[metric] == pytest.approx(value, rel=RTOL), (
                f"subject {expected['subject_id']} {metric}"
            )


def test_metrics_are_not_saturated(result):
    """A test that only ever sees 0.0 or 1.0 cannot detect drift."""
    window = result["config_results"][0]["window_mean_scores"]
    assert 0.0 < window["accuracy"] < 1.0
    assert 0.0 < window["f1"] < 1.0


def test_run_is_deterministic(result):
    assert run_loso()["selection_score"] == pytest.approx(
        result["selection_score"], rel=1e-12
    )


# --------------------------------------------------------------------------
# The held-out-user callback wiring
# --------------------------------------------------------------------------

def test_consolidated_oracle_callback_is_wired_without_monkeypatching():
    """The split replaced an import-time monkeypatch with an explicit import.

    ``training_outputs`` used to rebind ``cross_val.HeldOutUserOracleMetrics`` at
    import time, and the LOSO fold runner relied on reading that global *after*
    the rebind had fired. Moving the base class into ``cross_validation.callbacks``
    would have pointed that rebind at the wrong module and silently downgraded
    LOSO to the base callback. The fold runner now imports the consolidated class
    directly, so the wiring is visible in the source rather than in import order.
    """
    from eegproc.deep_learning import training_outputs
    from eegproc.deep_learning.cross_validation import callbacks, loso_fold

    consolidated = training_outputs.HeldOutUserOracleMetrics
    base = callbacks.HeldOutUserOracleMetrics

    assert consolidated is not base, "the consolidated subclass must not be the base"
    assert issubclass(consolidated, base)
    assert loso_fold.HeldOutUserOracleMetrics is consolidated, (
        "the LOSO fold runner must use the consolidated oracle callback"
    )

    source = Path(training_outputs.__file__).read_text()
    assert "_cross_val_module" not in source, "the import-time monkeypatch is back"


def test_loso_constructs_the_consolidated_oracle_callback(monkeypatch):
    """Behavioural form of the above: prove the callback LOSO builds is the subclass."""
    from eegproc.deep_learning import training_outputs

    constructed = []
    original_init = training_outputs.HeldOutUserOracleMetrics.__init__

    def recording_init(self, *args, **kwargs):
        constructed.append(self)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(
        training_outputs.HeldOutUserOracleMetrics, "__init__", recording_init
    )
    run_loso()

    assert constructed, (
        "loso_cv never constructed training_outputs.HeldOutUserOracleMetrics — "
        "the consolidated oracle callback was silently downgraded"
    )
    assert all(
        isinstance(cb, training_outputs.HeldOutUserOracleMetrics) for cb in constructed
    )
