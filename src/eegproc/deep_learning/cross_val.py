from __future__ import annotations

# Joint-loss variant: early stopping may monitor ``val_loss`` and flat/nested
# hyperparameter search may rank configurations by ``joint_loss``, the complete
# weighted Keras VAE + variational-classifier objective.

import gc
import itertools
import multiprocessing as mp
import os
import queue
import traceback
from itertools import combinations
from pprint import pformat
from typing import Callable, Literal, Mapping

import numpy as np
import tensorflow as tf
from joblib.externals import cloudpickle
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

_FIT_RESERVED_KEYS = frozenset({"epochs", "batch_size"})
_CLASSIFICATION_METRICS = frozenset(
    {
        "accuracy",
        # MTLFuseNet-compatible binary metrics: class 1 is positive.
        "f1",
        "precision",
        "recall",
        # Explicit class-balanced alternatives retained for diagnostics.
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        # Backward-compatible aliases for the canonical binary metrics.
        "binary_f1",
        "binary_precision",
        "binary_recall",
        "roc_auc",
    }
)

# Sequence-valued encoder settings need architecture-aware nesting rules.
# The integer is the nesting depth of one architecture value:
#   depth 1: [16, 32] or [2, 2]
#   depth 2: [[3, 3], [3, 3]]
# One additional outer level enumerates multiple candidate architectures.
_DEFAULT_SEQUENCE_HYPERPARAMETER_DEPTHS = {
    "conv_filters": 1,
    "kernel_sizes": 1,
    "pool_after_layers": 1,
    "pool_sizes": 1,
    "gcn_units": 1,
    "temporal_pool_sizes": 1,
    "spatial_pool_sizes": 2,
}
_EMPTY_SEQUENCE_ALLOWED_KEYS = frozenset(
    {
        "pool_after_layers",
        "pool_sizes",
        "spatial_pool_sizes",
    }
)

# Changing these coefficients changes the numerical scale and definition of
# the complete joint objective. Direct joint-loss comparisons are therefore
# most interpretable when these values remain fixed across configurations.
_JOINT_LOSS_WEIGHT_KEYS = frozenset(
    {
        "ae_loss_weight",
        "vc_loss_weight",
        "vae_beta",
        "vc_alpha",
        "vc_beta",
        "vc_gamma",
        "vc_lambda",
        "subject_loss_weight",
        "mldg_meta_test_weight",
        "use_subject_adversarial",
        "label_smoothing",
    }
)


def _warn_if_joint_loss_weights_vary(
    grid_configs: list[dict],
    selection_metric: str,
) -> None:
    """Warn when direct joint-loss comparisons change the objective itself."""
    if selection_metric != "joint_loss" or len(grid_configs) < 2:
        return

    varying_keys: list[str] = []
    for key in sorted(_JOINT_LOSS_WEIGHT_KEYS):
        explicit_values = {
            repr(config[key])
            for config in grid_configs
            if key in config
        }
        if len(explicit_values) > 1:
            varying_keys.append(key)

    if varying_keys:
        print(
            "Warning: selecting hyperparameters by joint_loss while varying "
            f"{varying_keys} compares differently weighted objectives. Lower "
            "scores may reflect smaller penalty coefficients rather than a "
            "better model. Keep these weights fixed for an apples-to-apples "
            "joint-loss search, or treat the result as exploratory.",
            flush=True,
        )


def _sequence_structure_depth(value) -> int:
    """Return the maximum list/tuple nesting depth of one value."""
    if not isinstance(value, (list, tuple)):
        return 0
    if not value:
        return 1
    return 1 + max(_sequence_structure_depth(item) for item in value)


def _copy_sequence_value(value):
    """Copy nested list/tuple values into JSON-friendly lists."""
    if isinstance(value, (list, tuple)):
        return [_copy_sequence_value(item) for item in value]
    return value


def _hyperparameter_candidates(
    key: str,
    value,
    sequence_hyperparameter_depths: Mapping[str, int] | None = None,
) -> list:
    """Return candidate values while preserving architecture sequences.

    ``sequence_hyperparameter_depths`` resolves otherwise ambiguous nested
    values. For CNN2D, for example, one ``kernel_sizes`` architecture has
    depth two (``[[3, 3], [3, 3]]``), while a depth-three value enumerates
    several kernel schedules. For CNN1D the same key has depth one.
    """
    sequence_depths = dict(_DEFAULT_SEQUENCE_HYPERPARAMETER_DEPTHS)
    if sequence_hyperparameter_depths:
        for sequence_key, expected_depth in sequence_hyperparameter_depths.items():
            expected_depth = int(expected_depth)
            if expected_depth < 1:
                raise ValueError(
                    "Sequence hyperparameter depths must be >= 1; got "
                    f"{sequence_key!r}: {expected_depth}."
                )
            sequence_depths[str(sequence_key)] = expected_depth

    if key not in sequence_depths:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(
                    f"Hyperparameter {key!r} has an empty candidate list."
                )
            return list(value)
        return [value]

    if value is None:
        return [None]

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"Sequence hyperparameter {key!r} must be a list or tuple, "
            f"got {type(value).__name__}."
    )
    if not value:
        if key in _EMPTY_SEQUENCE_ALLOWED_KEYS:
            return [[]]
        raise ValueError(f"Sequence hyperparameter {key!r} cannot be empty.")

    expected_depth = sequence_depths[key]
    actual_depth = _sequence_structure_depth(value)

    # A shallower representation is accepted for reusable atomic settings,
    # such as CNN2D kernel_sizes=[3, 3] or spatial_pool_sizes=[2, 2].
    if actual_depth <= expected_depth:
        return [_copy_sequence_value(value)]

    if actual_depth == expected_depth + 1:
        candidates = [_copy_sequence_value(item) for item in value]
        if (
            key not in _EMPTY_SEQUENCE_ALLOWED_KEYS
            and any(isinstance(candidate, list) and not candidate for candidate in candidates)
        ):
            raise ValueError(
                f"Sequence hyperparameter {key!r} contains an empty candidate."
            )
        return candidates

    raise ValueError(
        f"Sequence hyperparameter {key!r} has nesting depth {actual_depth}, "
        f"but one architecture expects depth {expected_depth}. Use depth "
        f"{expected_depth} for one architecture or {expected_depth + 1} "
        "to enumerate candidates."
    )


def _expand_hyperparameter_grid(
    hp: dict | None,
    sequence_hyperparameter_depths: Mapping[str, int] | None = None,
) -> list[dict]:
    """Expand a hyperparameter dictionary into a Cartesian-product grid."""
    if not hp:
        return [{}]

    keys = list(hp)
    candidate_values = [
        _hyperparameter_candidates(
            key,
            hp[key],
            sequence_hyperparameter_depths=sequence_hyperparameter_depths,
        )
        for key in keys
    ]
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*candidate_values)
    ]



def _split_config(config: dict) -> tuple[dict, dict]:
    """Split a flat config into model-builder kwargs and model.fit kwargs."""
    model_hp = {k: v for k, v in config.items() if k not in _FIT_RESERVED_KEYS}
    fit_hp = {k: v for k, v in config.items() if k in _FIT_RESERVED_KEYS}
    return model_hp, fit_hp




class AlternatingSubjectSetSequence(tf.keras.utils.Sequence):
    """Yield batches alternately from two disjoint subject sets.

    Each epoch shuffles samples independently inside both environments. Batch
    index parity determines the environment: even batches come from set A and
    odd batches from set B. The shorter environment is cycled so both sets
    contribute the same number of optimizer updates per epoch.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        subject_set_a: np.ndarray,
        subject_set_b: np.ndarray,
        batch_size: int,
        *,
        model: tf.keras.Model,
        class_weight: dict[int, float] | None = None,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.subject_ids = np.asarray(subject_ids).reshape(-1)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self.model = model
        self.fit_inputs = _prepare_fit_inputs_with_subject_ids(
            model,
            self.X,
            self.subject_ids,
        )
        self.class_weight = dict(class_weight or {})
        self.rng = np.random.default_rng(seed)

        self.subject_set_a = np.asarray(subject_set_a)
        self.subject_set_b = np.asarray(subject_set_b)
        overlap = np.intersect1d(self.subject_set_a, self.subject_set_b)
        if len(overlap):
            raise ValueError(f"Alternating subject sets overlap: {overlap.tolist()}.")
        self.indices_a = np.where(np.isin(self.subject_ids, self.subject_set_a))[0]
        self.indices_b = np.where(np.isin(self.subject_ids, self.subject_set_b))[0]
        if not len(self.indices_a) or not len(self.indices_b):
            raise ValueError("Both alternating subject sets must contain samples.")
        self._order_a = self.indices_a.copy()
        self._order_b = self.indices_b.copy()
        self.on_epoch_end()

    def __len__(self) -> int:
        batches_a = int(np.ceil(len(self.indices_a) / self.batch_size))
        batches_b = int(np.ceil(len(self.indices_b) / self.batch_size))
        return 2 * max(batches_a, batches_b)

    @staticmethod
    def _cycled_batch(order: np.ndarray, batch_index: int, batch_size: int) -> np.ndarray:
        start = batch_index * batch_size
        positions = (np.arange(start, start + batch_size) % len(order)).astype(np.int64)
        return order[positions]

    def __getitem__(self, index: int):
        environment_batch = int(index) // 2
        order = self._order_a if int(index) % 2 == 0 else self._order_b
        indices = self._cycled_batch(order, environment_batch, self.batch_size)
        y_batch = self.y[indices]
        if isinstance(self.fit_inputs, Mapping):
            x_for_fit = {
                key: np.asarray(value)[indices]
                for key, value in self.fit_inputs.items()
            }
        else:
            x_for_fit = np.asarray(self.fit_inputs)[indices]
        if not self.class_weight:
            return x_for_fit, y_batch
        y_ids = _as_numpy_1d(y_batch).astype(np.int64)
        sample_weight = np.asarray(
            [self.class_weight.get(int(label), 1.0) for label in y_ids],
            dtype=np.float32,
        )
        return x_for_fit, y_batch, sample_weight

    def on_epoch_end(self) -> None:
        self.rng.shuffle(self._order_a)
        self.rng.shuffle(self._order_b)



class MetaLearningSubjectSequence(tf.keras.utils.Sequence):
    """Yield subject-stratified first-order MLDG episodes with natural labels.

    Every item contains two disjoint groups sampled from the current fold's
    gradient-training subjects:

    ``meta_train`` (A)
        Supplies the inner classification/adversarial/SupCon gradient.
    ``meta_test`` (B)
        Supplies an emotion-only gradient after the model has been moved to
        temporary fast weights computed from A.

    Subject IDs are remapped once over the complete fold-local training pool so
    A and B use one consistent subject-class vocabulary. The true LOSO test
    subject and any validation subjects are absent because this sequence is
    constructed only from ``X_fit_train``.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        *,
        model: tf.keras.Model,
        meta_train_subjects: int = 6,
        meta_test_subjects: int = 2,
        samples_per_subject: int = 4,
        class_weight: dict[int, float] | None = None,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.y_ids = _as_numpy_1d(self.y).astype(np.int64)
        self.subject_ids = np.asarray(subject_ids).reshape(-1)
        if not (len(self.X) == len(self.y) == len(self.subject_ids)):
            raise ValueError(
                "MLDG X, y, and subject_ids must have matching lengths."
            )

        self.meta_train_subjects = int(meta_train_subjects)
        self.meta_test_subjects = int(meta_test_subjects)
        self.samples_per_subject = int(samples_per_subject)
        if self.meta_train_subjects < 1 or self.meta_test_subjects < 1:
            raise ValueError("MLDG requires at least one A and one B subject.")
        if self.samples_per_subject < 1:
            raise ValueError("samples_per_subject must be at least 1.")

        self.unique_subjects = np.sort(np.unique(self.subject_ids))
        required_subjects = self.meta_train_subjects + self.meta_test_subjects
        if required_subjects > len(self.unique_subjects):
            raise ValueError(
                "MLDG episode requests "
                f"{required_subjects} subjects, but this fold has only "
                f"{len(self.unique_subjects)} gradient-training subjects."
            )

        self.class_weight = dict(class_weight or {})
        self.base_seed = 0 if seed is None else int(seed)
        self.epoch_index = 0
        total_episode_samples = required_subjects * self.samples_per_subject
        self.steps_per_epoch = max(
            1,
            int(np.ceil(len(self.X) / total_episode_samples)),
        )

        # Configure the fold-local adversarial head, when enabled, and retain a
        # contiguous subject vocabulary for both A and B.
        prepared = _prepare_fit_inputs_with_subject_ids(
            model,
            self.X,
            self.subject_ids,
        )
        if isinstance(prepared, Mapping):
            self.eeg_for_fit = np.asarray(prepared["eeg"])
            prepared_subject_ids = prepared.get("subject_id")
            if prepared_subject_ids is None:
                raise ValueError(
                    "prepare_fit_inputs returned a mapping without subject_id."
                )
            self.subject_classes = np.asarray(prepared_subject_ids, dtype=np.int32)
        else:
            self.eeg_for_fit = np.asarray(prepared)
            subject_to_class = {
                value.item() if isinstance(value, np.generic) else value: index
                for index, value in enumerate(self.unique_subjects)
            }
            self.subject_classes = np.asarray(
                [
                    subject_to_class[
                        value.item() if isinstance(value, np.generic) else value
                    ]
                    for value in self.subject_ids
                ],
                dtype=np.int32,
            )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _sample_subject_windows(
        self,
        subject,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample uniformly from all windows belonging to one subject.

        No class is selected explicitly. Consequently, both meta-train A and
        meta-test B preserve each selected subject's empirical class
        distribution in expectation while every selected subject still
        contributes exactly ``samples_per_subject`` windows.
        """
        subject_indices = np.flatnonzero(self.subject_ids == subject)
        if subject_indices.size == 0:
            raise RuntimeError(
                f"MLDG selected subject {subject!r} without any windows."
            )
        return rng.choice(
            subject_indices,
            size=self.samples_per_subject,
            replace=subject_indices.size < self.samples_per_subject,
        ).astype(np.int64)

    def _sample_group_indices(
        self,
        subjects: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        indices = np.concatenate(
            [self._sample_subject_windows(subject, rng) for subject in subjects],
            axis=0,
        )
        rng.shuffle(indices)
        return indices

    def _sample_weights(self, indices: np.ndarray) -> np.ndarray:
        if not self.class_weight:
            return np.ones(len(indices), dtype=np.float32)
        return np.asarray(
            [self.class_weight.get(int(label), 1.0) for label in self.y_ids[indices]],
            dtype=np.float32,
        )

    def __getitem__(self, index: int):
        seed_sequence = np.random.SeedSequence(
            [self.base_seed, int(self.epoch_index), int(index)]
        )
        rng = np.random.default_rng(seed_sequence)
        selected = rng.choice(
            self.unique_subjects,
            size=self.meta_train_subjects + self.meta_test_subjects,
            replace=False,
        )
        subjects_a = np.asarray(selected[: self.meta_train_subjects])
        subjects_b = np.asarray(selected[self.meta_train_subjects :])
        indices_a = self._sample_group_indices(subjects_a, rng)
        indices_b = self._sample_group_indices(subjects_b, rng)

        x_batch = {
            "meta_train": {
                "eeg": self.eeg_for_fit[indices_a],
                "subject_id": self.subject_classes[indices_a],
            },
            "meta_test": {
                "eeg": self.eeg_for_fit[indices_b],
                "subject_id": self.subject_classes[indices_b],
            },
        }
        y_batch = {
            "meta_train": self.y[indices_a],
            "meta_test": self.y[indices_b],
        }
        sample_weight = {
            "meta_train": self._sample_weights(indices_a),
            "meta_test": self._sample_weights(indices_b),
        }
        return x_batch, y_batch, sample_weight

    def on_epoch_end(self) -> None:
        self.epoch_index += 1

def _balanced_two_subject_sets(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split fold-local subjects into two nearly equal, label-balanced sets."""
    subjects = np.asarray(subject_ids).reshape(-1)
    y_ids = _as_numpy_1d(labels).astype(np.int64)
    unique_subjects = np.sort(np.unique(subjects))
    if len(unique_subjects) < 2:
        raise ValueError("Alternating optimization requires at least two subjects.")

    rows = []
    for subject in unique_subjects:
        mask = subjects == subject
        subject_labels = y_ids[mask]
        rows.append(
            (
                subject,
                int(np.sum(mask)),
                float(np.mean(subject_labels == 1)) if len(subject_labels) else 0.0,
            )
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows.sort(key=lambda row: (row[2], row[1]), reverse=True)

    sets = [[], []]
    counts = [0, 0]
    positives = [0.0, 0.0]
    target_sizes = [len(unique_subjects) // 2, len(unique_subjects) - len(unique_subjects) // 2]
    for subject, count, positive_fraction in rows:
        candidates = [idx for idx in (0, 1) if len(sets[idx]) < target_sizes[idx]]
        choice = min(
            candidates,
            key=lambda idx: (
                positives[idx] / max(counts[idx], 1),
                counts[idx],
                len(sets[idx]),
            ),
        )
        sets[choice].append(subject)
        counts[choice] += count
        positives[choice] += positive_fraction * count

    return np.sort(np.asarray(sets[0])), np.sort(np.asarray(sets[1]))


def _prepare_fit_inputs_with_subject_ids(
    model: tf.keras.Model,
    X: np.ndarray,
    subject_ids: np.ndarray,
):
    """Attach fold-local subject labels only when the model requests them.

    Subject-adversarial models expose ``prepare_fit_inputs``. The method maps
    the fitting subjects to contiguous fold-local classes and returns a Keras
    input dictionary. Ordinary models continue receiving the original EEG
    tensor unchanged. Validation and test inputs are intentionally left raw so
    held-out identities never contribute to the adversarial loss.
    """
    prepare = getattr(model, "prepare_fit_inputs", None)
    if prepare is None or not getattr(model, "use_subject_adversarial", False):
        return X
    return prepare(X, subject_ids)


def _choose_best_config_index(
    mean_scores: list[dict],
    selection_metric: str,
    maximize_metric: bool,
) -> int:
    """Choose the best hyperparameter config from inner-CV mean scores."""
    metric_values = [scores[selection_metric] for scores in mean_scores]

    if maximize_metric:
        return int(np.argmax(metric_values))

    return int(np.argmin(metric_values))


def _apply_preprocessing_strategy(
    preprocessing_strategy: Callable | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
):
    """Apply optional preprocessing inside a CV fold.

    The preprocessing strategy is deliberately fold-local so that anything
    fit inside the strategy is fit only on the training partition.
    """
    if preprocessing_strategy is None:
        return X_train, y_train, X_eval, y_eval

    result = preprocessing_strategy(
        X_train,
        y_train,
        X_eval,
        y_eval,
        train_indices,
        eval_indices,
    )

    if not isinstance(result, tuple):
        raise ValueError(
            "preprocessing_strategy must return a tuple with either 2 or 4 values."
        )

    if len(result) == 2:
        X_train_processed, X_eval_processed = result
        return X_train_processed, y_train, X_eval_processed, y_eval

    if len(result) == 4:
        return result

    raise ValueError(
        "preprocessing_strategy must return either "
        "(X_train, X_eval) or (X_train, y_train, X_eval, y_eval)."
    )


def _as_numpy_1d(values: np.ndarray) -> np.ndarray:
    """Return labels as a 1D numpy array.

    Supports integer labels shaped (n,), binary labels shaped (n, 1), and
    one-hot labels shaped (n, n_classes).
    """
    values = np.asarray(values)

    if values.ndim == 1:
        return values

    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]

    if values.ndim == 2 and values.shape[1] > 1:
        return np.argmax(values, axis=1)

    raise ValueError(
        f"Expected labels with shape (n,), (n, 1), or (n, c). Got {values.shape}."
    )


def _to_probabilities(model_output: np.ndarray) -> np.ndarray:
    """Convert model output to class probabilities.

    Handles:
        - binary sigmoid probabilities/logits with shape (n,) or (n, 1)
        - multiclass softmax probabilities with shape (n, c)
        - multiclass logits with shape (n, c)
    """
    output = np.asarray(model_output)

    if output.ndim == 1:
        output = output.reshape(-1, 1)

    if output.ndim != 2:
        raise ValueError(
            f"Expected model output with shape (n,), (n, 1), or (n, c). Got {output.shape}."
        )

    if output.shape[1] == 1:
        p1 = output[:, 0].astype(np.float64)

        # If values are outside [0, 1], assume logits and sigmoid them.
        if np.any(p1 < 0.0) or np.any(p1 > 1.0):
            p1 = 1.0 / (1.0 + np.exp(-p1))

        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    row_sums = output.sum(axis=1)

    # Already probabilities.
    if (
        np.all(output >= 0.0)
        and np.all(output <= 1.0)
        and np.allclose(row_sums, 1.0, atol=1e-4)
    ):
        return output.astype(np.float64)

    # Otherwise assume logits and softmax.
    shifted = output - np.max(output, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _predict_mc_probability_samples(
    model,
    X: np.ndarray,
    n_samples: int,
    batch_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return posterior-sampled probabilities shaped ``(S, N, C)``.

    Joint VAE models can expose ``predict_mc_probabilities`` to encode each
    input batch once and vectorize the recurrent/classifier work across latent
    samples. A slower generic fallback is retained for compatible custom models.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    X = np.asarray(X)
    effective_batch_size = len(X) if batch_size is None else int(batch_size)
    if effective_batch_size < 1:
        raise ValueError("batch_size must be at least 1 when provided.")

    sample_batches: list[np.ndarray] = []
    for batch_index, start in enumerate(range(0, len(X), effective_batch_size)):
        X_batch = X[start : start + effective_batch_size]
        batch_seed = None if seed is None else (int(seed), int(batch_index))

        if hasattr(model, "predict_mc_probabilities"):
            mc_output = model.predict_mc_probabilities(
                X_batch,
                n_samples=n_samples,
                seed=batch_seed,
            )
            probability_samples = mc_output["probability_samples"]
            if hasattr(probability_samples, "numpy"):
                probability_samples = probability_samples.numpy()
            probability_samples = np.asarray(probability_samples, dtype=np.float64)
        else:
            probability_draws: list[np.ndarray] = []
            for sample_index in range(n_samples):
                if seed is not None:
                    tf.random.set_seed(int(seed) + batch_index * n_samples + sample_index)
                try:
                    raw_output = model(
                        tf.convert_to_tensor(X_batch, dtype=tf.float32),
                        training=False,
                        sample_latent=True,
                    )
                except TypeError as exc:
                    raise TypeError(
                        "Monte Carlo latent prediction requires the model to "
                        "implement predict_mc_probabilities(...) or accept "
                        "sample_latent=True in call(...)."
                    ) from exc
                raw_output = _extract_classifier_output(raw_output)
                if hasattr(raw_output, "numpy"):
                    raw_output = raw_output.numpy()
                probability_draws.append(_to_probabilities(raw_output))
            probability_samples = np.stack(probability_draws, axis=0)

        if probability_samples.ndim != 3:
            raise ValueError(
                "Monte Carlo probabilities must have shape "
                f"(n_samples, batch, n_classes); got {probability_samples.shape}."
            )
        sample_batches.append(probability_samples)

    return np.concatenate(sample_batches, axis=1)


def _predict_probabilities(
    model,
    X,
    batch_size=None,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
):
    """Return class probabilities using posterior means or MC latent draws.

    ``n_prediction_latent_samples=0`` preserves deterministic posterior-mean
    inference. Positive values average that many samples from ``q(z|x)``;
    ``1`` therefore means one random latent draw and one classifier pass.
    """
    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")

    if n_prediction_latent_samples > 0:
        probability_samples = _predict_mc_probability_samples(
            model=model,
            X=X,
            n_samples=n_prediction_latent_samples,
            batch_size=batch_size,
            seed=latent_sampling_seed,
        )
        return probability_samples.mean(axis=0)

    if hasattr(model, "predict_proba"):
        raw_pred = model.predict_proba(X)
    else:
        predict_kwargs = {"verbose": 0}

        if batch_size is not None:
            predict_kwargs["batch_size"] = batch_size

        raw_pred = model.predict(X, **predict_kwargs)

    if isinstance(raw_pred, Mapping):
        if "probabilities" in raw_pred:
            raw_pred = raw_pred["probabilities"]
        elif "logits" in raw_pred:
            raw_pred = raw_pred["logits"]
        else:
            raise ValueError(
                "Model.predict() returned a dictionary, but it did not contain "
                "'logits' or 'probabilities'. "
                f"Available outputs: {list(raw_pred.keys())}"
            )

    return _to_probabilities(raw_pred)


def _normalize_decision_thresholds(
    thresholds: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[float, ...]:
    """Validate, deduplicate, and sort binary class-1 thresholds."""
    values = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("decision_thresholds must contain at least one value.")
    if not np.isfinite(values).all():
        raise ValueError("decision_thresholds must contain only finite values.")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("Every decision threshold must be strictly between 0 and 1.")
    return tuple(float(value) for value in np.unique(values))


def _predict_labels(
    probabilities: np.ndarray,
    decision_threshold: float = 0.5,
) -> np.ndarray:
    """Convert probabilities to labels using a binary class-1 threshold."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape (n_samples, n_classes); got "
            f"{probabilities.shape}."
        )
    if probabilities.shape[1] == 2:
        threshold = float(decision_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("decision_threshold must be strictly between 0 and 1.")
        return (probabilities[:, 1] >= threshold).astype(np.int64)
    if not np.isclose(float(decision_threshold), 0.5):
        raise ValueError(
            "Custom decision thresholds are supported only for binary models."
        )
    return np.argmax(probabilities, axis=1).astype(np.int64)


def _threshold_metric_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
) -> float:
    """Score one validation threshold without using test labels."""
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    y_pred = _as_numpy_1d(y_pred).astype(np.int64)
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        # MTLFuseNet convention: binary F1 for class 1.
        return float(
            f1_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        )
    if metric == "balanced_accuracy":
        return float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        )
    if metric == "binary_f1":
        return float(
            f1_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        )
    raise ValueError(
        "threshold_selection_metric must be accuracy, f1, "
        "balanced_accuracy, or binary_f1. Here f1 follows the "
        "MTLFuseNet binary class-1 convention."
    )


def _select_binary_decision_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    thresholds: tuple[float, ...],
    metric: str,
) -> tuple[float, float, list[dict]]:
    """Select a threshold on validation data with deterministic tie-breaking."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        if len(thresholds) > 1 or not np.isclose(thresholds[0], 0.5):
            raise ValueError(
                "Threshold search requires a binary two-probability output."
            )
        return 0.5, float("nan"), []

    rows: list[dict] = []
    for threshold in thresholds:
        y_pred = _predict_labels(
            probabilities,
            decision_threshold=threshold,
        )
        score = _threshold_metric_value(y_true, y_pred, metric)
        rows.append(
            {
                "threshold": float(threshold),
                "score": float(score),
                "predicted_class_1_fraction": float(np.mean(y_pred == 1)),
            }
        )

    # Maximize score; ties prefer the threshold closest to the conventional 0.5,
    # then the lower threshold for a stable deterministic result.
    best = min(
        rows,
        key=lambda row: (
            -row["score"],
            abs(row["threshold"] - 0.5),
            row["threshold"],
        ),
    )
    return float(best["threshold"]), float(best["score"]), rows


def _stratified_diagnostic_indices(
    y: np.ndarray,
    max_samples: int,
    seed: int | None,
) -> np.ndarray:
    """Choose a deterministic approximately class-balanced diagnostic subset."""
    y_ids = _as_numpy_1d(y).astype(np.int64)
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1.")
    if len(y_ids) <= max_samples:
        return np.arange(len(y_ids), dtype=np.int64)

    rng = np.random.default_rng(seed)
    classes = np.unique(y_ids)
    per_class = max(1, max_samples // max(1, len(classes)))
    selected: list[int] = []

    for class_id in classes:
        class_indices = np.where(y_ids == class_id)[0]
        take = min(per_class, len(class_indices))
        selected.extend(
            rng.choice(class_indices, size=take, replace=False).tolist()
        )

    selected_array = np.asarray(sorted(set(selected)), dtype=np.int64)
    remaining_slots = max_samples - len(selected_array)
    if remaining_slots > 0:
        remaining = np.setdiff1d(
            np.arange(len(y_ids), dtype=np.int64),
            selected_array,
            assume_unique=False,
        )
        if len(remaining):
            extra = rng.choice(
                remaining,
                size=min(remaining_slots, len(remaining)),
                replace=False,
            )
            selected_array = np.sort(
                np.concatenate([selected_array, extra.astype(np.int64)])
            )

    return selected_array[:max_samples]


def _numpy_value(value):
    """Convert tensors and array-like values to numpy arrays."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _diagnostic_model_outputs(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int | None,
) -> dict[str, np.ndarray]:
    """Return probabilities and any available internal classifier tensors."""
    if hasattr(model, "predict_diagnostics"):
        raw_outputs = model.predict_diagnostics(X, batch_size=batch_size)
    else:
        inputs = tf.convert_to_tensor(X, dtype=tf.float32)
        try:
            raw_outputs = model(
                inputs,
                training=False,
                sample_latent=False,
                include_reconstruction=False,
            )
        except TypeError:
            raw_outputs = model(inputs, training=False)

    if isinstance(raw_outputs, Mapping):
        outputs = {
            str(key): _numpy_value(value)
            for key, value in raw_outputs.items()
            if value is not None
        }
        if "probabilities" in outputs:
            probabilities = _to_probabilities(outputs["probabilities"])
        elif "logits" in outputs:
            probabilities = _to_probabilities(outputs["logits"])
        else:
            classifier_output = _extract_classifier_output(raw_outputs)
            probabilities = _to_probabilities(_numpy_value(classifier_output))
        outputs["probabilities"] = probabilities
        return outputs

    classifier_output = _extract_classifier_output(raw_outputs)
    return {
        "probabilities": _to_probabilities(_numpy_value(classifier_output)),
    }


def _prediction_diagnostic_summary(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold_tolerance: float = 0.01,
    internal_outputs: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Summarize confidence, threshold collapse, and internal feature spread."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_ids = _as_numpy_1d(y_true).astype(np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(y_ids):
        raise ValueError(
            "Diagnostic probabilities must have shape (n, c) and align with "
            f"labels; got {probabilities.shape} and {len(y_ids)} labels."
        )
    if threshold_tolerance < 0.0:
        raise ValueError("threshold_tolerance must be non-negative.")

    y_pred = _predict_labels(probabilities)
    confidence = np.max(probabilities, axis=1)

    summary: dict[str, float | int] = {
        "n_samples": int(len(y_ids)),
        "accuracy": float(np.mean(y_pred == y_ids)),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_std": float(np.std(confidence)),
    }

    for class_index in range(probabilities.shape[1]):
        class_probabilities = probabilities[:, class_index]
        summary[f"true_class_{class_index}_fraction"] = float(
            np.mean(y_ids == class_index)
        )
        summary[f"predicted_class_{class_index}_fraction"] = float(
            np.mean(y_pred == class_index)
        )
    return summary


def _print_probability_diagnostics(
    label: str,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold_tolerance: float = 0.01,
) -> dict[str, float | int]:
    """Print a compact probability-distribution diagnostic line."""
    summary = _prediction_diagnostic_summary(
        probabilities=probabilities,
        y_true=y_true,
        threshold_tolerance=threshold_tolerance,
    )
    parts = [
        f"n={summary['n_samples']}",
        f"accuracy={summary['accuracy']:.4f}",
        f"confidence={summary['confidence_mean']:.4f}",
    ]
    if probabilities.shape[1] == 2:
        parts.extend(
            [
                f"pred1={summary['predicted_class_1_fraction']:.4f}",
                f"true1={summary['true_class_1_fraction']:.4f}",
            ]
        )
    print(f"\nPrediction diagnostics [{label}]: " + "  ".join(parts), flush=True)
    return summary


class PredictionDiagnostics(tf.keras.callbacks.Callback):
    """Inspect deterministic train/validation predictions during training.

    Only a fixed, approximately class-balanced subset is evaluated, so the
    callback remains inexpensive relative to a full validation pass. It records
    exact probability spread and, when the model exposes ``predict_diagnostics``,
    latent and logit spread as well.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        fold_number: int | None = None,
        batch_size: int | None = None,
        every_n_epochs: int = 1,
        max_samples: int = 256,
        threshold_tolerance: float = 0.01,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be at least 1.")
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        if threshold_tolerance < 0.0:
            raise ValueError("threshold_tolerance must be non-negative.")

        train_indices = _stratified_diagnostic_indices(
            y_train,
            max_samples=max_samples,
            seed=seed,
        )
        self.X_train = np.asarray(X_train)[train_indices]
        self.y_train = np.asarray(y_train)[train_indices]

        self.X_val = None
        self.y_val = None
        if X_val is not None and y_val is not None and len(X_val):
            validation_seed = None if seed is None else int(seed) + 1
            val_indices = _stratified_diagnostic_indices(
                y_val,
                max_samples=max_samples,
                seed=validation_seed,
            )
            self.X_val = np.asarray(X_val)[val_indices]
            self.y_val = np.asarray(y_val)[val_indices]

        self.fold_number = fold_number
        self.batch_size = batch_size
        self.every_n_epochs = int(every_n_epochs)
        self.threshold_tolerance = float(threshold_tolerance)
        self.history: list[dict] = []

    def _report_split(
        self,
        split: str,
        X: np.ndarray,
        y: np.ndarray,
        epoch_number: int,
        logs: dict,
    ) -> None:
        internal_outputs = _diagnostic_model_outputs(
            model=self.model,
            X=X,
            batch_size=self.batch_size,
        )
        summary = _prediction_diagnostic_summary(
            probabilities=internal_outputs["probabilities"],
            y_true=y,
            threshold_tolerance=self.threshold_tolerance,
            internal_outputs=internal_outputs,
        )
        row = {
            "fold": None if self.fold_number is None else int(self.fold_number),
            "epoch": int(epoch_number),
            "split": split,
            **summary,
        }
        # Keep diagnostics in the dedicated callback history only. They are
        # intentionally not inserted into Keras logs and not printed separately;
        # CompactEpochLogger owns the human-readable epoch output.
        self.history.append(row)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        epoch_number = int(epoch) + 1
        if epoch_number % self.every_n_epochs != 0:
            return
        if logs is None:
            logs = {}

        self._report_split(
            split="train",
            X=self.X_train,
            y=self.y_train,
            epoch_number=epoch_number,
            logs=logs,
        )
        if self.X_val is not None and self.y_val is not None:
            self._report_split(
                split="validation",
                X=self.X_val,
                y=self.y_val,
                epoch_number=epoch_number,
                logs=logs,
            )


class CompactEpochLogger(tf.keras.callbacks.Callback):
    """Print each epoch as compact metric, class-balance, and loss rows.

    Keras' built-in ``verbose=2`` logger places every metric on one very long
    line. This callback groups the same epoch logs into readable categories and
    leaves ``history.history`` unchanged.
    """

    _PREFERRED_METRIC_ORDER = (
        "accuracy",
        "val_accuracy",
        "trial_f1",
        "val_trial_f1",
        "trial_balanced_accuracy",
        "val_trial_balanced_accuracy",
        "balanced_accuracy",
        "val_balanced_accuracy",
        "decoder_accuracy",
        "val_decoder_accuracy",
        "subject_accuracy",
        "val_subject_accuracy",
        "precision",
        "val_precision",
        "recall",
        "val_recall",
        "f1",
        "val_f1",
        "roc_auc",
        "val_roc_auc",
        "learning_rate",
        "lr",
    )

    _KNOWN_LOSS_NAMES = frozenset(
        {
            "loss",
            "base_total_loss",
            "regularization_loss",
            "autoencoder_loss",
            "reconstruction_loss",
            "kl_loss",
            "weighted_kl_loss",
            "vc_loss",
            "vc_cross_entropy",
            "weighted_vc_cross_entropy",
            "vc_latent_kl",
            "weighted_vc_latent_kl",
            "vc_class_prior_kl",
            "weighted_vc_class_prior_kl",
            "vc_discriminator_kl",
            "weighted_vc_discriminator_kl",
            "vc_discriminator_loss",
            "subject_loss",
            "weighted_subject_loss",
            "trial_loss",
        }
    )

    def __init__(
        self,
        fold_number: int | None = None,
        context: str | None = None,
    ) -> None:
        super().__init__()
        self.fold_number = fold_number
        self.context = context

    @staticmethod
    def _float_value(value) -> float | None:
        if value is None:
            return None
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value)
        if array.ndim != 0:
            return None
        result = float(array)
        return result if np.isfinite(result) else None

    @staticmethod
    def _format_value(value: float) -> str:
        absolute = abs(value)
        if absolute != 0.0 and (absolute < 1e-4 or absolute >= 1e4):
            return f"{value:.3e}"
        return f"{value:.4f}"

    @staticmethod
    def _base_name(name: str) -> str:
        return name[4:] if name.startswith("val_") else name

    @classmethod
    def _is_class_fraction(cls, name: str) -> bool:
        base = cls._base_name(name)
        return (
            base.startswith("predicted_class_")
            or base.startswith("true_class_")
        ) and base.endswith("_fraction")

    @classmethod
    def _is_loss_like(cls, name: str) -> bool:
        base = cls._base_name(name)
        return (
            base in cls._KNOWN_LOSS_NAMES
            or base.endswith("_loss")
            or base.endswith("_kl")
            or "cross_entropy" in base
            or base.startswith("weighted_")
        )

    def _prefix(self, epoch_number: int) -> str:
        total_epochs = int(self.params.get("epochs", epoch_number))
        parts: list[str] = []
        if self.fold_number is not None:
            parts.append(f"Fold {int(self.fold_number)}")
        if self.context:
            parts.append(str(self.context))
        parts.append(f"Epoch {epoch_number}/{total_epochs}")
        return "[" + "][".join(parts) + "]"

    def _log_value(
        self,
        logs: Mapping[str, object],
        name: str,
    ) -> float | None:
        return self._float_value(logs.get(name))

    def _model_weight(self, name: str) -> float | None:
        return self._float_value(getattr(self.model, name, None))

    def _format_performance(self, logs: Mapping[str, object]) -> str | None:
        metric_names = [
            name
            for name in logs
            if not self._is_class_fraction(name)
            and not self._is_loss_like(name)
        ]
        ordered_names: list[str] = []
        for name in self._PREFERRED_METRIC_ORDER:
            if name in metric_names:
                ordered_names.append(name)
        ordered_names.extend(
            sorted(name for name in metric_names if name not in ordered_names)
        )

        parts = []
        for name in ordered_names:
            value = self._log_value(logs, name)
            if value is not None:
                parts.append(f"{name}={self._format_value(value)}")
        return " | ".join(parts) if parts else None

    def _format_distribution(
        self,
        logs: Mapping[str, object],
        validation: bool,
    ) -> str | None:
        prefix = "val_" if validation else ""
        class_ids: set[int] = set()
        for name in logs:
            if not name.startswith(prefix):
                continue
            base = name[len(prefix):]
            for distribution_name in ("predicted_class_", "true_class_"):
                if base.startswith(distribution_name) and base.endswith("_fraction"):
                    class_text = base[
                        len(distribution_name) : -len("_fraction")
                    ]
                    try:
                        class_ids.add(int(class_text))
                    except ValueError:
                        pass

        if not class_ids:
            return None

        def values(kind: str) -> str:
            entries = []
            for class_id in sorted(class_ids):
                key = f"{prefix}{kind}_class_{class_id}_fraction"
                value = self._log_value(logs, key)
                if value is not None:
                    entries.append(f"{class_id}:{self._format_value(value)}")
            return "{" + ", ".join(entries) + "}"

        split = "val" if validation else "train"
        return f"{split} pred={values('predicted')} true={values('true')}"

    def _weighted_contribution(
        self,
        logs: Mapping[str, object],
        prefix: str,
        raw_name: str,
        weight_name: str,
    ) -> tuple[float | None, float | None, float | None]:
        raw = self._log_value(logs, f"{prefix}{raw_name}")
        weight = self._model_weight(weight_name)
        contribution = None
        if raw is not None and weight is not None:
            contribution = raw * weight
        return contribution, raw, weight

    def _format_loss_split(
        self,
        logs: Mapping[str, object],
        validation: bool,
    ) -> str | None:
        prefix = "val_" if validation else ""
        split = "val" if validation else "train"
        handled: set[str] = set()
        parts: list[str] = []

        def read(name: str) -> float | None:
            handled.add(f"{prefix}{name}")
            return self._log_value(logs, f"{prefix}{name}")

        total = read("loss")
        base_total = read("base_total_loss")
        regularization = read("regularization_loss")
        if total is not None:
            parts.append(f"total={self._format_value(total)}")
        if base_total is not None:
            parts.append(f"base={self._format_value(base_total)}")
        if regularization is not None:
            parts.append(f"reg={self._format_value(regularization)}")

        ae_contribution, ae_raw, ae_weight = self._weighted_contribution(
            logs, prefix, "autoencoder_loss", "ae_loss_weight"
        )
        handled.add(f"{prefix}autoencoder_loss")
        reconstruction = read("reconstruction_loss")
        raw_kl = read("kl_loss")
        weighted_kl = read("weighted_kl_loss")
        if ae_raw is not None:
            ae_head = (
                ae_contribution if ae_contribution is not None else ae_raw
            )
            ae_details = [f"raw={self._format_value(ae_raw)}"]
            if ae_weight is not None:
                ae_details.append(f"w={self._format_value(ae_weight)}")
            if reconstruction is not None:
                ae_details.append(f"recon={self._format_value(reconstruction)}")
            if raw_kl is not None:
                ae_details.append(f"KL={self._format_value(raw_kl)}")
            if weighted_kl is not None:
                ae_details.append(f"wKL={self._format_value(weighted_kl)}")
            parts.append(
                f"AE={self._format_value(ae_head)}[" + ",".join(ae_details) + "]"
            )

        vc_contribution, vc_raw, vc_weight = self._weighted_contribution(
            logs, prefix, "vc_loss", "vc_loss_weight"
        )
        handled.add(f"{prefix}vc_loss")
        vc_terms = (
            ("weighted_vc_cross_entropy", "CE"),
            ("weighted_vc_latent_kl", "latent"),
            ("weighted_vc_class_prior_kl", "prior"),
            ("weighted_vc_discriminator_kl", "disc"),
        )
        vc_details: list[str] = []
        if vc_raw is not None:
            vc_details.append(f"raw={self._format_value(vc_raw)}")
            if vc_weight is not None:
                vc_details.append(f"w={self._format_value(vc_weight)}")
        for key, label in vc_terms:
            value = read(key)
            if value is not None:
                vc_details.append(f"{label}={self._format_value(value)}")
        # Mark raw diagnostic VC terms as handled so they do not appear again.
        for key in (
            "vc_cross_entropy",
            "vc_latent_kl",
            "vc_class_prior_kl",
            "vc_discriminator_kl",
            "vc_discriminator_loss",
        ):
            handled.add(f"{prefix}{key}")
        if vc_raw is not None or vc_details:
            vc_head = vc_contribution if vc_contribution is not None else vc_raw
            if vc_head is None:
                vc_head = sum(
                    value
                    for key, _ in vc_terms
                    if (value := self._log_value(logs, f"{prefix}{key}"))
                    is not None
                )
            parts.append(
                f"VC={self._format_value(vc_head)}[" + ",".join(vc_details) + "]"
            )

        subject_raw = read("subject_loss")
        weighted_subject = read("weighted_subject_loss")
        if weighted_subject is not None:
            subject_text = f"subject={self._format_value(weighted_subject)}"
            if subject_raw is not None and not np.isclose(
                subject_raw, weighted_subject
            ):
                subject_text += f"[raw={self._format_value(subject_raw)}]"
            parts.append(subject_text)
        elif subject_raw is not None:
            parts.append(f"subject={self._format_value(subject_raw)}")

        trial_loss = read("trial_loss")
        if trial_loss is not None:
            parts.append(f"trial={self._format_value(trial_loss)}")

        extras: list[str] = []
        for name in sorted(logs):
            if name in handled or not name.startswith(prefix):
                continue
            # Do not let the train row consume validation-prefixed values.
            if not validation and name.startswith("val_"):
                continue
            if self._is_loss_like(name):
                value = self._log_value(logs, name)
                if value is not None:
                    display_name = name[len(prefix):] if prefix else name
                    extras.append(
                        f"{display_name}={self._format_value(value)}"
                    )
        if extras:
            parts.append("extra[" + ",".join(extras) + "]")

        return f"{split} " + " | ".join(parts) if parts else None

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        logs = logs or {}
        epoch_number = int(epoch) + 1
        prefix = self._prefix(epoch_number)

        performance = self._format_performance(logs)
        if performance:
            print(f"{prefix} METRICS | {performance}", flush=True)

        distributions = [
            value
            for value in (
                self._format_distribution(logs, validation=False),
                self._format_distribution(logs, validation=True),
            )
            if value is not None
        ]
        if distributions:
            print(
                f"{prefix} CLASSES | " + " | ".join(distributions),
                flush=True,
            )

        train_loss = self._format_loss_split(logs, validation=False)
        if train_loss:
            print(f"{prefix} LOSS | {train_loss}", flush=True)
        validation_loss = self._format_loss_split(logs, validation=True)
        if validation_loss:
            print(f"{prefix} LOSS | {validation_loss}", flush=True)


def _is_trial_tensor(X: np.ndarray) -> bool:
    """Return True for hierarchical inputs shaped ``(N, W, T, F)``."""
    return np.asarray(X).ndim == 4


def _count_windows_for_indices(
    feature_array: np.ndarray,
    indices: np.ndarray,
) -> int:
    """Count underlying windows represented by selected samples.

    Rank-4 hierarchical inputs contain one trial per first-axis sample and one
    window axis at position 1. Rank-3 legacy inputs contain one window per
    first-axis sample.
    """
    features = np.asarray(feature_array)
    selected_count = int(len(indices))
    if features.ndim == 4:
        return selected_count * int(features.shape[1])
    if features.ndim == 3:
        return selected_count
    raise ValueError(
        "feature_array must be rank 3 or 4 when counting windows; "
        f"got {features.shape}."
    )


def _direct_trial_aggregation(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    n_windows_per_trial: int,
    decision_threshold: float = 0.5,
) -> dict:
    """Build the trial-log structure when the model already predicts trials."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    subject_ids = np.asarray(subject_ids)
    trial_ids = np.asarray(trial_ids)
    lengths = (len(probabilities), len(y_true), len(subject_ids), len(trial_ids))
    if len(set(lengths)) != 1:
        raise ValueError(
            "Trial probabilities, labels, subject IDs, and trial IDs must "
            f"align; got lengths {lengths}."
        )
    return {
        "probabilities": probabilities,
        "y_true": y_true,
        "y_pred": _predict_labels(
            probabilities,
            decision_threshold=decision_threshold,
        ),
        "subject_ids": subject_ids,
        "trial_ids": trial_ids,
        "n_windows": np.full(len(y_true), int(n_windows_per_trial), dtype=np.int64),
        "window_indices": [],
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    metrics: list[str] | tuple[str, ...],
    n_classes: int,
) -> dict:
    """Compute selected classification metrics.

    For binary tasks, ``f1``, ``precision``, and ``recall`` follow the
    MTLFuseNet convention: class 1 is the positive class and no macro averaging
    is applied. ``binary_f1``, ``binary_precision``, and ``binary_recall`` are
    retained as backward-compatible aliases. Explicit ``macro_*`` metrics and
    ``balanced_accuracy`` remain available for class-balanced diagnostics.

    For multiclass tasks, the canonical metrics fall back to macro averaging
    because binary positive-class metrics are undefined. ``roc_auc`` uses the
    predicted probability for class 1 and is reported as NaN when a binary fold
    contains only one ground-truth class.
    """
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    y_pred = _as_numpy_1d(y_pred).astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}.")
    if probabilities.ndim != 2 or probabilities.shape != (len(y_true), n_classes):
        raise ValueError(
            "probabilities must have shape (n_samples, n_classes); got "
            f"{probabilities.shape} for {len(y_true)} labels and "
            f"{n_classes} classes."
        )

    expected_labels = list(range(n_classes))

    if np.any(y_true < 0) or np.any(y_true >= n_classes):
        raise ValueError(
            f"y_true contains labels outside the expected range "
            f"[0, {n_classes - 1}]."
        )
    if np.any(y_pred < 0) or np.any(y_pred >= n_classes):
        raise ValueError(
            f"y_pred contains labels outside the expected range "
            f"[0, {n_classes - 1}]."
        )

    binary_metric_names = {
        "binary_f1",
        "binary_precision",
        "binary_recall",
        "roc_auc",
    }
    if n_classes != 2 and any(metric in binary_metric_names for metric in metrics):
        raise ValueError(
            "binary_f1, binary_precision, binary_recall, and roc_auc require "
            f"exactly two classes; got n_classes={n_classes}."
        )

    scores: dict[str, float] = {}

    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported classification metric: {metric}. "
                f"Supported metrics: {sorted(_CLASSIFICATION_METRICS)}"
            )

        if metric == "accuracy":
            scores["accuracy"] = float(accuracy_score(y_true, y_pred))

        elif metric == "f1":
            if n_classes == 2:
                value = f1_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            else:
                value = f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            scores["f1"] = float(value)

        elif metric == "precision":
            if n_classes == 2:
                value = precision_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            else:
                value = precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            scores["precision"] = float(value)

        elif metric == "recall":
            if n_classes == 2:
                value = recall_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            else:
                value = recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            scores["recall"] = float(value)

        elif metric == "macro_f1":
            scores["macro_f1"] = float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "macro_precision":
            scores["macro_precision"] = float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "macro_recall":
            scores["macro_recall"] = float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "balanced_accuracy":
            # For binary/multiclass classification this is macro recall over
            # the complete expected label set, including an absent class as 0.
            scores["balanced_accuracy"] = float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "binary_f1":
            scores["binary_f1"] = float(
                f1_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            )

        elif metric == "binary_precision":
            scores["binary_precision"] = float(
                precision_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            )

        elif metric == "binary_recall":
            scores["binary_recall"] = float(
                recall_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            )

        elif metric == "roc_auc":
            if len(np.unique(y_true)) < 2:
                scores["roc_auc"] = float("nan")
            else:
                scores["roc_auc"] = float(
                    roc_auc_score(y_true, probabilities[:, 1])
                )

    return scores



def _aggregate_window_probabilities_by_trial(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    decision_threshold: float = 0.5,
) -> dict:
    """Aggregate window probabilities into one prediction per subject/trial.

    The model is still trained and run at the window level. For evaluation, all
    window probabilities belonging to the same ``(subject_id, trial_id)`` are
    averaged, and the class with the highest mean probability becomes the trial
    prediction.

    Every window in one trial must have the same ground-truth label. Subject ID
    is included in the grouping key because trial numbers commonly repeat across
    subjects.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    subject_ids = np.asarray(subject_ids)
    trial_ids = np.asarray(trial_ids)

    n_windows = len(y_true)
    if probabilities.ndim != 2 or len(probabilities) != n_windows:
        raise ValueError(
            "probabilities must have shape (n_windows, n_classes) and align "
            f"with y_true. Got probabilities={probabilities.shape}, "
            f"n_windows={n_windows}."
        )
    if len(subject_ids) != n_windows or len(trial_ids) != n_windows:
        raise ValueError(
            "subject_ids and trial_ids must contain one value per window. "
            f"Got {len(subject_ids)} subjects, {len(trial_ids)} trials, "
            f"and {n_windows} labels."
        )

    grouped_indices: dict[tuple, list[int]] = {}
    for index, (subject_id, trial_id) in enumerate(zip(subject_ids, trial_ids)):
        key = (_python_scalar(subject_id), _python_scalar(trial_id))
        grouped_indices.setdefault(key, []).append(index)

    trial_probabilities: list[np.ndarray] = []
    trial_y_true: list[int] = []
    trial_subject_ids: list = []
    output_trial_ids: list = []
    trial_window_counts: list[int] = []
    trial_window_indices: list[np.ndarray] = []

    for (subject_id, trial_id), indices_list in grouped_indices.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        labels = np.unique(y_true[indices])

        if len(labels) != 1:
            raise ValueError(
                "All windows in one trial must share one ground-truth label. "
                f"Subject {subject_id!r}, trial {trial_id!r} contains labels "
                f"{labels.tolist()}."
            )

        trial_probabilities.append(probabilities[indices].mean(axis=0))
        trial_y_true.append(int(labels[0]))
        trial_subject_ids.append(subject_id)
        output_trial_ids.append(trial_id)
        trial_window_counts.append(int(len(indices)))
        trial_window_indices.append(indices)

    trial_probabilities_array = np.stack(trial_probabilities, axis=0)

    return {
        "probabilities": trial_probabilities_array,
        "y_true": np.asarray(trial_y_true, dtype=np.int64),
        "y_pred": _predict_labels(
            trial_probabilities_array,
            decision_threshold=decision_threshold,
        ),
        "subject_ids": np.asarray(trial_subject_ids),
        "trial_ids": np.asarray(output_trial_ids),
        "n_windows": np.asarray(trial_window_counts, dtype=np.int64),
        "window_indices": trial_window_indices,
    }


def _probability_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Return multiclass log loss for a probability matrix."""
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected probabilities with shape (n, c), got {probabilities.shape}."
        )

    return float(
        log_loss(
            y_true,
            probabilities,
            labels=list(range(probabilities.shape[1])),
        )
    )


class TrialValidationMetrics(tf.keras.callbacks.Callback):
    """Compute deterministic trial-level validation metrics each epoch.

    Hierarchical models are scored directly from one output per trial. Legacy
    window models are still aggregated within each (subject_id, trial_id) pair.
    The resulting values are added to the Keras epoch logs as
    ``val_trial_f1``, ``val_trial_balanced_accuracy``, and ``val_trial_loss``
    so callbacks such as EarlyStopping can monitor them.
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        subject_ids_val: np.ndarray,
        trial_ids_val: np.ndarray,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        self.subject_ids_val = np.asarray(subject_ids_val)
        self.trial_ids_val = np.asarray(trial_ids_val)
        self.batch_size = batch_size

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            return

        # Use posterior-mean inference here. It is deterministic, inexpensive,
        # and avoids Monte Carlo sampling noise in the stopping decision.
        probabilities_model = _predict_probabilities(
            model=self.model,
            X=self.X_val,
            batch_size=self.batch_size,
            n_prediction_latent_samples=0,
            latent_sampling_seed=None,
        )

        if _is_trial_tensor(self.X_val):
            trial_aggregation = _direct_trial_aggregation(
                probabilities=probabilities_model,
                y_true=self.y_val,
                subject_ids=self.subject_ids_val,
                trial_ids=self.trial_ids_val,
                n_windows_per_trial=self.X_val.shape[1],
            )
        else:
            trial_aggregation = _aggregate_window_probabilities_by_trial(
                probabilities=probabilities_model,
                y_true=self.y_val,
                subject_ids=self.subject_ids_val,
                trial_ids=self.trial_ids_val,
            )

        probabilities_trial = trial_aggregation["probabilities"]
        y_true_trial = trial_aggregation["y_true"]
        y_pred_trial = trial_aggregation["y_pred"]
        expected_labels = list(range(probabilities_trial.shape[1]))

        if probabilities_trial.shape[1] == 2:
            val_trial_f1 = f1_score(
                y_true_trial,
                y_pred_trial,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        else:
            val_trial_f1 = f1_score(
                y_true_trial,
                y_pred_trial,
                average="macro",
                labels=expected_labels,
                zero_division=0,
            )
        logs["val_trial_f1"] = float(val_trial_f1)
        logs["val_trial_macro_f1"] = float(
            f1_score(
                y_true_trial,
                y_pred_trial,
                average="macro",
                labels=expected_labels,
                zero_division=0,
            )
        )
        logs["val_trial_balanced_accuracy"] = float(
            recall_score(
                y_true_trial,
                y_pred_trial,
                average="macro",
                labels=expected_labels,
                zero_division=0,
            )
        )
        logs["val_trial_loss"] = _probability_log_loss(
            y_true=y_true_trial,
            probabilities=probabilities_trial,
        )


def _level_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    metrics: list[str] | tuple[str, ...],
) -> dict:
    """Compute loss and requested metrics for one evaluation level."""
    scores = {
        "loss": _probability_log_loss(y_true, probabilities),
    }
    scores.update(
        _classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            metrics=metrics,
            n_classes=probabilities.shape[1],
        )
    )
    return scores


def _prefix_scores(scores: dict, prefix: str) -> dict:
    """Prefix metric names, for example ``accuracy`` -> ``trial_accuracy``."""
    return {f"{prefix}_{key}": value for key, value in scores.items()}


def _scores_with_prefix(scores: Mapping[str, float], prefix: str) -> dict:
    """Return prefixed score fields with the prefix removed from their keys."""
    token = f"{prefix}_"
    return {
        key[len(token):]: value
        for key, value in scores.items()
        if key.startswith(token)
    }


def _validate_evaluation_level(level: str, parameter_name: str) -> None:
    """Validate a window/trial evaluation-level parameter."""
    if level not in {"window", "trial"}:
        raise ValueError(
            f"{parameter_name} must be 'window' or 'trial', got {level!r}."
        )


def _validate_processed_alignment(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    partition_name: str,
) -> None:
    """Ensure fold-local preprocessing preserved sample order and count."""
    lengths = (len(X), len(y), len(subject_ids), len(trial_ids))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Preprocessing changed the number of {partition_name} samples or "
            "misaligned labels/IDs. Sample creation, removal, reordering, and "
            "resampling must occur before nested_lnso_cv. Got lengths "
            f"X/y/subject/trial={lengths}."
        )


def _keras_evaluation_results(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int | None = None,
) -> dict[str, float]:
    """Evaluate once and return all scalar Keras metrics as Python floats."""
    eval_output = model.evaluate(
        X,
        y,
        batch_size=batch_size,
        verbose=0,
        return_dict=True,
    )

    if "loss" not in eval_output:
        raise ValueError(
            f"model.evaluate(..., return_dict=True) did not return 'loss': {eval_output}"
        )

    scalar_results: dict[str, float] = {}
    for metric_name, metric_value in eval_output.items():
        value_array = np.asarray(metric_value)
        if value_array.ndim != 0:
            raise ValueError(
                "Keras evaluation metrics must be scalar. "
                f"Metric {metric_name!r} returned shape {value_array.shape}."
            )
        scalar_results[str(metric_name)] = float(value_array)

    return scalar_results


def _keras_loss_value(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int | None = None,
) -> float:
    """Evaluate and return the Keras loss value."""
    return _keras_evaluation_results(
        model=model,
        X=X,
        y=y,
        batch_size=batch_size,
    )["loss"]


def _make_prediction_log(
    fold_index: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
) -> list[dict]:
    """Create one prediction-log row per evaluated window."""
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    y_pred = _as_numpy_1d(y_pred).astype(np.int64)

    rows: list[dict] = []

    for i in range(len(y_true)):
        pred_class = int(y_pred[i])
        row = {
            "fold": int(fold_index),
            "window_index": int(i),
            # Backwards-compatible alias retained for existing exports.
            "sample_index": int(i),
            "subject_id": _python_scalar(subject_ids[i]),
            "trial_id": _python_scalar(trial_ids[i]),
            "y_true": int(y_true[i]),
            "y_pred": pred_class,
            "correct": int(pred_class == int(y_true[i])),
            "p_pred": float(probabilities[i, pred_class]),
            "confidence": float(np.max(probabilities[i])),
        }
        for class_idx in range(probabilities.shape[1]):
            row[f"p_class_{class_idx}"] = float(probabilities[i, class_idx])

        rows.append(row)

    return rows


def _make_trial_prediction_log(
    fold_index: int,
    trial_aggregation: dict,
) -> list[dict]:
    """Create one prediction-log row per evaluated subject/trial."""
    rows: list[dict] = []
    probabilities = trial_aggregation["probabilities"]
    y_true = trial_aggregation["y_true"]
    y_pred = trial_aggregation["y_pred"]

    for i in range(len(y_true)):
        pred_class = int(y_pred[i])
        row = {
            "fold": int(fold_index),
            "trial_index": int(i),
            "subject_id": _python_scalar(trial_aggregation["subject_ids"][i]),
            "trial_id": _python_scalar(trial_aggregation["trial_ids"][i]),
            "n_windows": int(trial_aggregation["n_windows"][i]),
            "y_true": int(y_true[i]),
            "y_pred": pred_class,
            "correct": int(pred_class == int(y_true[i])),
            "p_pred": float(probabilities[i, pred_class]),
            "confidence": float(np.max(probabilities[i])),
        }
        for class_idx in range(probabilities.shape[1]):
            row[f"p_class_{class_idx}"] = float(probabilities[i, class_idx])

        rows.append(row)

    return rows


def _extract_classifier_output(raw_output):
    """Extract classifier logits/probabilities from a model call or prediction."""
    if isinstance(raw_output, Mapping):
        if "probabilities" in raw_output:
            return raw_output["probabilities"]
        if "logits" in raw_output:
            return raw_output["logits"]
        raise ValueError(
            "Model output dictionary did not contain 'logits' or "
            f"'probabilities'. Available outputs: {list(raw_output.keys())}"
        )

    if isinstance(raw_output, (tuple, list)):
        return raw_output[0]

    return raw_output


def _make_variational_interval_logs(
    model: tf.keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    fold_index: int,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    decision_threshold: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    """Log Monte Carlo mean probabilities for windows and trials.

    Trial means are calculated by averaging windows within each stochastic
    forward pass before averaging across posterior samples. ``ci_level`` is
    retained for call compatibility but is no longer serialized or used.
    """
    if n_uncertainty_samples < 2:
        raise ValueError(
            "n_uncertainty_samples must be >= 2 when interval logging is enabled."
        )

    y_true = _as_numpy_1d(y_true).astype(np.int64)

    if _is_trial_tensor(X):
        trial_samples = _predict_mc_probability_samples(
            model=model,
            X=X,
            n_samples=n_uncertainty_samples,
            batch_size=None,
            seed=None,
        )
        trial_mean = trial_samples.mean(axis=0)
        trial_pred = _predict_labels(
            trial_mean, decision_threshold=decision_threshold
        )

        trial_rows: list[dict] = []
        for i in range(len(y_true)):
            pred_class = int(trial_pred[i])
            row = {
                "fold": int(fold_index),
                "trial_index": int(i),
                "subject_id": _python_scalar(subject_ids[i]),
                "trial_id": _python_scalar(trial_ids[i]),
                "n_windows": int(X.shape[1]),
                "y_true": int(y_true[i]),
                "y_pred": pred_class,
                "p_pred_mean": float(trial_mean[i, pred_class]),
            }
            for class_idx in range(trial_mean.shape[1]):
                row[f"p_class_{class_idx}_mean"] = float(
                    trial_mean[i, class_idx]
                )
            trial_rows.append(row)
        return [], trial_rows

    window_samples = _predict_mc_probability_samples(
        model=model,
        X=X,
        n_samples=n_uncertainty_samples,
        batch_size=None,
        seed=None,
    )
    window_mean = window_samples.mean(axis=0)

    window_pred = _predict_labels(
        window_mean, decision_threshold=decision_threshold
    )

    window_rows: list[dict] = []
    for i in range(len(y_true)):
        pred_class = int(window_pred[i])
        row = {
            "fold": int(fold_index),
            "window_index": int(i),
            "sample_index": int(i),
            "subject_id": _python_scalar(subject_ids[i]),
            "trial_id": _python_scalar(trial_ids[i]),
            "y_true": int(y_true[i]),
            "y_pred": pred_class,
            "p_pred_mean": float(window_mean[i, pred_class]),
        }
        for class_idx in range(window_mean.shape[1]):
            row[f"p_class_{class_idx}_mean"] = float(window_mean[i, class_idx])
        window_rows.append(row)

    reference_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=window_mean,
        y_true=y_true,
        subject_ids=subject_ids,
        trial_ids=trial_ids,
        decision_threshold=decision_threshold,
    )

    trial_sample_list: list[np.ndarray] = []
    for sample_index in range(window_samples.shape[0]):
        trial_sample_list.append(
            np.stack(
                [
                    window_samples[sample_index, indices].mean(axis=0)
                    for indices in reference_aggregation["window_indices"]
                ],
                axis=0,
            )
        )

    trial_samples = np.stack(trial_sample_list, axis=0)
    trial_mean = trial_samples.mean(axis=0)
    trial_pred = _predict_labels(
        trial_mean, decision_threshold=decision_threshold
    )

    trial_rows: list[dict] = []
    for i in range(len(reference_aggregation["y_true"])):
        pred_class = int(trial_pred[i])
        row = {
            "fold": int(fold_index),
            "trial_index": int(i),
            "subject_id": _python_scalar(reference_aggregation["subject_ids"][i]),
            "trial_id": _python_scalar(reference_aggregation["trial_ids"][i]),
            "n_windows": int(reference_aggregation["n_windows"][i]),
            "y_true": int(reference_aggregation["y_true"][i]),
            "y_pred": pred_class,
            "p_pred_mean": float(trial_mean[i, pred_class]),
        }
        for class_idx in range(trial_mean.shape[1]):
            row[f"p_class_{class_idx}_mean"] = float(trial_mean[i, class_idx])
        trial_rows.append(row)

    return window_rows, trial_rows

def _python_scalar(value):
    """Convert numpy scalars to plain Python scalars for logs/JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _print_fold_header(fold_number: int, total_folds: int, description: str) -> None:
    """Print a readable progress line for the current fold."""
    print(f"\n[Fold {fold_number:>3} / {total_folds}] {description}")


def _print_config(title: str, config: dict) -> None:
    """Pretty-print a config dict without terminal truncation."""
    print(title)
    print(pformat(config, indent=4, width=120, sort_dicts=False))


def _print_metric_row(title: str, row: dict) -> None:
    """Pretty-print a metric row."""
    print("\n" + title)
    print("-" * len(title))

    for key, value in row.items():
        if isinstance(value, float):
            print(f"{key:>24}: {value:.6f}")
        else:
            print(f"{key:>24}: {value}")


def _print_user_metrics(user_metric_rows: list[dict]) -> None:
    """Print compact per-user metrics."""
    print("\nPer-user metrics")
    print("-" * 100)

    for row in user_metric_rows:
        parts = [
            f"fold={row['fold']}",
            f"subject={row['subject_id']}",
            f"n={row['n_samples']}",
        ]

        for key, value in row.items():
            if key in {"fold", "subject_id", "n_samples"}:
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}={value}")

        print("  " + "  ".join(parts))


def _mean_std_rows(rows: list[dict], metric_names: list[str]) -> tuple[dict, dict]:
    """Compute mean/std for selected metric fields across row dicts."""
    mean_scores: dict[str, float] = {}
    std_scores: dict[str, float] = {}

    for metric_name in metric_names:
        values = [row[metric_name] for row in rows if metric_name in row]

        if not values:
            continue

        numeric_values = np.asarray(values, dtype=np.float64)
        finite_values = numeric_values[np.isfinite(numeric_values)]
        if not len(finite_values):
            mean_scores[metric_name] = float("nan")
            std_scores[metric_name] = float("nan")
            continue

        mean_scores[metric_name] = float(np.mean(finite_values))
        std_scores[metric_name] = float(np.std(finite_values))

    return mean_scores, std_scores


# ---------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------


def _evaluate_trial_tensor_fold(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subject_ids_test: np.ndarray,
    trial_ids_test: np.ndarray,
    fold_index: int,
    metrics: list[str] | tuple[str, ...],
    batch_size: int | None,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    log_predictions: bool,
    log_variational_intervals: bool,
    n_uncertainty_samples: int,
    ci_level: float,
    decision_threshold: float = 0.5,
) -> dict:
    """Evaluate a model that emits one classifier prediction per trial."""
    y_true_trial = _as_numpy_1d(y_test).astype(np.int64)
    probabilities_trial = _predict_probabilities(
        model=model,
        X=X_test,
        batch_size=batch_size,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
    )
    y_pred_trial = _predict_labels(
        probabilities_trial,
        decision_threshold=decision_threshold,
    )
    _print_probability_diagnostics(
        label=f"fold {fold_index} test trial",
        probabilities=probabilities_trial,
        y_true=y_true_trial,
    )
    keras_evaluation = _keras_evaluation_results(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
    )
    keras_model_loss = float(keras_evaluation["loss"])
    decoder_accuracy = keras_evaluation.get("decoder_accuracy")

    trial_scores = _level_scores(
        y_true=y_true_trial,
        y_pred=y_pred_trial,
        probabilities=probabilities_trial,
        metrics=metrics,
    )
    trial_scores["joint_loss"] = keras_model_loss
    if decoder_accuracy is not None:
        trial_scores["decoder_accuracy"] = float(decoder_accuracy)

    n_trials = int(len(y_true_trial))
    n_windows_per_trial = int(X_test.shape[1])
    n_windows = n_trials * n_windows_per_trial
    fold_scores = {
        "fold": int(fold_index),
        "evaluation_level": "trial",
        "classification_level": "trial",
        "n_samples": n_trials,
        "n_windows": n_windows,
        "n_trials": n_trials,
        "windows_per_trial": n_windows_per_trial,
        "keras_model_loss": keras_model_loss,
        "joint_loss": keras_model_loss,
        **(
            {"decoder_accuracy": float(decoder_accuracy)}
            if decoder_accuracy is not None
            else {}
        ),
        "prediction_latent_samples": int(n_prediction_latent_samples),
        "decision_threshold": float(decision_threshold),
        **trial_scores,
        **_prefix_scores(trial_scores, "trial"),
    }

    window_fold_metrics = {
        "fold": int(fold_index),
        "n_windows": n_windows,
        "classification_available": False,
        "joint_loss": keras_model_loss,
        **(
            {"decoder_accuracy": float(decoder_accuracy)}
            if decoder_accuracy is not None
            else {}
        ),
    }
    trial_fold_metrics = {
        "fold": int(fold_index),
        "n_trials": n_trials,
        "windows_per_trial": n_windows_per_trial,
        **trial_scores,
    }

    user_rows: list[dict] = []
    for subject_id in np.unique(subject_ids_test):
        trial_mask = subject_ids_test == subject_id
        user_trial_scores = _level_scores(
            y_true=y_true_trial[trial_mask],
            y_pred=y_pred_trial[trial_mask],
            probabilities=probabilities_trial[trial_mask],
            metrics=metrics,
        )
        user_rows.append(
            {
                "fold": int(fold_index),
                "subject_id": _python_scalar(subject_id),
                "evaluation_level": "trial",
                "classification_level": "trial",
                "n_samples": int(trial_mask.sum()),
                "n_windows": int(trial_mask.sum()) * n_windows_per_trial,
                "n_trials": int(trial_mask.sum()),
                **user_trial_scores,
                **_prefix_scores(user_trial_scores, "trial"),
            }
        )

    trial_aggregation = _direct_trial_aggregation(
        probabilities=probabilities_trial,
        y_true=y_true_trial,
        subject_ids=subject_ids_test,
        trial_ids=trial_ids_test,
        n_windows_per_trial=n_windows_per_trial,
        decision_threshold=decision_threshold,
    )
    trial_prediction_rows = (
        _make_trial_prediction_log(fold_index, trial_aggregation)
        if log_predictions
        else []
    )

    window_interval_rows: list[dict] = []
    trial_interval_rows: list[dict] = []
    if log_variational_intervals:
        window_interval_rows, trial_interval_rows = _make_variational_interval_logs(
            model=model,
            X=X_test,
            y_true=y_true_trial,
            subject_ids=subject_ids_test,
            trial_ids=trial_ids_test,
            fold_index=fold_index,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=decision_threshold,
        )

    _print_metric_row(
        title=f"Fold {fold_index} metrics (trial primary)",
        row=fold_scores,
    )
    _print_user_metrics(user_rows)

    return {
        "fold_metrics": fold_scores,
        "window_fold_metrics": window_fold_metrics,
        "trial_fold_metrics": trial_fold_metrics,
        "user_metrics": user_rows,
        "window_prediction_log": [],
        "trial_prediction_log": trial_prediction_rows,
        "window_variational_interval_log": window_interval_rows,
        "trial_variational_interval_log": trial_interval_rows,
    }


def _evaluate_classification_fold(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subject_ids_test: np.ndarray,
    trial_ids_test: np.ndarray,
    fold_index: int,
    metrics: list[str] | tuple[str, ...],
    evaluation_level: Literal["window", "trial"] = "trial",
    batch_size: int | None = None,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    decision_threshold: float = 0.5,
) -> dict:
    """Evaluate one outer fold at the model's native classification level."""
    _validate_evaluation_level(evaluation_level, "evaluation_level")
    if _is_trial_tensor(X_test):
        if evaluation_level != "trial":
            raise ValueError(
                "Hierarchical rank-4 inputs produce trial-level classifier "
                "outputs; evaluation_level must be 'trial'."
            )
        return _evaluate_trial_tensor_fold(
            model=model,
            X_test=X_test,
            y_test=y_test,
            subject_ids_test=subject_ids_test,
            trial_ids_test=trial_ids_test,
            fold_index=fold_index,
            metrics=metrics,
            batch_size=batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=decision_threshold,
        )
    y_true_window = _as_numpy_1d(y_test).astype(np.int64)

    probabilities_window = _predict_probabilities(
        model=model,
        X=X_test,
        batch_size=batch_size,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
    )
    y_pred_window = _predict_labels(
        probabilities_window,
        decision_threshold=decision_threshold,
    )
    _print_probability_diagnostics(
        label=f"fold {fold_index} test window",
        probabilities=probabilities_window,
        y_true=y_true_window,
    )

    # model.evaluate() is retained as a diagnostic because joint Keras models
    # may include reconstruction/regularization terms beyond classification.
    # It also exposes decoder_accuracy for continuous reconstruction quality.
    keras_evaluation = _keras_evaluation_results(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
    )
    keras_model_loss = keras_evaluation["loss"]
    decoder_accuracy = keras_evaluation.get("decoder_accuracy")

    window_scores = _level_scores(
        y_true=y_true_window,
        y_pred=y_pred_window,
        probabilities=probabilities_window,
        metrics=metrics,
    )
    # ``loss`` above is classifier probability log loss. ``joint_loss`` is
    # the model's complete weighted VAE + VC objective returned by Keras.
    window_scores["joint_loss"] = float(keras_model_loss)
    if decoder_accuracy is not None:
        window_scores["decoder_accuracy"] = float(decoder_accuracy)

    trial_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=probabilities_window,
        y_true=y_true_window,
        subject_ids=subject_ids_test,
        trial_ids=trial_ids_test,
        decision_threshold=decision_threshold,
    )
    _print_probability_diagnostics(
        label=f"fold {fold_index} test trial-aggregated",
        probabilities=trial_aggregation["probabilities"],
        y_true=trial_aggregation["y_true"],
    )
    trial_scores = _level_scores(
        y_true=trial_aggregation["y_true"],
        y_pred=trial_aggregation["y_pred"],
        probabilities=trial_aggregation["probabilities"],
        metrics=metrics,
    )

    primary_scores = trial_scores if evaluation_level == "trial" else window_scores
    fold_scores = {
        "fold": int(fold_index),
        "n_samples": int(
            len(trial_aggregation["y_true"])
            if evaluation_level == "trial"
            else len(y_true_window)
        ),
        "n_windows": int(len(y_true_window)),
        "n_trials": int(len(trial_aggregation["y_true"])),
        "keras_model_loss": float(keras_model_loss),
        "joint_loss": float(keras_model_loss),
        **(
            {"decoder_accuracy": float(decoder_accuracy)}
            if decoder_accuracy is not None
            else {}
        ),
        "prediction_latent_samples": int(n_prediction_latent_samples),
        "decision_threshold": float(decision_threshold),
        **primary_scores,
        **_prefix_scores(window_scores, "window"),
        **_prefix_scores(trial_scores, "trial"),
    }

    window_fold_metrics = {
        "fold": int(fold_index),
        "n_windows": int(len(y_true_window)),
        "keras_model_loss": float(keras_model_loss),
        "joint_loss": float(keras_model_loss),
        "decision_threshold": float(decision_threshold),
        **window_scores,
    }
    trial_fold_metrics = {
        "fold": int(fold_index),
        "n_trials": int(len(trial_aggregation["y_true"])),
        "decision_threshold": float(decision_threshold),
        **trial_scores,
    }

    user_rows: list[dict] = []
    for subject_id in np.unique(subject_ids_test):
        window_mask = subject_ids_test == subject_id
        trial_mask = trial_aggregation["subject_ids"] == subject_id

        user_window_scores = _level_scores(
            y_true=y_true_window[window_mask],
            y_pred=y_pred_window[window_mask],
            probabilities=probabilities_window[window_mask],
            metrics=metrics,
        )
        user_trial_scores = _level_scores(
            y_true=trial_aggregation["y_true"][trial_mask],
            y_pred=trial_aggregation["y_pred"][trial_mask],
            probabilities=trial_aggregation["probabilities"][trial_mask],
            metrics=metrics,
        )
        user_primary_scores = (
            user_trial_scores if evaluation_level == "trial" else user_window_scores
        )

        user_rows.append(
            {
                "fold": int(fold_index),
                "subject_id": _python_scalar(subject_id),
                "evaluation_level": evaluation_level,
                "n_samples": int(trial_mask.sum() if evaluation_level == "trial" else window_mask.sum()),
                "n_windows": int(window_mask.sum()),
                "n_trials": int(trial_mask.sum()),
                **user_primary_scores,
                **_prefix_scores(user_window_scores, "window"),
                **_prefix_scores(user_trial_scores, "trial"),
            }
        )

    window_prediction_rows: list[dict] = []
    trial_prediction_rows: list[dict] = []
    if log_predictions:
        window_prediction_rows = _make_prediction_log(
            fold_index=fold_index,
            y_true=y_true_window,
            y_pred=y_pred_window,
            probabilities=probabilities_window,
            subject_ids=subject_ids_test,
            trial_ids=trial_ids_test,
        )
        trial_prediction_rows = _make_trial_prediction_log(
            fold_index=fold_index,
            trial_aggregation=trial_aggregation,
        )

    window_interval_rows: list[dict] = []
    trial_interval_rows: list[dict] = []
    if log_variational_intervals:
        window_interval_rows, trial_interval_rows = _make_variational_interval_logs(
            model=model,
            X=X_test,
            y_true=y_true_window,
            subject_ids=subject_ids_test,
            trial_ids=trial_ids_test,
            fold_index=fold_index,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
        )

    _print_metric_row(
        title=f"Fold {fold_index} metrics ({evaluation_level} primary)",
        row=fold_scores,
    )
    _print_user_metrics(user_rows)

    return {
        "fold_metrics": fold_scores,
        "window_fold_metrics": window_fold_metrics,
        "trial_fold_metrics": trial_fold_metrics,
        "user_metrics": user_rows,
        "window_prediction_log": window_prediction_rows,
        "trial_prediction_log": trial_prediction_rows,
        "window_variational_interval_log": window_interval_rows,
        "trial_variational_interval_log": trial_interval_rows,
    }


def _evaluate_inner_config(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subject_ids_val: np.ndarray,
    trial_ids_val: np.ndarray,
    metrics: list[str] | tuple[str, ...],
    selection_level: Literal["window", "trial"] = "trial",
    batch_size: int | None = None,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
) -> dict:
    """Evaluate an inner fold at the model's native classification level."""
    _validate_evaluation_level(selection_level, "selection_level")
    if _is_trial_tensor(X_val):
        if selection_level != "trial":
            raise ValueError(
                "Hierarchical rank-4 inputs produce trial-level classifier "
                "outputs; selection_level must be 'trial'."
            )
        y_true_trial = _as_numpy_1d(y_val).astype(np.int64)
        probabilities_trial = _predict_probabilities(
            model,
            X_val,
            batch_size=batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
        )
        y_pred_trial = _predict_labels(probabilities_trial)
        trial_scores = _level_scores(
            y_true=y_true_trial,
            y_pred=y_pred_trial,
            probabilities=probabilities_trial,
            metrics=metrics,
        )
        keras_evaluation = _keras_evaluation_results(
            model, X_val, y_val, batch_size=batch_size
        )
        trial_scores["joint_loss"] = float(keras_evaluation["loss"])
        decoder_accuracy = keras_evaluation.get("decoder_accuracy")
        if decoder_accuracy is not None:
            trial_scores["decoder_accuracy"] = float(decoder_accuracy)
        primary_metric_keys = ["loss", "joint_loss", *metrics]
        return {
            **{key: trial_scores[key] for key in primary_metric_keys},
            **(
                {"decoder_accuracy": float(decoder_accuracy)}
                if decoder_accuracy is not None
                else {}
            ),
            **_prefix_scores(trial_scores, "trial"),
            "selection_level": "trial",
            "classification_level": "trial",
            "n_val_windows": int(len(y_true_trial) * X_val.shape[1]),
            "n_val_trials": int(len(y_true_trial)),
        }

    y_true_window = _as_numpy_1d(y_val).astype(np.int64)
    probabilities_window = _predict_probabilities(
        model,
        X_val,
        batch_size=batch_size,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
    )
    y_pred_window = _predict_labels(probabilities_window)

    window_scores = _level_scores(
        y_true=y_true_window,
        y_pred=y_pred_window,
        probabilities=probabilities_window,
        metrics=metrics,
    )
    keras_evaluation = _keras_evaluation_results(
        model, X_val, y_val, batch_size=batch_size
    )
    window_scores["keras_model_loss"] = keras_evaluation["loss"]
    window_scores["joint_loss"] = keras_evaluation["loss"]
    decoder_accuracy = keras_evaluation.get("decoder_accuracy")
    if decoder_accuracy is not None:
        window_scores["decoder_accuracy"] = float(decoder_accuracy)

    trial_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=probabilities_window,
        y_true=y_true_window,
        subject_ids=subject_ids_val,
        trial_ids=trial_ids_val,
    )
    trial_scores = _level_scores(
        y_true=trial_aggregation["y_true"],
        y_pred=trial_aggregation["y_pred"],
        probabilities=trial_aggregation["probabilities"],
        metrics=metrics,
    )

    primary_scores = trial_scores if selection_level == "trial" else window_scores
    primary_metric_keys = ["loss", *metrics]
    if "joint_loss" in primary_scores:
        primary_metric_keys.append("joint_loss")

    return {
        **{key: primary_scores[key] for key in primary_metric_keys},
        **(
            {"decoder_accuracy": float(decoder_accuracy)}
            if decoder_accuracy is not None
            else {}
        ),
        **_prefix_scores(window_scores, "window"),
        **_prefix_scores(trial_scores, "trial"),
        "selection_level": selection_level,
        "n_val_windows": int(len(y_true_window)),
        "n_val_trials": int(len(trial_aggregation["y_true"])),
    }


# ---------------------------------------------------------------------
# Concurrent outer-fold execution
# ---------------------------------------------------------------------


def _resolve_cuda_device_token(gpu_id: int) -> str:
    """Resolve a local GPU index to the token inherited by a child process.

    Slurm commonly sets ``CUDA_VISIBLE_DEVICES`` to physical ordinals or GPU
    UUIDs. Public ``gpu_ids`` are interpreted as local indices into that visible
    list, so ``gpu_ids=(0, 1)`` always means the first and second GPUs allocated
    to the job rather than physical devices 0 and 1 on the node.
    """
    gpu_id = int(gpu_id)
    if gpu_id < 0:
        raise ValueError(f"GPU indices must be non-negative, got {gpu_id}.")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return str(gpu_id)

    tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if not tokens or tokens == ["-1"]:
        raise ValueError(
            "gpu_ids were supplied, but CUDA_VISIBLE_DEVICES disables all GPUs."
        )
    if gpu_id >= len(tokens):
        raise ValueError(
            f"Requested local GPU index {gpu_id}, but CUDA_VISIBLE_DEVICES="
            f"{visible_devices!r} exposes only {len(tokens)} device(s)."
        )

    return tokens[gpu_id]


def _count_visible_gpus() -> int:
    """Return the number of GPUs visible to the current Slurm/job process."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        tokens = [
            token.strip()
            for token in visible_devices.split(",")
            if token.strip()
        ]
        if not tokens or tokens == ["-1"]:
            return 0
        return len(tokens)

    # Outside Slurm, fall back to TensorFlow's physical-device discovery.
    return len(tf.config.list_physical_devices("GPU"))


def _auto_assign_gpu_ids(n_workers: int) -> tuple[int, ...] | None:
    """Assign one local visible GPU to each worker when GPUs are available."""
    visible_gpu_count = _count_visible_gpus()
    if visible_gpu_count == 0:
        return None

    if n_workers > visible_gpu_count:
        print(
            f"Requested {n_workers} workers, but only {visible_gpu_count} GPU(s) "
            "are visible. Reducing the worker count to one worker per GPU.",
            flush=True,
        )

    return tuple(range(min(n_workers, visible_gpu_count)))


def _start_device_bound_process(
    context,
    target: Callable,
    target_args_prefix: tuple,
    requested_gpu_id: int | None,
    cpus_per_worker: int | None,
    name: str,
) -> mp.Process:
    """Start one spawned process with its GPU mask set before TensorFlow import.

    ``spawn`` launches a fresh interpreter that imports this module. Temporarily
    changing the parent's environment around ``Process.start`` ensures the child
    sees only its assigned GPU before importing TensorFlow. Inside a GPU-bound
    worker that device is therefore always worker-local GPU 0.
    """
    previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if requested_gpu_id is None:
        child_cuda_visible_devices = "-1"
        worker_local_gpu_id = None
        assigned_device_label = "CPU"
    else:
        cuda_token = _resolve_cuda_device_token(requested_gpu_id)
        child_cuda_visible_devices = cuda_token
        worker_local_gpu_id = 0
        assigned_device_label = (
            f"GPU {int(requested_gpu_id)} "
            f"(CUDA_VISIBLE_DEVICES={cuda_token})"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = child_cuda_visible_devices
    try:
        process = context.Process(
            target=target,
            args=(
                *target_args_prefix,
                worker_local_gpu_id,
                cpus_per_worker,
                assigned_device_label,
            ),
            name=name,
        )
        process.start()
    finally:
        if previous_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible_devices

    return process


def _configure_tensorflow_worker(
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None = None,
) -> None:
    """Configure TensorFlow before a worker constructs any model.

    A GPU worker is started with a one-device ``CUDA_VISIBLE_DEVICES`` mask, so
    ``gpu_id`` is normally 0 inside that process. A CPU worker is started with
    ``CUDA_VISIBLE_DEVICES=-1``. This prevents every process from probing or
    allocating memory on every GPU in a multi-GPU Slurm allocation.
    """
    if cpus_per_worker is not None:
        if cpus_per_worker < 1:
            raise ValueError("cpus_per_worker must be >= 1 when provided.")

        tf.config.threading.set_intra_op_parallelism_threads(cpus_per_worker)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    physical_gpus = tf.config.list_physical_devices("GPU")

    if gpu_id is None:
        tf.config.set_visible_devices([], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        if logical_gpus:
            raise RuntimeError(
                "CPU-only worker still has visible logical GPUs after TensorFlow "
                "configuration."
            )
        device_description = assigned_device_label or "CPU"
    else:
        gpu_id = int(gpu_id)

        if gpu_id < 0 or gpu_id >= len(physical_gpus):
            raise ValueError(
                f"Worker requested local GPU index {gpu_id}, but TensorFlow sees "
                f"{len(physical_gpus)} GPU(s). The child process should have been "
                "started with exactly one assigned CUDA device."
            )

        selected_gpu = physical_gpus[gpu_id]
        tf.config.set_visible_devices(selected_gpu, "GPU")
        tf.config.experimental.set_memory_growth(selected_gpu, True)

        logical_gpus = tf.config.list_logical_devices("GPU")
        if len(logical_gpus) != 1:
            raise RuntimeError(
                "A GPU worker must see exactly one logical GPU after isolation; "
                f"TensorFlow sees {len(logical_gpus)}."
            )

        device_description = assigned_device_label or f"GPU {gpu_id}"

    print(
        f"[{mp.current_process().name}] initialized on {device_description}",
        flush=True,
    )


def _collect_spawned_results(
    result_queue,
    processes: list[mp.Process],
    expected_results: int,
    worker_description: str,
) -> list[dict]:
    """Collect fold results without hanging forever after a worker crash."""
    outputs_by_fold: dict[int, dict] = {}

    while len(outputs_by_fold) < expected_results:
        try:
            status, fold_number, payload = result_queue.get(timeout=1.0)
        except queue.Empty:
            failed_processes = [
                process
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed_processes:
                failures = ", ".join(
                    f"{process.name} exitcode={process.exitcode}"
                    for process in failed_processes
                )
                raise RuntimeError(
                    f"A spawned {worker_description} process exited without "
                    f"returning a Python traceback: {failures}. This commonly "
                    "indicates an OS-level kill, CUDA failure, or out-of-memory "
                    "condition."
                )

            if all(process.exitcode is not None for process in processes):
                missing = expected_results - len(outputs_by_fold)
                raise RuntimeError(
                    f"All spawned {worker_description} processes exited, but "
                    f"{missing} fold result(s) were never returned."
                )
            continue

        if status == "error":
            location = (
                f" while running fold {fold_number}"
                if fold_number >= 0
                else " during TensorFlow worker initialization"
            )
            raise RuntimeError(
                f"A spawned {worker_description} worker failed{location}.\n\n"
                f"{payload}"
            )

        if status != "ok":
            raise RuntimeError(
                f"Unknown worker status {status!r} from fold {fold_number}."
            )

        if fold_number in outputs_by_fold:
            raise RuntimeError(
                f"Received duplicate result for fold {fold_number}."
            )

        outputs_by_fold[int(fold_number)] = payload

    return [outputs_by_fold[index] for index in sorted(outputs_by_fold)]


def _run_spawned_fold_pool(
    worker_target: Callable,
    worker_state: dict,
    tasks: list[tuple],
    n_workers: int,
    gpu_ids: tuple[int, ...] | None,
    cpus_per_worker: int | None,
    worker_name_prefix: str,
    worker_description: str,
) -> list[dict]:
    """Run fold tasks using persistent, device-isolated spawned workers."""
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    processes: list[mp.Process] = []
    completed_successfully = False

    try:
        try:
            worker_state_payload = cloudpickle.dumps(worker_state)
        except BaseException as exc:
            raise RuntimeError(
                "Could not serialize the cross-validation worker state. "
                "The model builder, preprocessing strategy, callbacks, and "
                "captured configuration must be cloudpickle-serializable."
            ) from exc

        payload_size_mb = len(worker_state_payload) / (1024 ** 2)
        if payload_size_mb >= 256.0:
            print(
                f"Warning: serialized worker state is {payload_size_mb:.1f} MiB. "
                "Each spawned worker will hold its own host-memory copy.",
                flush=True,
            )

        for worker_index in range(n_workers):
            requested_gpu_id = (
                gpu_ids[worker_index] if gpu_ids is not None else None
            )
            process = _start_device_bound_process(
                context=context,
                target=worker_target,
                target_args_prefix=(worker_state_payload, task_queue, result_queue),
                requested_gpu_id=requested_gpu_id,
                cpus_per_worker=cpus_per_worker,
                name=f"{worker_name_prefix}-{worker_index + 1}",
            )
            processes.append(process)

        for task in tasks:
            task_queue.put(task)

        for _ in processes:
            task_queue.put(None)

        outputs = _collect_spawned_results(
            result_queue=result_queue,
            processes=processes,
            expected_results=len(tasks),
            worker_description=worker_description,
        )
        completed_successfully = True
        return outputs
    finally:
        if not completed_successfully:
            for process in processes:
                if process.is_alive():
                    process.terminate()

        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

        for multiprocessing_queue in (task_queue, result_queue):
            try:
                if not completed_successfully:
                    multiprocessing_queue.cancel_join_thread()
                multiprocessing_queue.close()
            except (AttributeError, OSError, ValueError):
                pass


def _outer_fold_process_main(
    worker_state_payload: bytes,
    task_queue,
    result_queue,
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None,
) -> None:
    """Run outer-fold tasks in one persistent spawned process."""
    try:
        _configure_tensorflow_worker(
            gpu_id=gpu_id,
            cpus_per_worker=cpus_per_worker,
            assigned_device_label=assigned_device_label,
        )
        worker_state = cloudpickle.loads(worker_state_payload)

        while True:
            task = task_queue.get()

            if task is None:
                return

            outer_fold_number, outer_test_subjects = task

            try:
                fold_output = _run_outer_fold(
                    outer_fold_number=outer_fold_number,
                    outer_test_subjects=np.asarray(outer_test_subjects),
                    **worker_state,
                )
                result_queue.put(("ok", int(outer_fold_number), fold_output))
            except BaseException:
                result_queue.put(
                    (
                        "error",
                        int(outer_fold_number),
                        traceback.format_exc(),
                    )
                )
                return

    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def _run_outer_fold(
    outer_fold_number: int,
    outer_test_subjects: np.ndarray,
    total_outer_folds: int,
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    n_inner_subjects_to_leave_out: int,
    grid_configs: list[dict],
    batch_size: int,
    preprocessing_strategy: Callable | None,
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    evaluation_level: Literal["window", "trial"],
    maximize_metric: bool,
    metrics: tuple[str, ...],
    log_predictions: bool,
    log_variational_intervals: bool,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    n_uncertainty_samples: int,
    ci_level: float,
    verbose: int,
    extra_fit_kwargs: dict,
) -> dict:
    """Run one complete outer fold, including its inner grid search."""
    outer_test_mask = np.isin(subject_id_array, outer_test_subjects)
    outer_train_mask = ~outer_test_mask

    outer_train_indices = np.where(outer_train_mask)[0]
    outer_test_indices = np.where(outer_test_mask)[0]

    outer_train_subject_ids = subject_id_array[outer_train_indices]
    unique_outer_train_subjects = np.sort(np.unique(outer_train_subject_ids))

    inner_subject_splits = list(
        combinations(unique_outer_train_subjects, n_inner_subjects_to_leave_out)
    )

    sample_level = "trials" if feature_array.ndim == 4 else "windows"
    _print_fold_header(
        outer_fold_number,
        total_outer_folds,
        f"outer test subjects={outer_test_subjects.tolist()} "
        f"(outer_train={len(outer_train_indices)}, "
        f"outer_test={len(outer_test_indices)} {sample_level})",
    )

    inner_scores_by_config: list[list[dict]] = [[] for _ in grid_configs]
    inner_fold_results: list[dict] = []

    # -----------------------------------------------------------------
    # Inner CV: choose hyperparameters.
    # -----------------------------------------------------------------
    for inner_fold_number, inner_val_subjects in enumerate(
        inner_subject_splits,
        start=1,
    ):
        inner_val_subjects = np.asarray(inner_val_subjects)

        inner_val_mask_relative = np.isin(
            outer_train_subject_ids,
            inner_val_subjects,
        )
        inner_train_mask_relative = ~inner_val_mask_relative

        inner_train_indices = outer_train_indices[inner_train_mask_relative]
        inner_val_indices = outer_train_indices[inner_val_mask_relative]

        X_inner_train = feature_array[inner_train_indices]
        y_inner_train = label_array[inner_train_indices]
        X_inner_val = feature_array[inner_val_indices]
        y_inner_val = label_array[inner_val_indices]
        subject_ids_inner_train = subject_id_array[inner_train_indices]
        subject_ids_inner_val = subject_id_array[inner_val_indices]
        trial_ids_inner_train = trial_id_array[inner_train_indices]
        trial_ids_inner_val = trial_id_array[inner_val_indices]

        (
            X_inner_train,
            y_inner_train,
            X_inner_val,
            y_inner_val,
        ) = _apply_preprocessing_strategy(
            preprocessing_strategy=preprocessing_strategy,
            X_train=X_inner_train,
            y_train=y_inner_train,
            X_eval=X_inner_val,
            y_eval=y_inner_val,
            train_indices=inner_train_indices,
            eval_indices=inner_val_indices,
        )

        _validate_processed_alignment(
            X_inner_train, y_inner_train, subject_ids_inner_train,
            trial_ids_inner_train, "inner-training"
        )
        _validate_processed_alignment(
            X_inner_val, y_inner_val, subject_ids_inner_val,
            trial_ids_inner_val, "inner-validation"
        )

        config_results_this_inner_fold: list[dict] = []

        for config_index, config in enumerate(grid_configs):
            model_hp, fit_hp = _split_config(config)
            current_batch_size = fit_hp.get("batch_size", batch_size)

            tf.keras.backend.clear_session()
            model = model_builder_function(**model_hp)

            try:
                X_inner_train_for_fit = _prepare_fit_inputs_with_subject_ids(
                    model,
                    X_inner_train,
                    subject_ids_inner_train,
                )
                fit_kwargs = dict(fit_hp)
                fit_kwargs["validation_data"] = (X_inner_val, y_inner_val)
                current_extra_fit_kwargs = dict(extra_fit_kwargs)
                fit_callbacks = list(
                    current_extra_fit_kwargs.pop("callbacks", [])
                )
                if verbose:
                    fit_callbacks.append(
                        CompactEpochLogger(
                            fold_number=outer_fold_number,
                            context=(
                                f"Inner {inner_fold_number} "
                                f"Config {config_index + 1}"
                            ),
                        )
                    )
                if fit_callbacks:
                    current_extra_fit_kwargs["callbacks"] = fit_callbacks

                y_inner_train_ids = _as_numpy_1d(y_inner_train)
                classes, counts = np.unique(y_inner_train_ids, return_counts=True)

                class_weight = {
                    int(class_id): len(y_inner_train_ids) / (len(classes) * count)
                    for class_id, count in zip(classes, counts)
                }

                model.fit(
                    X_inner_train_for_fit,
                    y_inner_train,
                    class_weight=class_weight,
                    verbose=0,
                    **fit_kwargs,
                    **current_extra_fit_kwargs,
                )

                val_scores = _evaluate_inner_config(
                    model=model,
                    X_val=X_inner_val,
                    y_val=y_inner_val,
                    subject_ids_val=subject_ids_inner_val,
                    trial_ids_val=trial_ids_inner_val,
                    metrics=metrics,
                    selection_level=selection_level,
                    batch_size=current_batch_size,
                    n_prediction_latent_samples=n_prediction_latent_samples,
                    latent_sampling_seed=latent_sampling_seed,
                )

                config_result = {
                    "config_index": int(config_index),
                    "window_scores": _scores_with_prefix(val_scores, "window"),
                    "trial_scores": _scores_with_prefix(val_scores, "trial"),
                }
                config_result = {
                    key: value
                    for key, value in config_result.items()
                    if value != {}
                }

                config_results_this_inner_fold.append(config_result)
                inner_scores_by_config[config_index].append(val_scores)

            finally:
                del model
                gc.collect()
                tf.keras.backend.clear_session()

        inner_fold_results.append(
            {
                "inner_fold": int(inner_fold_number),
                "left_out_subjects": inner_val_subjects.tolist(),
                "n_train_windows": _count_windows_for_indices(feature_array, inner_train_indices),
                "n_val_windows": _count_windows_for_indices(feature_array, inner_val_indices),
                "n_train_trials": int(len(set(zip(
                    subject_ids_inner_train.tolist(), trial_ids_inner_train.tolist()
                )))),
                "n_val_trials": int(len(set(zip(
                    subject_ids_inner_val.tolist(), trial_ids_inner_val.tolist()
                )))),
                "configs": config_results_this_inner_fold,
            }
        )

    # -----------------------------------------------------------------
    # Aggregate inner-CV scores and choose the best configuration.
    # -----------------------------------------------------------------
    inner_mean_scores: list[dict] = []
    inner_std_scores: list[dict] = []
    score_metric_names = [
        "loss", "joint_loss", *metrics, "decoder_accuracy",
        "window_loss", "window_joint_loss", "window_keras_model_loss", "window_decoder_accuracy",
        *[f"window_{metric}" for metric in metrics],
        "trial_loss", "trial_joint_loss", "trial_decoder_accuracy",
        *[f"trial_{metric}" for metric in metrics],
    ]

    for config_index, config in enumerate(grid_configs):
        mean_scores_for_config, std_scores_for_config = _mean_std_rows(
            inner_scores_by_config[config_index],
            score_metric_names,
        )

        inner_mean_scores.append(
            {
                "config_index": int(config_index),
                **mean_scores_for_config,
            }
        )
        inner_std_scores.append(
            {
                "config_index": int(config_index),
                **std_scores_for_config,
            }
        )

    best_config_index = _choose_best_config_index(
        mean_scores=inner_mean_scores,
        selection_metric=selection_metric,
        maximize_metric=maximize_metric,
    )
    best_config = grid_configs[best_config_index]
    inner_config_scores = []
    for config_index in range(len(grid_configs)):
        mean_row = inner_mean_scores[config_index]
        std_row = inner_std_scores[config_index]
        score_row = {
            "config_index": int(config_index),
            "selection_score": float(mean_row[selection_metric]),
            "selection_score_std": float(std_row[selection_metric]),
            "window_mean_scores": _scores_with_prefix(mean_row, "window"),
            "window_std_scores": _scores_with_prefix(std_row, "window"),
            "trial_mean_scores": _scores_with_prefix(mean_row, "trial"),
            "trial_std_scores": _scores_with_prefix(std_row, "trial"),
        }
        inner_config_scores.append(
            {key: value for key, value in score_row.items() if value != {}}
        )

    print(
        f"\nBest config from inner CV for outer fold {outer_fold_number}: "
        f"{selection_metric}="
        f"{inner_mean_scores[best_config_index][selection_metric]:.6f}",
        flush=True,
    )
    _print_config("Best config:", best_config)

    best_config_result = {
        "outer_fold": int(outer_fold_number),
        "best_config_index": int(best_config_index),
        "selection_score": float(
            inner_mean_scores[best_config_index][selection_metric]
        ),
    }

    inner_cv_result = {
        "outer_fold": int(outer_fold_number),
        "inner_fold_results": inner_fold_results,
        "inner_config_scores": inner_config_scores,
    }

    # -----------------------------------------------------------------
    # Final outer training and testing.
    # -----------------------------------------------------------------
    X_outer_train = feature_array[outer_train_indices]
    y_outer_train = label_array[outer_train_indices]
    X_outer_test = feature_array[outer_test_indices]
    y_outer_test = label_array[outer_test_indices]
    subject_ids_outer_train = subject_id_array[outer_train_indices]
    subject_ids_outer_test = subject_id_array[outer_test_indices]
    trial_ids_outer_train = trial_id_array[outer_train_indices]
    trial_ids_outer_test = trial_id_array[outer_test_indices]

    (
        X_outer_train,
        y_outer_train,
        X_outer_test,
        y_outer_test,
    ) = _apply_preprocessing_strategy(
        preprocessing_strategy=preprocessing_strategy,
        X_train=X_outer_train,
        y_train=y_outer_train,
        X_eval=X_outer_test,
        y_eval=y_outer_test,
        train_indices=outer_train_indices,
        eval_indices=outer_test_indices,
    )

    _validate_processed_alignment(
        X_outer_train, y_outer_train, subject_ids_outer_train,
        trial_ids_outer_train, "outer-training"
    )
    _validate_processed_alignment(
        X_outer_test, y_outer_test, subject_ids_outer_test,
        trial_ids_outer_test, "outer-test"
    )

    model_hp, fit_hp = _split_config(best_config)
    current_batch_size = fit_hp.get("batch_size", batch_size)

    tf.keras.backend.clear_session()
    final_model = model_builder_function(**model_hp)

    try:
        X_outer_train_for_fit = _prepare_fit_inputs_with_subject_ids(
            final_model,
            X_outer_train,
            subject_ids_outer_train,
        )
        y_outer_train_ids = _as_numpy_1d(y_outer_train)
        classes, counts = np.unique(y_outer_train_ids, return_counts=True)

        class_weight = {
            int(class_id): len(y_outer_train_ids) / (len(classes) * count)
            for class_id, count in zip(classes, counts)
        }

        final_extra_fit_kwargs = dict(extra_fit_kwargs)
        final_callbacks = list(
            final_extra_fit_kwargs.pop("callbacks", [])
        )
        if verbose:
            final_callbacks.append(
                CompactEpochLogger(
                    fold_number=outer_fold_number,
                    context="Outer final",
                )
            )
        if final_callbacks:
            final_extra_fit_kwargs["callbacks"] = final_callbacks

        final_model.fit(
            X_outer_train_for_fit,
            y_outer_train,
            class_weight=class_weight,
            verbose=0,
            **fit_hp,
            **final_extra_fit_kwargs,
        )

        fold_result = _evaluate_classification_fold(
            model=final_model,
            X_test=X_outer_test,
            y_test=y_outer_test,
            subject_ids_test=subject_ids_outer_test,
            trial_ids_test=trial_ids_outer_test,
            fold_index=outer_fold_number,
            metrics=metrics,
            evaluation_level=evaluation_level,
            batch_size=current_batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
        )

    finally:
        del final_model
        gc.collect()
        tf.keras.backend.clear_session()

    fold_record = {
        "outer_fold_number": int(outer_fold_number),
        "left_out_subjects": outer_test_subjects.tolist(),
        "n_outer_train_windows": _count_windows_for_indices(
            feature_array, outer_train_indices
        ),
        "n_outer_test_windows": _count_windows_for_indices(
            feature_array, outer_test_indices
        ),
        "n_outer_train_trials": int(len(set(zip(
            subject_ids_outer_train.tolist(), trial_ids_outer_train.tolist()
        )))),
        "n_outer_test_trials": int(len(set(zip(
            subject_ids_outer_test.tolist(), trial_ids_outer_test.tolist()
        )))),
    }

    return {
        "outer_fold_number": int(outer_fold_number),
        "fold_record": fold_record,
        "best_config_result": best_config_result,
        "inner_cv_result": inner_cv_result,
        **fold_result,
    }


# ---------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------


def nested_lnso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray | None = None,
    n_outer_subjects_to_leave_out: int = 1,
    n_inner_subjects_to_leave_out: int = 1,
    n_epochs: int = 50,
    batch_size: int = 2,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    selection_metric: str = "f1",
    selection_level: Literal["window", "trial"] = "trial",
    evaluation_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "binary_f1",
        "binary_precision",
        "binary_recall",
        "roc_auc",
    ),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
) -> dict:
    """Run nested Leave-N-Subjects-Out CV.

    Outer folds can be executed concurrently using multiprocessing with the
    ``spawn`` start method. Inner folds and hyperparameter configurations remain
    sequential inside each worker, preventing nested process oversubscription.

    Trial-level evaluation
    ----------------------
    Rank-4 inputs contain one complete trial per sample and one trial ID per
    sample. The classifier is evaluated directly at trial level while decoder
    diagnostics remain window based. Rank-3 legacy inputs retain the previous
    behavior of averaging window probabilities within each trial.

    Parameters added for concurrency
    --------------------------------
    n_jobs:
        Number of persistent outer-fold worker processes. ``1`` preserves the
        original sequential behavior unless ``gpu_ids`` is supplied.
    gpu_ids:
        Local GPU indices assigned one-per-worker. For example,
        ``gpu_ids=(0, 1, 2, 3)`` with ``n_jobs=4``. When this is ``None`` and
        multiple workers are requested, visible Slurm/TensorFlow GPUs are
        assigned automatically, one per worker.
    cpus_per_worker:
        TensorFlow intra-op CPU threads available to each worker. Keep
        ``n_jobs * cpus_per_worker`` within the CPUs allocated by Slurm.

    Notes
    -----
    Worker state is serialized with cloudpickle, so locally defined model
    builders and preprocessing callables are supported. The training entry
    point must still be protected by ``if __name__ == "__main__":``.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    if "validation_data" in extra_fit_kwargs:
        raise ValueError(
            "Do not pass validation_data in extra_fit_kwargs. "
            "nested_lnso_cv creates validation_data from the inner folds."
        )

    if subject_id_array is None:
        raise ValueError("subject_id_array is required for nested LNSO CV.")
    if trial_id_array is None:
        raise ValueError(
            "trial_id_array is required for trial-level prediction and metrics. "
            "Pass one trial ID per sample, aligned with feature_array."
        )

    _validate_evaluation_level(selection_level, "selection_level")
    _validate_evaluation_level(evaluation_level, "evaluation_level")

    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    if feature_array.ndim not in {3, 4}:
        raise ValueError(
            "feature_array must be rank 3 for window samples or rank 4 for "
            f"grouped trial samples; got {feature_array.shape}."
        )
    if feature_array.ndim == 4:
        if selection_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require selection_level='trial'."
            )
        if evaluation_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require evaluation_level='trial'."
            )

    input_lengths = (
        len(feature_array), len(label_array), len(subject_id_array), len(trial_id_array)
    )
    if len(set(input_lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            "must have the same first dimension. Got lengths "
            f"{input_lengths}."
        )

    metrics = tuple(metrics)

    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. Supported metrics: "
                f"{sorted(_CLASSIFICATION_METRICS)}"
            )

    allowed_selection_metrics = {"loss", "joint_loss", *metrics}

    if selection_metric not in allowed_selection_metrics:
        raise ValueError(
            f"selection_metric='{selection_metric}' is not available. "
            f"Use 'loss', 'joint_loss', or one of metrics={list(metrics)}."
        )

    if (
        feature_array.ndim == 3
        and selection_metric == "joint_loss"
        and selection_level != "window"
    ):
        raise ValueError(
            "Window-level models require selection_level='window' when "
            "selecting by joint_loss. Hierarchical rank-4 models may select "
            "trial-level joint_loss directly."
        )

    if maximize_metric is None:
        maximize_metric = selection_metric not in {"loss", "joint_loss"}

    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")

    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")

    unique_subjects = np.sort(np.unique(subject_id_array))

    if n_outer_subjects_to_leave_out < 1:
        raise ValueError("n_outer_subjects_to_leave_out must be >= 1.")

    if n_inner_subjects_to_leave_out < 1:
        raise ValueError("n_inner_subjects_to_leave_out must be >= 1.")

    if n_outer_subjects_to_leave_out >= len(unique_subjects):
        raise ValueError(
            "n_outer_subjects_to_leave_out must be smaller than the number "
            f"of unique subjects. Got {n_outer_subjects_to_leave_out} for "
            f"{len(unique_subjects)} subjects."
        )

    n_outer_train_subjects = len(unique_subjects) - n_outer_subjects_to_leave_out

    if n_inner_subjects_to_leave_out >= n_outer_train_subjects:
        raise ValueError(
            "n_inner_subjects_to_leave_out must be smaller than the number "
            "of subjects available in each outer-training pool. Got "
            f"{n_inner_subjects_to_leave_out} for {n_outer_train_subjects} "
            "outer-training subjects."
        )

    if hyperparameters is None:
        hyperparameters = {}

    effective_hyperparameters = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        **hyperparameters,
    }

    sequence_hyperparameter_depths = getattr(
        model_builder_function,
        "_sequence_hyperparameter_depths",
        None,
    )
    grid_configs = _expand_hyperparameter_grid(
        effective_hyperparameters,
        sequence_hyperparameter_depths=sequence_hyperparameter_depths,
    )
    _warn_if_joint_loss_weights_vary(grid_configs, selection_metric)
    outer_subject_splits = list(
        combinations(unique_subjects, n_outer_subjects_to_leave_out)
    )
    total_outer_folds = len(outer_subject_splits)
    effective_n_jobs = min(n_jobs, total_outer_folds)

    normalized_gpu_ids: tuple[int, ...] | None = None

    if gpu_ids is None and effective_n_jobs > 1:
        normalized_gpu_ids = _auto_assign_gpu_ids(effective_n_jobs)
        if normalized_gpu_ids is not None:
            effective_n_jobs = len(normalized_gpu_ids)
    elif gpu_ids is not None:
        normalized_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)

        if not normalized_gpu_ids:
            raise ValueError("gpu_ids must contain at least one GPU index.")
        if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
            raise ValueError("gpu_ids must not contain duplicate GPU indices.")
        if effective_n_jobs > len(normalized_gpu_ids):
            raise ValueError(
                f"n_jobs={effective_n_jobs} requires at least that many GPU IDs, "
                f"but gpu_ids={normalized_gpu_ids}. Use one GPU per worker."
            )

        normalized_gpu_ids = normalized_gpu_ids[:effective_n_jobs]

    results = {
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "configs": [dict(config) for config in grid_configs],
        "fold_metrics": [],
        "user_metrics": [],
        "best_configs": [],
        "inner_cv_results": [],
        "fold_results": [],
        "window_mean_scores": {},
        "window_std_scores": {},
        "trial_mean_scores": {},
        "trial_std_scores": {},
    }

    if log_predictions:
        if feature_array.ndim == 3:
            results["window_prediction_log"] = []
        results["trial_prediction_log"] = []
    if log_variational_intervals:
        if feature_array.ndim == 3:
            results["window_variational_interval_log"] = []
        results["trial_variational_interval_log"] = []

    print(
        f"\nNested LNSO CV — {total_outer_folds} outer folds, "
        f"{len(grid_configs)} hyperparameter config"
        f"{'s' if len(grid_configs) != 1 else ''}"
    )
    print(f"Requested metrics: {list(metrics)}")
    print(
        f"Selection metric: {selection_level}-level {selection_metric} "
        f"({'maximize' if maximize_metric else 'minimize'})"
    )
    print(f"Primary reported metrics: {evaluation_level}-level")
    print(f"Prediction logging: {log_predictions}")
    print(f"Variational interval logging: {log_variational_intervals}")
    prediction_mode = (
        "posterior mean"
        if n_prediction_latent_samples == 0
        else f"MC average over {n_prediction_latent_samples} latent sample(s)"
    )
    print(f"Prediction latent mode: {prediction_mode}")
    print(f"Outer-fold workers: {effective_n_jobs}")

    if effective_n_jobs > 1 and normalized_gpu_ids is None:
        print("Worker devices: CPU-only")
    elif normalized_gpu_ids is not None:
        print(f"Worker devices: GPUs {list(normalized_gpu_ids)}")
    else:
        print("Worker device: current TensorFlow default")

    tasks = [
        (fold_number, tuple(outer_test_subjects))
        for fold_number, outer_test_subjects in enumerate(
            outer_subject_splits,
            start=1,
        )
    ]

    worker_state = {
        "total_outer_folds": total_outer_folds,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "n_inner_subjects_to_leave_out": n_inner_subjects_to_leave_out,
        "grid_configs": grid_configs,
        "batch_size": batch_size,
        "preprocessing_strategy": preprocessing_strategy,
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "maximize_metric": bool(maximize_metric),
        "metrics": metrics,
        "log_predictions": log_predictions,
        "log_variational_intervals": log_variational_intervals,
        "n_prediction_latent_samples": n_prediction_latent_samples,
        "latent_sampling_seed": latent_sampling_seed,
        "n_uncertainty_samples": n_uncertainty_samples,
        "ci_level": ci_level,
        "verbose": verbose,
        "extra_fit_kwargs": extra_fit_kwargs,
    }

    # Preserve the old in-process behavior for n_jobs=1 with no explicit GPU
    # assignment. This avoids spawn/pickling overhead for ordinary runs.
    if effective_n_jobs == 1 and normalized_gpu_ids is None:
        fold_outputs = [
            _run_outer_fold(
                outer_fold_number=fold_number,
                outer_test_subjects=np.asarray(outer_test_subjects),
                **worker_state,
            )
            for fold_number, outer_test_subjects in tasks
        ]
    else:
        fold_outputs = _run_spawned_fold_pool(
            worker_target=_outer_fold_process_main,
            worker_state=worker_state,
            tasks=tasks,
            n_workers=effective_n_jobs,
            gpu_ids=normalized_gpu_ids,
            cpus_per_worker=cpus_per_worker,
            worker_name_prefix="OuterFoldWorker",
            worker_description="outer-fold",
        )

    # Results arrive in completion order, so restore deterministic fold order.
    fold_outputs.sort(key=lambda row: row["outer_fold_number"])
    window_fold_metric_rows: list[dict] = []
    trial_fold_metric_rows: list[dict] = []

    for fold_output in fold_outputs:
        results["fold_metrics"].append(fold_output["fold_metrics"])
        window_fold_metric_rows.append(fold_output["window_fold_metrics"])
        trial_fold_metric_rows.append(fold_output["trial_fold_metrics"])
        results["user_metrics"].extend(fold_output["user_metrics"])
        if log_predictions:
            if feature_array.ndim == 3:
                results["window_prediction_log"].extend(
                    fold_output["window_prediction_log"]
                )
            results["trial_prediction_log"].extend(
                fold_output["trial_prediction_log"]
            )
        if log_variational_intervals:
            if feature_array.ndim == 3:
                results["window_variational_interval_log"].extend(
                    fold_output["window_variational_interval_log"]
                )
            results["trial_variational_interval_log"].extend(
                fold_output["trial_variational_interval_log"]
            )
        results["best_configs"].append(fold_output["best_config_result"])
        results["inner_cv_results"].append(fold_output["inner_cv_result"])
        results["fold_results"].append(fold_output["fold_record"])

    window_mean_scores, window_std_scores = _mean_std_rows(
        window_fold_metric_rows,
        ["loss", "joint_loss", "keras_model_loss", *metrics, "decoder_accuracy"],
    )
    trial_mean_scores, trial_std_scores = _mean_std_rows(
        trial_fold_metric_rows,
        ["loss", "joint_loss", "keras_model_loss", *metrics, "decoder_accuracy"],
    )
    results["window_mean_scores"] = window_mean_scores
    results["window_std_scores"] = window_std_scores
    results["trial_mean_scores"] = trial_mean_scores
    results["trial_std_scores"] = trial_std_scores

    print("\nNested LNSO CV complete")
    print("=" * 80)
    primary_mean_scores = (
        trial_mean_scores if evaluation_level == "trial" else window_mean_scores
    )
    primary_std_scores = (
        trial_std_scores if evaluation_level == "trial" else window_std_scores
    )
    print("Primary-level mean scores:")
    print(pformat(primary_mean_scores, indent=4, width=120, sort_dicts=False))
    print("Primary-level score standard deviations:")
    print(pformat(primary_std_scores, indent=4, width=120, sort_dicts=False))
    print("Window-level mean scores:")
    print(pformat(window_mean_scores, indent=4, width=120, sort_dicts=False))
    print("Trial-level mean scores:")
    print(pformat(trial_mean_scores, indent=4, width=120, sort_dicts=False))

    return results


# ---------------------------------------------------------------------
# Plain Leave-One-Subject-Out cross-validation
# ---------------------------------------------------------------------


def _run_loso_fold(
    fold_number: int,
    test_subject,
    total_folds: int,
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict,
    batch_size: int,
    preprocessing_strategy: Callable | None,
    evaluation_level: Literal["window", "trial"],
    metrics: tuple[str, ...],
    log_predictions: bool,
    log_variational_intervals: bool,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    n_uncertainty_samples: int,
    ci_level: float,
    validation_subjects_per_fold: int,
    validation_seed: int | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_monitor: str,
    early_stopping_mode: Literal["auto", "min", "max"],
    restore_best_weights: bool,
    prediction_diagnostics: bool,
    prediction_diagnostics_every_n_epochs: int,
    prediction_diagnostics_max_samples: int,
    prediction_diagnostics_threshold_tolerance: float,
    prediction_diagnostics_seed: int | None,
    decision_thresholds: tuple[float, ...],
    threshold_selection_metric: str,
    threshold_selection_level: Literal["window", "trial"],
    verbose: int,
    extra_fit_kwargs: dict,
    test_indices_override: np.ndarray | None = None,
    left_out_subjects_override: list | tuple | np.ndarray | None = None,
    held_out_trials: list[dict] | None = None,
    validation_excluded_subjects: list | tuple | np.ndarray | None = None,
    fold_description: str | None = None,
    cv_strategy: str = "loso",
    alternate_subject_sets: bool = False,
    alternating_subject_seed: int | None = 42,
    use_mldg: bool = False,
    mldg_meta_train_subjects: int = 6,
    mldg_meta_test_subjects: int = 2,
    mldg_samples_per_subject: int = 4,
    mldg_seed: int | None = 42,
) -> dict:
    """Train and evaluate one LOSO fold with optional seeded validation.

    The LOSO test subject is never used by ``model.fit``. When
    ``validation_subjects_per_fold`` is positive, that many subjects are drawn
    deterministically from the outer-training pool and excluded from gradient
    updates. They provide ``validation_data`` for early stopping without adding
    another model fit.
    """
    if test_indices_override is None:
        test_mask = subject_id_array == test_subject
        test_indices = np.where(test_mask)[0]
    else:
        test_indices = np.asarray(test_indices_override, dtype=np.int64).reshape(-1)
        if len(test_indices) == 0:
            raise ValueError("test_indices_override must contain at least one index.")
        if np.any(test_indices < 0) or np.any(test_indices >= len(feature_array)):
            raise ValueError(
                "test_indices_override contains an index outside the feature array."
            )
        test_indices = np.unique(test_indices)
        test_mask = np.zeros(len(feature_array), dtype=bool)
        test_mask[test_indices] = True

    outer_train_mask = ~test_mask
    outer_train_indices = np.where(outer_train_mask)[0]

    if len(outer_train_indices) == 0 or len(test_indices) == 0:
        split_name = "LOSO" if test_indices_override is None else cv_strategy
        raise ValueError(
            f"Invalid {split_name} split: train={len(outer_train_indices)}, "
            f"test={len(test_indices)} samples."
        )

    outer_train_subjects = np.sort(
        np.unique(subject_id_array[outer_train_indices])
    )
    validation_candidate_subjects = outer_train_subjects
    if validation_excluded_subjects is not None:
        validation_candidate_subjects = np.setdiff1d(
            outer_train_subjects,
            np.asarray(validation_excluded_subjects),
            assume_unique=False,
        )
    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if alternate_subject_sets and use_mldg:
        raise ValueError(
            "alternate_subject_sets and use_mldg are mutually exclusive."
        )
    if mldg_meta_train_subjects < 1 or mldg_meta_test_subjects < 1:
        raise ValueError("MLDG A/B subject counts must both be at least 1.")
    if mldg_samples_per_subject < 1:
        raise ValueError("mldg_samples_per_subject must be at least 1.")
    if mldg_seed is not None and mldg_seed < 0:
        raise ValueError("mldg_seed must be >= 0 or None.")
    if alternate_subject_sets and validation_subjects_per_fold != 0:
        raise ValueError(
            "alternate_subject_sets uses all non-test subjects and therefore "
            "requires validation_subjects_per_fold=0."
        )
    if (
        validation_subjects_per_fold > 0
        and validation_subjects_per_fold >= len(validation_candidate_subjects)
    ):
        raise ValueError(
            "validation_subjects_per_fold must leave at least one eligible "
            "subject outside validation. Got "
            f"{validation_subjects_per_fold} validation subjects from "
            f"{len(validation_candidate_subjects)} eligible subjects."
        )

    if validation_subjects_per_fold > 0:
        base_seed = 0 if validation_seed is None else int(validation_seed)
        fold_seed = np.random.SeedSequence([base_seed, int(fold_number)])
        rng = np.random.default_rng(fold_seed)
        validation_subjects = np.sort(
            rng.choice(
                validation_candidate_subjects,
                size=validation_subjects_per_fold,
                replace=False,
            )
        )
        validation_mask_relative = np.isin(
            subject_id_array[outer_train_indices],
            validation_subjects,
        )
        validation_indices = outer_train_indices[validation_mask_relative]
        fit_train_indices = outer_train_indices[~validation_mask_relative]
    else:
        validation_subjects = np.asarray([], dtype=outer_train_subjects.dtype)
        validation_indices = np.asarray([], dtype=np.int64)
        fit_train_indices = outer_train_indices

    sample_level = "trials" if feature_array.ndim == 4 else "windows"
    if fold_description is None:
        fold_description = (
            f"LOSO test subject={_python_scalar(test_subject)!r} "
            f"(fit_train={len(fit_train_indices)}, "
            f"validation={len(validation_indices)}, "
            f"test={len(test_indices)} {sample_level})"
        )
    else:
        fold_description = (
            f"{fold_description} (fit_train={len(fit_train_indices)}, "
            f"validation={len(validation_indices)}, "
            f"test={len(test_indices)} {sample_level})"
        )
    _print_fold_header(
        fold_number,
        total_folds,
        fold_description,
    )
    if len(validation_subjects):
        print(
            "Seeded validation subjects: "
            f"{[_python_scalar(value) for value in validation_subjects]}",
            flush=True,
        )

    # The current preprocessing callback API supports only one train/eval pair.
    # Refuse an ambiguous three-way fit rather than leaking validation subjects
    # into a fitted transform or fitting inconsistent transforms for val/test.
    if validation_subjects_per_fold > 0 and preprocessing_strategy is not None:
        raise ValueError(
            "Seeded subject-level validation currently requires "
            "preprocessing_strategy=None. Preprocess before loso_cv or extend "
            "the strategy API to transform train/validation/test from one "
            "fold-local fitted state."
        )

    X_fit_train = feature_array[fit_train_indices]
    y_fit_train = label_array[fit_train_indices]
    X_validation = feature_array[validation_indices]
    y_validation = label_array[validation_indices]
    X_test = feature_array[test_indices]
    y_test = label_array[test_indices]

    subject_ids_fit_train = subject_id_array[fit_train_indices]
    subject_ids_validation = subject_id_array[validation_indices]
    subject_ids_test = subject_id_array[test_indices]
    trial_ids_fit_train = trial_id_array[fit_train_indices]
    trial_ids_validation = trial_id_array[validation_indices]
    trial_ids_test = trial_id_array[test_indices]

    if validation_subjects_per_fold == 0:
        X_fit_train, y_fit_train, X_test, y_test = _apply_preprocessing_strategy(
            preprocessing_strategy=preprocessing_strategy,
            X_train=X_fit_train,
            y_train=y_fit_train,
            X_eval=X_test,
            y_eval=y_test,
            train_indices=fit_train_indices,
            eval_indices=test_indices,
        )

    _validate_processed_alignment(
        X_fit_train,
        y_fit_train,
        subject_ids_fit_train,
        trial_ids_fit_train,
        "LOSO-fit-training",
    )
    if validation_subjects_per_fold > 0:
        _validate_processed_alignment(
            X_validation,
            y_validation,
            subject_ids_validation,
            trial_ids_validation,
            "LOSO-validation",
        )
    _validate_processed_alignment(
        X_test,
        y_test,
        subject_ids_test,
        trial_ids_test,
        "LOSO-test",
    )

    model_hp, fit_hp = _split_config(fixed_config)
    current_batch_size = int(fit_hp.get("batch_size", batch_size))

    duplicate_fit_keys = set(fit_hp).intersection(extra_fit_kwargs)
    if duplicate_fit_keys:
        raise ValueError(
            "The following model.fit arguments were supplied in both the fixed "
            f"configuration and extra_fit_kwargs: {sorted(duplicate_fit_keys)}"
        )

    fit_call_kwargs = dict(extra_fit_kwargs)
    callbacks = list(fit_call_kwargs.pop("callbacks", []))
    prediction_diagnostics_callback: PredictionDiagnostics | None = None

    if prediction_diagnostics:
        prediction_diagnostics_callback = PredictionDiagnostics(
            X_train=X_fit_train,
            y_train=y_fit_train,
            X_val=(X_validation if validation_subjects_per_fold > 0 else None),
            y_val=(y_validation if validation_subjects_per_fold > 0 else None),
            fold_number=fold_number,
            batch_size=current_batch_size,
            every_n_epochs=prediction_diagnostics_every_n_epochs,
            max_samples=prediction_diagnostics_max_samples,
            threshold_tolerance=prediction_diagnostics_threshold_tolerance,
            seed=(
                None
                if prediction_diagnostics_seed is None
                else int(prediction_diagnostics_seed) + int(fold_number)
            ),
        )
        callbacks.append(prediction_diagnostics_callback)

    if validation_subjects_per_fold > 0:
        if early_stopping_monitor in {
            "val_trial_f1",
            "val_trial_balanced_accuracy",
            "val_trial_loss",
        }:
            # This callback must run before CompactEpochLogger and EarlyStopping
            # so the custom metric is available to both callbacks.
            callbacks.append(
                TrialValidationMetrics(
                    X_val=X_validation,
                    y_val=y_validation,
                    subject_ids_val=subject_ids_validation,
                    trial_ids_val=trial_ids_validation,
                    batch_size=current_batch_size,
                )
            )

    if verbose:
        callbacks.append(CompactEpochLogger(fold_number=fold_number))

    if validation_subjects_per_fold > 0 and early_stopping_patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                patience=int(early_stopping_patience),
                min_delta=float(early_stopping_min_delta),
                mode=early_stopping_mode,
                restore_best_weights=bool(restore_best_weights),
                verbose=1 if verbose else 0,
            )
        )

    if callbacks:
        fit_call_kwargs["callbacks"] = callbacks

    tf.keras.backend.clear_session()
    model = model_builder_function(**model_hp)
    X_fit_train_for_fit = _prepare_fit_inputs_with_subject_ids(
        model,
        X_fit_train,
        subject_ids_fit_train,
    )

    epochs_ran = 0
    best_epoch: int | None = None
    best_monitored_value: float | None = None
    stopped_early = False

    try:
        y_fit_train_ids = _as_numpy_1d(y_fit_train)
        classes, counts = np.unique(y_fit_train_ids, return_counts=True)

        class_weight = {
            int(class_id): len(y_fit_train_ids) / (len(classes) * count)
            for class_id, count in zip(classes, counts)
        }

        validation_data = (
            (X_validation, y_validation)
            if validation_subjects_per_fold > 0
            else None
        )
        if use_mldg:
            fold_mldg_seed = (
                None
                if mldg_seed is None
                else int(mldg_seed) + int(fold_number)
            )
            effective_class_weight = (
                class_weight
                if bool(getattr(model, "use_class_weight", True))
                else None
            )
            mldg_sequence = MetaLearningSubjectSequence(
                X=X_fit_train,
                y=y_fit_train,
                subject_ids=subject_ids_fit_train,
                model=model,
                meta_train_subjects=mldg_meta_train_subjects,
                meta_test_subjects=mldg_meta_test_subjects,
                samples_per_subject=mldg_samples_per_subject,
                class_weight=effective_class_weight,
                seed=fold_mldg_seed,
            )
            print(
                "First-order MLDG episodes (natural within-subject labels): "
                f"A_subjects={mldg_meta_train_subjects}, "
                f"B_subjects={mldg_meta_test_subjects}, "
                f"samples_per_subject={mldg_samples_per_subject}, "
                f"steps_per_epoch={len(mldg_sequence)}",
                flush=True,
            )
            subject_set_a = np.asarray([], dtype=subject_ids_fit_train.dtype)
            subject_set_b = np.asarray([], dtype=subject_ids_fit_train.dtype)
            history = model.fit(
                mldg_sequence,
                validation_data=validation_data,
                verbose=0,
                **fit_hp,
                **fit_call_kwargs,
            )
        elif alternate_subject_sets:
            if validation_subjects_per_fold > 0:
                raise ValueError(
                    "alternate_subject_sets requires validation_subjects_per_fold=0."
                )
            fold_alt_seed = (
                None
                if alternating_subject_seed is None
                else int(alternating_subject_seed) + int(fold_number)
            )
            subject_set_a, subject_set_b = _balanced_two_subject_sets(
                subject_ids_fit_train,
                y_fit_train,
                seed=fold_alt_seed,
            )
            print(
                "Alternating subject sets: "
                f"A={[_python_scalar(v) for v in subject_set_a]} | "
                f"B={[_python_scalar(v) for v in subject_set_b]}",
                flush=True,
            )
            alternating_sequence = AlternatingSubjectSetSequence(
                X=X_fit_train,
                y=y_fit_train,
                subject_ids=subject_ids_fit_train,
                subject_set_a=subject_set_a,
                subject_set_b=subject_set_b,
                batch_size=current_batch_size,
                model=model,
                class_weight=class_weight,
                seed=fold_alt_seed,
            )
            history = model.fit(
                alternating_sequence,
                validation_data=None,
                verbose=0,
                **fit_hp,
                **fit_call_kwargs,
            )
        else:
            subject_set_a = np.asarray([], dtype=subject_ids_fit_train.dtype)
            subject_set_b = np.asarray([], dtype=subject_ids_fit_train.dtype)
            history = model.fit(
                X_fit_train_for_fit,
                y_fit_train,
                validation_data=validation_data,
                class_weight=class_weight,
                verbose=0,
                **fit_hp,
                **fit_call_kwargs,
            )

        epochs_ran = int(len(history.history.get("loss", [])))
        requested_epochs = int(fit_hp.get("epochs", epochs_ran))
        stopped_early = bool(epochs_ran < requested_epochs)

        monitored_history = history.history.get(early_stopping_monitor)
        if monitored_history:
            monitored_values = np.asarray(monitored_history, dtype=np.float64)
            finite_mask = np.isfinite(monitored_values)
            if np.any(finite_mask):
                candidate_indices = np.where(finite_mask)[0]
                candidate_values = monitored_values[finite_mask]
                if early_stopping_mode == "max":
                    local_best = int(np.argmax(candidate_values))
                elif early_stopping_mode == "min":
                    local_best = int(np.argmin(candidate_values))
                else:
                    maximize_tokens = (
                        "acc", "auc", "f1", "precision", "recall"
                    )
                    maximize = any(
                        token in early_stopping_monitor.lower()
                        for token in maximize_tokens
                    )
                    local_best = int(
                        np.argmax(candidate_values)
                        if maximize
                        else np.argmin(candidate_values)
                    )
                best_index = int(candidate_indices[local_best])
                best_epoch = best_index + 1
                best_monitored_value = float(monitored_values[best_index])
        if best_epoch is None and epochs_ran > 0:
            best_epoch = epochs_ran

        selected_decision_threshold = float(decision_thresholds[0])
        threshold_validation_score: float | None = None
        threshold_search_results: list[dict] = []
        if validation_subjects_per_fold > 0:
            validation_probabilities = _predict_probabilities(
                model=model,
                X=X_validation,
                batch_size=current_batch_size,
                n_prediction_latent_samples=n_prediction_latent_samples,
                latent_sampling_seed=latent_sampling_seed,
            )
            if threshold_selection_level == "trial":
                if _is_trial_tensor(X_validation):
                    threshold_validation = _direct_trial_aggregation(
                        probabilities=validation_probabilities,
                        y_true=y_validation,
                        subject_ids=subject_ids_validation,
                        trial_ids=trial_ids_validation,
                        n_windows_per_trial=X_validation.shape[1],
                    )
                else:
                    threshold_validation = _aggregate_window_probabilities_by_trial(
                        probabilities=validation_probabilities,
                        y_true=y_validation,
                        subject_ids=subject_ids_validation,
                        trial_ids=trial_ids_validation,
                    )
                threshold_probabilities = threshold_validation["probabilities"]
                threshold_y_true = threshold_validation["y_true"]
            else:
                threshold_probabilities = validation_probabilities
                threshold_y_true = _as_numpy_1d(y_validation).astype(np.int64)

            (
                selected_decision_threshold,
                threshold_validation_score,
                threshold_search_results,
            ) = _select_binary_decision_threshold(
                probabilities=threshold_probabilities,
                y_true=threshold_y_true,
                thresholds=decision_thresholds,
                metric=threshold_selection_metric,
            )
            print(
                f"Fold {fold_number} selected decision threshold "
                f"{selected_decision_threshold:.4f} from validation "
                f"{threshold_selection_level}_{threshold_selection_metric}="
                f"{threshold_validation_score:.6f}",
                flush=True,
            )

        evaluation = _evaluate_classification_fold(
            model=model,
            X_test=X_test,
            y_test=y_test,
            subject_ids_test=subject_ids_test,
            trial_ids_test=trial_ids_test,
            fold_index=fold_number,
            metrics=metrics,
            evaluation_level=evaluation_level,
            batch_size=current_batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=selected_decision_threshold,
        )
    finally:
        del model
        gc.collect()
        tf.keras.backend.clear_session()

    def count_trials(subject_ids: np.ndarray, trial_ids: np.ndarray) -> int:
        return int(len(set(zip(subject_ids.tolist(), trial_ids.tolist()))))

    subject_ids_outer_train = subject_id_array[outer_train_indices]
    trial_ids_outer_train = trial_id_array[outer_train_indices]

    if left_out_subjects_override is None:
        left_out_subjects = [_python_scalar(test_subject)]
        left_out_subject = _python_scalar(test_subject)
    else:
        left_out_subjects = [
            _python_scalar(value)
            for value in np.asarray(left_out_subjects_override).reshape(-1).tolist()
        ]
        left_out_subject = (
            left_out_subjects[0] if len(left_out_subjects) == 1 else None
        )

    fold_record = {
        "fold_number": int(fold_number),
        "left_out_subjects": left_out_subjects,
        "held_out_trials": [] if held_out_trials is None else held_out_trials,
        "validation_subjects": [
            _python_scalar(value) for value in validation_subjects.tolist()
        ],
        "n_train_windows": _count_windows_for_indices(feature_array, outer_train_indices),
        "n_fit_train_windows": _count_windows_for_indices(feature_array, fit_train_indices),
        "n_validation_windows": _count_windows_for_indices(feature_array, validation_indices),
        "n_test_windows": _count_windows_for_indices(feature_array, test_indices),
        "n_train_trials": count_trials(
            subject_ids_outer_train, trial_ids_outer_train
        ),
        "n_fit_train_trials": count_trials(
            subject_ids_fit_train, trial_ids_fit_train
        ),
        "n_validation_trials": count_trials(
            subject_ids_validation, trial_ids_validation
        ),
        "n_test_trials": count_trials(subject_ids_test, trial_ids_test),
        "epochs_ran": int(epochs_ran),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_monitored_value": best_monitored_value,
        "stopped_early": bool(stopped_early),
        "decision_threshold": float(selected_decision_threshold),
        "alternate_subject_sets": bool(alternate_subject_sets),
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects) if use_mldg else 0,
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects) if use_mldg else 0,
        "mldg_samples_per_subject": int(mldg_samples_per_subject) if use_mldg else 0,
        "subject_set_a": [
            _python_scalar(value) for value in subject_set_a.tolist()
        ] if alternate_subject_sets else [],
        "subject_set_b": [
            _python_scalar(value) for value in subject_set_b.tolist()
        ] if alternate_subject_sets else [],
    }

    prediction_diagnostics_log = (
        []
        if prediction_diagnostics_callback is None
        else list(prediction_diagnostics_callback.history)
    )

    return {
        "outer_fold_number": int(fold_number),
        "fold_record": fold_record,
        "prediction_diagnostics_log": prediction_diagnostics_log,
        **evaluation,
    }

def _loso_fold_process_main(
    worker_state_payload: bytes,
    task_queue,
    result_queue,
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None,
) -> None:
    """Run ordinary LOSO folds in one persistent spawned process."""
    try:
        _configure_tensorflow_worker(
            gpu_id=gpu_id,
            cpus_per_worker=cpus_per_worker,
            assigned_device_label=assigned_device_label,
        )
        worker_state = cloudpickle.loads(worker_state_payload)

        while True:
            task = task_queue.get()

            if task is None:
                return

            fold_number, test_subject = task

            try:
                fold_output = _run_loso_fold(
                    fold_number=fold_number,
                    test_subject=test_subject,
                    **worker_state,
                )
                result_queue.put(("ok", int(fold_number), fold_output))
            except BaseException:
                result_queue.put(
                    (
                        "error",
                        int(fold_number),
                        traceback.format_exc(),
                    )
                )
                return

    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))
    finally:
        tf.keras.backend.clear_session()
        gc.collect()

def _compact_loso_training_result(fold_output: dict) -> dict:
    """Return the small per-fold training summary for one configuration."""
    fold_record = fold_output["fold_record"]
    return {
        "fold_number": int(fold_record["fold_number"]),
        "epochs_ran": int(fold_record["epochs_ran"]),
        "best_epoch": fold_record["best_epoch"],
        "best_monitored_value": fold_record["best_monitored_value"],
        "stopped_early": bool(fold_record["stopped_early"]),
        "decision_threshold": float(fold_record["decision_threshold"]),
    }

def _aggregate_loso_config_result(
    config_index: int,
    config: dict,
    fold_outputs: list[dict],
    metrics: tuple[str, ...],
    selection_metric: str,
    selection_level: Literal["window", "trial"],
) -> dict:
    """Aggregate a complete LOSO evaluation for one configuration."""
    fold_outputs = sorted(
        fold_outputs,
        key=lambda row: int(row["outer_fold_number"]),
    )

    fold_metrics = [dict(row["fold_metrics"]) for row in fold_outputs]
    window_fold_metrics = [
        dict(row["window_fold_metrics"]) for row in fold_outputs
    ]
    trial_fold_metrics = [
        dict(row["trial_fold_metrics"]) for row in fold_outputs
    ]

    mean_scores, std_scores = _mean_std_rows(
        fold_metrics,
        ["loss", "joint_loss", "keras_model_loss", *metrics, "decoder_accuracy"],
    )
    window_mean_scores, window_std_scores = _mean_std_rows(
        window_fold_metrics,
        ["loss", "joint_loss", "keras_model_loss", *metrics, "decoder_accuracy"],
    )
    trial_mean_scores, trial_std_scores = _mean_std_rows(
        trial_fold_metrics,
        ["loss", "joint_loss", "keras_model_loss", *metrics, "decoder_accuracy"],
    )

    selection_means = (
        trial_mean_scores if selection_level == "trial" else window_mean_scores
    )
    selection_stds = (
        trial_std_scores if selection_level == "trial" else window_std_scores
    )

    if selection_metric not in selection_means:
        raise ValueError(
            f"Selection metric {selection_metric!r} was not produced for "
            f"configuration {config_index}. Available metrics: "
            f"{sorted(selection_means)}"
        )

    return {
        "config_index": int(config_index),
        "config": dict(config),
        "selection_score": float(selection_means[selection_metric]),
        "selection_score_std": float(selection_stds[selection_metric]),
        "window_mean_scores": window_mean_scores,
        "window_std_scores": window_std_scores,
        "trial_mean_scores": trial_mean_scores,
        "trial_std_scores": trial_std_scores,
        "fold_metrics": fold_metrics,
        "fold_training": [
            _compact_loso_training_result(row) for row in fold_outputs
        ],
    }


def _loso_config_sort_key(
    config_result: dict,
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    maximize_metric: bool,
) -> tuple[float, float, float, int]:
    """Return a deterministic ranking key for flat LOSO grid search.

    The primary criterion is the mean selected metric across held-out subjects.
    Ties are resolved by lower between-subject standard deviation, then lower
    mean log loss, then the earlier configuration index.
    """
    mean_key = f"{selection_level}_mean_scores"
    std_key = f"{selection_level}_std_scores"
    mean_scores = config_result[mean_key]
    std_scores = config_result[std_key]

    primary = float(mean_scores[selection_metric])
    primary_std = float(std_scores[selection_metric])
    mean_loss = float(mean_scores.get("loss", np.inf))

    if not np.isfinite(primary):
        primary_rank = np.inf
    else:
        primary_rank = -primary if maximize_metric else primary

    if not np.isfinite(primary_std):
        primary_std = np.inf
    if not np.isfinite(mean_loss):
        mean_loss = np.inf

    return (
        float(primary_rank),
        float(primary_std),
        float(mean_loss),
        int(config_result["config_index"]),
    )


def _choose_best_loso_config_index(
    config_results: list[dict],
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    maximize_metric: bool,
) -> int:
    """Choose the global configuration after every config completes LOSO."""
    if not config_results:
        raise ValueError("No LOSO configuration results were produced.")

    best_result = min(
        config_results,
        key=lambda row: _loso_config_sort_key(
            config_result=row,
            selection_metric=selection_metric,
            selection_level=selection_level,
            maximize_metric=maximize_metric,
        ),
    )

    best_score = float(best_result["selection_score"])
    if not np.isfinite(best_score):
        raise RuntimeError(
            "All LOSO configurations produced a non-finite selection score."
        )

    return int(best_result["config_index"])


def loso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray | None = None,
    n_epochs: int = 50,
    batch_size: int = 2,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    evaluation_level: Literal["window", "trial"] = "trial",
    selection_metric: str = "f1",
    selection_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "binary_f1",
        "binary_precision",
        "binary_recall",
        "roc_auc",
    ),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    validation_subjects_per_fold: int = 0,
    validation_seed: int | None = 42,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: Literal["auto", "min", "max"] = "min",
    restore_best_weights: bool = True,
    prediction_diagnostics: bool = False,
    prediction_diagnostics_every_n_epochs: int = 1,
    prediction_diagnostics_max_samples: int = 256,
    prediction_diagnostics_threshold_tolerance: float = 0.01,
    prediction_diagnostics_seed: int | None = 42,
    decision_thresholds: list[float] | tuple[float, ...] = (0.5,),
    threshold_selection_metric: Literal[
        "accuracy", "f1", "balanced_accuracy", "binary_f1"
    ] = "f1",
    threshold_selection_level: Literal["window", "trial"] = "trial",
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
    max_folds: int | None = None,
    alternate_subject_sets: bool = False,
    alternating_subject_seed: int | None = 42,
    use_mldg: bool = False,
    mldg_meta_train_subjects: int = 6,
    mldg_meta_test_subjects: int = 2,
    mldg_samples_per_subject: int = 4,
    mldg_seed: int | None = 42,
) -> dict:
    """Run a flat hyperparameter search using complete LOSO evaluations.

    For every Cartesian-product hyperparameter configuration, each unique
    subject is held out exactly once. The configuration is therefore evaluated
    on the same complete set of subject-wise folds. After all configurations
    finish, one global configuration is selected from its mean LOSO metric.

    This is *not* nested cross-validation: the held-out LOSO results are used
    both to compare configurations and to report the selected configuration's
    cross-validation performance. This behavior is intentional for a practical
    flat LOSO hyperparameter search.

    Hyperparameter grid
    -------------------
    Scalar values may be supplied directly or as candidate lists/tuples. The
    Cartesian product is evaluated with a complete LOSO run per configuration.

    Sequence-valued encoder settings preserve one complete architecture before
    the Cartesian product is expanded. ``sequence_hyperparameter_depths``
    specifies the nesting depth of one value, resolving CNN1D/CNN2D ambiguity
    for keys such as ``kernel_sizes``. GCN ``gcn_units`` and temporal/spatial
    pooling schedules are preserved in the same way. One additional outer list
    level enumerates multiple architecture candidates.

    ``n_epochs`` and ``batch_size`` provide defaults and are overridden when
    ``hyperparameters`` contains ``epochs`` or ``batch_size``.

    Seeded validation and early stopping
    ------------------------------------
    When ``validation_subjects_per_fold`` is positive, that many subjects are
    sampled deterministically from each outer-training pool. They are excluded
    from gradient updates and passed to ``model.fit`` as ``validation_data``.
    The same fold-local validation subjects are reused for every hyperparameter
    configuration, while the LOSO test subject remains untouched. This adds no
    extra fits; it only changes each fit from train/test to train/validation/test.

    Selection
    ---------
    ``selection_level`` determines whether configurations are ranked using
    window- or trial-level scores. Hierarchical rank-4 inputs require trial-level
    selection. For binary tasks, ``selection_metric='f1'`` uses the MTLFuseNet
    convention: class 1 is positive. ``precision`` and ``recall`` follow the same
    convention. Explicit ``macro_*`` metrics and ``balanced_accuracy`` remain
    available for class-balanced diagnostics, while ``roc_auc`` uses the class-1
    probability.
    Classification metrics are maximized; probability loss and joint loss are minimized unless
    ``maximize_metric`` is explicitly supplied. Ties use lower between-subject
    standard deviation, lower mean log loss, then the earlier grid index.

    Returned results
    ----------------
    ``config_results`` contains per-fold and aggregate metrics for every
    configuration. Top-level prediction logs, user metrics, and fold metadata
    correspond only to the globally selected configuration. Selected fold
    metrics remain available through ``config_results[best_config_index]``.

    Concurrency
    -----------
    LOSO folds for one configuration run concurrently. The next configuration
    starts after the current configuration's folds complete. With one worker per
    GPU, this prevents multiple models from competing for the same GPU while
    bounding parent-process memory to approximately one configuration's logs.

    Smoke testing
    -------------
    ``max_folds`` deterministically limits every configuration to the first N
    sorted subjects. Leave it as ``None`` for complete LOSO evaluation.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    if "validation_data" in extra_fit_kwargs:
        raise ValueError(
            "Do not pass a fixed validation_data array to loso_cv. It would not "
            "be reconstructed fold-locally and could create leakage."
        )

    if subject_id_array is None:
        raise ValueError("subject_id_array is required for LOSO CV.")
    if trial_id_array is None:
        raise ValueError(
            "trial_id_array is required for trial-level prediction and metrics. "
            "Pass one trial ID per sample, aligned with feature_array."
        )

    _validate_evaluation_level(evaluation_level, "evaluation_level")
    _validate_evaluation_level(selection_level, "selection_level")

    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    if feature_array.ndim not in {3, 4}:
        raise ValueError(
            "feature_array must be rank 3 for window samples or rank 4 for "
            f"grouped trial samples; got {feature_array.shape}."
        )
    if feature_array.ndim == 4:
        if selection_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require selection_level='trial'."
            )
        if evaluation_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require evaluation_level='trial'."
            )

    input_lengths = (
        len(feature_array),
        len(label_array),
        len(subject_id_array),
        len(trial_id_array),
    )
    if len(set(input_lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            "must have the same first dimension. Got lengths "
            f"{input_lengths}."
        )

    metrics = tuple(metrics)
    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. Supported metrics: "
                f"{sorted(_CLASSIFICATION_METRICS)}"
            )

    allowed_selection_metrics = {"loss", "joint_loss", *metrics}
    if selection_metric not in allowed_selection_metrics:
        raise ValueError(
            f"selection_metric={selection_metric!r} is unavailable. "
            f"Use 'loss', 'joint_loss', or one of metrics={list(metrics)}."
        )

    if (
        feature_array.ndim == 3
        and selection_metric == "joint_loss"
        and selection_level != "window"
    ):
        raise ValueError(
            "Window-level models require selection_level='window' when "
            "selecting by joint_loss. Hierarchical rank-4 models may select "
            "trial-level joint_loss directly."
        )

    if maximize_metric is None:
        maximize_metric = selection_metric not in {"loss", "joint_loss"}

    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")

    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")

    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if alternate_subject_sets and use_mldg:
        raise ValueError(
            "alternate_subject_sets and use_mldg are mutually exclusive."
        )
    if mldg_meta_train_subjects < 1 or mldg_meta_test_subjects < 1:
        raise ValueError("MLDG A/B subject counts must both be at least 1.")
    if mldg_samples_per_subject < 1:
        raise ValueError("mldg_samples_per_subject must be at least 1.")
    if mldg_seed is not None and mldg_seed < 0:
        raise ValueError("mldg_seed must be >= 0 or None.")
    if validation_seed is not None and validation_seed < 0:
        raise ValueError("validation_seed must be >= 0 or None.")
    if early_stopping_patience is not None and early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0 or None.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be >= 0.")
    if early_stopping_mode not in {"auto", "min", "max"}:
        raise ValueError(
            "early_stopping_mode must be 'auto', 'min', or 'max'."
        )
    if not early_stopping_monitor:
        raise ValueError("early_stopping_monitor must be a non-empty string.")
    if prediction_diagnostics_every_n_epochs < 1:
        raise ValueError(
            "prediction_diagnostics_every_n_epochs must be at least 1."
        )
    if prediction_diagnostics_max_samples < 1:
        raise ValueError("prediction_diagnostics_max_samples must be at least 1.")
    if prediction_diagnostics_threshold_tolerance < 0.0:
        raise ValueError(
            "prediction_diagnostics_threshold_tolerance must be non-negative."
        )
    decision_thresholds = _normalize_decision_thresholds(decision_thresholds)
    if threshold_selection_metric not in {
        "accuracy", "f1", "balanced_accuracy", "binary_f1"
    }:
        raise ValueError(
            "threshold_selection_metric must be accuracy, f1, "
            "balanced_accuracy, or binary_f1."
        )
    _validate_evaluation_level(
        threshold_selection_level,
        "threshold_selection_level",
    )
    if len(decision_thresholds) > 1 and validation_subjects_per_fold == 0:
        raise ValueError(
            "Testing multiple decision thresholds requires fold-local validation "
            "subjects. Set validation_subjects_per_fold >= 1."
        )
    if (
        early_stopping_monitor in {
            "val_trial_f1",
            "val_trial_balanced_accuracy",
            "val_trial_loss",
        }
        and validation_subjects_per_fold == 0
        and early_stopping_patience is not None
    ):
        raise ValueError(
            f"{early_stopping_monitor} requires at least one fold-local "
            "validation subject. Set validation_subjects_per_fold >= 1."
        )

    unique_subjects = np.sort(np.unique(subject_id_array))
    if len(unique_subjects) < 2:
        raise ValueError(
            "LOSO CV requires at least two unique subjects. "
            f"Got {len(unique_subjects)}."
        )
    if validation_subjects_per_fold >= len(unique_subjects) - 1:
        raise ValueError(
            "validation_subjects_per_fold must leave at least one gradient-"
            "training subject after the LOSO test subject is removed. Got "
            f"{validation_subjects_per_fold} validation subjects for "
            f"{len(unique_subjects)} total subjects."
        )
    if use_mldg:
        available_mldg_subjects = (
            len(unique_subjects) - 1 - validation_subjects_per_fold
        )
        required_mldg_subjects = (
            int(mldg_meta_train_subjects) + int(mldg_meta_test_subjects)
        )
        if required_mldg_subjects > available_mldg_subjects:
            raise ValueError(
                "MLDG requires "
                f"{required_mldg_subjects} episodic subjects, but each LOSO "
                f"fold leaves only {available_mldg_subjects} gradient-training "
                "subjects after removing test and validation subjects."
            )

    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max_folds must be >= 1 when provided.")
        test_subjects = unique_subjects[: min(max_folds, len(unique_subjects))]
    else:
        test_subjects = unique_subjects

    effective_hyperparameters = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        **(hyperparameters or {}),
    }
    sequence_hyperparameter_depths = getattr(
        model_builder_function,
        "_sequence_hyperparameter_depths",
        None,
    )
    grid_configs = _expand_hyperparameter_grid(
        effective_hyperparameters,
        sequence_hyperparameter_depths=sequence_hyperparameter_depths,
    )
    _warn_if_joint_loss_weights_vary(grid_configs, selection_metric)
    if not grid_configs:
        raise ValueError("The hyperparameter grid produced no configurations.")

    total_folds = len(test_subjects)
    total_model_fits = len(grid_configs) * total_folds
    effective_n_jobs = min(n_jobs, total_folds)

    normalized_gpu_ids: tuple[int, ...] | None = None
    if gpu_ids is None and effective_n_jobs > 1:
        normalized_gpu_ids = _auto_assign_gpu_ids(effective_n_jobs)
        if normalized_gpu_ids is not None:
            effective_n_jobs = len(normalized_gpu_ids)
    elif gpu_ids is not None:
        normalized_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)

        if not normalized_gpu_ids:
            raise ValueError("gpu_ids must contain at least one GPU index.")
        if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
            raise ValueError("gpu_ids must not contain duplicate GPU indices.")
        if effective_n_jobs > len(normalized_gpu_ids):
            raise ValueError(
                f"n_jobs={effective_n_jobs} requires at least that many GPU IDs, "
                f"but gpu_ids={normalized_gpu_ids}. Use one GPU per worker."
            )

        normalized_gpu_ids = normalized_gpu_ids[:effective_n_jobs]

    print(
        f"\nFlat LOSO hyperparameter search — {len(grid_configs)} "
        f"configuration{'s' if len(grid_configs) != 1 else ''}, "
        f"{total_folds} fold{'s' if total_folds != 1 else ''} each"
    )
    print(f"Total available subjects: {len(unique_subjects)}")
    print(f"Total LOSO model fits: {total_model_fits}")
    if max_folds is not None:
        print(
            f"Smoke-test fold limit: {total_folds} of "
            f"{len(unique_subjects)} subjects per configuration"
        )
    print(f"Requested metrics: {list(metrics)}")
    print(
        f"Configuration selection: {selection_level}-level "
        f"{selection_metric} "
        f"({'maximize' if maximize_metric else 'minimize'})"
    )
    print(f"Primary reported metrics: {evaluation_level}-level")
    print(f"Prediction logging: {log_predictions}")
    print(f"Prediction diagnostics: {prediction_diagnostics}")
    print(f"Variational interval logging: {log_variational_intervals}")
    print(
        "Decision thresholds: "
        f"{list(decision_thresholds)}; selection="
        f"{threshold_selection_level}_{threshold_selection_metric}"
    )
    prediction_mode = (
        "posterior mean"
        if n_prediction_latent_samples == 0
        else f"MC average over {n_prediction_latent_samples} latent sample(s)"
    )
    print(f"Prediction latent mode: {prediction_mode}")
    if validation_subjects_per_fold > 0:
        print(
            "Per-fold validation: "
            f"{validation_subjects_per_fold} seeded subject(s), "
            f"seed={validation_seed}, monitor={early_stopping_monitor}, "
            f"patience={early_stopping_patience}, "
            f"restore_best_weights={restore_best_weights}"
        )
    else:
        print("Per-fold validation: disabled")
    if use_mldg:
        print(
            "Optimization: first-order MLDG with natural within-subject labels "
            f"(A={mldg_meta_train_subjects} subjects, "
            f"B={mldg_meta_test_subjects} subjects, "
            f"samples/subject={mldg_samples_per_subject}, seed={mldg_seed})"
        )
    elif alternate_subject_sets:
        print("Optimization: alternating fixed subject sets")
    else:
        print("Optimization: ordinary shuffled minibatches")
    print(f"Fold workers: {effective_n_jobs}")

    if effective_n_jobs > 1 and normalized_gpu_ids is None:
        print("Worker devices: CPU-only")
    elif normalized_gpu_ids is not None:
        print(f"Worker devices: GPUs {list(normalized_gpu_ids)}")
    else:
        print("Worker device: current TensorFlow default")

    tasks = [
        (fold_number, _python_scalar(test_subject))
        for fold_number, test_subject in enumerate(test_subjects, start=1)
    ]

    common_worker_state = {
        "total_folds": total_folds,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "batch_size": batch_size,
        "preprocessing_strategy": preprocessing_strategy,
        "evaluation_level": evaluation_level,
        "metrics": metrics,
        "log_predictions": log_predictions,
        "log_variational_intervals": log_variational_intervals,
        "n_prediction_latent_samples": n_prediction_latent_samples,
        "latent_sampling_seed": latent_sampling_seed,
        "n_uncertainty_samples": n_uncertainty_samples,
        "ci_level": ci_level,
        "validation_subjects_per_fold": validation_subjects_per_fold,
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "restore_best_weights": restore_best_weights,
        "prediction_diagnostics": bool(prediction_diagnostics),
        "prediction_diagnostics_every_n_epochs": int(
            prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": int(
            prediction_diagnostics_max_samples
        ),
        "prediction_diagnostics_threshold_tolerance": float(
            prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": prediction_diagnostics_seed,
        "decision_thresholds": decision_thresholds,
        "threshold_selection_metric": threshold_selection_metric,
        "threshold_selection_level": threshold_selection_level,
        "verbose": verbose,
        "extra_fit_kwargs": extra_fit_kwargs,
        "alternate_subject_sets": bool(alternate_subject_sets),
        "alternating_subject_seed": alternating_subject_seed,
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects),
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects),
        "mldg_samples_per_subject": int(mldg_samples_per_subject),
        "mldg_seed": mldg_seed,
    }

    config_results: list[dict] = []
    best_so_far_result: dict | None = None
    best_fold_outputs: list[dict] | None = None

    for config_index, config in enumerate(grid_configs):
        print("\n" + "#" * 80)
        print(
            f"Configuration {config_index + 1} / {len(grid_configs)} "
            f"({total_folds} LOSO fits)"
        )
        _print_config("Configuration:", config)

        worker_state = {
            **common_worker_state,
            "fixed_config": config,
        }

        if effective_n_jobs == 1 and normalized_gpu_ids is None:
            fold_outputs = [
                _run_loso_fold(
                    fold_number=fold_number,
                    test_subject=test_subject,
                    **worker_state,
                )
                for fold_number, test_subject in tasks
            ]
        else:
            fold_outputs = _run_spawned_fold_pool(
                worker_target=_loso_fold_process_main,
                worker_state=worker_state,
                tasks=tasks,
                n_workers=effective_n_jobs,
                gpu_ids=normalized_gpu_ids,
                cpus_per_worker=cpus_per_worker,
                worker_name_prefix=f"LOSOConfig{config_index + 1}Worker",
                worker_description=(
                    f"LOSO-fold for configuration {config_index + 1}"
                ),
            )

        fold_outputs.sort(key=lambda row: row["outer_fold_number"])
        config_result = _aggregate_loso_config_result(
            config_index=config_index,
            config=config,
            fold_outputs=fold_outputs,
            metrics=metrics,
            selection_metric=selection_metric,
            selection_level=selection_level,
        )
        config_results.append(config_result)

        if (
            best_so_far_result is None
            or _loso_config_sort_key(
                config_result=config_result,
                selection_metric=selection_metric,
                selection_level=selection_level,
                maximize_metric=bool(maximize_metric),
            )
            < _loso_config_sort_key(
                config_result=best_so_far_result,
                selection_metric=selection_metric,
                selection_level=selection_level,
                maximize_metric=bool(maximize_metric),
            )
        ):
            best_so_far_result = config_result
            best_fold_outputs = fold_outputs

        print(
            f"\nConfiguration {config_index + 1} complete: "
            f"mean {selection_level}_{selection_metric}="
            f"{config_result['selection_score']:.6f} ± "
            f"{config_result['selection_score_std']:.6f}",
            flush=True,
        )

    best_config_index = _choose_best_loso_config_index(
        config_results=config_results,
        selection_metric=selection_metric,
        selection_level=selection_level,
        maximize_metric=bool(maximize_metric),
    )
    best_config_result = config_results[best_config_index]
    best_config = dict(best_config_result["config"])

    if (
        best_so_far_result is None
        or best_fold_outputs is None
        or int(best_so_far_result["config_index"]) != best_config_index
    ):
        raise RuntimeError(
            "Internal LOSO grid-search error: selected configuration logs "
            "were not retained correctly."
        )

    # Surface one canonical copy of each selected-configuration artifact.
    results = {
        "cv_strategy": "flat_loso_hyperparameter_search",
        "hyperparameter_search": True,
        "n_configs": int(len(grid_configs)),
        "n_subjects": int(len(unique_subjects)),
        "n_evaluated_folds_per_config": int(total_folds),
        "n_total_loso_fits": int(total_model_fits),
        "max_folds": max_folds,
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "maximize_metric": bool(maximize_metric),
        "selection_score": float(best_config_result["selection_score"]),
        "selection_score_std": float(best_config_result["selection_score_std"]),
        "n_prediction_latent_samples": int(n_prediction_latent_samples),
        "latent_sampling_seed": latent_sampling_seed,
        "validation_subjects_per_fold": int(validation_subjects_per_fold),
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "restore_best_weights": bool(restore_best_weights),
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects) if use_mldg else 0,
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects) if use_mldg else 0,
        "mldg_samples_per_subject": int(mldg_samples_per_subject) if use_mldg else 0,
        "mldg_seed": mldg_seed if use_mldg else None,
        "config_results": config_results,
        "best_config_index": int(best_config_index),
        "best_config": best_config,
        "user_metrics": [],
        "fold_results": [],
    }
    if log_predictions:
        if feature_array.ndim == 3:
            results["window_prediction_log"] = []
        results["trial_prediction_log"] = []
    if log_variational_intervals:
        if feature_array.ndim == 3:
            results["window_variational_interval_log"] = []
        results["trial_variational_interval_log"] = []
    if prediction_diagnostics:
        results["prediction_diagnostics_log"] = []

    for fold_output in best_fold_outputs:
        results["user_metrics"].extend(fold_output["user_metrics"])
        if log_predictions:
            if feature_array.ndim == 3:
                results["window_prediction_log"].extend(
                    fold_output["window_prediction_log"]
                )
            results["trial_prediction_log"].extend(
                fold_output["trial_prediction_log"]
            )
        if log_variational_intervals:
            if feature_array.ndim == 3:
                results["window_variational_interval_log"].extend(
                    fold_output["window_variational_interval_log"]
                )
            results["trial_variational_interval_log"].extend(
                fold_output["trial_variational_interval_log"]
            )
        if prediction_diagnostics:
            results["prediction_diagnostics_log"].extend(
                fold_output.get("prediction_diagnostics_log", [])
            )
        results["fold_results"].append(dict(fold_output["fold_record"]))

    print("\nFlat LOSO hyperparameter search complete")
    print("=" * 80)
    print(
        f"Selected configuration {best_config_index + 1} / "
        f"{len(grid_configs)} using {selection_level}-level "
        f"{selection_metric}."
    )
    _print_config("Best configuration:", best_config)
    print(
        f"Selection score: {best_config_result['selection_score']:.6f} ± "
        f"{best_config_result['selection_score_std']:.6f}"
    )
    print("Selected configuration primary mean scores:")
    print(
        pformat(
            best_config_result[f"{evaluation_level}_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration primary score standard deviations:")
    print(
        pformat(
            best_config_result[f"{evaluation_level}_std_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration window-level mean scores:")
    print(
        pformat(
            best_config_result["window_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration trial-level mean scores:")
    print(
        pformat(
            best_config_result["trial_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )

    return results



def fixed_loso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict,
    n_epochs: int,
    batch_size: int,
    *,
    preprocessing_strategy: Callable | None = None,
    evaluation_level: Literal["window", "trial"] = "trial",
    selection_metric: str = "balanced_accuracy",
    selection_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
    ),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    decision_threshold: float = 0.5,
    prediction_diagnostics: bool = False,
    prediction_diagnostics_every_n_epochs: int = 1,
    prediction_diagnostics_max_samples: int = 256,
    prediction_diagnostics_threshold_tolerance: float = 0.01,
    prediction_diagnostics_seed: int | None = 42,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
    max_folds: int | None = None,
    alternate_subject_sets: bool = False,
    alternating_subject_seed: int | None = 42,
    use_mldg: bool = False,
    mldg_meta_train_subjects: int = 6,
    mldg_meta_test_subjects: int = 2,
    mldg_samples_per_subject: int = 4,
    mldg_seed: int | None = 42,
) -> dict:
    """Evaluate one fixed configuration with strict LOSOCV and no validation.

    Every fold trains for exactly ``n_epochs`` on all non-test subjects. No
    validation subjects are removed, no validation data are passed to Keras,
    no early-stopping callback is installed, and the supplied decision
    threshold is applied unchanged to every held-out subject.

    This is intended as a post-selection diagnostic after another CV run has
    already chosen the hyperparameters, epoch count, and threshold. It does not
    perform another hyperparameter or threshold search.
    """
    if n_epochs < 1:
        raise ValueError("n_epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    decision_threshold = float(decision_threshold)
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must be strictly between 0 and 1.")

    model_config = dict(fixed_config)
    # The explicit post-selection values must override anything retained from
    # the original search result.
    model_config.pop("epochs", None)
    model_config.pop("batch_size", None)

    print(
        "\nFixed-config no-validation LOSOCV — "
        f"epochs={int(n_epochs)}, batch_size={int(batch_size)}, "
        f"decision_threshold={decision_threshold:.4f}",
        flush=True,
    )

    results = loso_cv(
        model_builder_function=model_builder_function,
        feature_array=feature_array,
        label_array=label_array,
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        hyperparameters=model_config,
        preprocessing_strategy=preprocessing_strategy,
        evaluation_level=evaluation_level,
        selection_metric=selection_metric,
        selection_level=selection_level,
        maximize_metric=maximize_metric,
        metrics=metrics,
        log_predictions=log_predictions,
        log_variational_intervals=log_variational_intervals,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
        n_uncertainty_samples=n_uncertainty_samples,
        ci_level=ci_level,
        validation_subjects_per_fold=0,
        validation_seed=None,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        early_stopping_monitor="loss",
        early_stopping_mode="min",
        restore_best_weights=False,
        prediction_diagnostics=prediction_diagnostics,
        prediction_diagnostics_every_n_epochs=(
            prediction_diagnostics_every_n_epochs
        ),
        prediction_diagnostics_max_samples=prediction_diagnostics_max_samples,
        prediction_diagnostics_threshold_tolerance=(
            prediction_diagnostics_threshold_tolerance
        ),
        prediction_diagnostics_seed=prediction_diagnostics_seed,
        decision_thresholds=(decision_threshold,),
        threshold_selection_metric="balanced_accuracy",
        threshold_selection_level=selection_level,
        verbose=verbose,
        extra_fit_kwargs=extra_fit_kwargs,
        n_jobs=n_jobs,
        gpu_ids=gpu_ids,
        cpus_per_worker=cpus_per_worker,
        max_folds=max_folds,
        alternate_subject_sets=alternate_subject_sets,
        alternating_subject_seed=alternating_subject_seed,
        use_mldg=use_mldg,
        mldg_meta_train_subjects=mldg_meta_train_subjects,
        mldg_meta_test_subjects=mldg_meta_test_subjects,
        mldg_samples_per_subject=mldg_samples_per_subject,
        mldg_seed=mldg_seed,
    )

    if int(results.get("n_configs", 0)) != 1:
        raise RuntimeError(
            "fixed_loso_cv expected exactly one configuration, but loso_cv "
            f"reported {results.get('n_configs')}."
        )

    results.update(
        {
            "cv_strategy": "fixed_loso_no_validation",
            "hyperparameter_search": False,
            "post_selection_diagnostic": True,
            "fixed_epochs": int(n_epochs),
            "fixed_batch_size": int(batch_size),
            "fixed_decision_threshold": decision_threshold,
            "validation_subjects_per_fold": 0,
            "early_stopping_patience": None,
            "restore_best_weights": False,
        }
    )
    return results


# ---------------------------------------------------------------------
# Leave-N-Subjects-and-K-Trials-Out cross-validation
# ---------------------------------------------------------------------


def _stable_scalar_sort_key(value) -> tuple[str, str]:
    """Return a deterministic ordering key for mixed scalar ID types."""
    value = _python_scalar(value)
    return type(value).__name__, repr(value)


def _trial_metadata_by_subject(
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
) -> tuple[dict, dict, tuple[int, ...]]:
    """Return subject-to-trials and one class label per subject/trial key."""
    y_ids = _as_numpy_1d(label_array).astype(np.int64)
    subject_to_trials: dict = {}
    labels_by_key: dict[tuple, int] = {}

    grouped_labels: dict[tuple, set[int]] = {}
    for subject_id, trial_id, label in zip(
        subject_id_array,
        trial_id_array,
        y_ids,
    ):
        subject_value = _python_scalar(subject_id)
        trial_value = _python_scalar(trial_id)
        key = (subject_value, trial_value)
        grouped_labels.setdefault(key, set()).add(int(label))

    for (subject_id, trial_id), labels in grouped_labels.items():
        if len(labels) != 1:
            raise ValueError(
                "Every (subject_id, trial_id) group must have one label. "
                f"Subject {subject_id!r}, trial {trial_id!r} has labels "
                f"{sorted(labels)}."
            )
        labels_by_key[(subject_id, trial_id)] = int(next(iter(labels)))
        subject_to_trials.setdefault(subject_id, []).append(trial_id)

    for subject_id, trial_ids in subject_to_trials.items():
        subject_to_trials[subject_id] = tuple(
            sorted(set(trial_ids), key=_stable_scalar_sort_key)
        )

    classes = tuple(sorted(set(labels_by_key.values())))
    return subject_to_trials, labels_by_key, classes


def _lnskto_split_signature(held_out_trials: list[dict]) -> tuple:
    """Create a hashable canonical signature for one LNSKTO test split."""
    rows = []
    for row in held_out_trials:
        subject_id = _python_scalar(row["subject_id"])
        trial_ids = tuple(
            sorted(
                (_python_scalar(value) for value in row["trial_ids"]),
                key=_stable_scalar_sort_key,
            )
        )
        rows.append((subject_id, trial_ids))
    rows.sort(key=lambda item: _stable_scalar_sort_key(item[0]))
    return tuple(rows)


def _generate_lnskto_fold_specs(
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    n_subjects: int = 3,
    k_trials: int = 3,
    n_folds: int | None = None,
    split_seed: int | None = 42,
    require_all_classes_in_test: bool = True,
    candidate_pool_size: int = 256,
) -> list[dict]:
    """Generate reproducible, globally trial-disjoint LNSKTO folds.

    Each fold selects ``n_subjects`` subjects and exactly ``k_trials`` complete
    trials from each selected subject. The selected subjects remain represented
    in the fold's training data through all of their non-held-out trials.

    A ``(subject_id, trial_id)`` key may be used as test data in at most one
    fold. Consequently, the test-trial sets of every pair of generated folds
    are disjoint. The same subject may appear in several folds, but each time it
    must contribute previously untested trials.
    """
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)
    label_array = np.asarray(label_array)

    if n_subjects < 1:
        raise ValueError("n_subjects must be at least 1.")
    if k_trials < 1:
        raise ValueError("k_trials must be at least 1.")
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be at least 1.")
    if split_seed is not None and split_seed < 0:
        raise ValueError("split_seed must be non-negative or None.")

    subject_to_trials, labels_by_key, global_classes = _trial_metadata_by_subject(
        label_array=label_array,
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
    )
    unique_subjects = tuple(
        sorted(subject_to_trials, key=_stable_scalar_sort_key)
    )

    if n_subjects > len(unique_subjects):
        raise ValueError(
            f"n_subjects={n_subjects} exceeds the {len(unique_subjects)} "
            "available subjects."
        )

    insufficient = {
        subject_id: len(trials)
        for subject_id, trials in subject_to_trials.items()
        if len(trials) <= k_trials
    }
    if insufficient:
        details = ", ".join(
            f"{subject_id!r}:{count}"
            for subject_id, count in sorted(
                insufficient.items(),
                key=lambda item: _stable_scalar_sort_key(item[0]),
            )
        )
        raise ValueError(
            f"Every selectable subject needs at least {k_trials + 1} unique "
            f"trials so that k_trials={k_trials} can be tested while at least "
            f"one other trial remains in that fold's training data. "
            f"Insufficient subjects: {details}."
        )

    if n_folds is None:
        # For DREAMER this produces 23 folds. With N=3 and K=3, that consumes
        # 207 of the 414 unique subject-trial keys without test-set reuse.
        n_folds = len(unique_subjects)
    n_folds = int(n_folds)
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")

    # Each subject can contribute at most floor(number_of_trials / k_trials)
    # disjoint K-trial groups. Each fold consumes one group from n_subjects.
    max_disjoint_folds_upper_bound = (
        sum(len(trials) // k_trials for trials in subject_to_trials.values())
        // n_subjects
    )
    if n_folds > max_disjoint_folds_upper_bound:
        raise ValueError(
            f"Requested n_folds={n_folds}, but at most "
            f"{max_disjoint_folds_upper_bound} folds can be formed without "
            "reusing a tested (subject_id, trial_id) key for "
            f"n_subjects={n_subjects} and k_trials={k_trials}."
        )

    if require_all_classes_in_test and len(global_classes) < 2:
        raise ValueError(
            "require_all_classes_in_test=True requires at least two classes."
        )
    if require_all_classes_in_test and n_subjects * k_trials < len(global_classes):
        raise ValueError(
            "The test fold is too small to contain all classes: "
            f"n_subjects * k_trials={n_subjects * k_trials}, "
            f"n_classes={len(global_classes)}."
        )

    rng = np.random.default_rng(split_seed)
    subject_use_count = {subject_id: 0 for subject_id in unique_subjects}
    used_test_trial_keys: set[tuple] = set()
    all_trial_keys = set(labels_by_key)
    fold_specs: list[dict] = []

    for fold_number in range(1, n_folds + 1):
        best_candidate: tuple | None = None
        best_score: tuple | None = None

        unused_trials_by_subject = {
            subject_id: tuple(
                trial_id
                for trial_id in subject_to_trials[subject_id]
                if (subject_id, trial_id) not in used_test_trial_keys
            )
            for subject_id in unique_subjects
        }
        eligible_subjects = tuple(
            subject_id
            for subject_id in unique_subjects
            if len(unused_trials_by_subject[subject_id]) >= k_trials
        )
        if len(eligible_subjects) < n_subjects:
            remaining = {
                _python_scalar(subject_id): len(unused_trials_by_subject[subject_id])
                for subject_id in unique_subjects
            }
            raise RuntimeError(
                "Could not generate another globally trial-disjoint LNSKTO "
                f"fold {fold_number}/{n_folds}: only {len(eligible_subjects)} "
                f"subjects retain at least {k_trials} never-tested trials. "
                f"Remaining unused trials by subject: {remaining}. Reduce "
                "n_folds or k_trials."
            )

        # Draw many valid candidates and select the candidate that best balances
        # test labels and subject reuse. Trial reuse is impossible because every
        # candidate is drawn only from globally unused subject-trial keys.
        attempts = max(candidate_pool_size, 64)
        for _ in range(attempts):
            selected_subject_indices = rng.choice(
                len(eligible_subjects),
                size=n_subjects,
                replace=False,
            )
            selected_subjects = [
                eligible_subjects[int(index)]
                for index in selected_subject_indices
            ]

            held_out_trials: list[dict] = []
            held_out_keys: set[tuple] = set()
            for subject_id in selected_subjects:
                available_trials = unused_trials_by_subject[subject_id]
                selected_trial_indices = rng.choice(
                    len(available_trials),
                    size=k_trials,
                    replace=False,
                )
                selected_trials = sorted(
                    (
                        available_trials[int(index)]
                        for index in selected_trial_indices
                    ),
                    key=_stable_scalar_sort_key,
                )
                held_out_trials.append(
                    {
                        "subject_id": _python_scalar(subject_id),
                        "trial_ids": [
                            _python_scalar(value) for value in selected_trials
                        ],
                    }
                )
                held_out_keys.update(
                    (subject_id, trial_id) for trial_id in selected_trials
                )

            if held_out_keys.intersection(used_test_trial_keys):
                raise RuntimeError(
                    "Internal LNSKTO error: a candidate contains a previously "
                    "tested subject-trial key."
                )

            held_out_trials.sort(
                key=lambda row: _stable_scalar_sort_key(row["subject_id"])
            )
            test_labels = [labels_by_key[key] for key in held_out_keys]
            train_keys = all_trial_keys.difference(held_out_keys)
            train_labels = {labels_by_key[key] for key in train_keys}
            if train_labels != set(global_classes):
                continue
            if (
                require_all_classes_in_test
                and set(test_labels) != set(global_classes)
            ):
                continue

            class_counts = np.asarray(
                [test_labels.count(class_id) for class_id in global_classes],
                dtype=np.int64,
            )
            class_imbalance = int(class_counts.max() - class_counts.min())
            subject_counts = [
                subject_use_count[subject_id]
                for subject_id in selected_subjects
            ]
            remaining_after_selection = [
                len(unused_trials_by_subject[subject_id]) - k_trials
                for subject_id in selected_subjects
            ]
            score = (
                class_imbalance,
                max(subject_counts, default=0),
                sum(subject_counts),
                # Prefer candidates that do not strand a subject with fewer
                # than K unused trials when another balanced choice exists.
                sum(0 < remaining < k_trials for remaining in remaining_after_selection),
                -min(remaining_after_selection, default=0),
                float(rng.random()),
            )

            if best_score is None or score < best_score:
                best_score = score
                best_candidate = (
                    held_out_trials,
                    held_out_keys,
                    test_labels,
                )

        if best_candidate is None:
            raise RuntimeError(
                "Could not generate another valid globally trial-disjoint "
                "LNSKTO fold. Reduce n_folds, disable "
                "require_all_classes_in_test, increase candidate_pool_size, "
                "or change split_seed."
            )

        held_out_trials, held_out_keys, test_labels = best_candidate
        overlap = held_out_keys.intersection(used_test_trial_keys)
        if overlap:
            raise RuntimeError(
                "Internal LNSKTO split-generation error: accepted fold reuses "
                f"previously tested keys {sorted(overlap, key=repr)}."
            )
        used_test_trial_keys.update(held_out_keys)

        selected_subjects = [row["subject_id"] for row in held_out_trials]
        for subject_id in selected_subjects:
            subject_use_count[subject_id] += 1

        test_mask = np.zeros(len(subject_id_array), dtype=bool)
        for row in held_out_trials:
            subject_mask = subject_id_array == row["subject_id"]
            trial_mask = np.isin(trial_id_array, row["trial_ids"])
            test_mask |= subject_mask & trial_mask
        test_indices = np.where(test_mask)[0].astype(np.int64)

        observed_test_keys = set(
            zip(
                subject_id_array[test_indices].tolist(),
                trial_id_array[test_indices].tolist(),
            )
        )
        expected_test_keys = {
            (_python_scalar(subject_id), _python_scalar(trial_id))
            for subject_id, trial_id in held_out_keys
        }
        if observed_test_keys != expected_test_keys:
            raise RuntimeError(
                "Internal LNSKTO split-generation error: selected trial keys "
                "did not map exactly to the test samples."
            )

        fold_specs.append(
            {
                "fold_number": int(fold_number),
                "test_indices": test_indices,
                "test_subjects": selected_subjects,
                "held_out_trials": held_out_trials,
                "test_trial_keys": [
                    {
                        "subject_id": _python_scalar(subject_id),
                        "trial_id": _python_scalar(trial_id),
                    }
                    for subject_id, trial_id in sorted(
                        held_out_keys,
                        key=lambda key: (
                            _stable_scalar_sort_key(key[0]),
                            _stable_scalar_sort_key(key[1]),
                        ),
                    )
                ],
                "n_test_trials": int(len(held_out_keys)),
                "test_class_counts": {
                    int(class_id): int(test_labels.count(class_id))
                    for class_id in global_classes
                },
                "cumulative_unique_test_trials": int(
                    len(used_test_trial_keys)
                ),
            }
        )

    # Independent final verification: no test key may occur in two fold specs.
    verified_test_keys: set[tuple] = set()
    for fold_spec in fold_specs:
        fold_keys = {
            (row["subject_id"], trial_id)
            for row in fold_spec["held_out_trials"]
            for trial_id in row["trial_ids"]
        }
        overlap = verified_test_keys.intersection(fold_keys)
        if overlap:
            raise RuntimeError(
                "Generated LNSKTO folds are not test-trial disjoint. Repeated "
                f"keys: {sorted(overlap, key=repr)}."
            )
        verified_test_keys.update(fold_keys)

    expected_unique_test_trials = n_folds * n_subjects * k_trials
    if len(verified_test_keys) != expected_unique_test_trials:
        raise RuntimeError(
            "Generated LNSKTO fold count does not match the number of unique "
            f"test trial keys: expected {expected_unique_test_trials}, got "
            f"{len(verified_test_keys)}."
        )

    return fold_specs


def _run_lnskto_fold_task(
    fold_number: int,
    fold_spec: dict,
    n_subjects: int,
    k_trials: int,
    **worker_state,
) -> dict:
    """Execute one LNSKTO split through the shared fold-training pipeline."""
    test_subjects = list(fold_spec["test_subjects"])
    held_out_trials = [dict(row) for row in fold_spec["held_out_trials"]]

    # Verify the intended partial-subject holdout before any model is built:
    # current test trials are absent from training, while the same subjects'
    # other trials remain available for gradient training.
    subject_id_array = np.asarray(worker_state["subject_id_array"])
    trial_id_array = np.asarray(worker_state["trial_id_array"])
    test_indices = np.asarray(fold_spec["test_indices"], dtype=np.int64)
    test_mask = np.zeros(len(subject_id_array), dtype=bool)
    test_mask[test_indices] = True
    train_mask = ~test_mask
    for row in held_out_trials:
        subject_id = row["subject_id"]
        held_out_trial_ids = np.asarray(row["trial_ids"])
        subject_mask = subject_id_array == subject_id
        held_out_mask = subject_mask & np.isin(trial_id_array, held_out_trial_ids)
        if np.any(train_mask & held_out_mask):
            raise RuntimeError(
                "LNSKTO leakage: a held-out subject-trial group remains in "
                f"training for subject={subject_id!r}, "
                f"trials={row['trial_ids']}."
            )
        same_subject_non_test_mask = (
            train_mask
            & subject_mask
            & ~np.isin(trial_id_array, held_out_trial_ids)
        )
        if not np.any(same_subject_non_test_mask):
            raise RuntimeError(
                "LNSKTO split removed the selected subject completely instead "
                f"of retaining non-test trials: subject={subject_id!r}."
            )

    trial_summary = "; ".join(
        f"subject={row['subject_id']!r}: trials={row['trial_ids']}"
        for row in held_out_trials
    )
    fold_output = _run_loso_fold(
        fold_number=fold_number,
        test_subject=f"LNSKTO-{fold_number}",
        test_indices_override=np.asarray(fold_spec["test_indices"], dtype=np.int64),
        left_out_subjects_override=test_subjects,
        held_out_trials=held_out_trials,
        validation_excluded_subjects=test_subjects,
        fold_description=(
            f"LNSKTO n={n_subjects}, k={k_trials}; {trial_summary}"
        ),
        cv_strategy="leave_n_subjects_k_trials_out",
        **worker_state,
    )

    fold_record = fold_output["fold_record"]
    fold_record.update(
        {
            "test_class_counts": dict(fold_spec["test_class_counts"]),
        }
    )
    return fold_output


def _lnskto_fold_process_main(
    worker_state_payload: bytes,
    task_queue,
    result_queue,
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None,
) -> None:
    """Run LNSKTO folds in one persistent spawned process."""
    try:
        _configure_tensorflow_worker(
            gpu_id=gpu_id,
            cpus_per_worker=cpus_per_worker,
            assigned_device_label=assigned_device_label,
        )
        worker_state = cloudpickle.loads(worker_state_payload)

        while True:
            task = task_queue.get()
            if task is None:
                return
            fold_number, fold_spec = task
            try:
                fold_output = _run_lnskto_fold_task(
                    fold_number=fold_number,
                    fold_spec=fold_spec,
                    **worker_state,
                )
                result_queue.put(("ok", int(fold_number), fold_output))
            except BaseException:
                result_queue.put(
                    ("error", int(fold_number), traceback.format_exc())
                )
                return
    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def lnskto_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray | None = None,
    n_subjects: int = 3,
    k_trials: int = 3,
    n_folds: int | None = None,
    split_seed: int | None = 42,
    require_all_classes_in_test: bool = True,
    n_epochs: int = 50,
    batch_size: int = 2,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    evaluation_level: Literal["window", "trial"] = "trial",
    selection_metric: str = "f1",
    selection_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "binary_f1",
        "binary_precision",
        "binary_recall",
        "roc_auc",
    ),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    validation_subjects_per_fold: int = 0,
    validation_seed: int | None = 42,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: Literal["auto", "min", "max"] = "min",
    restore_best_weights: bool = True,
    prediction_diagnostics: bool = False,
    prediction_diagnostics_every_n_epochs: int = 1,
    prediction_diagnostics_max_samples: int = 256,
    prediction_diagnostics_threshold_tolerance: float = 0.01,
    prediction_diagnostics_seed: int | None = 42,
    decision_thresholds: list[float] | tuple[float, ...] = (0.5,),
    threshold_selection_metric: Literal[
        "accuracy", "f1", "balanced_accuracy", "binary_f1"
    ] = "f1",
    threshold_selection_level: Literal["window", "trial"] = "trial",
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
) -> dict:
    """Run flat Leave-N-Subjects-and-K-Trials-Out cross-validation.

    In every fold, ``n_subjects`` subjects are selected and exactly
    ``k_trials`` complete trials are placed in the test set for each selected
    subject. The selected subjects are *not* removed completely: their other
    trials remain in the training pool. Thus this protocol measures
    generalization to unseen trials from partly observed subjects, not strict
    cross-subject generalization. Use ``loso_cv`` or ``nested_lnso_cv`` for a
    subject-independent estimate.

    The defaults are ``n_subjects=3`` and ``k_trials=3``, producing nine held-
    out trials per fold. When ``n_folds`` is ``None``, the number of folds is
    set to the total number of subjects (23 for DREAMER). Fold generation is
    deterministic for a fixed ``split_seed``. Subjects may recur across folds,
    but a ``(subject_id, trial_id)`` test key is globally consumed after one
    use and can never appear in another fold's test set. Entire trial groups are
    kept together, so rank-3 window data never leaks windows from the current
    fold's held-out trials into that fold's training data.

    Hyperparameter selection follows the same flat-search semantics as
    ``loso_cv``: every configuration is evaluated on all generated folds, and
    the globally best configuration is selected from the mean requested metric.
    This is not nested CV because the same folds are used for model selection
    and final performance reporting.

    When fold-local validation subjects are requested, subjects contributing
    held-out test trials are excluded from validation selection. Their remaining
    non-test trials therefore stay in gradient training, preserving the intended
    partial-subject holdout design.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    if "validation_data" in extra_fit_kwargs:
        raise ValueError(
            "Do not pass a fixed validation_data array to lnskto_cv. It would "
            "not be reconstructed fold-locally and could create leakage."
        )
    if subject_id_array is None:
        raise ValueError("subject_id_array is required for LNSKTO CV.")
    if trial_id_array is None:
        raise ValueError(
            "trial_id_array is required. Pass one trial ID per sample, aligned "
            "with feature_array."
        )

    _validate_evaluation_level(evaluation_level, "evaluation_level")
    _validate_evaluation_level(selection_level, "selection_level")

    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    if feature_array.ndim not in {3, 4}:
        raise ValueError(
            "feature_array must be rank 3 for window samples or rank 4 for "
            f"grouped trial samples; got {feature_array.shape}."
        )
    if feature_array.ndim == 4:
        if selection_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require selection_level='trial'."
            )
        if evaluation_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require evaluation_level='trial'."
            )

    input_lengths = (
        len(feature_array),
        len(label_array),
        len(subject_id_array),
        len(trial_id_array),
    )
    if len(set(input_lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            f"must have the same first dimension. Got lengths {input_lengths}."
        )

    metrics = tuple(metrics)
    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. Supported metrics: "
                f"{sorted(_CLASSIFICATION_METRICS)}"
            )

    allowed_selection_metrics = {"loss", "joint_loss", *metrics}
    if selection_metric not in allowed_selection_metrics:
        raise ValueError(
            f"selection_metric={selection_metric!r} is unavailable. Use "
            f"'loss', 'joint_loss', or one of metrics={list(metrics)}."
        )
    if (
        feature_array.ndim == 3
        and selection_metric == "joint_loss"
        and selection_level != "window"
    ):
        raise ValueError(
            "Window-level models require selection_level='window' when "
            "selecting by joint_loss."
        )
    if maximize_metric is None:
        maximize_metric = selection_metric not in {"loss", "joint_loss"}

    if n_subjects < 1:
        raise ValueError("n_subjects must be at least 1.")
    if k_trials < 1:
        raise ValueError("k_trials must be at least 1.")
    if n_folds is not None and n_folds < 1:
        raise ValueError("n_folds must be at least 1 or None.")
    if split_seed is not None and split_seed < 0:
        raise ValueError("split_seed must be non-negative or None.")
    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")

    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if alternate_subject_sets and use_mldg:
        raise ValueError(
            "alternate_subject_sets and use_mldg are mutually exclusive."
        )
    if mldg_meta_train_subjects < 1 or mldg_meta_test_subjects < 1:
        raise ValueError("MLDG A/B subject counts must both be at least 1.")
    if mldg_samples_per_subject < 1:
        raise ValueError("mldg_samples_per_subject must be at least 1.")
    if mldg_seed is not None and mldg_seed < 0:
        raise ValueError("mldg_seed must be >= 0 or None.")
    if validation_seed is not None and validation_seed < 0:
        raise ValueError("validation_seed must be >= 0 or None.")
    if early_stopping_patience is not None and early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0 or None.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be >= 0.")
    if early_stopping_mode not in {"auto", "min", "max"}:
        raise ValueError(
            "early_stopping_mode must be 'auto', 'min', or 'max'."
        )
    if not early_stopping_monitor:
        raise ValueError("early_stopping_monitor must be a non-empty string.")
    if prediction_diagnostics_every_n_epochs < 1:
        raise ValueError(
            "prediction_diagnostics_every_n_epochs must be at least 1."
        )
    if prediction_diagnostics_max_samples < 1:
        raise ValueError(
            "prediction_diagnostics_max_samples must be at least 1."
        )
    if prediction_diagnostics_threshold_tolerance < 0.0:
        raise ValueError(
            "prediction_diagnostics_threshold_tolerance must be non-negative."
        )

    decision_thresholds = _normalize_decision_thresholds(decision_thresholds)
    if threshold_selection_metric not in {
        "accuracy", "f1", "balanced_accuracy", "binary_f1"
    }:
        raise ValueError(
            "threshold_selection_metric must be accuracy, f1, "
            "balanced_accuracy, or binary_f1."
        )
    _validate_evaluation_level(
        threshold_selection_level,
        "threshold_selection_level",
    )
    if len(decision_thresholds) > 1 and validation_subjects_per_fold == 0:
        raise ValueError(
            "Testing multiple decision thresholds requires fold-local "
            "validation subjects. Set validation_subjects_per_fold >= 1."
        )
    if (
        early_stopping_monitor in {
            "val_trial_f1",
            "val_trial_balanced_accuracy",
            "val_trial_loss",
        }
        and validation_subjects_per_fold == 0
        and early_stopping_patience is not None
    ):
        raise ValueError(
            f"{early_stopping_monitor} requires at least one fold-local "
            "validation subject."
        )

    unique_subjects = np.unique(subject_id_array)
    if n_subjects > len(unique_subjects):
        raise ValueError(
            f"n_subjects={n_subjects} exceeds the {len(unique_subjects)} "
            "available subjects."
        )
    validation_candidate_count = len(unique_subjects) - n_subjects
    if (
        validation_subjects_per_fold > 0
        and validation_subjects_per_fold >= validation_candidate_count
    ):
        raise ValueError(
            "validation_subjects_per_fold must leave at least one non-test "
            "subject outside validation. With "
            f"{len(unique_subjects)} subjects and n_subjects={n_subjects}, "
            f"only {validation_candidate_count} validation candidates remain."
        )

    fold_specs = _generate_lnskto_fold_specs(
        label_array=label_array,
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        n_subjects=n_subjects,
        k_trials=k_trials,
        n_folds=n_folds,
        split_seed=split_seed,
        require_all_classes_in_test=require_all_classes_in_test,
    )
    total_folds = len(fold_specs)

    effective_hyperparameters = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        **(hyperparameters or {}),
    }
    sequence_hyperparameter_depths = getattr(
        model_builder_function,
        "_sequence_hyperparameter_depths",
        None,
    )
    grid_configs = _expand_hyperparameter_grid(
        effective_hyperparameters,
        sequence_hyperparameter_depths=sequence_hyperparameter_depths,
    )
    _warn_if_joint_loss_weights_vary(grid_configs, selection_metric)
    if not grid_configs:
        raise ValueError("The hyperparameter grid produced no configurations.")

    total_model_fits = len(grid_configs) * total_folds
    effective_n_jobs = min(n_jobs, total_folds)
    normalized_gpu_ids: tuple[int, ...] | None = None
    if gpu_ids is None and effective_n_jobs > 1:
        normalized_gpu_ids = _auto_assign_gpu_ids(effective_n_jobs)
        if normalized_gpu_ids is not None:
            effective_n_jobs = len(normalized_gpu_ids)
    elif gpu_ids is not None:
        normalized_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
        if not normalized_gpu_ids:
            raise ValueError("gpu_ids must contain at least one GPU index.")
        if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
            raise ValueError("gpu_ids must not contain duplicate GPU indices.")
        if effective_n_jobs > len(normalized_gpu_ids):
            raise ValueError(
                f"n_jobs={effective_n_jobs} requires at least that many GPU "
                f"IDs, but gpu_ids={normalized_gpu_ids}."
            )
        normalized_gpu_ids = normalized_gpu_ids[:effective_n_jobs]

    print(
        f"\nFlat LNSKTO hyperparameter search — n={n_subjects}, "
        f"k={k_trials}, {len(grid_configs)} configuration"
        f"{'s' if len(grid_configs) != 1 else ''}, {total_folds} fold"
        f"{'s' if total_folds != 1 else ''} each"
    )
    print(
        f"Each test fold contains {n_subjects * k_trials} complete trials; "
        "selected subjects' remaining trials stay in training."
    )
    print(f"Split seed: {split_seed}")
    print(f"Require all classes in test: {require_all_classes_in_test}")
    print(f"Total model fits: {total_model_fits}")
    print(f"Requested metrics: {list(metrics)}")
    print(
        f"Configuration selection: {selection_level}-level "
        f"{selection_metric} "
        f"({'maximize' if maximize_metric else 'minimize'})"
    )
    print(f"Primary reported metrics: {evaluation_level}-level")
    print(f"Prediction logging: {log_predictions}")
    print(f"Prediction diagnostics: {prediction_diagnostics}")
    print(f"Variational interval logging: {log_variational_intervals}")
    print(
        "Decision thresholds: "
        f"{list(decision_thresholds)}; selection="
        f"{threshold_selection_level}_{threshold_selection_metric}"
    )
    prediction_mode = (
        "posterior mean"
        if n_prediction_latent_samples == 0
        else f"MC average over {n_prediction_latent_samples} latent sample(s)"
    )
    print(f"Prediction latent mode: {prediction_mode}")
    if validation_subjects_per_fold > 0:
        print(
            "Per-fold validation: "
            f"{validation_subjects_per_fold} seeded non-test subject(s), "
            f"seed={validation_seed}, monitor={early_stopping_monitor}, "
            f"patience={early_stopping_patience}, "
            f"restore_best_weights={restore_best_weights}"
        )
    else:
        print("Per-fold validation: disabled")
    print(f"Fold workers: {effective_n_jobs}")
    if effective_n_jobs > 1 and normalized_gpu_ids is None:
        print("Worker devices: CPU-only")
    elif normalized_gpu_ids is not None:
        print(f"Worker devices: GPUs {list(normalized_gpu_ids)}")
    else:
        print("Worker device: current TensorFlow default")

    tasks = [
        (int(spec["fold_number"]), spec)
        for spec in fold_specs
    ]
    common_worker_state = {
        "n_subjects": int(n_subjects),
        "k_trials": int(k_trials),
        "total_folds": total_folds,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "batch_size": batch_size,
        "preprocessing_strategy": preprocessing_strategy,
        "evaluation_level": evaluation_level,
        "metrics": metrics,
        "log_predictions": log_predictions,
        "log_variational_intervals": log_variational_intervals,
        "n_prediction_latent_samples": n_prediction_latent_samples,
        "latent_sampling_seed": latent_sampling_seed,
        "n_uncertainty_samples": n_uncertainty_samples,
        "ci_level": ci_level,
        "validation_subjects_per_fold": validation_subjects_per_fold,
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "restore_best_weights": restore_best_weights,
        "prediction_diagnostics": bool(prediction_diagnostics),
        "prediction_diagnostics_every_n_epochs": int(
            prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": int(
            prediction_diagnostics_max_samples
        ),
        "prediction_diagnostics_threshold_tolerance": float(
            prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": prediction_diagnostics_seed,
        "decision_thresholds": decision_thresholds,
        "threshold_selection_metric": threshold_selection_metric,
        "threshold_selection_level": threshold_selection_level,
        "verbose": verbose,
        "extra_fit_kwargs": extra_fit_kwargs,
    }

    config_results: list[dict] = []
    best_so_far_result: dict | None = None
    best_fold_outputs: list[dict] | None = None

    for config_index, config in enumerate(grid_configs):
        print("\n" + "#" * 80)
        print(
            f"Configuration {config_index + 1} / {len(grid_configs)} "
            f"({total_folds} LNSKTO fits)"
        )
        _print_config("Configuration:", config)
        worker_state = {
            **common_worker_state,
            "fixed_config": config,
        }

        if effective_n_jobs == 1 and normalized_gpu_ids is None:
            fold_outputs = [
                _run_lnskto_fold_task(
                    fold_number=fold_number,
                    fold_spec=fold_spec,
                    **worker_state,
                )
                for fold_number, fold_spec in tasks
            ]
        else:
            fold_outputs = _run_spawned_fold_pool(
                worker_target=_lnskto_fold_process_main,
                worker_state=worker_state,
                tasks=tasks,
                n_workers=effective_n_jobs,
                gpu_ids=normalized_gpu_ids,
                cpus_per_worker=cpus_per_worker,
                worker_name_prefix=f"LNSKTOConfig{config_index + 1}Worker",
                worker_description=(
                    f"LNSKTO fold for configuration {config_index + 1}"
                ),
            )

        fold_outputs.sort(key=lambda row: row["outer_fold_number"])
        config_result = _aggregate_loso_config_result(
            config_index=config_index,
            config=config,
            fold_outputs=fold_outputs,
            metrics=metrics,
            selection_metric=selection_metric,
            selection_level=selection_level,
        )
        config_results.append(config_result)

        if (
            best_so_far_result is None
            or _loso_config_sort_key(
                config_result=config_result,
                selection_metric=selection_metric,
                selection_level=selection_level,
                maximize_metric=bool(maximize_metric),
            )
            < _loso_config_sort_key(
                config_result=best_so_far_result,
                selection_metric=selection_metric,
                selection_level=selection_level,
                maximize_metric=bool(maximize_metric),
            )
        ):
            best_so_far_result = config_result
            best_fold_outputs = fold_outputs

        print(
            f"\nConfiguration {config_index + 1} complete: mean "
            f"{selection_level}_{selection_metric}="
            f"{config_result['selection_score']:.6f} ± "
            f"{config_result['selection_score_std']:.6f}",
            flush=True,
        )

    best_config_index = _choose_best_loso_config_index(
        config_results=config_results,
        selection_metric=selection_metric,
        selection_level=selection_level,
        maximize_metric=bool(maximize_metric),
    )
    best_config_result = config_results[best_config_index]
    best_config = dict(best_config_result["config"])
    if (
        best_so_far_result is None
        or best_fold_outputs is None
        or int(best_so_far_result["config_index"]) != best_config_index
    ):
        raise RuntimeError(
            "Internal LNSKTO grid-search error: selected configuration logs "
            "were not retained correctly."
        )

    results = {
        "cv_strategy": "flat_leave_n_subjects_k_trials_out_hyperparameter_search",
        "hyperparameter_search": True,
        "n_subjects": int(n_subjects),
        "k_trials": int(k_trials),
        "n_test_trials_per_fold": int(n_subjects * k_trials),
        "n_folds": int(total_folds),
        "split_seed": split_seed,
        "require_all_classes_in_test": bool(require_all_classes_in_test),
        "selected_subjects_remain_in_training": True,
        "test_trial_keys_are_globally_unique": True,
        "n_unique_test_trial_keys": int(total_folds * n_subjects * k_trials),
        "n_configs": int(len(grid_configs)),
        "n_total_cv_fits": int(total_model_fits),
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "maximize_metric": bool(maximize_metric),
        "selection_score": float(best_config_result["selection_score"]),
        "selection_score_std": float(best_config_result["selection_score_std"]),
        "n_prediction_latent_samples": int(n_prediction_latent_samples),
        "latent_sampling_seed": latent_sampling_seed,
        "validation_subjects_per_fold": int(validation_subjects_per_fold),
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "restore_best_weights": bool(restore_best_weights),
        "config_results": config_results,
        "best_config_index": int(best_config_index),
        "best_config": best_config,
        "user_metrics": [],
        "fold_results": [],
    }
    if log_predictions:
        if feature_array.ndim == 3:
            results["window_prediction_log"] = []
        results["trial_prediction_log"] = []
    if log_variational_intervals:
        if feature_array.ndim == 3:
            results["window_variational_interval_log"] = []
        results["trial_variational_interval_log"] = []
    if prediction_diagnostics:
        results["prediction_diagnostics_log"] = []

    for fold_output in best_fold_outputs:
        results["user_metrics"].extend(fold_output["user_metrics"])
        if log_predictions:
            if feature_array.ndim == 3:
                results["window_prediction_log"].extend(
                    fold_output["window_prediction_log"]
                )
            results["trial_prediction_log"].extend(
                fold_output["trial_prediction_log"]
            )
        if log_variational_intervals:
            if feature_array.ndim == 3:
                results["window_variational_interval_log"].extend(
                    fold_output["window_variational_interval_log"]
                )
            results["trial_variational_interval_log"].extend(
                fold_output["trial_variational_interval_log"]
            )
        if prediction_diagnostics:
            results["prediction_diagnostics_log"].extend(
                fold_output.get("prediction_diagnostics_log", [])
            )
        results["fold_results"].append(dict(fold_output["fold_record"]))

    print("\nFlat LNSKTO hyperparameter search complete")
    print("=" * 80)
    print(
        f"Selected configuration {best_config_index + 1} / "
        f"{len(grid_configs)} using {selection_level}-level "
        f"{selection_metric}."
    )
    _print_config("Best configuration:", best_config)
    print(
        f"Selection score: {best_config_result['selection_score']:.6f} ± "
        f"{best_config_result['selection_score_std']:.6f}"
    )
    print("Selected configuration primary mean scores:")
    print(
        pformat(
            best_config_result[f"{evaluation_level}_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration primary score standard deviations:")
    print(
        pformat(
            best_config_result[f"{evaluation_level}_std_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration window-level mean scores:")
    print(
        pformat(
            best_config_result["window_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration trial-level mean scores:")
    print(
        pformat(
            best_config_result["trial_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    return results


# Descriptive alias for callers that prefer the full protocol name.
leave_n_subjects_k_trials_out_cv = lnskto_cv
