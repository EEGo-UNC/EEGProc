from __future__ import annotations
from typing import Mapping

import numpy as np
import tensorflow as tf

from ..cross_val import _as_numpy_1d, _prepare_fit_inputs_with_subject_ids

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