from __future__ import annotations
from typing import Mapping

import numpy as np
import tensorflow as tf

from ..cross_val import _as_numpy_1d, _prepare_fit_inputs_with_subject_ids

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

