from __future__ import annotations
from typing import Any, Mapping

import numpy as np
import tensorflow as tf

from ..cross_val import _as_numpy_1d, _prepare_fit_inputs_with_subject_ids


class SICMLDGEpisodeSequence(tf.keras.utils.Sequence):
    """Build balanced, complete-trial MLDG episodes for SIC.

    One sequence item is one MLDG update. Meta-train and meta-test subjects are
    disjoint, subject roles are balanced across an epoch, and selected trials
    are kept intact so window-level inputs never split a trial.
    """

    def __init__(
        self,
        *,
        eeg,
        labels,
        subject_ids,
        trial_ids,
        sample_weight,
        meta_train_subjects: int | None,
        meta_test_subjects: int,
        trials_per_subject: int,
        steps_per_epoch: int | None,
        seed: int | None,
    ) -> None:
        super().__init__()
        self.eeg = np.asarray(eeg)
        self.labels = np.asarray(labels)
        self.subject_ids = np.asarray(subject_ids).reshape(-1)
        self.trial_ids = np.asarray(trial_ids).reshape(-1)
        self.sample_weight = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        )
        n_samples = len(self.eeg)
        if not (
            len(self.labels)
            == len(self.subject_ids)
            == len(self.trial_ids)
            == n_samples
        ):
            raise ValueError("MLDG EEG, labels, subject IDs, and trial IDs must align.")
        if self.sample_weight is not None and len(self.sample_weight) != n_samples:
            raise ValueError("MLDG sample weights must align with EEG inputs.")

        self.subjects = np.sort(np.unique(self.subject_ids))
        self.meta_test_subjects = int(meta_test_subjects)
        if self.meta_test_subjects < 1:
            raise ValueError("mldg_meta_test_subjects must be >= 1.")
        if self.meta_test_subjects >= len(self.subjects):
            raise ValueError(
                "MLDG needs at least one meta-train subject in addition to its "
                f"{self.meta_test_subjects} meta-test subjects; only "
                f"{len(self.subjects)} source subjects are available."
            )
        self.meta_train_subjects = (
            len(self.subjects) - self.meta_test_subjects
            if meta_train_subjects is None
            else int(meta_train_subjects)
        )
        if self.meta_train_subjects < 1:
            raise ValueError("mldg_meta_train_subjects must be >= 1 or null.")
        if self.meta_train_subjects + self.meta_test_subjects > len(self.subjects):
            raise ValueError(
                "mldg_meta_train_subjects + mldg_meta_test_subjects cannot "
                f"exceed the {len(self.subjects)} available source subjects."
            )

        self.trials_per_subject = int(trials_per_subject)
        if self.trials_per_subject < 1:
            raise ValueError("mldg_trials_per_subject must be >= 1.")
        self._trial_groups: dict[Any, list[np.ndarray]] = {}
        total_trial_groups = 0
        for subject_id in self.subjects.tolist():
            subject_mask = self.subject_ids == subject_id
            groups = []
            for trial_id in np.unique(self.trial_ids[subject_mask]).tolist():
                indices = np.flatnonzero(
                    subject_mask & (self.trial_ids == trial_id)
                ).astype(np.int64)
                if len(indices):
                    groups.append(indices)
            if not groups:
                raise ValueError(f"Subject {subject_id!r} has no MLDG trials.")
            self._trial_groups[subject_id] = groups
            total_trial_groups += len(groups)

        subjects_per_episode = self.meta_train_subjects + self.meta_test_subjects
        default_steps = int(
            np.ceil(
                total_trial_groups
                / float(subjects_per_episode * self.trials_per_subject)
            )
        )
        self.steps_per_epoch = (
            max(1, default_steps) if steps_per_epoch is None else int(steps_per_epoch)
        )
        if self.steps_per_epoch < 1:
            raise ValueError("mldg_steps_per_epoch must be >= 1 or null.")
        self.seed = None if seed is None else int(seed)
        self._epoch = 0
        self._episodes: list[tuple[np.ndarray, np.ndarray]] = []
        self._build_epoch()

    def __len__(self) -> int:
        return self.steps_per_epoch

    @staticmethod
    def _balanced_subject_choice(subjects, counts, n_select, rng):
        candidates = list(subjects)
        rng.shuffle(candidates)
        candidates.sort(key=lambda subject_id: counts[subject_id])
        chosen = candidates[:n_select]
        for subject_id in chosen:
            counts[subject_id] += 1
        return chosen

    def _sample_complete_trials(self, subject_id, rng, trial_queues):
        groups = self._trial_groups[subject_id]
        chosen = []
        for _ in range(self.trials_per_subject):
            if not trial_queues[subject_id]:
                trial_queues[subject_id].extend(rng.permutation(len(groups)).tolist())
            chosen.append(trial_queues[subject_id].pop())
        return np.concatenate([groups[int(index)] for index in chosen])

    def _build_epoch(self) -> None:
        epoch_seed = None if self.seed is None else self.seed + self._epoch
        rng = np.random.default_rng(epoch_seed)
        meta_test_counts = {subject_id: 0 for subject_id in self.subjects.tolist()}
        meta_train_counts = {subject_id: 0 for subject_id in self.subjects.tolist()}
        episodes = []
        all_subjects = self.subjects.tolist()
        trial_queues = {subject_id: [] for subject_id in all_subjects}
        for _ in range(self.steps_per_epoch):
            meta_test = self._balanced_subject_choice(
                all_subjects,
                meta_test_counts,
                self.meta_test_subjects,
                rng,
            )
            meta_test_set = set(meta_test)
            remaining = [
                subject_id
                for subject_id in all_subjects
                if subject_id not in meta_test_set
            ]
            meta_train = self._balanced_subject_choice(
                remaining,
                meta_train_counts,
                self.meta_train_subjects,
                rng,
            )

            episode_indices = []
            episode_roles = []
            for role, selected_subjects in ((0, meta_train), (1, meta_test)):
                for subject_id in selected_subjects:
                    indices = self._sample_complete_trials(
                        subject_id,
                        rng,
                        trial_queues,
                    )
                    episode_indices.append(indices)
                    episode_roles.append(np.full(len(indices), role, dtype=np.int32))
            indices = np.concatenate(episode_indices)
            roles = np.concatenate(episode_roles)
            order = rng.permutation(len(indices))
            episodes.append((indices[order], roles[order]))
        self._episodes = episodes

    def __getitem__(self, index: int):
        indices, roles = self._episodes[int(index)]
        inputs = {
            "eeg": self.eeg[indices],
            "subject_id": self.subject_ids[indices],
            "mldg_role": roles,
        }
        if self.sample_weight is None:
            return inputs, self.labels[indices]
        return inputs, self.labels[indices], self.sample_weight[indices]

    def on_epoch_end(self) -> None:
        self._epoch += 1
        self._build_epoch()


def fit_sic_mldg(
    model,
    *,
    x,
    y,
    epochs,
    verbose,
    callbacks,
    validation_split,
    validation_data,
    class_weight,
    sample_weight,
    initial_epoch,
    steps_per_epoch,
    validation_steps,
    validation_batch_size,
    validation_freq,
    **kwargs,
):
    """Adapt ordinary SIC source arrays into complete MLDG episodes."""
    if validation_split not in (None, 0, 0.0):
        raise ValueError(
            "MLDG does not support sample-level validation_split. Use "
            "subject-disjoint validation_data or validation_subjects=0."
        )
    if not isinstance(x, Mapping):
        raise ValueError(
            "MLDG source fitting requires inputs prepared by "
            "model.prepare_fit_inputs(...)."
        )
    missing = {"eeg", "subject_id", "trial_id"}.difference(x)
    if missing:
        raise ValueError(
            "MLDG source inputs are missing "
            f"{sorted(missing)}. The model builder must receive aligned "
            "training_subject_ids and training_trial_ids."
        )
    if y is None:
        raise ValueError("MLDG source fitting requires labels.")

    resolved_sample_weight = (
        None
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    )
    if class_weight is not None:
        labels_array = np.asarray(y)
        if labels_array.ndim == 2 and labels_array.shape[-1] > 1:
            class_ids = np.argmax(labels_array, axis=-1)
        else:
            class_ids = labels_array.reshape(-1)
        class_weights = np.asarray(
            [class_weight.get(int(class_id), 1.0) for class_id in class_ids],
            dtype=np.float32,
        )
        resolved_sample_weight = (
            class_weights
            if resolved_sample_weight is None
            else resolved_sample_weight * class_weights
        )

    episode_steps = (
        model.mldg_steps_per_epoch
        if model.mldg_steps_per_epoch is not None
        else steps_per_epoch
    )
    episodes = SICMLDGEpisodeSequence(
        eeg=x["eeg"],
        labels=y,
        subject_ids=x["subject_id"],
        trial_ids=x["trial_id"],
        sample_weight=resolved_sample_weight,
        meta_train_subjects=model.mldg_meta_train_subjects,
        meta_test_subjects=model.mldg_meta_test_subjects,
        trials_per_subject=model.mldg_trials_per_subject,
        steps_per_epoch=episode_steps,
        seed=model.mldg_seed,
    )
    # One item is already one complete episode; Keras must not rebatch it
    #
    return tf.keras.Model.fit(
        model,
        x=episodes,
        y=None,
        batch_size=None,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=0.0,
        validation_data=validation_data,
        shuffle=False,
        class_weight=None,
        sample_weight=None,
        initial_epoch=initial_epoch,
        steps_per_epoch=None,
        validation_steps=validation_steps,
        validation_batch_size=validation_batch_size,
        validation_freq=validation_freq,
        **kwargs,
    )


def _dense_gradient(gradient):
    if gradient is None:
        return None
    if isinstance(gradient, tf.IndexedSlices):
        return tf.convert_to_tensor(gradient)
    return gradient


def _combine_first_order_gradients(meta_train, meta_test, beta):
    combined = []
    for train_gradient, test_gradient in zip(meta_train, meta_test):
        train_gradient = _dense_gradient(train_gradient)
        test_gradient = _dense_gradient(test_gradient)
        if train_gradient is None:
            gradient = (
                None
                if test_gradient is None
                else tf.cast(beta, test_gradient.dtype) * test_gradient
            )
        elif test_gradient is None:
            gradient = train_gradient
        else:
            gradient = (
                train_gradient + tf.cast(beta, test_gradient.dtype) * test_gradient
            )
        combined.append(gradient)
    return combined


def _gradient_cosine_similarity(left, right):
    dot = tf.zeros((), dtype=tf.float32)
    left_norm_sq = tf.zeros((), dtype=tf.float32)
    right_norm_sq = tf.zeros((), dtype=tf.float32)
    for left_gradient, right_gradient in zip(left, right):
        left_gradient = _dense_gradient(left_gradient)
        right_gradient = _dense_gradient(right_gradient)
        if left_gradient is None or right_gradient is None:
            continue
        left_flat = tf.cast(tf.reshape(left_gradient, [-1]), tf.float32)
        right_flat = tf.cast(tf.reshape(right_gradient, [-1]), tf.float32)
        dot += tf.reduce_sum(left_flat * right_flat)
        left_norm_sq += tf.reduce_sum(tf.square(left_flat))
        right_norm_sq += tf.reduce_sum(tf.square(right_flat))
    return tf.math.divide_no_nan(dot, tf.sqrt(left_norm_sq * right_norm_sq))


def run_sic_mldg_train_step(model, x, y_flat, sample_weight) -> None:
    """Execute SIC's first-order MLDG update outside the model definition."""
    if not isinstance(x, Mapping) or "mldg_role" not in x:
        raise ValueError(
            "MLDG train_step requires the episode roles produced by "
            "SICMLDGEpisodeSequence."
        )
    eeg_inputs, subject_ids = model._split_eeg_and_subject_inputs(x)
    if subject_ids is None:
        raise ValueError("MLDG source training requires subject IDs.")
    roles = tf.cast(tf.reshape(x["mldg_role"], [-1]), tf.int32)
    meta_train_mask = tf.equal(roles, 0)
    meta_test_mask = tf.equal(roles, 1)
    tf.debugging.assert_positive(
        tf.reduce_sum(tf.cast(meta_train_mask, tf.int32)),
        message="Every MLDG episode needs meta-train examples.",
    )
    tf.debugging.assert_positive(
        tf.reduce_sum(tf.cast(meta_test_mask, tf.int32)),
        message="Every MLDG episode needs meta-test examples.",
    )

    def masked(value, mask):
        return None if value is None else tf.boolean_mask(value, mask)

    meta_train_eeg = masked(eeg_inputs, meta_train_mask)
    meta_train_y = masked(y_flat, meta_train_mask)
    meta_train_subject_ids = masked(subject_ids, meta_train_mask)
    meta_train_weight = masked(sample_weight, meta_train_mask)
    meta_test_eeg = masked(eeg_inputs, meta_test_mask)
    meta_test_y = masked(y_flat, meta_test_mask)
    meta_test_subject_ids = masked(subject_ids, meta_test_mask)
    meta_test_weight = masked(sample_weight, meta_test_mask)

    # A keeps SIC's complete source objective. B measures emotion transfer only.
    with tf.GradientTape() as meta_train_tape:
        meta_train_outputs = model._encode(meta_train_eeg, training=True)
        meta_train_vc = model._vc_components(
            meta_train_outputs["classification_embedding"],
            meta_train_outputs["logits"],
            meta_train_y,
            meta_train_weight,
            calibration=False,
        )
        meta_train_subject = model._subject_components(
            meta_train_outputs["pooled_features"],
            meta_train_subject_ids,
            training=True,
            use_grl=True,
        )
        dtype = meta_train_vc["total_loss"].dtype
        meta_train_loss = (
            tf.cast(model.vc_loss_weight, dtype) * meta_train_vc["total_loss"]
            + tf.cast(model.subject_loss_weight, dtype)
            * tf.cast(meta_train_subject["subject_loss"], dtype)
            + model._regularization_loss(dtype)
        )
    variables = model.trainable_variables
    meta_train_gradients = meta_train_tape.gradient(meta_train_loss, variables)

    # The temporary inner update is detached: this is first-order MLDG.
    original_values = [tf.identity(variable) for variable in variables]
    for variable, gradient in zip(variables, meta_train_gradients):
        gradient = _dense_gradient(gradient)
        if gradient is not None:
            variable.assign_sub(
                tf.cast(model.mldg_inner_learning_rate, gradient.dtype)
                * tf.stop_gradient(gradient)
            )

    with tf.GradientTape() as meta_test_tape:
        meta_test_outputs = model._encode(meta_test_eeg, training=True)
        meta_test_vc = model._vc_components(
            meta_test_outputs["classification_embedding"],
            meta_test_outputs["logits"],
            meta_test_y,
            meta_test_weight,
            calibration=False,
        )
        meta_test_loss = (
            tf.cast(model.vc_loss_weight, meta_test_vc["total_loss"].dtype)
            * meta_test_vc["total_loss"]
        )
    meta_test_gradients = meta_test_tape.gradient(meta_test_loss, variables)

    for variable, original_value in zip(variables, original_values):
        variable.assign(original_value)
    combined_gradients = _combine_first_order_gradients(
        meta_train_gradients,
        meta_test_gradients,
        model.mldg_meta_test_weight,
    )
    model._apply_gradients(model.main_optimizer, combined_gradients, variables)

    if model.update_vc_discriminator:
        if model.vc_discriminator_optimizer is None:
            raise RuntimeError(
                "update_vc_discriminator=True requires a discriminator optimizer."
            )
        embedding_frozen = tf.stop_gradient(
            meta_train_outputs["classification_embedding"]
        )
        with tf.GradientTape() as disc_tape:
            disc_loss = model.vc_target.discriminator_loss(
                embedding_frozen,
                meta_train_y,
            )
        disc_variables = model._vc_discriminator_variables()
        disc_gradients = disc_tape.gradient(disc_loss, disc_variables)
        model._apply_gradients(
            model.vc_discriminator_optimizer,
            disc_gradients,
            disc_variables,
        )

    outer_loss = (
        meta_train_loss
        + tf.cast(model.mldg_meta_test_weight, meta_test_loss.dtype) * meta_test_loss
    )
    model._update_metrics(
        total_loss=outer_loss,
        vc_components=meta_train_vc,
        outputs=meta_train_outputs,
        y_flat=meta_train_y,
        sample_weight=meta_train_weight,
        subject_components=meta_train_subject,
        vrex_components=None,
    )
    model.mldg_meta_train_loss_tracker.update_state(meta_train_loss)
    model.mldg_meta_test_loss_tracker.update_state(meta_test_loss)
    model.mldg_meta_train_subjects_tracker.update_state(
        tf.cast(tf.size(tf.unique(meta_train_subject_ids).y), tf.float32)
    )
    model.mldg_meta_test_subjects_tracker.update_state(
        tf.cast(tf.size(tf.unique(meta_test_subject_ids).y), tf.float32)
    )
    model.mldg_gradient_cosine_tracker.update_state(
        _gradient_cosine_similarity(meta_train_gradients, meta_test_gradients)
    )


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
            raise ValueError("MLDG X, y, and subject_ids must have matching lengths.")

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
