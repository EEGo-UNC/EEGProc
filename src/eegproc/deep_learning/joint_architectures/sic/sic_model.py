"""SIC: Subject Invariant Calibrator.

MTLFuseNet GCN/GRU -> temporal BiLSTM -> variational z -> dense-softmax.

Architecture
------------
For each EEG window::

    channel-band EEG
        -> MTLFuseNet-style fixed-MI shared GCN
        -> spectral GRU across frequency bands
        -> temporal BiLSTM
        -> q(z | x) = N(z_mean, diag(exp(z_log_var)))

The pooled z representation feeds a conventional dense -> softmax emotion
classifier.  EEGProc's existing VariationalClassifier is used as an auxiliary
VC target/regularizer on the dense classifier embedding; the dense logits are
still the actual prediction logits.

The generative path uses EEGProc's existing graph-aware MTL decoder::

    z sequence
        -> temporal projection / upsampling
        -> fixed-MI MTL-style graph decoding
        -> reconstructed channel-band EEG

The decoder is intentionally simpler than the encoder: it is a reconstruction
module, not a claimed mathematical inverse of the GCN/GRU/BiLSTM encoder.

Subject invariance can be imposed directly on pooled z with ordinary gradient
reversal.  There is no adversarial takeover/recovery controller: emotion/VC,
VAE, and (when enabled) subject-adversarial objectives are optimized together
in the same source update.

V-REx can additionally regularize source training.  Each source subject present
in a minibatch is treated as an environment.  SIC computes the focal risk
separately for each subject and adds the variance of those risks to the ordinary
source objective.  This uses the same batched forward pass and a single
backward/optimizer step.

First-order MLDG is available as a mutually exclusive source-training method.
Every source epoch is composed of subject-disjoint meta-train/meta-test
episodes.  A temporary plain-SGD step is taken on the meta-train objective,
the meta-test focal-classification gradient is evaluated at those adapted
parameters, and the original parameters receive one combined outer-optimizer
update.  The temporary assignment is detached, so no Hessian or
gradient-through-gradient path is constructed.

Subject calibration
-------------------
``prepare_for_subject_calibration`` freezes the complete representation,
posterior, decoder, subject adversary, and VC target parameters.  It then
unfreezes only the last ``calibration_unfreeze_layers`` prediction layers:

    1 -> softmax/logits only
    2 -> last dense hidden block + softmax
    3 -> last two dense hidden blocks + softmax
    ...

The calibration train step uses the dense-softmax classification objective and,
when enabled, the same frozen VC target.  Therefore VC regularization can shape
any unfrozen dense hidden representation while a softmax-only calibration
reduces naturally to fitting the output decision boundary.

This file is designed for ``cross_val.subject_calibration_cv``.  Its builder
accepts ``training_features`` and computes the fixed MI adjacency from source
subjects only, avoiding target-subject leakage.

Ablations are controlled by ``use_gcn_gru_branch``, ``use_bilstm_branch``, and
``use_decoder``. At least one encoder branch must remain enabled. Disabling the
decoder removes reconstruction and decoder parameters while retaining the
variational posterior KL term, which isolates the decoder/reconstruction
contribution without converting the classifier into a deterministic model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from ...supervised.variational_classifier import VariationalClassifier
    from ...unsupervised.GNN.GCNMTL import (
        GCNMTLEncoder,
        GCNMTLDecoder,
        compute_mtl_shared_mi_adjacency,
    )
except ImportError:
    from eegproc.deep_learning.supervised.variational_classifier import (
        VariationalClassifier,
    )
    from eegproc.deep_learning.unsupervised.GNN.GCNMTL import (
        GCNMTLEncoder,
        GCNMTLDecoder,
        compute_mtl_shared_mi_adjacency,
    )


SIC_BUILDER_API_VERSION = 7
JOINT_V6_BUILDER_API_VERSION = SIC_BUILDER_API_VERSION


_TRAINING_METHOD_ALIASES = {
    "normal": "erm",
    "standard": "erm",
    "joint": "erm",
    "erm": "erm",
    "vrex": "vrex",
    "v-rex": "vrex",
    "v_rex": "vrex",
    "mldg": "mldg",
    "fo_mldg": "mldg",
    "first_order_mldg": "mldg",
}


def _resolve_training_method(
    training_method: str | None,
    use_vrex: bool,
) -> str:
    """Resolve the new method selector while preserving old V-REx configs."""
    if training_method is None:
        return "vrex" if bool(use_vrex) else "erm"
    normalized = str(training_method).strip().lower().replace("-", "_")
    normalized = _TRAINING_METHOD_ALIASES.get(normalized, normalized)
    if normalized not in {"erm", "vrex", "mldg"}:
        raise ValueError(
            "training_method must be one of 'erm', 'vrex', or 'mldg'; "
            f"got {training_method!r}."
        )
    if bool(use_vrex) and normalized != "vrex":
        raise ValueError(
            "use_vrex=true conflicts with training_method="
            f"{training_method!r}. Use training_method='vrex' or remove the "
            "legacy use_vrex setting."
        )
    return normalized


class _MLDGEpisodeSequence(tf.keras.utils.Sequence):
    """Precompute balanced, complete-trial first-order MLDG episodes.

    Every item is one meta-learning episode and therefore one Keras train step.
    Subject roles are balanced across an epoch.  Trial groups are sampled as
    indivisible units, so window-level training never splits a selected trial.
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
    ):
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

    def __len__(self):
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

    def _build_epoch(self):
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
            remaining = [
                subject_id
                for subject_id in all_subjects
                if subject_id not in set(meta_test)
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

    def __getitem__(self, index):
        indices, roles = self._episodes[int(index)]
        inputs = {
            "eeg": self.eeg[indices],
            "subject_id": self.subject_ids[indices],
            "mldg_role": roles,
        }
        if self.sample_weight is None:
            return inputs, self.labels[indices]
        return inputs, self.labels[indices], self.sample_weight[indices]

    def on_epoch_end(self):
        self._epoch += 1
        self._build_epoch()


def _build_optimizer(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> tf.keras.optimizers.Optimizer:
    optimizer_name = str(optimizer_name).lower()
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer_name must be 'adam' or 'adamw'.")


def _as_positive_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    output = tuple(int(value) for value in values)
    if not output or any(value < 1 for value in output):
        raise ValueError(f"{name} must contain positive integers; got {values}.")
    return output


def _resolve_temporal_pool_sizes(
    temporal_pool_sizes: Sequence[int] | None,
    t_down: int,
) -> tuple[int, ...]:
    t_down = int(t_down)
    if t_down < 1:
        raise ValueError("t_down must be >= 1.")
    if temporal_pool_sizes is None:
        pools = () if t_down == 1 else (t_down,)
    else:
        pools = tuple(int(value) for value in temporal_pool_sizes)
    if any(value < 1 for value in pools):
        raise ValueError("temporal_pool_sizes values must be >= 1.")
    effective = int(np.prod(pools, dtype=np.int64)) if pools else 1
    if effective != t_down:
        raise ValueError(
            f"t_down={t_down}, but temporal_pool_sizes={pools} gives {effective}."
        )
    return pools


def _deduplicate_variables(variables):
    seen: set[int] = set()
    output = []
    for variable in variables:
        identifier = id(variable)
        if identifier not in seen:
            seen.add(identifier)
            output.append(variable)
    return output


def _serialize_keras_object(value):
    if value is None:
        return None
    return tf.keras.utils.serialize_keras_object(value)


def _deserialize_keras_object(value):
    if value is None:
        return None
    return tf.keras.utils.deserialize_keras_object(value)


def _source_only_mi_adjacency(
    training_features,
    *,
    n_channels: int,
    n_bands: int,
    n_neighbors: int,
    random_state: int,
    zero_diagonal: bool,
    band_reduction: str,
    max_observations: int | None,
) -> np.ndarray:
    """Compute one fixed MI graph from source-training features only.

    ``training_features`` may be rank 3 (windows, time, features) or rank 4
    (trials, windows, time, features).  All leading axes are observations for
    MI estimation.  Optional row subsampling keeps this step practical.
    """
    x = np.asarray(training_features, dtype=np.float32)
    expected_features = int(n_channels) * int(n_bands)
    if x.ndim < 2 or x.shape[-1] != expected_features:
        raise ValueError(
            "training_features must end in n_channels*n_bands features; "
            f"got {x.shape}, expected last dimension {expected_features}."
        )
    observations = x.reshape(-1, expected_features)
    if max_observations is not None:
        max_observations = int(max_observations)
        if max_observations < 4:
            raise ValueError("mi_max_observations must be >= 4 or None.")
        if len(observations) > max_observations:
            rng = np.random.default_rng(int(random_state))
            indices = rng.choice(
                len(observations),
                size=max_observations,
                replace=False,
            )
            observations = observations[indices]

    return compute_mtl_shared_mi_adjacency(
        observations,
        n_channels=int(n_channels),
        n_bands=int(n_bands),
        n_neighbors=int(n_neighbors),
        random_state=int(random_state),
        zero_diagonal=bool(zero_diagonal),
        band_reduction=str(band_reduction),
    )


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class StreamingBalancedAccuracy(tf.keras.metrics.Metric):
    """Balanced accuracy accumulated from one epoch-wide confusion matrix."""

    def __init__(self, n_classes: int = 2, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = int(n_classes)
        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(self.n_classes, self.n_classes),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.convert_to_tensor(y_pred)
        if y_pred.shape.rank is not None and y_pred.shape.rank > 1:
            y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        else:
            y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.int32)
        y_pred = tf.reshape(y_pred, [-1])

        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)

        matrix = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.n_classes,
            weights=weights,
            dtype=tf.float32,
        )
        self.confusion_matrix.assign_add(matrix)

    def result(self):
        true_counts = tf.reduce_sum(self.confusion_matrix, axis=1)
        recalls = tf.math.divide_no_nan(
            tf.linalg.diag_part(self.confusion_matrix),
            true_counts,
        )
        return tf.reduce_mean(recalls)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def get_config(self):
        config = super().get_config()
        config.update({"n_classes": self.n_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class StreamingR2(tf.keras.metrics.Metric):
    """Epoch-wide R^2 for continuous decoder reconstruction."""

    def __init__(self, name="decoder_r2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name="ss_res", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_y_sq = self.add_weight(name="sum_y_sq", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        y_true = tf.cast(tf.reshape(y_true, [-1]), self.dtype)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), self.dtype)
        self.ss_res.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.sum_y.assign_add(tf.reduce_sum(y_true))
        self.sum_y_sq.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.count.assign_add(tf.cast(tf.size(y_true), self.dtype))

    def result(self):
        ss_tot = self.sum_y_sq - tf.math.divide_no_nan(
            tf.square(self.sum_y),
            self.count,
        )
        eps = tf.cast(tf.keras.backend.epsilon(), self.dtype)
        return tf.where(
            ss_tot > eps,
            1.0 - tf.math.divide_no_nan(self.ss_res, ss_tot),
            tf.zeros((), dtype=self.dtype),
        )

    def reset_state(self):
        for variable in (self.ss_res, self.sum_y, self.sum_y_sq, self.count):
            variable.assign(tf.zeros_like(variable))


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class GradientReversal(tf.keras.layers.Layer):
    """Identity forward pass with a negative scaled encoder gradient."""

    def __init__(self, adversarial_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if float(adversarial_weight) < 0.0:
            raise ValueError("adversarial_weight must be non-negative.")
        self.adversarial_weight = float(adversarial_weight)

    def call(self, inputs):
        scale = self.adversarial_weight

        @tf.custom_gradient
        def reverse_gradient(x):
            def gradient(dy):
                return -tf.cast(scale, dy.dtype) * dy

            return x, gradient

        return reverse_gradient(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"adversarial_weight": self.adversarial_weight})
        return config


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class SICModel(tf.keras.Model):
    """SIC with serial or parallel feature-fusion encoder and focal classifier."""

    def __init__(
        self,
        *,
        graph_encoder: GCNMTLEncoder | None,
        decoder: GCNMTLDecoder | None,
        classification_level: str = "trial",
        n_classes: int = 2,
        bilstm_units: int = 128,
        n_bilstm_layers: int = 1,
        bilstm_dropout: float = 0.30,
        architecture_mode: str = "serial",
        use_gcn_gru_branch: bool = True,
        use_bilstm_branch: bool = True,
        use_decoder: bool = True,
        fusion_units: int = 128,
        fusion_dropout: float = 0.20,
        temporal_downsample_factor: int = 1,
        z_dim: int = 64,
        z_log_var_clip_min: float = -20.0,
        z_log_var_clip_max: float = 20.0,
        classification_hidden_units: Sequence[int] = (128,),
        classification_dropout: float = 0.20,
        activation: str = "relu",
        label_smoothing: float = 0.0,
        focal_gamma: float = 1.0,
        focal_alpha: float | None = None,
        vc_loss_weight: float = 1.0,
        vc_alpha: float = 1.0,
        vc_beta: float = 0.5,
        vc_gamma: float = 0.0,
        vc_lambda: float = 0.0,
        update_vc_discriminator: bool = False,
        vae_loss_weight: float = 0.10,
        vae_beta: float = 0.05,
        training_method: str | None = None,
        use_vrex: bool = False,
        vrex_penalty_weight: float = 1.0,
        mldg_meta_train_subjects: int | None = None,
        mldg_meta_test_subjects: int = 4,
        mldg_trials_per_subject: int = 1,
        mldg_steps_per_epoch: int | None = None,
        mldg_inner_learning_rate: float = 1e-4,
        mldg_meta_test_weight: float = 1.0,
        mldg_seed: int | None = 42,
        use_subject_adversarial: bool = True,
        n_subject_classes: int | None = None,
        subject_adversarial_weight: float = 0.8,
        subject_loss_weight: float = 1.0,
        subject_hidden_units: int = 64,
        subject_dropout: float = 0.0,
        calibration_unfreeze_layers: int = 1,
        calibration_use_vc_target: bool = True,
        calibration_vc_alpha: float | None = None,
        calibration_vc_beta: float | None = None,
        calibration_vc_gamma: float | None = None,
        calibration_vc_lambda: float | None = None,
        use_class_weight: bool = False,
        name: str = "sic_subject_invariant_calibrator",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        classification_level = str(classification_level).lower()
        if classification_level not in {"window", "trial"}:
            raise ValueError("classification_level must be 'window' or 'trial'.")
        use_gcn_gru_branch = bool(use_gcn_gru_branch)
        use_bilstm_branch = bool(use_bilstm_branch)
        use_decoder = bool(use_decoder)
        if not use_gcn_gru_branch and not use_bilstm_branch:
            raise ValueError(
                "At least one of use_gcn_gru_branch/use_bilstm_branch must be true."
            )
        if use_gcn_gru_branch and graph_encoder is None:
            raise ValueError("graph_encoder is required when use_gcn_gru_branch=true.")
        if use_decoder and decoder is None:
            raise ValueError("decoder is required when use_decoder=true.")
        if int(n_classes) < 2:
            raise ValueError("n_classes must be >= 2.")
        if int(bilstm_units) < 1 or int(n_bilstm_layers) < 1 or int(z_dim) < 1:
            raise ValueError("BiLSTM and z dimensions must be positive.")
        architecture_mode = str(architecture_mode).lower()
        if architecture_mode not in {"serial", "feature_fusion"}:
            raise ValueError("architecture_mode must be 'serial' or 'feature_fusion'.")
        if int(fusion_units) < 1:
            raise ValueError("fusion_units must be positive.")
        if not 0.0 <= float(fusion_dropout) < 1.0:
            raise ValueError("fusion_dropout must be in [0, 1).")
        if int(temporal_downsample_factor) < 1:
            raise ValueError("temporal_downsample_factor must be >= 1.")
        if float(focal_gamma) < 0.0:
            raise ValueError("focal_gamma must be non-negative.")
        if focal_alpha is not None and not 0.0 <= float(focal_alpha) <= 1.0:
            raise ValueError("focal_alpha must be in [0, 1] or None.")
        if float(z_log_var_clip_min) >= float(z_log_var_clip_max):
            raise ValueError("z_log_var_clip_min must be less than max.")
        hidden_units = tuple(int(value) for value in classification_hidden_units)
        if any(value < 1 for value in hidden_units):
            raise ValueError("classification_hidden_units must be positive.")
        if not 0.0 <= float(classification_dropout) < 1.0:
            raise ValueError("classification_dropout must be in [0, 1).")
        if not 0.0 <= float(bilstm_dropout) < 1.0:
            raise ValueError("bilstm_dropout must be in [0, 1).")
        if not 0.0 <= float(label_smoothing) < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        for loss_name, value in (
            ("vc_loss_weight", vc_loss_weight),
            ("vae_loss_weight", vae_loss_weight),
            ("vae_beta", vae_beta),
            ("subject_adversarial_weight", subject_adversarial_weight),
            ("subject_loss_weight", subject_loss_weight),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{loss_name} must be non-negative.")
        if int(subject_hidden_units) < 1:
            raise ValueError("subject_hidden_units must be positive.")
        if not 0.0 <= float(subject_dropout) < 1.0:
            raise ValueError("subject_dropout must be in [0, 1).")
        if float(vrex_penalty_weight) < 0.0:
            raise ValueError("vrex_penalty_weight must be non-negative.")
        resolved_training_method = _resolve_training_method(
            training_method,
            use_vrex,
        )
        if mldg_meta_train_subjects is not None and int(mldg_meta_train_subjects) < 1:
            raise ValueError("mldg_meta_train_subjects must be >= 1 or null.")
        if int(mldg_meta_test_subjects) < 1:
            raise ValueError("mldg_meta_test_subjects must be >= 1.")
        if int(mldg_trials_per_subject) < 1:
            raise ValueError("mldg_trials_per_subject must be >= 1.")
        if mldg_steps_per_epoch is not None and int(mldg_steps_per_epoch) < 1:
            raise ValueError("mldg_steps_per_epoch must be >= 1 or null.")
        if float(mldg_inner_learning_rate) <= 0.0:
            raise ValueError("mldg_inner_learning_rate must be positive.")
        if float(mldg_meta_test_weight) < 0.0:
            raise ValueError("mldg_meta_test_weight must be non-negative.")
        calibration_unfreeze_layers = int(calibration_unfreeze_layers)
        max_calibration_layers = len(hidden_units) + 1
        if not 1 <= calibration_unfreeze_layers <= max_calibration_layers:
            raise ValueError(
                "calibration_unfreeze_layers must be between 1 and "
                f"{max_calibration_layers}; got {calibration_unfreeze_layers}."
            )

        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.classification_level = classification_level
        self.n_classes = int(n_classes)
        self.bilstm_units = int(bilstm_units)
        self.n_bilstm_layers = int(n_bilstm_layers)
        self.bilstm_dropout_rate = float(bilstm_dropout)
        self.architecture_mode = architecture_mode
        self.use_gcn_gru_branch = use_gcn_gru_branch
        self.use_bilstm_branch = use_bilstm_branch
        self.use_decoder = use_decoder
        self.fusion_units = int(fusion_units)
        self.fusion_dropout_rate = float(fusion_dropout)
        self.temporal_downsample_factor = int(temporal_downsample_factor)
        self.z_dim = int(z_dim)
        self.z_log_var_clip_min = float(z_log_var_clip_min)
        self.z_log_var_clip_max = float(z_log_var_clip_max)
        self.classification_hidden_units = hidden_units
        self.classification_dropout_rate = float(classification_dropout)
        self.activation_name = str(activation)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)

        self.vc_loss_weight = float(vc_loss_weight)
        self.vc_alpha = float(vc_alpha)
        self.vc_beta = float(vc_beta)
        self.vc_gamma = float(vc_gamma)
        self.vc_lambda = float(vc_lambda)
        self.update_vc_discriminator = bool(update_vc_discriminator)
        self.vae_loss_weight = float(vae_loss_weight)
        self.vae_beta = float(vae_beta)
        self.use_class_weight = bool(use_class_weight)

        self.training_method = resolved_training_method
        self.use_vrex = self.training_method == "vrex"
        self.use_mldg = self.training_method == "mldg"
        self.vrex_penalty_weight = float(vrex_penalty_weight)
        self.mldg_meta_train_subjects = (
            None if mldg_meta_train_subjects is None else int(mldg_meta_train_subjects)
        )
        self.mldg_meta_test_subjects = int(mldg_meta_test_subjects)
        self.mldg_trials_per_subject = int(mldg_trials_per_subject)
        self.mldg_steps_per_epoch = (
            None if mldg_steps_per_epoch is None else int(mldg_steps_per_epoch)
        )
        self.mldg_inner_learning_rate = float(mldg_inner_learning_rate)
        self.mldg_meta_test_weight = float(mldg_meta_test_weight)
        self.mldg_seed = None if mldg_seed is None else int(mldg_seed)

        self.subject_adversarial_enabled = bool(use_subject_adversarial)
        self.n_subject_classes = (
            None if n_subject_classes is None else int(n_subject_classes)
        )
        self.subject_adversarial_weight = float(subject_adversarial_weight)
        self.subject_loss_weight = float(subject_loss_weight)
        self.subject_hidden_units = int(subject_hidden_units)
        self.subject_dropout_rate = float(subject_dropout)

        self.calibration_unfreeze_layers = calibration_unfreeze_layers
        self.calibration_use_vc_target = bool(calibration_use_vc_target)
        self.calibration_vc_alpha = (
            self.vc_alpha
            if calibration_vc_alpha is None
            else float(calibration_vc_alpha)
        )
        self.calibration_vc_beta = (
            self.vc_beta if calibration_vc_beta is None else float(calibration_vc_beta)
        )
        self.calibration_vc_gamma = (
            self.vc_gamma
            if calibration_vc_gamma is None
            else float(calibration_vc_gamma)
        )
        self.calibration_vc_lambda = (
            self.vc_lambda
            if calibration_vc_lambda is None
            else float(calibration_vc_lambda)
        )
        self.calibration_mode = False

        self.requires_subject_ids = (
            self.subject_adversarial_enabled or self.use_vrex or self.use_mldg
        )
        # Current EEGProc cross_val uses ``use_subject_adversarial`` as the
        # compatibility gate for attaching source subject IDs.  Keep that gate
        # true whenever V-REx or MLDG needs subject metadata; actual GRL
        # computation is still controlled exclusively by
        # ``subject_adversarial_enabled``.
        self.use_subject_adversarial = self.requires_subject_ids
        self._source_subject_ids = None
        self._source_trial_ids = None

        # MTL encoder already performs GCN + spectral GRU.  This recurrent
        # stack is exclusively temporal, preserving the reduced time axis.
        self.temporal_bilstms: list[tf.keras.layers.Layer] = []
        self.temporal_norms: list[tf.keras.layers.Layer] = []
        self.temporal_dropouts: list[tf.keras.layers.Layer] = []
        for index in range(self.n_bilstm_layers):
            self.temporal_bilstms.append(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        self.bilstm_units,
                        return_sequences=True,
                        name=f"v6_temporal_lstm_{index}",
                    ),
                    merge_mode="concat",
                    name=f"v6_temporal_bilstm_{index}",
                )
            )
            self.temporal_norms.append(
                tf.keras.layers.LayerNormalization(
                    axis=-1,
                    name=f"v6_temporal_bilstm_ln_{index}",
                )
            )
            self.temporal_dropouts.append(
                tf.keras.layers.Dropout(
                    self.bilstm_dropout_rate,
                    name=f"v6_temporal_bilstm_dropout_{index}",
                )
            )

        # In serial mode the BiLSTM consumes the GCN-GRU sequence.
        # In feature_fusion mode the BiLSTM is an independent raw-EEG branch;
        # it is downsampled to the GCN-GRU temporal resolution and fused before z.
        self.parallel_temporal_pool = tf.keras.layers.AveragePooling1D(
            pool_size=self.temporal_downsample_factor,
            strides=self.temporal_downsample_factor,
            padding="same",
            name="v6_parallel_bilstm_pool",
        )
        self.fusion_projection = tf.keras.layers.Dense(
            self.fusion_units,
            activation=self.activation_name,
            name="v6_feature_fusion_dense",
        )
        self.fusion_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            name="v6_feature_fusion_ln",
        )
        self.fusion_dropout_layer = tf.keras.layers.Dropout(
            self.fusion_dropout_rate,
            name="v6_feature_fusion_dropout",
        )

        self.z_mean_projection = tf.keras.layers.Dense(
            self.z_dim,
            activation=None,
            name="v6_z_mean",
        )
        self.z_log_var_projection = tf.keras.layers.Dense(
            self.z_dim,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="v6_z_log_var",
        )
        self.z_pool = tf.keras.layers.GlobalAveragePooling1D(name="v6_z_pool")

        # Dense-softmax prediction head. Hidden blocks are explicit so the
        # calibration policy can unfreeze exactly the last k prediction layers.
        self.classification_dense_layers: list[tf.keras.layers.Layer] = []
        self.classification_norm_layers: list[tf.keras.layers.Layer] = []
        self.classification_dropout_layers: list[tf.keras.layers.Layer] = []
        for index, units in enumerate(self.classification_hidden_units):
            self.classification_dense_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation=self.activation_name,
                    name=f"v6_classifier_dense_{index}",
                )
            )
            self.classification_norm_layers.append(
                tf.keras.layers.LayerNormalization(
                    axis=-1,
                    name=f"v6_classifier_ln_{index}",
                )
            )
            self.classification_dropout_layers.append(
                tf.keras.layers.Dropout(
                    self.classification_dropout_rate,
                    name=f"v6_classifier_dropout_{index}",
                )
            )
        self.logits_layer = tf.keras.layers.Dense(
            self.n_classes,
            activation=None,
            name="v6_classifier_logits",
        )

        vc_dim = (
            self.classification_hidden_units[-1]
            if self.classification_hidden_units
            else self.z_dim
        )
        self.vc_target = VariationalClassifier(
            n_classes=self.n_classes,
            latent_dim=vc_dim,
            label_smoothing=self.label_smoothing,
            name="v6_vc_target",
        )
        # We pass externally produced dense logits into vc_loss_components, so
        # explicitly build the target's Gaussian/discriminator parameters.
        self.vc_target.build(tf.TensorShape([None, vc_dim]))

        self.subject_gradient_reversal = None
        self.subject_hidden = None
        self.subject_dropout_layer = None
        self.subject_logits_layer = None
        if self.subject_adversarial_enabled and self.n_subject_classes is not None:
            self._configure_subject_head(self.n_subject_classes)

        self.main_optimizer = None
        self.vc_discriminator_optimizer = None

        # Keras metrics.  The same classification metrics are emitted both
        # during source training and calibration; cross_val additionally logs
        # paired zero-shot and post-calibration trial metrics.
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.vc_loss_tracker = tf.keras.metrics.Mean(name="vc_loss")
        self.focal_loss_tracker = tf.keras.metrics.Mean(name="focal_loss")
        metric_prefix = self.classification_level
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name=f"{metric_prefix}_accuracy"
        )
        self.balanced_accuracy_tracker = StreamingBalancedAccuracy(
            n_classes=self.n_classes,
            name=f"{metric_prefix}_balanced_accuracy",
        )
        # Exact epoch-wide class fractions. During validation Keras exposes
        # these as val_predicted_class_1_fraction / val_true_class_1_fraction.
        self.predicted_class_1_fraction_tracker = tf.keras.metrics.Mean(
            name="predicted_class_1_fraction"
        )
        self.true_class_1_fraction_tracker = tf.keras.metrics.Mean(
            name="true_class_1_fraction"
        )
        self.vae_loss_tracker = tf.keras.metrics.Mean(name="vae_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.decoder_r2_tracker = StreamingR2(name="decoder_r2")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.vrex_penalty_tracker = tf.keras.metrics.Mean(name="vrex_penalty")
        self.vrex_subject_risk_mean_tracker = tf.keras.metrics.Mean(
            name="vrex_subject_risk_mean"
        )
        self.vrex_subjects_per_batch_tracker = tf.keras.metrics.Mean(
            name="vrex_subjects_per_batch"
        )
        self.mldg_meta_train_loss_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_train_loss"
        )
        self.mldg_meta_test_loss_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_test_loss"
        )
        self.mldg_meta_train_subjects_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_train_subjects"
        )
        self.mldg_meta_test_subjects_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_test_subjects"
        )
        self.mldg_gradient_cosine_tracker = tf.keras.metrics.Mean(
            name="mldg_gradient_cosine"
        )
        self.subject_loss_tracker = tf.keras.metrics.Mean(name="subject_loss")
        self.subject_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="subject_accuracy"
        )

    @property
    def metrics(self):
        output = [
            self.loss_tracker,
            self.vc_loss_tracker,
            self.focal_loss_tracker,
            self.accuracy_tracker,
            self.balanced_accuracy_tracker,
            self.predicted_class_1_fraction_tracker,
            self.true_class_1_fraction_tracker,
            self.vae_loss_tracker,
            self.kl_loss_tracker,
        ]
        if self.use_decoder:
            output.extend([self.reconstruction_loss_tracker, self.decoder_r2_tracker])
        if self.use_vrex:
            output.extend(
                [
                    self.vrex_penalty_tracker,
                    self.vrex_subject_risk_mean_tracker,
                    self.vrex_subjects_per_batch_tracker,
                ]
            )
        if self.use_mldg:
            output.extend(
                [
                    self.mldg_meta_train_loss_tracker,
                    self.mldg_meta_test_loss_tracker,
                    self.mldg_meta_train_subjects_tracker,
                    self.mldg_meta_test_subjects_tracker,
                    self.mldg_gradient_cosine_tracker,
                ]
            )
        if self.subject_adversarial_enabled:
            output.extend(
                [
                    self.subject_loss_tracker,
                    self.subject_accuracy_tracker,
                ]
            )
        return output

    def compile(
        self,
        main_optimizer,
        vc_discriminator_optimizer=None,
        **kwargs,
    ):
        if main_optimizer is None:
            raise ValueError("main_optimizer is required.")
        kwargs.setdefault("jit_compile", False)
        super().compile(optimizer=main_optimizer, **kwargs)
        self.main_optimizer = main_optimizer
        self.vc_discriminator_optimizer = vc_discriminator_optimizer

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        **kwargs,
    ):
        if not self.use_class_weight:
            class_weight = None
        if not self.use_mldg or self.calibration_mode:
            return super().fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_split=validation_split,
                validation_data=validation_data,
                shuffle=shuffle,
                class_weight=class_weight,
                sample_weight=sample_weight,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                validation_batch_size=validation_batch_size,
                validation_freq=validation_freq,
                **kwargs,
            )

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
            self.mldg_steps_per_epoch
            if self.mldg_steps_per_epoch is not None
            else steps_per_epoch
        )
        episodes = _MLDGEpisodeSequence(
            eeg=x["eeg"],
            labels=y,
            subject_ids=x["subject_id"],
            trial_ids=x["trial_id"],
            sample_weight=resolved_sample_weight,
            meta_train_subjects=self.mldg_meta_train_subjects,
            meta_test_subjects=self.mldg_meta_test_subjects,
            trials_per_subject=self.mldg_trials_per_subject,
            steps_per_epoch=episode_steps,
            seed=self.mldg_seed,
        )
        # One Sequence item is already one complete MLDG episode. Ordinary
        # source_batch_size/shuffle semantics must not rebatch or split it.
        return super().fit(
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

    @staticmethod
    def _flatten_labels(labels):
        labels = tf.convert_to_tensor(labels)
        if (
            labels.shape.rank == 2
            and labels.shape[-1] is not None
            and labels.shape[-1] > 1
        ):
            return tf.argmax(labels, axis=-1, output_type=tf.int32)
        return tf.cast(tf.reshape(labels, [-1]), tf.int32)

    @staticmethod
    def _split_eeg_and_subject_inputs(inputs):
        if isinstance(inputs, Mapping):
            if "eeg" not in inputs:
                raise ValueError("Input mappings must contain an 'eeg' key.")
            return inputs["eeg"], inputs.get("subject_id")
        return inputs, None

    def _configure_subject_head(self, n_subject_classes: int):
        n_subject_classes = int(n_subject_classes)
        if n_subject_classes < 2:
            raise ValueError("Subject adversity requires at least two subjects.")
        if self.subject_logits_layer is not None:
            if self.n_subject_classes != n_subject_classes:
                raise ValueError(
                    "Subject head already configured for "
                    f"{self.n_subject_classes}, not {n_subject_classes}."
                )
            return
        self.n_subject_classes = n_subject_classes
        self.subject_gradient_reversal = GradientReversal(
            adversarial_weight=self.subject_adversarial_weight,
            name="v6_subject_gradient_reversal",
        )
        self.subject_hidden = tf.keras.layers.Dense(
            self.subject_hidden_units,
            activation=self.activation_name,
            name="v6_subject_hidden",
        )
        self.subject_dropout_layer = tf.keras.layers.Dropout(
            self.subject_dropout_rate,
            name="v6_subject_dropout",
        )
        self.subject_logits_layer = tf.keras.layers.Dense(
            self.n_subject_classes,
            activation=None,
            name="v6_subject_logits",
        )

    def set_source_training_metadata(self, subject_ids, trial_ids):
        """Retain source-only IDs so MLDG can keep complete trials together."""
        if subject_ids is None or trial_ids is None:
            self._source_subject_ids = None
            self._source_trial_ids = None
            return
        subjects = np.asarray(subject_ids).reshape(-1)
        trials = np.asarray(trial_ids).reshape(-1)
        if len(subjects) != len(trials):
            raise ValueError("Source subject IDs and trial IDs must align.")
        self._source_subject_ids = subjects.copy()
        self._source_trial_ids = trials.copy()

    def prepare_fit_inputs(self, eeg_inputs, subject_ids):
        """Attach contiguous source-fold subject labels for adversarial training."""
        if not self.requires_subject_ids:
            return eeg_inputs
        eeg_array = np.asarray(eeg_inputs)
        subjects = np.asarray(subject_ids).reshape(-1)
        if len(eeg_array) != len(subjects):
            raise ValueError("EEG inputs and subject IDs must align.")
        unique_subjects = np.sort(np.unique(subjects))
        if self.subject_adversarial_enabled:
            self._configure_subject_head(len(unique_subjects))
        mapping = {
            value.item() if isinstance(value, np.generic) else value: index
            for index, value in enumerate(unique_subjects)
        }
        remapped = np.asarray(
            [
                mapping[value.item() if isinstance(value, np.generic) else value]
                for value in subjects
            ],
            dtype=np.int32,
        )
        output = {"eeg": eeg_array, "subject_id": remapped}
        matching_trial_ids = None
        if self._source_subject_ids is not None and self._source_trial_ids is not None:
            if len(subjects) == len(self._source_subject_ids) and np.array_equal(
                subjects,
                self._source_subject_ids,
            ):
                matching_trial_ids = self._source_trial_ids
            else:
                # cross_val may build from all source subjects and subsequently
                # reserve whole validation subjects. Boolean subject filtering
                # preserves row order, so recover the aligned trial IDs without
                # exposing target-subject data or guessing individual windows.
                source_mask = np.isin(
                    self._source_subject_ids,
                    np.unique(subjects),
                )
                candidate_subjects = self._source_subject_ids[source_mask]
                if len(candidate_subjects) == len(subjects) and np.array_equal(
                    candidate_subjects,
                    subjects,
                ):
                    matching_trial_ids = self._source_trial_ids[source_mask]
        if self.use_mldg and matching_trial_ids is not None:
            output["trial_id"] = np.asarray(matching_trial_ids).copy()
        return output

    def prepare_calibration_inputs(self, eeg_inputs):
        """Calibration intentionally has no subject-adversarial input."""
        return np.asarray(eeg_inputs)

    def _flatten_trial_windows(self, eeg_inputs):
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        if eeg_inputs.shape.rank != 4:
            raise ValueError(
                "Trial mode expects (batch, windows, timesteps, features); "
                f"got {eeg_inputs.shape}."
            )
        shape = tf.shape(eeg_inputs)
        return (
            tf.reshape(eeg_inputs, [shape[0] * shape[1], shape[2], shape[3]]),
            shape[0],
            shape[1],
        )

    def _temporal_encode(self, graph_sequence, training: bool):
        x = graph_sequence
        for bilstm, norm, dropout in zip(
            self.temporal_bilstms,
            self.temporal_norms,
            self.temporal_dropouts,
        ):
            x = bilstm(x, training=training)
            x = norm(x)
            x = dropout(x, training=training)
        return x

    def _posterior_from_flat_windows(self, flat_windows, training: bool):
        graph_sequence = (
            self.graph_encoder(flat_windows, training=training)
            if self.use_gcn_gru_branch
            else None
        )
        bilstm_sequence = None

        if self.architecture_mode == "serial":
            if self.use_gcn_gru_branch and self.use_bilstm_branch:
                bilstm_sequence = self._temporal_encode(
                    graph_sequence,
                    training=training,
                )
                fused_sequence = bilstm_sequence
            elif self.use_gcn_gru_branch:
                fused_sequence = graph_sequence
            else:
                bilstm_sequence = self._temporal_encode(
                    flat_windows,
                    training=training,
                )
                fused_sequence = self.parallel_temporal_pool(bilstm_sequence)
        else:
            pooled_bilstm = None
            if self.use_bilstm_branch:
                # Independent temporal branch directly from the raw EEG feature
                # sequence. No GCN/GRU features enter this BiLSTM branch.
                bilstm_sequence = self._temporal_encode(
                    flat_windows,
                    training=training,
                )
                pooled_bilstm = self.parallel_temporal_pool(bilstm_sequence)

            if self.use_gcn_gru_branch and self.use_bilstm_branch:
                tf.debugging.assert_equal(
                    tf.shape(graph_sequence)[1],
                    tf.shape(pooled_bilstm)[1],
                    message=(
                        "GCN-GRU and BiLSTM branch sequence lengths do not match "
                        "for feature fusion."
                    ),
                )
                fused_sequence = tf.concat(
                    [graph_sequence, pooled_bilstm],
                    axis=-1,
                )
                fused_sequence = self.fusion_projection(fused_sequence)
                fused_sequence = self.fusion_norm(fused_sequence)
                fused_sequence = self.fusion_dropout_layer(
                    fused_sequence,
                    training=training,
                )
            elif self.use_gcn_gru_branch:
                fused_sequence = graph_sequence
            else:
                fused_sequence = pooled_bilstm
        temporal_sequence = fused_sequence

        z_mean = self.z_mean_projection(fused_sequence)
        raw_log_var = self.z_log_var_projection(fused_sequence)
        z_log_var = tf.clip_by_value(
            raw_log_var,
            self.z_log_var_clip_min,
            self.z_log_var_clip_max,
        )
        return {
            "graph_sequence": graph_sequence,
            "temporal_sequence": temporal_sequence,
            "bilstm_sequence": bilstm_sequence,
            "fused_sequence": fused_sequence,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
        }

    @staticmethod
    def _reparameterize(z_mean, z_log_var, seed=None):
        if seed is None:
            epsilon = tf.random.normal(tf.shape(z_mean), dtype=z_mean.dtype)
        else:
            if isinstance(seed, tuple):
                stateless_seed = tf.constant(seed, dtype=tf.int32)
            else:
                stateless_seed = tf.constant([int(seed), 0], dtype=tf.int32)
            epsilon = tf.random.stateless_normal(
                tf.shape(z_mean),
                seed=stateless_seed,
                dtype=z_mean.dtype,
            )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def _pool_z_for_prediction(self, flat_z, batch_size=None, n_windows=None):
        window_z = self.z_pool(flat_z)
        if self.classification_level == "window":
            return window_z, window_z
        window_z = tf.reshape(window_z, [batch_size, n_windows, self.z_dim])
        trial_z = tf.reduce_mean(window_z, axis=1)
        return trial_z, window_z

    def _classifier_forward(self, pooled_z, training: bool):
        x = pooled_z
        for dense, norm, dropout in zip(
            self.classification_dense_layers,
            self.classification_norm_layers,
            self.classification_dropout_layers,
        ):
            x = dense(x)
            x = norm(x)
            # During calibration, frozen earlier classifier blocks are also
            # deterministic. Dropout remains active only in blocks that were
            # explicitly selected for fine-tuning.
            block_training = bool(training) and bool(dense.trainable)
            x = dropout(x, training=block_training)
        classification_embedding = x
        logits = self.logits_layer(classification_embedding)
        return classification_embedding, logits

    def _encode(
        self,
        eeg_inputs,
        *,
        training: bool,
        sample_latent: bool,
        seed=None,
        classifier_training: bool | None = None,
    ):
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        if classifier_training is None:
            classifier_training = bool(training)
        if self.classification_level == "window":
            if eeg_inputs.shape.rank != 3:
                raise ValueError(
                    "Window mode expects (batch, timesteps, features); "
                    f"got {eeg_inputs.shape}."
                )
            flat_windows = eeg_inputs
            batch_size = n_windows = None
        else:
            flat_windows, batch_size, n_windows = self._flatten_trial_windows(
                eeg_inputs
            )

        posterior = self._posterior_from_flat_windows(
            flat_windows,
            training=training,
        )
        z = (
            self._reparameterize(
                posterior["z_mean"],
                posterior["z_log_var"],
                seed=seed,
            )
            if sample_latent
            else posterior["z_mean"]
        )
        pooled_z, window_z = self._pool_z_for_prediction(
            z,
            batch_size=batch_size,
            n_windows=n_windows,
        )
        pooled_z_mean, window_z_mean = self._pool_z_for_prediction(
            posterior["z_mean"],
            batch_size=batch_size,
            n_windows=n_windows,
        )
        classification_embedding, logits = self._classifier_forward(
            pooled_z,
            training=bool(classifier_training),
        )
        posterior.update(
            {
                "flat_windows": flat_windows,
                "z": z,
                "pooled_z": pooled_z,
                "window_z": window_z,
                "pooled_z_mean": pooled_z_mean,
                "window_z_mean": window_z_mean,
                "classification_embedding": classification_embedding,
                "logits": logits,
                "probabilities": tf.nn.softmax(logits, axis=-1),
                "batch_size": batch_size,
                "n_windows": n_windows,
            }
        )
        return posterior

    def call(self, inputs, training=False, sample_latent: bool | None = None):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        if sample_latent is None:
            sample_latent = bool(training) and not self.calibration_mode
        outputs = self._encode(
            eeg_inputs,
            training=bool(training),
            sample_latent=bool(sample_latent),
        )
        return outputs["logits"]

    def _per_sample_focal_loss(self, logits, y_flat):
        """Sparse binary/multiclass focal loss from deterministic logits."""
        y_flat = tf.cast(tf.reshape(y_flat, [-1]), tf.int32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        probs = tf.exp(log_probs)
        row_ids = tf.range(tf.shape(y_flat)[0], dtype=tf.int32)
        gather_ids = tf.stack([row_ids, y_flat], axis=1)
        p_t = tf.gather_nd(probs, gather_ids)
        log_p_t = tf.gather_nd(log_probs, gather_ids)
        modulating = tf.pow(
            tf.maximum(1.0 - p_t, tf.keras.backend.epsilon()),
            tf.cast(self.focal_gamma, p_t.dtype),
        )
        # Direct focal objective: -(1 - p_t)^gamma * log(p_t).
        loss = -modulating * log_p_t

        if self.focal_alpha is not None:
            if self.n_classes != 2:
                raise ValueError(
                    "Scalar focal_alpha is currently defined only for binary SIC."
                )
            alpha = tf.cast(self.focal_alpha, loss.dtype)
            alpha_t = tf.where(
                tf.equal(y_flat, 1),
                alpha,
                1.0 - alpha,
            )
            loss = alpha_t * loss
        return loss

    @staticmethod
    def _weighted_mean(values, sample_weight):
        if sample_weight is None:
            return tf.reduce_mean(values)
        weights = tf.cast(tf.reshape(sample_weight, [-1]), values.dtype)
        return tf.math.divide_no_nan(
            tf.reduce_sum(values * weights),
            tf.reduce_sum(weights),
        )

    def _vc_components(
        self,
        classification_embedding,
        logits,
        y_flat,
        sample_weight,
        *,
        calibration: bool,
    ):
        # The deterministic classifier term is focal loss. The VC target is
        # asked for variational regularizers only, so no second categorical
        # classification objective is calculated or optimized.
        focal_per_sample = self._per_sample_focal_loss(logits, y_flat)
        focal_loss = self._weighted_mean(focal_per_sample, sample_weight)

        if calibration and not self.calibration_use_vc_target:
            zero = tf.zeros((), dtype=focal_loss.dtype)
            return {
                "total_loss": focal_loss,
                "classification_loss": focal_loss,
                "weighted_classification_loss": focal_loss,
                "focal_loss": focal_loss,
                "weighted_focal_loss": focal_loss,
                "latent_posterior_kl": zero,
                "weighted_latent_posterior_kl": zero,
                "discriminator_kl": zero,
                "weighted_discriminator_kl": zero,
                "class_prior_kl": zero,
                "weighted_class_prior_kl": zero,
            }

        alpha = self.calibration_vc_alpha if calibration else self.vc_alpha
        components = self.vc_target.vc_loss_components(
            mh=classification_embedding,
            y=y_flat,
            alpha=alpha,
            beta=(self.calibration_vc_beta if calibration else self.vc_beta),
            gamma=(self.calibration_vc_gamma if calibration else self.vc_gamma),
            lambda_=(self.calibration_vc_lambda if calibration else self.vc_lambda),
            logits=logits,
            sample_weight=sample_weight,
            include_classification=False,
        )

        # Add the model's focal objective to the VC regularizers. Because the
        # VC target was called with include_classification=False, its total is
        # regularization-only and there is no classification term to subtract.
        dtype = components["total_loss"].dtype
        focal_term = tf.cast(alpha, dtype) * tf.cast(focal_loss, dtype)
        components = dict(components)
        components["total_loss"] = tf.cast(components["total_loss"], dtype) + focal_term
        components["classification_loss"] = tf.cast(focal_loss, dtype)
        components["weighted_classification_loss"] = focal_term
        components["focal_loss"] = tf.cast(focal_loss, dtype)
        components["weighted_focal_loss"] = focal_term
        # Do not expose the VariationalClassifier's legacy cross-entropy
        # aliases: older code used those names for the focal value, which made
        # logs incorrectly show identical cross_entropy and focal_loss fields.
        components.pop("cross_entropy", None)
        components.pop("weighted_cross_entropy", None)
        return components

    def _vae_components(self, outputs, training: bool):
        z_mean = outputs["z_mean"]
        z_log_var = outputs["z_log_var"]
        kl_values = -0.5 * (1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_values)
        if self.use_decoder:
            reconstruction = self.decoder(outputs["z"], training=training)
            reconstruction_loss = tf.reduce_mean(
                tf.square(outputs["flat_windows"] - reconstruction)
            )
        else:
            reconstruction = None
            reconstruction_loss = tf.zeros((), dtype=kl_loss.dtype)
        vae_loss = (
            reconstruction_loss
            + tf.cast(
                self.vae_beta,
                kl_loss.dtype,
            )
            * kl_loss
        )
        return {
            "vae_loss": vae_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "reconstruction": reconstruction,
        }

    def _subject_logits(self, pooled_z, training: bool, use_grl: bool):
        if self.subject_logits_layer is None:
            raise RuntimeError("Subject head has not been configured.")
        x = self.subject_gradient_reversal(pooled_z) if use_grl else pooled_z
        x = self.subject_hidden(x)
        x = self.subject_dropout_layer(x, training=training)
        return self.subject_logits_layer(x)

    def _subject_components(self, pooled_z, subject_ids, training: bool, use_grl: bool):
        if not self.subject_adversarial_enabled or subject_ids is None:
            zero = tf.zeros((), dtype=pooled_z.dtype)
            return {
                "subject_loss": zero,
                "subject_logits": None,
                "subject_targets": None,
            }
        targets = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
        logits = self._subject_logits(
            pooled_z,
            training=training,
            use_grl=use_grl,
        )
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )
        )
        return {
            "subject_loss": loss,
            "subject_logits": logits,
            "subject_targets": targets,
        }

    def _regularization_loss(self, dtype):
        if not self.losses:
            return tf.zeros((), dtype=dtype)
        return tf.add_n([tf.cast(value, dtype) for value in self.losses])

    @staticmethod
    def _apply_gradients(optimizer, gradients, variables):
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if pairs:
            optimizer.apply_gradients(pairs)

    @staticmethod
    def _dense_gradient(gradient):
        if gradient is None:
            return None
        if isinstance(gradient, tf.IndexedSlices):
            return tf.convert_to_tensor(gradient)
        return gradient

    @classmethod
    def _combine_first_order_gradients(cls, meta_train, meta_test, beta):
        combined = []
        for train_gradient, test_gradient in zip(meta_train, meta_test):
            train_gradient = cls._dense_gradient(train_gradient)
            test_gradient = cls._dense_gradient(test_gradient)
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
                    train_gradient
                    + tf.cast(
                        beta,
                        test_gradient.dtype,
                    )
                    * test_gradient
                )
            combined.append(gradient)
        return combined

    @classmethod
    def _gradient_cosine_similarity(cls, left, right):
        dot = tf.zeros((), dtype=tf.float32)
        left_norm_sq = tf.zeros((), dtype=tf.float32)
        right_norm_sq = tf.zeros((), dtype=tf.float32)
        for left_gradient, right_gradient in zip(left, right):
            left_gradient = cls._dense_gradient(left_gradient)
            right_gradient = cls._dense_gradient(right_gradient)
            if left_gradient is None or right_gradient is None:
                continue
            left_flat = tf.cast(tf.reshape(left_gradient, [-1]), tf.float32)
            right_flat = tf.cast(tf.reshape(right_gradient, [-1]), tf.float32)
            dot += tf.reduce_sum(left_flat * right_flat)
            left_norm_sq += tf.reduce_sum(tf.square(left_flat))
            right_norm_sq += tf.reduce_sum(tf.square(right_flat))
        return tf.math.divide_no_nan(
            dot,
            tf.sqrt(left_norm_sq * right_norm_sq),
        )

    def _subject_head_variables(self):
        if self.subject_logits_layer is None:
            return []
        variables = []
        for component in (self.subject_hidden, self.subject_logits_layer):
            variables.extend(component.trainable_variables)
        return _deduplicate_variables(variables)

    def _vc_discriminator_variables(self):
        if not self.update_vc_discriminator:
            return []
        variables = []
        for attribute in ("disc_w", "disc_b"):
            variable = getattr(self.vc_target, attribute, None)
            if variable is not None:
                variables.append(variable)
        return variables

    def _vrex_components(self, logits, y_flat, subject_ids, sample_weight):
        """Return subject-wise classification risks and the V-REx penalty.

        The deterministic SIC classifier uses focal loss. V-REx therefore
        adds ``lambda * Var(R_s)`` where ``R_s`` is the mean focal risk for one
        source subject represented in the current minibatch.
        """
        dtype = logits.dtype
        zero = tf.zeros((), dtype=dtype)
        if not self.use_vrex or subject_ids is None:
            return {
                "penalty": zero,
                "mean_subject_risk": zero,
                "n_subjects": zero,
                "subject_risks": tf.zeros((0,), dtype=dtype),
            }

        targets = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
        per_sample = self._per_sample_focal_loss(logits, y_flat)
        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), per_sample.dtype)

        unique_subjects = tf.unique(targets).y

        def risk_for_subject(subject_id):
            mask = tf.cast(tf.equal(targets, subject_id), per_sample.dtype)
            if weights is None:
                return tf.math.divide_no_nan(
                    tf.reduce_sum(per_sample * mask),
                    tf.reduce_sum(mask),
                )
            subject_weights = weights * mask
            return tf.math.divide_no_nan(
                tf.reduce_sum(per_sample * subject_weights),
                tf.reduce_sum(subject_weights),
            )

        subject_risks = tf.map_fn(
            risk_for_subject,
            unique_subjects,
            fn_output_signature=per_sample.dtype,
        )
        mean_subject_risk = tf.reduce_mean(subject_risks)
        penalty = tf.math.reduce_variance(subject_risks)
        return {
            "penalty": penalty,
            "mean_subject_risk": mean_subject_risk,
            "n_subjects": tf.cast(tf.size(unique_subjects), dtype),
            "subject_risks": subject_risks,
        }

    def _update_metrics(
        self,
        *,
        total_loss,
        vc_components,
        outputs,
        y_flat,
        sample_weight,
        vae_components=None,
        subject_components=None,
        vrex_components=None,
    ):
        zero = tf.zeros((), dtype=total_loss.dtype)
        self.loss_tracker.update_state(total_loss)
        self.vc_loss_tracker.update_state(vc_components["total_loss"])
        self.focal_loss_tracker.update_state(vc_components["focal_loss"])
        # Reporting metrics are deliberately unweighted. Class/sample weights
        # may shape the optimization loss, but "window_accuracy" and
        # "window_balanced_accuracy" should describe the model's natural
        # predictions on the observed windows.
        self.accuracy_tracker.update_state(
            y_flat,
            outputs["logits"],
        )
        self.balanced_accuracy_tracker.update_state(
            y_flat,
            outputs["logits"],
        )
        predicted_ids = tf.argmax(
            outputs["logits"],
            axis=-1,
            output_type=tf.int32,
        )
        self.predicted_class_1_fraction_tracker.update_state(
            tf.cast(tf.equal(predicted_ids, 1), tf.float32)
        )
        self.true_class_1_fraction_tracker.update_state(
            tf.cast(tf.equal(tf.cast(y_flat, tf.int32), 1), tf.float32)
        )
        self.vae_loss_tracker.update_state(
            zero if vae_components is None else vae_components["vae_loss"]
        )
        if self.use_decoder:
            self.reconstruction_loss_tracker.update_state(
                zero
                if vae_components is None
                else vae_components["reconstruction_loss"]
            )
        if (
            vae_components is not None
            and vae_components.get("reconstruction") is not None
        ):
            self.decoder_r2_tracker.update_state(
                outputs["flat_windows"],
                vae_components["reconstruction"],
            )
        self.kl_loss_tracker.update_state(
            zero if vae_components is None else vae_components["kl_loss"]
        )
        if self.use_vrex:
            self.vrex_penalty_tracker.update_state(
                zero if vrex_components is None else vrex_components["penalty"]
            )
            self.vrex_subject_risk_mean_tracker.update_state(
                zero
                if vrex_components is None
                else vrex_components["mean_subject_risk"]
            )
            self.vrex_subjects_per_batch_tracker.update_state(
                zero if vrex_components is None else vrex_components["n_subjects"]
            )
        if self.subject_adversarial_enabled:
            subject_loss = (
                zero
                if subject_components is None
                else subject_components["subject_loss"]
            )
            self.subject_loss_tracker.update_state(subject_loss)
            if (
                subject_components is not None
                and subject_components["subject_logits"] is not None
            ):
                self.subject_accuracy_tracker.update_state(
                    subject_components["subject_targets"],
                    subject_components["subject_logits"],
                )

    def _source_train_step(self, x, y_flat, sample_weight):
        eeg_inputs, subject_ids = self._split_eeg_and_subject_inputs(x)
        if self.use_vrex and subject_ids is None:
            raise ValueError(
                "V-REx source training requires subject IDs in each minibatch."
            )

        with tf.GradientTape() as tape:
            outputs = self._encode(
                eeg_inputs,
                training=True,
                sample_latent=True,
            )
            vc_components = self._vc_components(
                outputs["classification_embedding"],
                outputs["logits"],
                y_flat,
                sample_weight,
                calibration=False,
            )
            vae_components = self._vae_components(outputs, training=True)
            subject_components = self._subject_components(
                outputs["pooled_z_mean"],
                subject_ids,
                training=True,
                use_grl=True,
            )
            vrex_components = self._vrex_components(
                outputs["logits"],
                y_flat,
                subject_ids,
                sample_weight,
            )
            dtype = vc_components["total_loss"].dtype
            total = (
                tf.cast(self.vc_loss_weight, dtype) * vc_components["total_loss"]
                + tf.cast(self.vae_loss_weight, dtype)
                * tf.cast(vae_components["vae_loss"], dtype)
                + tf.cast(self.subject_loss_weight, dtype)
                * tf.cast(subject_components["subject_loss"], dtype)
                + tf.cast(self.vrex_penalty_weight, dtype)
                * tf.cast(vrex_components["penalty"], dtype)
                + self._regularization_loss(dtype)
            )
        variables = self.trainable_variables
        gradients = tape.gradient(total, variables)
        self._apply_gradients(self.main_optimizer, gradients, variables)

        if self.update_vc_discriminator:
            if self.vc_discriminator_optimizer is None:
                raise RuntimeError(
                    "update_vc_discriminator=True requires a discriminator optimizer."
                )
            embedding_frozen = tf.stop_gradient(outputs["classification_embedding"])
            with tf.GradientTape() as disc_tape:
                disc_loss = self.vc_target.discriminator_loss(
                    embedding_frozen,
                    y_flat,
                )
            disc_variables = self._vc_discriminator_variables()
            disc_gradients = disc_tape.gradient(disc_loss, disc_variables)
            self._apply_gradients(
                self.vc_discriminator_optimizer,
                disc_gradients,
                disc_variables,
            )

        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
            vae_components=vae_components,
            subject_components=subject_components,
            vrex_components=vrex_components,
        )

    def _mldg_train_step(self, x, y_flat, sample_weight):
        if not isinstance(x, Mapping) or "mldg_role" not in x:
            raise ValueError(
                "MLDG train_step requires the episode roles produced by the "
                "MLDG fit adapter."
            )
        eeg_inputs, subject_ids = self._split_eeg_and_subject_inputs(x)
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

        # The inner/meta-train loss retains SIC's full source objective. The
        # virtual-unseen loss intentionally measures focal classification only:
        # reconstruction, VC distribution regularizers, and subject
        # identification are source auxiliaries, not evidence that emotion
        # prediction transferred.
        with tf.GradientTape() as meta_train_tape:
            meta_train_outputs = self._encode(
                meta_train_eeg,
                training=True,
                sample_latent=True,
            )
            meta_train_vc = self._vc_components(
                meta_train_outputs["classification_embedding"],
                meta_train_outputs["logits"],
                meta_train_y,
                meta_train_weight,
                calibration=False,
            )
            meta_train_vae = self._vae_components(
                meta_train_outputs,
                training=True,
            )
            meta_train_subject = self._subject_components(
                meta_train_outputs["pooled_z_mean"],
                meta_train_subject_ids,
                training=True,
                use_grl=True,
            )
            dtype = meta_train_vc["total_loss"].dtype
            meta_train_loss = (
                tf.cast(self.vc_loss_weight, dtype) * meta_train_vc["total_loss"]
                + tf.cast(self.vae_loss_weight, dtype)
                * tf.cast(meta_train_vae["vae_loss"], dtype)
                + tf.cast(self.subject_loss_weight, dtype)
                * tf.cast(meta_train_subject["subject_loss"], dtype)
                + self._regularization_loss(dtype)
            )
        variables = self.trainable_variables
        meta_train_gradients = meta_train_tape.gradient(
            meta_train_loss,
            variables,
        )

        # Assigning stop-gradient tensors into the existing variables creates
        # theta' without retaining a differentiable path back through g_A.
        # That is the explicit first-order approximation.
        original_values = [tf.identity(variable) for variable in variables]
        for variable, gradient in zip(variables, meta_train_gradients):
            gradient = self._dense_gradient(gradient)
            if gradient is not None:
                variable.assign_sub(
                    tf.cast(self.mldg_inner_learning_rate, gradient.dtype)
                    * tf.stop_gradient(gradient)
                )

        with tf.GradientTape() as meta_test_tape:
            meta_test_outputs = self._encode(
                meta_test_eeg,
                training=True,
                sample_latent=True,
            )
            meta_test_vc = self._vc_components(
                meta_test_outputs["classification_embedding"],
                meta_test_outputs["logits"],
                meta_test_y,
                meta_test_weight,
                calibration=False,
            )
            meta_test_loss = (
                tf.cast(
                    self.vc_loss_weight,
                    meta_test_vc["weighted_focal_loss"].dtype,
                )
                * meta_test_vc["weighted_focal_loss"]
            )
        meta_test_gradients = meta_test_tape.gradient(
            meta_test_loss,
            variables,
        )

        # Restore theta before the sole persistent optimizer update.
        for variable, original_value in zip(variables, original_values):
            variable.assign(original_value)
        combined_gradients = self._combine_first_order_gradients(
            meta_train_gradients,
            meta_test_gradients,
            self.mldg_meta_test_weight,
        )
        self._apply_gradients(
            self.main_optimizer,
            combined_gradients,
            variables,
        )

        if self.update_vc_discriminator:
            if self.vc_discriminator_optimizer is None:
                raise RuntimeError(
                    "update_vc_discriminator=True requires a discriminator optimizer."
                )
            embedding_frozen = tf.stop_gradient(
                meta_train_outputs["classification_embedding"]
            )
            with tf.GradientTape() as disc_tape:
                disc_loss = self.vc_target.discriminator_loss(
                    embedding_frozen,
                    meta_train_y,
                )
            disc_variables = self._vc_discriminator_variables()
            disc_gradients = disc_tape.gradient(disc_loss, disc_variables)
            self._apply_gradients(
                self.vc_discriminator_optimizer,
                disc_gradients,
                disc_variables,
            )

        outer_loss = (
            meta_train_loss
            + tf.cast(
                self.mldg_meta_test_weight,
                meta_test_loss.dtype,
            )
            * meta_test_loss
        )
        self._update_metrics(
            total_loss=outer_loss,
            vc_components=meta_train_vc,
            outputs=meta_train_outputs,
            y_flat=meta_train_y,
            sample_weight=meta_train_weight,
            vae_components=meta_train_vae,
            subject_components=meta_train_subject,
            vrex_components=None,
        )
        self.mldg_meta_train_loss_tracker.update_state(meta_train_loss)
        self.mldg_meta_test_loss_tracker.update_state(meta_test_loss)
        self.mldg_meta_train_subjects_tracker.update_state(
            tf.cast(tf.size(tf.unique(meta_train_subject_ids).y), tf.float32)
        )
        self.mldg_meta_test_subjects_tracker.update_state(
            tf.cast(tf.size(tf.unique(meta_test_subject_ids).y), tf.float32)
        )
        self.mldg_gradient_cosine_tracker.update_state(
            self._gradient_cosine_similarity(
                meta_train_gradients,
                meta_test_gradients,
            )
        )

    def _calibration_train_step(self, x, y_flat, sample_weight):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(x)
        # Backbone layers are frozen and posterior mean is used so six-trial
        # calibration fits a stable subject-specific decision boundary.
        with tf.GradientTape() as tape:
            outputs = self._encode(
                eeg_inputs,
                training=False,
                sample_latent=False,
                classifier_training=True,
            )
            vc_components = self._vc_components(
                outputs["classification_embedding"],
                outputs["logits"],
                y_flat,
                sample_weight,
                calibration=True,
            )
            dtype = vc_components["total_loss"].dtype
            total = tf.cast(self.vc_loss_weight, dtype) * vc_components[
                "total_loss"
            ] + self._regularization_loss(dtype)
        variables = self.trainable_variables
        gradients = tape.gradient(total, variables)
        self._apply_gradients(self.main_optimizer, gradients, variables)
        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
            vae_components=None,
            subject_components=None,
            vrex_components=None,
        )

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_flat = self._flatten_labels(y)
        if self.main_optimizer is None:
            raise RuntimeError("Call model.compile(...) before model.fit(...).")
        if self.calibration_mode:
            self._calibration_train_step(x, y_flat, sample_weight)
        elif self.use_mldg:
            self._mldg_train_step(x, y_flat, sample_weight)
        else:
            self._source_train_step(x, y_flat, sample_weight)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_flat = self._flatten_labels(y)
        eeg_inputs, subject_ids = self._split_eeg_and_subject_inputs(x)
        outputs = self._encode(
            eeg_inputs,
            training=False,
            sample_latent=False,
        )
        vc_components = self._vc_components(
            outputs["classification_embedding"],
            outputs["logits"],
            y_flat,
            sample_weight,
            calibration=self.calibration_mode,
        )
        if self.calibration_mode:
            vae_components = None
            subject_components = None
            total = (
                tf.cast(self.vc_loss_weight, vc_components["total_loss"].dtype)
                * vc_components["total_loss"]
            )
        else:
            vae_components = self._vae_components(outputs, training=False)
            subject_components = self._subject_components(
                outputs["pooled_z_mean"],
                subject_ids,
                training=False,
                use_grl=False,
            )
            dtype = vc_components["total_loss"].dtype
            total = (
                tf.cast(self.vc_loss_weight, dtype) * vc_components["total_loss"]
                + tf.cast(self.vae_loss_weight, dtype)
                * tf.cast(vae_components["vae_loss"], dtype)
                + tf.cast(self.subject_loss_weight, dtype)
                * tf.cast(subject_components["subject_loss"], dtype)
            )
        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
            vae_components=vae_components,
            subject_components=subject_components,
            vrex_components=None,
        )
        return {metric.name: metric.result() for metric in self.metrics}

    def predict_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        return self(x, training=False, sample_latent=False)

    def prepare_for_zero_shot_evaluation(self):
        """Restore source-evaluation semantics after a calibration fold.

        ``subject_calibration_cv`` restores source weights between folds.  This
        hook resets the loss/inference mode as well, so paired zero-shot metrics
        are evaluated as the original population model rather than in the
        calibration-only loss mode.
        """
        self.calibration_mode = False
        return self

    def prepare_for_subject_calibration(
        self,
        *,
        learning_rate: float,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.0,
        unfreeze_layers: int | None = None,
    ):
        """Freeze the population model and unfreeze only the final k head layers.

        ``unfreeze_layers`` counts prediction layers backward from the output:
        1 means logits/softmax only, 2 means the final hidden dense block plus
        logits, and so on.  The VC target is frozen and acts as a source-trained
        target for any unfrozen hidden classification representation.
        """
        if unfreeze_layers is None:
            unfreeze_layers = self.calibration_unfreeze_layers
        unfreeze_layers = int(unfreeze_layers)
        max_layers = len(self.classification_dense_layers) + 1
        if not 1 <= unfreeze_layers <= max_layers:
            raise ValueError(
                f"unfreeze_layers must be in [1, {max_layers}], got "
                f"{unfreeze_layers}."
            )

        self.calibration_mode = True

        # Freeze every major source-trained subsystem first.
        if self.graph_encoder is not None:
            self.graph_encoder.trainable = False
        for layer in self.temporal_bilstms:
            layer.trainable = False
        for layer in self.temporal_norms:
            layer.trainable = False
        for layer in self.temporal_dropouts:
            layer.trainable = False
        self.parallel_temporal_pool.trainable = False
        self.fusion_projection.trainable = False
        self.fusion_norm.trainable = False
        self.fusion_dropout_layer.trainable = False
        self.z_mean_projection.trainable = False
        self.z_log_var_projection.trainable = False
        if self.decoder is not None:
            self.decoder.trainable = False
        self.vc_target.trainable = False
        if self.subject_gradient_reversal is not None:
            self.subject_gradient_reversal.trainable = False
        if self.subject_hidden is not None:
            self.subject_hidden.trainable = False
        if self.subject_dropout_layer is not None:
            self.subject_dropout_layer.trainable = False
        if self.subject_logits_layer is not None:
            self.subject_logits_layer.trainable = False

        # Freeze the whole dense prediction head, then unfreeze exactly the
        # requested suffix.  A hidden block consists of Dense + LN + Dropout.
        for dense, norm, dropout in zip(
            self.classification_dense_layers,
            self.classification_norm_layers,
            self.classification_dropout_layers,
        ):
            dense.trainable = False
            norm.trainable = False
            dropout.trainable = False
        self.logits_layer.trainable = True

        hidden_to_unfreeze = max(0, unfreeze_layers - 1)
        if hidden_to_unfreeze:
            start = len(self.classification_dense_layers) - hidden_to_unfreeze
            for index in range(start, len(self.classification_dense_layers)):
                self.classification_dense_layers[index].trainable = True
                self.classification_norm_layers[index].trainable = True
                self.classification_dropout_layers[index].trainable = True

        optimizer = _build_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        # A fresh optimizer is required because each calibration fold restores
        # source weights and must not inherit optimizer moments from pretraining
        # or another target calibration fold.
        self.compile(
            main_optimizer=optimizer,
            vc_discriminator_optimizer=None,
            run_eagerly=False,
            jit_compile=False,
        )
        return {
            "calibration_unfreeze_layers": unfreeze_layers,
            "trainable_variables": [
                variable.name for variable in self.trainable_variables
            ],
            "calibration_use_vc_target": self.calibration_use_vc_target,
        }

    def predict_mc_probabilities(self, inputs, n_samples: int = 30, seed=None):
        """Posterior predictive probabilities from VAE latent sampling."""
        if int(n_samples) < 1:
            raise ValueError("n_samples must be >= 1.")
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)

        if self.classification_level == "window":
            flat_windows = eeg_inputs
            batch_size = tf.shape(eeg_inputs)[0]
            n_windows = None
        else:
            flat_windows, batch_size, n_windows = self._flatten_trial_windows(
                eeg_inputs
            )
        posterior = self._posterior_from_flat_windows(flat_windows, training=False)
        z_mean = posterior["z_mean"]
        z_log_var = posterior["z_log_var"]

        if seed is None:
            epsilon = tf.random.normal(
                tf.concat(
                    [tf.constant([int(n_samples)], dtype=tf.int32), tf.shape(z_mean)],
                    axis=0,
                ),
                dtype=z_mean.dtype,
            )
        else:
            if isinstance(seed, tuple):
                stateless_seed = tf.constant(seed, dtype=tf.int32)
            else:
                stateless_seed = tf.constant([int(seed), 0], dtype=tf.int32)
            epsilon = tf.random.stateless_normal(
                tf.concat(
                    [tf.constant([int(n_samples)], dtype=tf.int32), tf.shape(z_mean)],
                    axis=0,
                ),
                seed=stateless_seed,
                dtype=z_mean.dtype,
            )
        z_samples = (
            z_mean[tf.newaxis, ...] + tf.exp(0.5 * z_log_var[tf.newaxis, ...]) * epsilon
        )

        # Pool each sampled latent sequence to one vector per window.
        pooled_windows = tf.reduce_mean(z_samples, axis=2)
        if self.classification_level == "trial":
            pooled_windows = tf.reshape(
                pooled_windows,
                [int(n_samples), batch_size, n_windows, self.z_dim],
            )
            pooled = tf.reduce_mean(pooled_windows, axis=2)
        else:
            pooled = pooled_windows

        sample_count = tf.shape(pooled)[0]
        sample_batch = tf.shape(pooled)[1]
        flat_pooled = tf.reshape(pooled, [sample_count * sample_batch, self.z_dim])
        embedding, logits = self._classifier_forward(flat_pooled, training=False)
        del embedding
        probabilities = tf.nn.softmax(logits, axis=-1)
        probabilities = tf.reshape(
            probabilities,
            [sample_count, sample_batch, self.n_classes],
        )
        return {
            "probability_samples": probabilities,
            "mean_probabilities": tf.reduce_mean(probabilities, axis=0),
        }

    def get_latent_distribution(self, inputs):
        """Return the counterfactual VAE posterior and pooled representation."""
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        outputs = self._encode(
            eeg_inputs,
            training=False,
            sample_latent=False,
        )
        return {
            "z_mean": outputs["z_mean"],
            "z_log_var": outputs["z_log_var"],
            "pooled_z": outputs["pooled_z_mean"],
            "classification_embedding": outputs["classification_embedding"],
            "probabilities": outputs["probabilities"],
        }

    def decode_latent(self, latent_sequence):
        if not self.use_decoder or self.decoder is None:
            raise RuntimeError("Decoder is disabled for this SIC ablation.")
        return self.decoder(latent_sequence, training=False)

    def get_adjacency_matrices(self):
        if self.graph_encoder is None:
            return {}
        return {
            "mtl_raw_adjacency": self.graph_encoder.get_raw_adjacency_matrix(),
            "mtl_normalized_adjacency": self.graph_encoder.get_adjacency_matrix(),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "graph_encoder": (
                    None
                    if self.graph_encoder is None
                    else _serialize_keras_object(self.graph_encoder)
                ),
                "decoder": (
                    None
                    if self.decoder is None
                    else _serialize_keras_object(self.decoder)
                ),
                "classification_level": self.classification_level,
                "n_classes": self.n_classes,
                "bilstm_units": self.bilstm_units,
                "n_bilstm_layers": self.n_bilstm_layers,
                "bilstm_dropout": self.bilstm_dropout_rate,
                "architecture_mode": self.architecture_mode,
                "use_gcn_gru_branch": self.use_gcn_gru_branch,
                "use_bilstm_branch": self.use_bilstm_branch,
                "use_decoder": self.use_decoder,
                "fusion_units": self.fusion_units,
                "fusion_dropout": self.fusion_dropout_rate,
                "temporal_downsample_factor": self.temporal_downsample_factor,
                "z_dim": self.z_dim,
                "z_log_var_clip_min": self.z_log_var_clip_min,
                "z_log_var_clip_max": self.z_log_var_clip_max,
                "classification_hidden_units": self.classification_hidden_units,
                "classification_dropout": self.classification_dropout_rate,
                "activation": self.activation_name,
                "label_smoothing": self.label_smoothing,
                "focal_gamma": self.focal_gamma,
                "focal_alpha": self.focal_alpha,
                "vc_loss_weight": self.vc_loss_weight,
                "vc_alpha": self.vc_alpha,
                "vc_beta": self.vc_beta,
                "vc_gamma": self.vc_gamma,
                "vc_lambda": self.vc_lambda,
                "update_vc_discriminator": self.update_vc_discriminator,
                "vae_loss_weight": self.vae_loss_weight,
                "vae_beta": self.vae_beta,
                "training_method": self.training_method,
                "use_vrex": self.use_vrex,
                "vrex_penalty_weight": self.vrex_penalty_weight,
                "mldg_meta_train_subjects": self.mldg_meta_train_subjects,
                "mldg_meta_test_subjects": self.mldg_meta_test_subjects,
                "mldg_trials_per_subject": self.mldg_trials_per_subject,
                "mldg_steps_per_epoch": self.mldg_steps_per_epoch,
                "mldg_inner_learning_rate": self.mldg_inner_learning_rate,
                "mldg_meta_test_weight": self.mldg_meta_test_weight,
                "mldg_seed": self.mldg_seed,
                "use_subject_adversarial": self.subject_adversarial_enabled,
                "n_subject_classes": self.n_subject_classes,
                "subject_adversarial_weight": self.subject_adversarial_weight,
                "subject_loss_weight": self.subject_loss_weight,
                "subject_hidden_units": self.subject_hidden_units,
                "subject_dropout": self.subject_dropout_rate,
                "calibration_unfreeze_layers": self.calibration_unfreeze_layers,
                "calibration_use_vc_target": self.calibration_use_vc_target,
                "calibration_vc_alpha": self.calibration_vc_alpha,
                "calibration_vc_beta": self.calibration_vc_beta,
                "calibration_vc_gamma": self.calibration_vc_gamma,
                "calibration_vc_lambda": self.calibration_vc_lambda,
                "use_class_weight": self.use_class_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["graph_encoder"] = (
            None
            if config["graph_encoder"] is None
            else _deserialize_keras_object(config["graph_encoder"])
        )
        config["decoder"] = (
            None
            if config["decoder"] is None
            else _deserialize_keras_object(config["decoder"])
        )
        return cls(**config)


def build_sic_model(
    input_shape,
    *,
    training_features=None,
    training_labels=None,
    training_subject_ids=None,
    training_trial_ids=None,
    adjacency=None,
    classification_level: str = "trial",
    n_classes: int = 2,
    n_channels: int = 14,
    n_bands: int = 3,
    t_down: int = 2,
    temporal_pool_sizes: Sequence[int] | None = (2,),
    gcn_units: Sequence[int] = (64, 32),
    gcn_dropout: float = 0.10,
    gcn_activation: str = "relu",
    gcn_use_batch_norm: bool = False,
    spectral_gru_units: int = 384,
    spectral_gru_dropout: float = 0.20,
    graph_add_self_loops: bool = True,
    graph_symmetrize: bool = True,
    graph_epsilon: float = 1e-8,
    mi_n_neighbors: int = 3,
    mi_random_state: int = 42,
    mi_zero_diagonal: bool = False,
    mi_band_reduction: str = "mean",
    mi_max_observations: int | None = 15000,
    bilstm_units: int = 128,
    n_bilstm_layers: int = 1,
    bilstm_dropout: float = 0.30,
    architecture_mode: str = "serial",
    use_gcn_gru_branch: bool = True,
    use_bilstm_branch: bool = True,
    use_decoder: bool = True,
    fusion_units: int = 128,
    fusion_dropout: float = 0.20,
    z_dim: int = 64,
    z_log_var_clip_min: float = -20.0,
    z_log_var_clip_max: float = 20.0,
    classification_hidden_units: Sequence[int] = (128,),
    classification_dropout: float = 0.20,
    activation: str = "relu",
    label_smoothing: float = 0.0,
    focal_gamma: float = 1.0,
    focal_alpha: float | None = None,
    vc_loss_weight: float = 1.0,
    vc_alpha: float = 1.0,
    vc_beta: float = 0.5,
    vc_gamma: float = 0.0,
    vc_lambda: float = 0.0,
    update_vc_discriminator: bool = False,
    vae_loss_weight: float = 0.10,
    vae_beta: float = 0.05,
    training_method: str | None = None,
    use_vrex: bool = False,
    vrex_penalty_weight: float = 1.0,
    mldg_meta_train_subjects: int | None = None,
    mldg_meta_test_subjects: int = 4,
    mldg_trials_per_subject: int = 1,
    mldg_steps_per_epoch: int | None = None,
    mldg_inner_learning_rate: float = 1e-4,
    mldg_meta_test_weight: float = 1.0,
    mldg_seed: int | None = 42,
    use_subject_adversarial: bool = True,
    n_subject_classes: int | None = None,
    subject_adversarial_weight: float = 0.8,
    subject_loss_weight: float = 1.0,
    subject_hidden_units: int = 64,
    subject_dropout: float = 0.0,
    calibration_unfreeze_layers: int = 1,
    calibration_use_vc_target: bool = True,
    calibration_vc_alpha: float | None = None,
    calibration_vc_beta: float | None = None,
    calibration_vc_gamma: float | None = None,
    calibration_vc_lambda: float | None = None,
    decoder_dropout: float = 0.10,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    vc_discriminator_learning_rate: float | None = None,
    weight_decay: float = 5e-5,
    use_class_weight: bool = False,
    model_name: str = "sic_subject_invariant_calibrator",
    **unused_kwargs,
) -> SICModel:
    """Build SIC, computing its fixed MI graph from source data when needed.

    The ``training_*`` arguments are accepted explicitly for
    ``subject_calibration_cv``. ``training_features`` estimates the source-only
    graph, ``training_subject_ids`` configures source metadata/adversity, and
    ``training_trial_ids`` keeps complete MLDG trials together. Labels are
    accepted to keep the builder contract uniform and leakage-auditable.
    """
    del training_labels, unused_kwargs

    classification_level = str(classification_level).lower()
    input_shape = tuple(int(value) for value in input_shape)
    if classification_level == "window":
        if len(input_shape) != 2:
            raise ValueError(
                "Window-level v6 expects input_shape=(timesteps, features); "
                f"got {input_shape}."
            )
        timesteps, n_features = input_shape
        dummy_shape = (1, timesteps, n_features)
    elif classification_level == "trial":
        if len(input_shape) != 3:
            raise ValueError(
                "Trial-level v6 expects input_shape=(windows, timesteps, features); "
                f"got {input_shape}."
            )
        n_trial_windows, timesteps, n_features = input_shape
        dummy_shape = (1, n_trial_windows, timesteps, n_features)
    else:
        raise ValueError("classification_level must be 'window' or 'trial'.")

    expected_features = int(n_channels) * int(n_bands)
    if n_features != expected_features:
        raise ValueError(
            f"Input features={n_features}, expected {n_channels}*{n_bands}="
            f"{expected_features}."
        )
    pools = _resolve_temporal_pool_sizes(temporal_pool_sizes, t_down)
    gcn_units = _as_positive_tuple("gcn_units", gcn_units)

    use_gcn_gru_branch = bool(use_gcn_gru_branch)
    use_bilstm_branch = bool(use_bilstm_branch)
    use_decoder = bool(use_decoder)
    if not use_gcn_gru_branch and not use_bilstm_branch:
        raise ValueError(
            "At least one of use_gcn_gru_branch/use_bilstm_branch must be true."
        )

    needs_graph = use_gcn_gru_branch or use_decoder
    if adjacency is None and needs_graph:
        if training_features is None:
            raise ValueError(
                "SIC requires either adjacency=... or training_features=... so "
                "the MTLFuseNet MI graph can be estimated from source data only."
            )
        adjacency = _source_only_mi_adjacency(
            training_features,
            n_channels=int(n_channels),
            n_bands=int(n_bands),
            n_neighbors=int(mi_n_neighbors),
            random_state=int(mi_random_state),
            zero_diagonal=bool(mi_zero_diagonal),
            band_reduction=str(mi_band_reduction),
            max_observations=mi_max_observations,
        )
    elif adjacency is not None:
        adjacency = np.asarray(adjacency, dtype=np.float32)

    if n_subject_classes is None and training_subject_ids is not None:
        n_subject_classes = int(len(np.unique(np.asarray(training_subject_ids))))

    graph_encoder = None
    if use_gcn_gru_branch:
        graph_encoder = GCNMTLEncoder(
            timesteps=int(timesteps),
            t_down=int(t_down),
            adjacency=adjacency,
            n_channels=int(n_channels),
            n_bands=int(n_bands),
            gcn_units=gcn_units,
            temporal_pool_sizes=pools,
            emb_dim=None,
            dropout=float(gcn_dropout),
            activation=str(gcn_activation),
            use_batch_norm=bool(gcn_use_batch_norm),
            use_spectral_gru=True,
            spectral_gru_units=int(spectral_gru_units),
            spectral_gru_dropout=float(spectral_gru_dropout),
            graph_add_self_loops=bool(graph_add_self_loops),
            graph_symmetrize=bool(graph_symmetrize),
            graph_epsilon=float(graph_epsilon),
            name="v6_mtl_gcn_gru_encoder",
        )

    decoder = None
    if use_decoder:
        decoder = GCNMTLDecoder(
            timesteps=int(timesteps),
            n_channels=int(n_channels),
            n_bands=int(n_bands),
            t_down=int(t_down),
            gcn_units=gcn_units,
            temporal_pool_sizes=pools,
            adjacency=adjacency,
            emb_dim=int(z_dim),
            dropout=float(decoder_dropout),
            activation=str(activation),
            use_batch_norm=bool(gcn_use_batch_norm),
            graph_add_self_loops=bool(graph_add_self_loops),
            graph_symmetrize=bool(graph_symmetrize),
            graph_epsilon=float(graph_epsilon),
            name="sic_mtl_graph_decoder",
        )

    model = SICModel(
        graph_encoder=graph_encoder,
        decoder=decoder,
        classification_level=classification_level,
        n_classes=int(n_classes),
        bilstm_units=int(bilstm_units),
        n_bilstm_layers=int(n_bilstm_layers),
        bilstm_dropout=float(bilstm_dropout),
        architecture_mode=str(architecture_mode),
        use_gcn_gru_branch=use_gcn_gru_branch,
        use_bilstm_branch=use_bilstm_branch,
        use_decoder=use_decoder,
        fusion_units=int(fusion_units),
        fusion_dropout=float(fusion_dropout),
        temporal_downsample_factor=int(t_down),
        z_dim=int(z_dim),
        z_log_var_clip_min=float(z_log_var_clip_min),
        z_log_var_clip_max=float(z_log_var_clip_max),
        classification_hidden_units=tuple(
            int(value) for value in classification_hidden_units
        ),
        classification_dropout=float(classification_dropout),
        activation=str(activation),
        label_smoothing=float(label_smoothing),
        focal_gamma=float(focal_gamma),
        focal_alpha=focal_alpha,
        vc_loss_weight=float(vc_loss_weight),
        vc_alpha=float(vc_alpha),
        vc_beta=float(vc_beta),
        vc_gamma=float(vc_gamma),
        vc_lambda=float(vc_lambda),
        update_vc_discriminator=bool(update_vc_discriminator),
        vae_loss_weight=float(vae_loss_weight),
        vae_beta=float(vae_beta),
        training_method=training_method,
        use_vrex=bool(use_vrex),
        vrex_penalty_weight=float(vrex_penalty_weight),
        mldg_meta_train_subjects=mldg_meta_train_subjects,
        mldg_meta_test_subjects=int(mldg_meta_test_subjects),
        mldg_trials_per_subject=int(mldg_trials_per_subject),
        mldg_steps_per_epoch=mldg_steps_per_epoch,
        mldg_inner_learning_rate=float(mldg_inner_learning_rate),
        mldg_meta_test_weight=float(mldg_meta_test_weight),
        mldg_seed=mldg_seed,
        use_subject_adversarial=bool(use_subject_adversarial),
        n_subject_classes=n_subject_classes,
        subject_adversarial_weight=float(subject_adversarial_weight),
        subject_loss_weight=float(subject_loss_weight),
        subject_hidden_units=int(subject_hidden_units),
        subject_dropout=float(subject_dropout),
        calibration_unfreeze_layers=int(calibration_unfreeze_layers),
        calibration_use_vc_target=bool(calibration_use_vc_target),
        calibration_vc_alpha=calibration_vc_alpha,
        calibration_vc_beta=calibration_vc_beta,
        calibration_vc_gamma=calibration_vc_gamma,
        calibration_vc_lambda=calibration_vc_lambda,
        use_class_weight=bool(use_class_weight),
        name=model_name,
    )

    main_optimizer = _build_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    resolved_disc_lr = (
        float(learning_rate)
        if vc_discriminator_learning_rate is None
        else float(vc_discriminator_learning_rate)
    )
    vc_discriminator_optimizer = (
        _build_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=resolved_disc_lr,
            weight_decay=float(weight_decay),
        )
        if bool(update_vc_discriminator)
        else None
    )

    model.compile(
        main_optimizer=main_optimizer,
        vc_discriminator_optimizer=vc_discriminator_optimizer,
        run_eagerly=False,
        jit_compile=False,
    )
    model.set_source_training_metadata(
        training_subject_ids,
        training_trial_ids,
    )

    # Build every stateful branch before the first fit. This avoids Keras 3
    # creating new nested-layer variables after the outer model has been marked
    # built (the failure mode that affected earlier subject-adversarial models).
    if bool(use_subject_adversarial) and n_subject_classes is not None:
        _ = model._subject_logits(
            tf.zeros((1, int(z_dim)), dtype=tf.float32),
            training=False,
            use_grl=False,
        )
    latent_timesteps = int(np.ceil(float(timesteps) / float(t_down)))
    if decoder is not None:
        _ = decoder(
            tf.zeros((1, latent_timesteps, int(z_dim)), dtype=tf.float32),
            training=False,
        )
    if not bool(use_subject_adversarial) or n_subject_classes is not None:
        _ = model(tf.zeros(dummy_shape, dtype=tf.float32), training=False)
    return model


# Compatibility aliases for earlier v6 naming.
JointV6Model = SICModel
JointSTSModelV6 = SICModel
build_joint_v6_model = build_sic_model
build_joint_sts_model_v6 = build_sic_model
