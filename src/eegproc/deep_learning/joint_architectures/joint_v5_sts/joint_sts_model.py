"""Joint v5 STS: MTLFuseNet-style spatio-spectral GCN -> GRU -> classifier.

This is intentionally a classification-only baseline corresponding to the upper
spatio-spectral branch in the MTLFuseNet architecture diagram:

    channel/band EEG -> shared fixed-MI GCN -> GRU across frequency bands
        -> direct GRU latent (384-D by default) -> simple non-recurrent pooling
        -> dense classifier -> logits

There is NO BiLSTM, VAE, decoder, feature fusion, subject adversary, SupCon,
or alternating/meta-learning objective in v5.0.

Important leakage rule
----------------------
The mutual-information adjacency is adapted lazily inside ``fit`` from the data
passed to that fit call. Under EEGProc LOSO this means each model constructs A
from that fold's TRAINING DATA ONLY; validation/test data never enter A.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from ...unsupervised.GNN.GCNMTL import (
        GCNMTLEncoder,
        compute_mtl_shared_mi_adjacency,
    )
except ImportError:
    from eegproc.deep_learning.unsupervised.GNN.GCNMTL import (
        GCNMTLEncoder,
        compute_mtl_shared_mi_adjacency,
    )


JOINT_STS_BUILDER_API_VERSION = 1


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Sparse categorical focal loss; gamma=0 recovers cross-entropy."""

    def __init__(
        self,
        gamma: float = 0.0,
        alpha=None,
        from_logits: bool = True,
        name: str = "sparse_categorical_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, reduction="sum_over_batch_size", **kwargs)
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = float(gamma)
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = tuple(float(v) for v in alpha)
        else:
            self.alpha = (float(alpha),)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        if self.from_logits:
            probs = tf.nn.softmax(y_pred, axis=-1)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_true, logits=y_pred
            )
        else:
            probs = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
            ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, probs)

        rows = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
        pt = tf.gather_nd(probs, tf.stack([rows, y_true], axis=1))
        focal = tf.pow(
            tf.maximum(1.0 - pt, tf.keras.backend.epsilon()), self.gamma
        )
        loss = tf.cast(ce, pt.dtype) * focal

        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=loss.dtype)
            if len(self.alpha) == 1:
                loss *= alpha[0]
            else:
                loss *= tf.gather(alpha, y_true)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "from_logits": self.from_logits,
            }
        )
        return config


def _build_optimizer(name: str, learning_rate: float, weight_decay: float):
    name = str(name).lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer_name must be 'adam' or 'adamw'.")


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class JointSTSModel(tf.keras.Model):
    """MTL-style GCN -> spectral GRU -> pooled dense classifier.

    ``classification_level='window'`` expects input ``(B,T,F)``.
    ``classification_level='trial'`` expects input ``(B,W,T,F)`` and uses only
    non-recurrent averaging across windows after producing one GCN-GRU embedding
    per window. Thus v5.0 contains no temporal RNN outside the paper-style GRU.
    """

    def __init__(
        self,
        *,
        timesteps: int,
        n_features: int,
        n_classes: int = 2,
        n_channels: int = 14,
        n_bands: int = 3,
        classification_level: str = "window",
        t_down: int = 2,
        temporal_pool_sizes: Sequence[int] | None = (2,),
        gcn_units: Sequence[int] = (32,),
        gcn_dropout: float = 0.10,
        gcn_activation: str = "relu",
        gcn_use_batch_norm: bool = False,
        spectral_gru_units: int = 384,
        spectral_gru_dropout: float = 0.0,
        graph_add_self_loops: bool = True,
        graph_symmetrize: bool = True,
        graph_epsilon: float = 1e-8,
        mi_n_neighbors: int = 3,
        mi_random_state: int = 42,
        mi_zero_diagonal: bool = False,
        mi_band_reduction: str = "mean",
        mi_max_observations: int | None = 50000,
        adjacency=None,
        classification_hidden_units: int = 128,
        classification_dropout: float = 0.30,
        activation: str = "relu",
        use_class_weight: bool = False,
        name: str = "joint_v5_sts_model",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.timesteps = int(timesteps)
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.classification_level = str(classification_level).lower()
        if self.classification_level not in {"window", "trial"}:
            raise ValueError("classification_level must be 'window' or 'trial'.")
        if self.n_features != self.n_channels * self.n_bands:
            raise ValueError(
                f"n_features={self.n_features} must equal n_channels*n_bands="
                f"{self.n_channels * self.n_bands}."
            )

        self.t_down = int(t_down)
        self.temporal_pool_sizes = (
            None
            if temporal_pool_sizes is None
            else tuple(int(v) for v in temporal_pool_sizes)
        )
        self.gcn_units = tuple(int(v) for v in gcn_units)
        self.gcn_dropout = float(gcn_dropout)
        self.gcn_activation = str(gcn_activation)
        self.gcn_use_batch_norm = bool(gcn_use_batch_norm)
        self.spectral_gru_units = int(spectral_gru_units)
        self.spectral_gru_dropout = float(spectral_gru_dropout)
        self.graph_add_self_loops = bool(graph_add_self_loops)
        self.graph_symmetrize = bool(graph_symmetrize)
        self.graph_epsilon = float(graph_epsilon)

        self.mi_n_neighbors = int(mi_n_neighbors)
        self.mi_random_state = int(mi_random_state)
        self.mi_zero_diagonal = bool(mi_zero_diagonal)
        self.mi_band_reduction = str(mi_band_reduction)
        self.mi_max_observations = (
            None if mi_max_observations is None else int(mi_max_observations)
        )
        if self.mi_max_observations is not None and self.mi_max_observations < 4:
            raise ValueError("mi_max_observations must be >= 4 or None.")

        self.classification_hidden_units = int(classification_hidden_units)
        self.classification_dropout_rate = float(classification_dropout)
        self.activation_name = str(activation)
        self.use_class_weight = bool(use_class_weight)

        self.graph_encoder: GCNMTLEncoder | None = None
        self._fitted_adjacency: np.ndarray | None = None
        if adjacency is not None:
            self._install_graph_encoder(np.asarray(adjacency, dtype=np.float32))

        # Non-recurrent pooling only. GCNMTL returns the GRU latent directly;
        # there is no post-GRU Conv1D/Dense projection.
        self.window_temporal_pool = tf.keras.layers.GlobalAveragePooling1D(
            name="v5_gcn_gru_temporal_mean"
        )
        self.classification_hidden = tf.keras.layers.Dense(
            self.classification_hidden_units,
            activation=self.activation_name,
            name="v5_classification_hidden",
        )
        self.classification_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            name="v5_classification_ln",
        )
        self.classification_dropout = tf.keras.layers.Dropout(
            self.classification_dropout_rate,
            name="v5_classification_dropout",
        )
        self.logits_layer = tf.keras.layers.Dense(
            self.n_classes,
            activation=None,
            name="v5_logits",
        )

        self.requires_subject_ids = False
        self.use_subject_adversarial = False

    @property
    def adjacency_is_fitted(self) -> bool:
        return self.graph_encoder is not None and self._fitted_adjacency is not None

    def _install_graph_encoder(self, adjacency: np.ndarray) -> None:
        adjacency = np.asarray(adjacency, dtype=np.float32)
        expected = (self.n_channels, self.n_channels)
        if adjacency.shape != expected:
            raise ValueError(f"adjacency must be {expected}, got {adjacency.shape}.")
        self._fitted_adjacency = adjacency.copy()
        self.graph_encoder = GCNMTLEncoder(
            timesteps=self.timesteps,
            t_down=self.t_down,
            adjacency=adjacency,
            n_channels=self.n_channels,
            n_bands=self.n_bands,
            gcn_units=self.gcn_units,
            temporal_pool_sizes=self.temporal_pool_sizes,
            dropout=self.gcn_dropout,
            activation=self.gcn_activation,
            use_batch_norm=self.gcn_use_batch_norm,
            use_spectral_gru=True,
            spectral_gru_units=self.spectral_gru_units,
            spectral_gru_dropout=self.spectral_gru_dropout,
            graph_add_self_loops=self.graph_add_self_loops,
            graph_symmetrize=self.graph_symmetrize,
            graph_epsilon=self.graph_epsilon,
            name="v5_mtl_gcn_gru_encoder",
        )

    def adapt_graph(self, training_inputs) -> np.ndarray:
        """Estimate and install a fixed MI graph from TRAINING INPUTS ONLY."""
        if isinstance(training_inputs, tf.data.Dataset):
            raise TypeError(
                "v5 graph adaptation expects an in-memory NumPy/Tensor training "
                "array so the MI graph can be computed before Keras fitting."
            )

        x = np.asarray(training_inputs, dtype=np.float32)
        if x.ndim not in {3, 4}:
            raise ValueError(
                "v5 expects training inputs shaped (N,T,F) or (N,W,T,F); "
                f"got {x.shape}."
            )
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x.shape[-1]}."
            )

        # Flatten sample/window/time axes into MI observations, then take a
        # deterministic training-only subsample when the raw fold is large.
        observations = x.reshape(-1, self.n_features)
        if (
            self.mi_max_observations is not None
            and len(observations) > self.mi_max_observations
        ):
            rng = np.random.default_rng(self.mi_random_state)
            idx = rng.choice(
                len(observations),
                size=self.mi_max_observations,
                replace=False,
            )
            observations = observations[idx]

        adjacency = compute_mtl_shared_mi_adjacency(
            observations,
            n_channels=self.n_channels,
            n_bands=self.n_bands,
            n_neighbors=self.mi_n_neighbors,
            random_state=self.mi_random_state,
            zero_diagonal=self.mi_zero_diagonal,
            band_reduction=self.mi_band_reduction,
        )
        self._install_graph_encoder(adjacency)
        return adjacency

    def fit(self, x=None, y=None, **kwargs):
        # Critical LOSO behavior: cross_val creates a fresh model for each fold,
        # then calls fit(X_fold_train,...). A is therefore fitted from that fold's
        # training subset and is frozen before validation/test inference.
        if not self.adjacency_is_fitted:
            self.adapt_graph(x)
        if not self.use_class_weight:
            kwargs.pop("class_weight", None)
        return super().fit(x=x, y=y, **kwargs)

    def _encode_windows(self, windows, training=False):
        if self.graph_encoder is None:
            raise RuntimeError(
                "The MI adjacency has not been fitted. Call model.fit(...) or "
                "model.adapt_graph(training_inputs) before inference."
            )
        seq = self.graph_encoder(windows, training=training)
        emb = self.window_temporal_pool(seq)
        return seq, emb

    def encode(self, inputs, training=False) -> dict[str, tf.Tensor]:
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)

        if self.classification_level == "window":
            if x.shape.rank != 3:
                raise ValueError(
                    "window mode expects (batch,timesteps,features); "
                    f"got {x.shape}."
                )
            seq, sample_embedding = self._encode_windows(x, training=training)
            return {
                "gcn_gru_sequence": seq,
                "window_embeddings": sample_embedding,
                "classification_embedding": sample_embedding,
            }

        if x.shape.rank != 4:
            raise ValueError(
                "trial mode expects (batch,windows,timesteps,features); "
                f"got {x.shape}."
            )
        batch_size = tf.shape(x)[0]
        n_windows = tf.shape(x)[1]
        flat = tf.reshape(
            x,
            (batch_size * n_windows, self.timesteps, self.n_features),
        )
        seq_flat, emb_flat = self._encode_windows(flat, training=training)
        window_embeddings = tf.reshape(
            emb_flat,
            (batch_size, n_windows, self.spectral_gru_units),
        )
        # No learned temporal module between windows in v5.0.
        trial_embedding = tf.reduce_mean(window_embeddings, axis=1)
        return {
            "gcn_gru_sequence": seq_flat,
            "window_embeddings": window_embeddings,
            "classification_embedding": trial_embedding,
        }

    def _classify_embedding(self, embedding, training=False):
        latent = self.classification_hidden(embedding)
        latent = self.classification_norm(latent)
        latent = self.classification_dropout(latent, training=training)
        logits = self.logits_layer(latent)
        return latent, logits

    def call(self, inputs, training=False):
        encoded = self.encode(inputs, training=training)
        _, logits = self._classify_embedding(
            encoded["classification_embedding"], training=training
        )
        return logits

    def predict_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        return self(x, training=False)

    def predict_diagnostics(self, inputs, batch_size=None):
        eeg = tf.convert_to_tensor(inputs, dtype=tf.float32)
        n = int(tf.shape(eeg)[0].numpy())
        bs = n if batch_size is None else int(batch_size)
        outputs: dict[str, list[tf.Tensor]] = {
            "classification_latent": [],
            "logits": [],
            "probabilities": [],
            "logit_margin": [],
        }
        for start in range(0, n, bs):
            batch = eeg[start : start + bs]
            encoded = self.encode(batch, training=False)
            latent, logits = self._classify_embedding(
                encoded["classification_embedding"], training=False
            )
            probs = tf.nn.softmax(logits, axis=-1)
            margin = (
                logits[:, 1] - logits[:, 0]
                if self.n_classes == 2
                else tf.math.top_k(logits, k=2).values[:, 0]
                - tf.math.top_k(logits, k=2).values[:, 1]
            )
            outputs["classification_latent"].append(latent)
            outputs["logits"].append(logits)
            outputs["probabilities"].append(probs)
            outputs["logit_margin"].append(margin)
        return {k: tf.concat(v, axis=0) for k, v in outputs.items()}

    def predict_mc_probabilities(self, inputs, n_samples=1, seed=None):
        del seed
        logits = self(inputs, training=False)
        probs = tf.nn.softmax(logits, axis=-1)
        return {
            "mean_probabilities": probs,
            "probability_samples": tf.repeat(
                probs[tf.newaxis, ...], repeats=int(n_samples), axis=0
            ),
        }

    def get_adjacency_matrices(self):
        if self.graph_encoder is None:
            return {}
        return {
            "mtl_mi_raw": self.graph_encoder.get_raw_adjacency_matrix(),
            "mtl_mi_normalized": self.graph_encoder.get_adjacency_matrix(),
        }

    def get_band_features(self, inputs, training=False):
        if self.graph_encoder is None:
            raise RuntimeError("Graph has not been adapted yet.")
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if x.shape.rank == 4:
            x = tf.reshape(x, (-1, self.timesteps, self.n_features))
        return self.graph_encoder.get_band_features(x, training=training)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "timesteps": self.timesteps,
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "classification_level": self.classification_level,
                "t_down": self.t_down,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "gcn_units": self.gcn_units,
                "gcn_dropout": self.gcn_dropout,
                "gcn_activation": self.gcn_activation,
                "gcn_use_batch_norm": self.gcn_use_batch_norm,
                "spectral_gru_units": self.spectral_gru_units,
                "spectral_gru_dropout": self.spectral_gru_dropout,
                "graph_add_self_loops": self.graph_add_self_loops,
                "graph_symmetrize": self.graph_symmetrize,
                "graph_epsilon": self.graph_epsilon,
                "mi_n_neighbors": self.mi_n_neighbors,
                "mi_random_state": self.mi_random_state,
                "mi_zero_diagonal": self.mi_zero_diagonal,
                "mi_band_reduction": self.mi_band_reduction,
                "mi_max_observations": self.mi_max_observations,
                "adjacency": (
                    None
                    if self._fitted_adjacency is None
                    else self._fitted_adjacency.tolist()
                ),
                "classification_hidden_units": self.classification_hidden_units,
                "classification_dropout": self.classification_dropout_rate,
                "activation": self.activation_name,
                "use_class_weight": self.use_class_weight,
            }
        )
        return config


def build_joint_sts_model(
    input_shape,
    *,
    n_classes=2,
    n_channels=14,
    n_bands=3,
    classification_level="window",
    t_down=2,
    temporal_pool_sizes=(2,),
    gcn_units=(32,),
    gcn_dropout=0.10,
    gcn_activation="relu",
    gcn_use_batch_norm=False,
    spectral_gru_units=384,
    spectral_gru_dropout=0.0,
    graph_add_self_loops=True,
    graph_symmetrize=True,
    graph_epsilon=1e-8,
    mi_n_neighbors=3,
    mi_random_state=42,
    mi_zero_diagonal=False,
    mi_band_reduction="mean",
    mi_max_observations=50000,
    adjacency=None,
    classification_hidden_units=128,
    classification_dropout=0.30,
    activation="relu",
    focal_gamma=0.0,
    focal_alpha=None,
    use_class_weight=False,
    optimizer_name="adamw",
    classification_learning_rate=1e-4,
    weight_decay=1e-4,
    model_name="joint_v5_sts_model",
    **unused_kwargs,
):
    """Build v5.0 MTL-GCN -> spectral GRU -> classifier.

    ``adjacency=None`` is intentional for CV: the model lazily estimates A from
    the training data received by ``fit``. Supplying adjacency is useful for
    reloads/tests where a precomputed training-only graph is already available.
    """
    classification_level = str(classification_level).lower()
    shape = tuple(int(v) for v in input_shape)
    if classification_level == "window":
        if len(shape) != 2:
            raise ValueError(f"window input_shape must be (T,F), got {shape}.")
        timesteps, n_features = shape
    elif classification_level == "trial":
        if len(shape) != 3:
            raise ValueError(f"trial input_shape must be (W,T,F), got {shape}.")
        _, timesteps, n_features = shape
    else:
        raise ValueError("classification_level must be 'window' or 'trial'.")

    model = JointSTSModel(
        timesteps=timesteps,
        n_features=n_features,
        n_classes=int(n_classes),
        n_channels=int(n_channels),
        n_bands=int(n_bands),
        classification_level=classification_level,
        t_down=int(t_down),
        temporal_pool_sizes=(
            None
            if temporal_pool_sizes is None
            else tuple(int(v) for v in temporal_pool_sizes)
        ),
        gcn_units=tuple(int(v) for v in gcn_units),
        gcn_dropout=float(gcn_dropout),
        gcn_activation=str(gcn_activation),
        gcn_use_batch_norm=bool(gcn_use_batch_norm),
        spectral_gru_units=int(spectral_gru_units),
        spectral_gru_dropout=float(spectral_gru_dropout),
        graph_add_self_loops=bool(graph_add_self_loops),
        graph_symmetrize=bool(graph_symmetrize),
        graph_epsilon=float(graph_epsilon),
        mi_n_neighbors=int(mi_n_neighbors),
        mi_random_state=int(mi_random_state),
        mi_zero_diagonal=bool(mi_zero_diagonal),
        mi_band_reduction=str(mi_band_reduction),
        mi_max_observations=(
            None if mi_max_observations is None else int(mi_max_observations)
        ),
        adjacency=adjacency,
        classification_hidden_units=int(classification_hidden_units),
        classification_dropout=float(classification_dropout),
        activation=str(activation),
        use_class_weight=bool(use_class_weight),
        name=model_name,
    )
    model.compile(
        optimizer=_build_optimizer(
            optimizer_name,
            float(classification_learning_rate),
            float(weight_decay),
        ),
        loss=SparseCategoricalFocalLoss(
            gamma=float(focal_gamma),
            alpha=focal_alpha,
            from_logits=True,
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )
    return model


__all__ = [
    "JOINT_STS_BUILDER_API_VERSION",
    "JointSTSModel",
    "SparseCategoricalFocalLoss",
    "build_joint_sts_model",
]
