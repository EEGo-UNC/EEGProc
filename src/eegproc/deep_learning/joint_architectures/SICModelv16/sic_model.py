"""Band-preserving SIC v16; reuse SIC's training core without its EEG encoders.

The inherited graph_encoder/gcn_gru_decoder slots are compatibility adapters:
they contain ThreeBandEncoder/ThreeBandDecoder, never a graph or spectral GRU.
This keeps existing SIC source training, MLDG and calibration steps unchanged.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..SICModelv15.sic_model import SICModel
from ..sic_common.model import SICModelCore, _build_optimizer
from ...unsupervised.Convolutions.CNN3D import (
    MTLFuseNet3DCNNEncoder, MTLFuseNet3DCNNDecoder,
)
from ...supervised.cnn_transformer import (
    LearnedTemporalPositionEmbedding, TemporalTransformerBlock,
)
from ...supervised.rnn_architectures import BiGRUClassifier, GRUClassifier
from ...supervised.variational_classifier import VariationalClassifier

BANDS = ("theta", "alpha", "beta")
SIC_BUILDER_API_VERSION = 16


def split_bands(inputs):
    """Channel-major/band-minor (..,42) -> (..,14,3), never contiguous slices."""
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    tf.debugging.assert_equal(tf.shape(inputs)[-1], 42)
    return tf.reshape(inputs, tf.concat([tf.shape(inputs)[:-1], [14, 3]], 0))


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class ThreeBandEncoder(tf.keras.Model):
    def __init__(self, timesteps=128, conv_filters=(32, 64),
                 spatial_pool_sizes=(2, 2), temporal_pool_sizes=(),
                 temporal_kernel_size=7, spatial_kernel_size=3, emb_dim=128,
                 dropout=0.2, use_transformer=False, transformer_layers=1,
                 attention_heads=4, feed_forward_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.settings = dict(timesteps=int(timesteps), conv_filters=tuple(conv_filters),
            spatial_pool_sizes=tuple(spatial_pool_sizes),
            temporal_pool_sizes=tuple(temporal_pool_sizes),
            temporal_kernel_size=int(temporal_kernel_size),
            spatial_kernel_size=int(spatial_kernel_size), emb_dim=int(emb_dim),
            dropout=float(dropout), use_transformer=bool(use_transformer),
            transformer_layers=int(transformer_layers), attention_heads=int(attention_heads),
            feed_forward_dim=int(feed_forward_dim))
        if len(conv_filters) != 2 or len(spatial_pool_sizes) != 2:
            raise ValueError("v16 uses exactly two Conv3D blocks per band.")
        if int(transformer_layers) < 1:
            raise ValueError("transformer_layers must be positive.")
        self.timesteps = int(timesteps)
        self.emb_dim = int(emb_dim)
        self.t_down = int(np.prod(temporal_pool_sizes)) if temporal_pool_sizes else 1
        self.encoders = []
        self.transformers = []
        for band in BANDS:
            encoder = MTLFuseNet3DCNNEncoder(
                timesteps=self.timesteps, n_channels=14, n_bands=1,
                t_down=self.t_down, conv_filters=conv_filters,
                spatial_pool_sizes=spatial_pool_sizes, temporal_pool_sizes=temporal_pool_sizes,
                temporal_kernel_size=temporal_kernel_size, spatial_kernel_size=spatial_kernel_size,
                emb_dim=emb_dim, dropout=dropout, name=f"{band}_cnn")
            self.encoders.append(encoder)
            if use_transformer:
                sequence = [LearnedTemporalPositionEmbedding(
                    encoder.output_timesteps, self.emb_dim, name=f"{band}_positions")]
                sequence.extend(TemporalTransformerBlock(
                    self.emb_dim, attention_heads, feed_forward_dim, dropout,
                    name=f"{band}_transformer_{index}") for index in range(transformer_layers))
                sequence.append(layers.LayerNormalization(epsilon=1e-6, name=f"{band}_norm"))
                self.transformers.append(tf.keras.Sequential(sequence, name=f"{band}_attention"))

    def encode(self, inputs, training=False):
        bands = split_bands(inputs)
        # Calling each CNN through __call__ preserves its build/serialization state.
        latents = [encoder(bands[..., index], training=training)
                   for index, encoder in enumerate(self.encoders)]
        attended = ([block(latent, training=training)
                     for block, latent in zip(self.transformers, latents)]
                    if self.transformers else latents)
        return tf.concat(latents, -1), tf.concat(attended, -1)

    def call(self, inputs, training=False):
        return self.encode(inputs, training=training)[1]

    def build(self, input_shape):
        self.encode(tf.zeros((1, self.timesteps, 42)), training=False)
        super().build(input_shape)

    def get_config(self):
        return {**super().get_config(), **self.settings}


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class ThreeBandDecoder(tf.keras.Model):
    def __init__(self, timesteps=128, emb_dim=128, temporal_pool_sizes=(),
                 dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.settings = dict(timesteps=int(timesteps), emb_dim=int(emb_dim),
            temporal_pool_sizes=tuple(temporal_pool_sizes), dropout=float(dropout))
        self.timesteps = int(timesteps)
        self.emb_dim = int(emb_dim)
        t_down = int(np.prod(temporal_pool_sizes)) if temporal_pool_sizes else 1
        self.latent_timesteps = (self.timesteps + t_down - 1) // t_down
        self.decoders = [MTLFuseNet3DCNNDecoder(
            timesteps=self.timesteps, n_channels=14, n_bands=1, t_down=t_down,
            temporal_pool_sizes=temporal_pool_sizes, emb_dim=emb_dim, dropout=dropout,
            name=f"{band}_decoder") for band in BANDS]

    def call(self, inputs, training=False):
        reconstructions = [decoder(latent, training=training) for decoder, latent
                           in zip(self.decoders, tf.split(inputs, 3, axis=-1))]
        # Restore original channel-major, band-minor feature order.
        stacked = tf.stack(reconstructions, axis=-1)
        return tf.reshape(stacked, [tf.shape(inputs)[0], self.timesteps, 42])

    def build(self, input_shape):
        self.call(tf.zeros((1, self.latent_timesteps, self.emb_dim * 3)))
        super().build(input_shape)

    def get_config(self):
        return {**super().get_config(), **self.settings}


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class ThreeBandSICModel(SICModel):
    def __init__(self, *, graph_encoder, gcn_gru_decoder=None,
                 use_recurrent_classifier=False, use_variational_head=False,
                 classifier_units=128, classifier_dropout=0.2, **kwargs):
        self.use_recurrent_classifier = bool(use_recurrent_classifier)
        self.use_variational_head = bool(use_variational_head)
        self.classifier_units = int(classifier_units)
        self.classifier_dropout = float(classifier_dropout)
        if self.classifier_units < 1 or not 0 <= self.classifier_dropout < 1:
            raise ValueError("Invalid dense classifier width or dropout.")
        # Reject configurations that would silently restore SIC's original branches.
        for key, expected in (("use_gcn_gru_branch", True), ("use_bilstm_branch", False)):
            if key in kwargs and kwargs[key] != expected:
                raise ValueError(f"{key} is an internal compatibility slot in v16.")
            kwargs[key] = expected
        if not isinstance(graph_encoder, ThreeBandEncoder):
            raise TypeError("v16 requires ThreeBandEncoder, not a SIC graph encoder.")
        if kwargs.get("bilstm_decoder") is not None:
            raise ValueError("v16 has no BiLSTM decoder.")
        if kwargs.get("update_vc_discriminator", False) and not self.use_variational_head:
            raise ValueError("VC discriminator requires use_variational_head=true.")
        kwargs["gcn_gru_feature_dim"] = 3 * graph_encoder.emb_dim
        kwargs.setdefault("training_method", "erm")
        kwargs.setdefault("use_subject_adversarial", False)
        kwargs.setdefault("focal_gamma", 0.0)
        super().__init__(graph_encoder=graph_encoder, gcn_gru_decoder=gcn_gru_decoder, **kwargs)
        if not self.use_variational_head:
            self.vc_loss_tracker = tf.keras.metrics.Mean(name="classification_objective")
        self.band_loss_trackers = [tf.keras.metrics.Mean(name=f"{band}_reconstruction_loss")
                                   for band in BANDS]
        # Build decoder weights on construction so load_model restores every branch,
        # even though inference never invokes the reconstruction path.
        if self.use_decoder:
            if not isinstance(gcn_gru_decoder, ThreeBandDecoder):
                raise TypeError("v16 reconstruction requires ThreeBandDecoder.")
            gcn_gru_decoder(tf.zeros((1, graph_encoder.encoders[0].output_timesteps,
                                     self.combined_feature_dim)))

    def _build_sequence_heads(self, *, vc_logit_scale=1.0):
        self.temporal_bilstms, self.temporal_norms, self.temporal_dropouts = [], [], []
        if self.use_recurrent_classifier:
            common = dict(timesteps=None, n_features=self.combined_feature_dim,
                          n_classes=self.n_classes, gru_units=self.classifier_rnn_units,
                          dropout=self.classifier_rnn_dropout_rate)
            if self.classifier_rnn_type == "bigru":
                builder = BiGRUClassifier(**common, n_bigru_layers=self.n_classifier_rnn_layers)
                self.classification_embedding_dim = 2 * self.classifier_rnn_units[-1]
            else:
                builder = GRUClassifier(**common, n_gru_layers=self.n_classifier_rnn_layers)
                self.classification_embedding_dim = self.classifier_rnn_units[-1]
            self.trial_recurrent_classifier = builder.build_sequence_summarizer()
        else:
            self.trial_recurrent_classifier = tf.keras.Sequential([
                layers.GlobalAveragePooling1D(),
                layers.Dense(self.classifier_units, activation="relu"),
                layers.Dropout(self.classifier_dropout),
            ], name="temporal_pool_dense")
            self.classification_embedding_dim = self.classifier_units
        if self.use_variational_head:
            self.vc_target = VariationalClassifier(
                n_classes=self.n_classes, latent_dim=self.classification_embedding_dim,
                label_smoothing=self.label_smoothing, focal_gamma=self.focal_gamma,
                focal_alpha=None if self.focal_alpha is None else (1-self.focal_alpha, self.focal_alpha),
                logit_scale=vc_logit_scale, name="sic_v16_vc_head")
        else:
            self.vc_target = layers.Dense(self.n_classes, name="sic_v16_dense_logits")
        self.vc_target.build(tf.TensorShape([None, self.classification_embedding_dim]))
        self.subject_gradient_reversal = self.subject_hidden = None
        self.subject_dropout_layer = self.subject_logits_layer = None
        if self.subject_adversarial_enabled and self.n_subject_classes is not None:
            self._configure_subject_head(self.n_subject_classes)
        self.main_optimizer = self.vc_discriminator_optimizer = None

    def _features_from_flat_windows(self, flat_windows, training):
        # Build the container as well as nested layers before checkpointing.
        if not self.graph_encoder.built:
            self.graph_encoder.build(flat_windows.shape)
        before_attention, after_attention = self.graph_encoder.encode(flat_windows, training=training)
        return {"graph_sequence": before_attention, "bilstm_sequence": None,
                "combined_feature_sequence": after_attention}

    def _vc_components(self, classification_embedding, logits, y_flat, sample_weight, *, calibration):
        if self.use_variational_head:
            return super()._vc_components(classification_embedding, logits, y_flat,
                                          sample_weight, calibration=calibration)
        per_sample = (self._per_sample_calibration_loss(logits, y_flat) if calibration
                      else self._per_sample_focal_loss(logits, y_flat,
                          gamma=self.focal_gamma, alpha=self.focal_alpha,
                          label_smoothing=self.label_smoothing))
        loss = tf.reduce_mean(per_sample)
        weighted = self._weighted_mean(per_sample, sample_weight)
        zero = tf.zeros((), logits.dtype)
        return dict(total_loss=weighted, classification_loss=loss,
            weighted_classification_loss=weighted, focal_loss=loss, weighted_focal_loss=weighted,
            latent_posterior_kl=zero, weighted_latent_posterior_kl=zero,
            discriminator_kl=zero, weighted_discriminator_kl=zero,
            class_prior_kl=zero, weighted_class_prior_kl=zero)

    def _reconstruction_components(self, outputs, training):
        # MSE over the reinterleaved 42 features is exactly mean(per-band MSE).
        components = SICModelCore._reconstruction_components(self, outputs, training)
        components["joint_reconstruction"] = None
        if self.use_decoder:
            target = split_bands(outputs["flat_windows"])
            reconstructed = split_bands(components["gcn_gru_reconstruction"])
            for index, band in enumerate(BANDS):
                components[f"{band}_reconstruction_loss"] = tf.reduce_mean(
                    tf.square(target[..., index] - reconstructed[..., index]))
        return components

    @property
    def metrics(self):
        # Hide legacy graph-slot metric names; these are three-band CNNs.
        return [m for m in super().metrics if not m.name.startswith("gcn_gru_")]

    def _extra_decoder_metrics(self):
        return getattr(self, "band_loss_trackers", [])

    def _update_extra_decoder_metrics(self, outputs, reconstruction_components, zero):
        if reconstruction_components is not None:
            for band, tracker in zip(BANDS, self.band_loss_trackers):
                tracker.update_state(reconstruction_components.get(f"{band}_reconstruction_loss", zero))

    def get_adjacency_matrices(self):
        return {}  # No MI graph is constructed or fitted in v16.

    def get_encoder_features(self, inputs):
        eeg, _ = self._split_eeg_and_subject_inputs(inputs)
        outputs = self._encode(eeg, training=False)
        cnn_parts = tf.split(outputs["graph_sequence"], 3, axis=-1)
        final_parts = tf.split(outputs["combined_feature_sequence"], 3, axis=-1)
        return {"cnn_latents": dict(zip(BANDS, cnn_parts)),
                "band_features": dict(zip(BANDS, final_parts)),
                **{key: outputs[key] for key in ("combined_feature_sequence", "classifier_sequence",
                    "pooled_features", "window_features", "classification_embedding", "probabilities")}}

    def decode_branch_feature_sequence(self, branch, feature_sequence):
        if not self.use_decoder:
            raise RuntimeError("Decoder is disabled.")
        if branch not in BANDS:
            raise ValueError(f"Expected one of {BANDS}.")
        return self.gcn_gru_decoder.decoders[BANDS.index(branch)](feature_sequence, training=False)

    def reconstruct_branches(self, inputs):
        full = super().reconstruct_branches(inputs)["gcn_gru"]
        separated = split_bands(full)
        return {band: separated[..., index] for index, band in enumerate(BANDS)}

    def get_config(self):
        return {**super().get_config(), "use_recurrent_classifier": self.use_recurrent_classifier,
                "use_variational_head": self.use_variational_head,
                "classifier_units": self.classifier_units, "classifier_dropout": self.classifier_dropout}

    @classmethod
    def from_config(cls, config):
        model = SICModelCore.from_config.__func__(cls, config)
        shape = (1, model.graph_encoder.timesteps, 42)
        if model.classification_level == "trial":
            shape = (1, 1, model.graph_encoder.timesteps, 42)
        model(tf.zeros(shape), training=False)
        return model

    def configure_fold_devices(self, devices, **kwargs):
        if len(devices) != 1:
            raise ValueError("v16 currently uses one GPU per fold; v15's graph sharding is incompatible.")


def build_sic_model(input_shape, *, training_features=None, training_labels=None,
                    training_subject_ids=None, training_trial_ids=None,
                    use_decoder=False, use_transformer=False,
                    conv_filters=(32, 64), spatial_pool_sizes=(2, 2), temporal_pool_sizes=(),
                    temporal_kernel_size=7, spatial_kernel_size=3, emb_dim=128, dropout=0.2,
                    transformer_layers=1, attention_heads=4, feed_forward_dim=256,
                    learning_rate=1e-4, weight_decay=5e-5, optimizer_name="adamw",
                    classification_level="trial", **kwargs):
    """SIC-compatible builder; training metadata never contains the held-out user."""
    input_shape = tuple(input_shape)
    rank = 3 if classification_level == "trial" else 2
    if len(input_shape) != rank or input_shape[-1] != 42:
        raise ValueError("Expected (windows,time,42) for trials or (time,42) for windows.")
    if any(d is None or int(d) < 1 for d in input_shape):
        raise ValueError("Builder requires positive, fixed input dimensions.")
    if kwargs.get("update_vc_discriminator", False):
        raise ValueError("v16 does not yet support a separate VC discriminator optimizer.")
    encoder = ThreeBandEncoder(timesteps=input_shape[-2], conv_filters=conv_filters,
        spatial_pool_sizes=spatial_pool_sizes, temporal_pool_sizes=temporal_pool_sizes,
        temporal_kernel_size=temporal_kernel_size, spatial_kernel_size=spatial_kernel_size,
        emb_dim=emb_dim, dropout=dropout, use_transformer=use_transformer,
        transformer_layers=transformer_layers, attention_heads=attention_heads,
        feed_forward_dim=feed_forward_dim, name="three_band_encoder")
    decoder = (ThreeBandDecoder(timesteps=input_shape[-2], emb_dim=emb_dim,
        temporal_pool_sizes=temporal_pool_sizes, dropout=dropout, name="three_band_decoder")
        if use_decoder else None)
    if kwargs.get("use_subject_adversarial", False) and training_subject_ids is not None:
        kwargs["n_subject_classes"] = len(np.unique(training_subject_ids))
    if kwargs.get("use_subject_adversarial", False) and kwargs.get("n_subject_classes") is None:
        raise ValueError("Provide training_subject_ids or n_subject_classes to build the adversary before fitting.")
    model = ThreeBandSICModel(graph_encoder=encoder, gcn_gru_decoder=decoder,
        use_decoder=use_decoder, classification_level=classification_level, **kwargs)
    model.set_source_training_metadata(training_subject_ids, training_trial_ids)
    model.compile(main_optimizer=_build_optimizer(optimizer_name=optimizer_name,
        learning_rate=learning_rate, weight_decay=weight_decay), jit_compile=False)
    # One window is sufficient to build trial heads; avoid allocating a full trial.
    dummy_shape = (1, 1, input_shape[-2], 42) if rank == 3 else (1, input_shape[-2], 42)
    model(tf.zeros(dummy_shape), training=False)
    if model.subject_logits_layer is not None:
        model._subject_logits(tf.zeros((1, model.classification_embedding_dim)),
                              training=False, use_grl=False)
    return model
