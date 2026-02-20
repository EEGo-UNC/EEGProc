import tensorflow as tf
from tensorflow.keras import layers, Model
from GraphConv import GraphConv


def encoder_1dcnn(
    timesteps: int,
    n_features: int,
    base_filters: int = 64,
    kernel_size: int = 7,
    emb_dim: int = 128,
    dropout: float = 0.10,
) -> tf.keras.Model:
    """Build a 1D convolutional sequence encoder over the time axis.

    Treats the feature vector at each timestep as a flat, unstructured input
    and learns temporal patterns via three Conv1D blocks. Two
    MaxPool1D layers downsample the time axis by a factor of 4 in total.
    This encoder makes no assumptions about the internal organisation of
    ``n_features`` and serves as a general-purpose temporal baseline.

    Parameters
    ----------
    timesteps : int
        Number of timesteps T.
    n_features : int
        Number of features per timestep (e.g. 84 for 14 electrodes x 6 bands).
    base_filters : int, optional
        Number of filters in the first Conv1D block. Subsequent blocks use
        ``base_filters * 2`` and ``base_filters * 4`` respectively.
        Default is 64.
    kernel_size : int, optional
        Temporal kernel size for the first Conv1D block. The second block uses
        ``kernel_size // 2 + 1``, the third uses 3. Default is 7.
    emb_dim : int, optional
        Dimensionality of the output embedding at each downsampled timestep.
        Default is 128.
    dropout : float, optional
        Dropout rate applied after each pooling operation. Default is 0.10.

    Returns
    -------
    tf.keras.Model
        Keras Model with input shape ``(batch, timesteps, n_features)`` and
        output shape ``(batch, ceil(timesteps / 4), emb_dim)``.

    Notes
    -----
    The output shape is compatible with ``training_autoencoder``, which
    accepts any encoder whose output is ``(batch, t_down, emb_dim)``.
    """
    x_in = layers.Input(shape=(timesteps, n_features), name="x")

    x = layers.Conv1D(
        base_filters, kernel_size, padding="same", activation="relu", name="enc_conv1"
    )(x_in)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_pool1")(x)
    x = layers.Dropout(dropout, name="enc_do1")(x)

    x = layers.Conv1D(
        base_filters * 2,
        kernel_size // 2 + 1,
        padding="same",
        activation="relu",
        name="enc_conv2",
    )(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_pool2")(x)
    x = layers.Dropout(dropout, name="enc_do2")(x)

    x = layers.Conv1D(
        base_filters * 4, 3, padding="same", activation="relu", name="enc_conv3"
    )(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)

    seq_emb = layers.Conv1D(
        emb_dim, 1, padding="same", activation=None, name="seq_emb"
    )(x)

    return Model(inputs=x_in, outputs=seq_emb, name="encoder_1dcnn")


def encoder_2dcnn(
    timesteps: int,
    n_channels: int = 14,
    n_bands: int = 6,
    base_filters: int = 32,
    emb_dim: int = 128,
    dropout: float = 0.10,
) -> tf.keras.Model:
    """Build a 2D convolutional sequence encoder over the electrode x band plane.

    Accepts the same flat input shape as ``encoder_1dcnn`` but internally
    reshapes it into a spatial grid of ``(n_channels, n_bands)`` at each
    timestep. Three ``TimeDistributed`` Conv2D blocks learn joint
    electrode-band patterns.

    After spatial feature extraction, a ``TimeDistributed``
    ``GlobalAveragePooling2D`` collapses the electrode x band grid into a
    single vector per timestep. Two ``MaxPool1D`` layers then downsample the
    time axis by a factor of 4, matching the temporal compression of
    ``encoder_1dcnn``. A final ``Conv1D`` projection maps the result to
    ``emb_dim``.

    Parameters
    ----------
    timesteps : int
        Number of timesteps.
    n_channels : int, optional
        Number of EEG electrode channels.
    n_bands : int, optional
        Number of frequency bands.
    base_filters : int, optional
        Number of filters in the first Conv2D block. Subsequent blocks use
        ``base_filters * 2`` and ``base_filters * 4`` respectively.
        Default is 32.
    emb_dim : int, optional
        Dimensionality of the output embedding at each downsampled timestep.
        Default is 128.
    dropout : float, optional
        Dropout rate applied after each spatial conv block and after the
        first temporal pooling step. Default is 0.10.

    Returns
    -------
    tf.keras.Model
        Keras Model with input shape ``(batch, timesteps, n_channels * n_bands)``
        and output shape ``(batch, ceil(timesteps / 4), emb_dim)``. The input
        shape is identical to ``encoder_1dcnn`` for the same feature count,
        making both encoders interchangeable as arguments to
        ``training_autoencoder``.

    Notes
    -----
    Conv2D kernel sizes are ``(3, 3)`` for the first two blocks and ``(2, 2)``
    for the third. ``same`` padding preserves the spatial grid dimensions
    throughout, so no spatial downsampling occurs and all downsampling occurs
    only on the time axis.

    ``TimeDistributed`` wraps every spatial layer so that the same weights
    are applied independently at each timestep. This weight sharing keeps the
    parameter count tractable and enforces electrode-band spatial relationship
    biases.

    Because ``TimeDistributed(Conv2D(...))`` processes ``T`` spatial grids
    per batch item, memory usage is substantially higher than
    ``encoder_1dcnn`` at equal filter counts. Reduce ``base_filters`` if
    out-of-memory errors occur during training.
    """
    n_features = n_channels * n_bands
    x_in = layers.Input(shape=(timesteps, n_features), name="x")

    x = layers.Reshape((timesteps, n_channels, n_bands, 1), name="to_grid")(x_in)

    x = layers.TimeDistributed(
        layers.Conv2D(base_filters, (3, 3), padding="same", activation="relu"),
        name="enc_conv2d_1",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="enc_bn2d_1")(x)
    x = layers.TimeDistributed(layers.Dropout(dropout), name="enc_do2d_1")(x)

    x = layers.TimeDistributed(
        layers.Conv2D(base_filters * 2, (3, 3), padding="same", activation="relu"),
        name="enc_conv2d_2",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="enc_bn2d_2")(x)
    x = layers.TimeDistributed(layers.Dropout(dropout), name="enc_do2d_2")(x)

    x = layers.TimeDistributed(
        layers.Conv2D(base_filters * 4, (2, 2), padding="same", activation="relu"),
        name="enc_conv2d_3",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="enc_bn2d_3")(x)

    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name="enc_gap2d")(x)

    x = layers.MaxPool1D(2, padding="same", name="enc_tpool1")(x)
    x = layers.Dropout(dropout, name="enc_tdo1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool2")(x)

    seq_emb = layers.Conv1D(
        emb_dim, 1, padding="same", activation=None, name="seq_emb"
    )(x)

    return Model(inputs=x_in, outputs=seq_emb, name="encoder_2dcnn")


def training_autoencoder(
    encoder: tf.keras.Model,
    timesteps: int,
    n_features: int,
    base_filters: int = 64,
    kernel_size: int = 7,
    dropout: float = 0.10,
    lr: float = 1e-3,
    loss: str = "mse",
) -> tf.keras.Model:
    """Attach a 1D convolutional decoder to a sequence encoder and compile an autoencoder.

    Builds a symmetric 1D decoder that mirrors the temporal structure of
    ``encoder_1dcnn``: two ``UpSampling1D`` layers restore the time axis from
    ``ceil(T / 4)`` back to ``T``, and a final ``Conv1D`` projects each
    timestep back to ``n_features``. A length-correction layer trims or pads
    the output to exactly ``timesteps`` in case upsampling introduces
    off-by-one differences.

    The decoder reads its input shape directly from ``encoder.output_shape``,
    so it is agnostic to which encoder is passed. All functions in ``unsupervised.py``
    produce ``(batch, ceil(T / 4), emb_dim)`` and are, therefore,
    fully interchangeable as arguments to this function.

    Parameters
    ----------
    encoder : tf.keras.Model
        A compiled or uncompiled Keras Model whose output shape is
        ``(batch, t_down, emb_dim)``. Usually should be the returned
        models from previous functions
    timesteps : int
        Original input sequence length T. Used by the length-correction layer
        to guarantee the reconstruction has exactly this many timesteps.
    n_features : int
        Number of features per timestep to reconstruct. Must match the feature
        dimension of the data passed to ``autoencoder.fit``.
    base_filters : int, optional
        Number of filters in the final decoder Conv1D block. Earlier blocks
        use ``base_filters * 2`` and ``base_filters * 4``. Default is 64.
    kernel_size : int, optional
        Kernel size used in the second-to-last decoder Conv1D block
        (``kernel_size // 2 + 1``). Default is 7.
    dropout : float, optional
        Dropout rate applied in the two intermediate decoder blocks.
        Default is 0.10.
    lr : float, optional
        Learning rate for the Adam optimiser. Default is 1e-3.
    loss : str, optional
        Keras loss identifier passed to ``model.compile``. Default is ``"mse"``.

    Returns
    -------
    tf.keras.Model
        Compiled Keras Model mapping ``(batch, timesteps, n_features)`` to a
        reconstruction of the same shape, optimised with Adam and monitored
        with mean absolute error alongside the primary loss.
    """
    seq_in = layers.Input(shape=encoder.output_shape[1:], name="seq_in")

    y = layers.Conv1D(
        base_filters * 4, 3, padding="same", activation="relu", name="dec_conv0"
    )(seq_in)
    y = layers.BatchNormalization(name="dec_bn0")(y)

    y = layers.UpSampling1D(2, name="dec_up1")(y)
    y = layers.Conv1D(
        base_filters * 2, 3, padding="same", activation="relu", name="dec_conv1"
    )(y)
    y = layers.BatchNormalization(name="dec_bn1")(y)
    y = layers.Dropout(dropout, name="dec_do1")(y)

    y = layers.UpSampling1D(2, name="dec_up2")(y)
    y = layers.Conv1D(
        base_filters,
        kernel_size // 2 + 1,
        padding="same",
        activation="relu",
        name="dec_conv2",
    )(y)
    y = layers.BatchNormalization(name="dec_bn2")(y)
    y = layers.Dropout(dropout, name="dec_do2")(y)

    x_hat = layers.Conv1D(n_features, 1, padding="same", activation=None, name="x_hat")(
        y
    )

    def fix_len(t):
        t = t[:, :timesteps, :]
        cur = tf.shape(t)[1]
        pad = tf.maximum(0, timesteps - cur)
        return tf.pad(t, [[0, 0], [0, pad], [0, 0]])

    x_hat = layers.Lambda(fix_len, name="fix_length")(x_hat)

    decoder = Model(seq_in, x_hat, name="decoder_seq")

    autoencoder = Model(encoder.input, decoder(encoder.output), name="autoencoder_seq")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    return autoencoder


def encoder_gcn(
    timesteps: int,
    n_channels: int = 14,
    n_bands: int = 6,
    gcn_units: tuple[int] = (32, 64),
    emb_dim: int = 128,
    dropout: float = 0.10,
) -> tf.keras.Model:
    # Partially generated with AI assistance (Claude, Anthropic) — reviewed and adapted.
    """Build a graph convolutional sequence encoder with a learned adjacency matrix.

    At each timestep the flat feature vector is reshaped into a node feature
    matrix ``X_t ∈ R^{N x F}`` (channels x bands). A stack of
    ``GraphConv`` layers — each with its own learnable adjacency
    matrix ``A_theta`` — performs neighbourhood aggregation across channels,
    allowing the model to discover functional connectivity patterns that are
    invisible to Conv1D or Conv2D encoders which assume a fixed or regular
    topology. All graph layers are applied via ``TimeDistributed`` so that the
    same weights operate independently at each timestep.

    After graph feature extraction, ``GlobalAveragePooling`` collapses the
    node dimension to a single vector per timestep. Two ``MaxPool1D`` layers
    then compress the time axis by a factor of 4, and a final ``Dense``
    projection maps each timestep to ``emb_dim``.

    Parameters
    ----------
    timesteps : int
        Number of timesteps T.
    n_channels : int, optional
        Number of EEG electrode nodes N. Default is 14 (DREAMER dataset).
    n_bands : int, optional
        Number of frequency bands F per electrode, forming the initial node
        feature dimension. Default is 6.
    gcn_units : tuple of int, optional
        Output dimensionality of each successive ``GraphConv``
        layer. The length of this tuple determines the number of GCN layers.
        Default is ``(32, 64)``.
    emb_dim : int, optional
        Dimensionality of the final output embedding per downsampled timestep.
        Default is 128.
    dropout : float, optional
        Dropout rate applied after each GCN layer and after the first temporal
        pooling step. Default is 0.10.

    Returns
    -------
    tf.keras.Model
        Keras Model with input shape ``(batch, timesteps, n_channels * n_bands)``
        and output shape ``(batch, ceil(timesteps / 4), emb_dim)``. The output
        shape will also fit in training_autoencoder.

    Notes
    -----
    Each ``GraphConv`` layer maintains its own ``A_theta ∈
    R^{N x N}`` adjacency matrix, initialized with Glorot uniform weights and
    updated end-to-end during training. With ``n_channels=14`` this adds
    only ``len(gcn_units) * 196`` parameters to the model — negligible
    overhead relative to the projection weights.

    If electrode topology priors are available (e.g. inverse scalp distances
    from the 10-20 system), each layer's adjacency matrix can be warm-started
    after construction:

        enc = encoder_gcn(...)
        prior = compute_distance_matrix()   # (14, 14)
        enc.get_layer("gcn_0").A_theta.assign(prior)

    ``TimeDistributed`` enforces that the learned graph structure is
    stationary across timesteps.
    """
    n_features = n_channels * n_bands
    x_in = layers.Input(shape=(timesteps, n_features), name="x")

    x = layers.Reshape((timesteps, n_channels, n_bands), name="to_nodes")(x_in)

    for i, units in enumerate(gcn_units):
        x = layers.TimeDistributed(
            GraphConv(units=units, n_nodes=n_channels),
            name=f"gcn_{i}",
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization(), name=f"gcn_bn_{i}")(x)
        x = layers.TimeDistributed(layers.Dropout(dropout), name=f"gcn_do_{i}")(x)

    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=2), name="gcn_pool_nodes")(x)

    x = layers.MaxPool1D(2, padding="same", name="enc_tpool1")(x)
    x = layers.Dropout(dropout, name="enc_tdo1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool2")(x)

    seq_emb = layers.Dense(emb_dim, activation=None, name="seq_emb")(x)

    return Model(inputs=x_in, outputs=seq_emb, name="encoder_gcn")
