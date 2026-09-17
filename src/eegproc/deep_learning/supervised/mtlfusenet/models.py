import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class Sampling(layers.Layer):
    """Reparameterization trick: samples a latent vector from (mean, log-variance)."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_encoder(input_shape=(9, 9, 128), latent_dim=128):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    return Model(inputs, [z_mean, z_log_var], name='vae_encoder')


def build_vae_decoder(latent_dim=128, output_shape=(9, 9, 128)):
    """Note: Dense(9*9*512) projection reverses the encoder's Flatten.
       sigmoid on the final layer assumes input EEG is normalized to 0-1 range."""
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(9 * 9 * 512, activation='relu')(latent_inputs)
    x = layers.Reshape((9, 9, 512))(x)
    x = layers.Conv2DTranspose(256, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(256, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, 3, strides=1, padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(128, 3, strides=1, padding='same', activation='sigmoid')(x)
    return Model(latent_inputs, outputs, name='vae_decoder')


class VAE(Model):
    """Wires encoder -> sampling -> decoder. Uses MSE as reconstruction loss
       (interpretation of paper's log p(x|z) term for Gaussian-assumed data)."""
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)

        return reconstruction


def build_vae():
    """Convenience function: builds and compiles the full VAE."""
    encoder = build_vae_encoder()
    decoder = build_vae_decoder()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam', loss='mse')
    return vae, encoder, decoder

class GCNLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.W = self.add_weight(shape=(feature_dim, self.output_dim),
                                  initializer='glorot_uniform', trainable=True, name='W')
        self.b = self.add_weight(shape=(self.output_dim,),
                                  initializer='zeros', trainable=True, name='b')

    def call(self, inputs):
        node_features, adj_normalized = inputs
        x = tf.matmul(adj_normalized, node_features)
        x = tf.matmul(x, self.W) + self.b
        return tf.nn.relu(x)


def normalize_adjacency(adj):
    """Symmetric normalization for GCN, per paper Eq. 15 (adds self-loops first)."""
    n = adj.shape[0]
    A_tilde = adj + np.eye(n)
    D_tilde = np.diag(A_tilde.sum(axis=1))
    D_tilde_inv_sqrt = np.linalg.inv(np.sqrt(D_tilde))
    return D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt


def compute_gcn_sequence(de_features, window_idx, gcn_layer, adj_matrices, bands=('theta', 'alpha', 'beta')):
    """Runs the GCN on all three bands for one time window, returns a (3, 14, output_dim) sequence."""
    band_outputs = []
    for band in bands:
        band_cols = [c for c in de_features.columns if c.endswith(f'_{band}')]
        features = de_features[band_cols].values[window_idx].reshape(-1, 1)
        feat_tensor = tf.constant(features, dtype=tf.float32)
        adj_tensor = tf.constant(adj_matrices[band], dtype=tf.float32)
        output = gcn_layer([feat_tensor, adj_tensor])
        band_outputs.append(output)
    return tf.stack(band_outputs, axis=0)


def build_gru(units=384):
    """Spatio-spectral latent dim = 384 for DREAMER, per paper Table 2."""
    return layers.GRU(units=units)