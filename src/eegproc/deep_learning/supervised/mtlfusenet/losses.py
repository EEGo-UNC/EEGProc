import tensorflow as tf


def focal_loss(y_true, y_pred, alpha=0.7, gamma=2.0):
    """Paper doesn't state alpha/gamma explicitly for DREAMER — using
       defaults from the original focal loss paper (Lin et al.)."""
    y_true_onehot = tf.one_hot(y_true, depth=2)
    y_true_onehot = tf.cast(y_true_onehot, tf.float32)

    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    ce = -y_true_onehot * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    fl = weight * ce

    return tf.reduce_sum(fl, axis=-1)


def triplet_center_loss(features, labels, centers, margin=1.0):
    """features: (batch, feature_dim), labels: (batch,) int, centers: (num_classes, feature_dim)"""
    num_classes = tf.shape(centers)[0]

    features_expanded = tf.expand_dims(features, axis=1)
    centers_expanded = tf.expand_dims(centers, axis=0)
    distances = 0.5 * tf.reduce_sum(tf.square(features_expanded - centers_expanded), axis=-1)

    labels = tf.reshape(labels, [-1])
    own_class_dist = tf.gather(distances, labels, axis=1, batch_dims=1)

    mask = tf.one_hot(labels, num_classes) * 1e9
    other_class_dist = tf.reduce_min(distances + mask, axis=1)

    loss = tf.maximum(own_class_dist + margin - other_class_dist, 0.0)
    return tf.reduce_mean(loss)


def total_loss(fl, tc, vae_loss, beta1=0.7, beta2=0.2, beta3=0.1):
    """Eq. 22: weighted combination of focal loss, triplet-center loss, VAE loss."""
    return beta1 * fl + beta2 * tc + beta3 * vae_loss