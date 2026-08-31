"""The mathematical objective for one complete SIC counterfactual trial.

Only z_prime is optimized. The original trial x, original encoder features z,
and all model parameters are constants. Distances are means, not sums, so
their scale does not automatically grow with trial length or feature width.
There is no VAE, KL divergence, classifier training, or physiological model.
"""

import math
from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class CounterfactualLoss:
    """Weighted target, latent, decoded-signal, and physiological penalties.

    The objective is target_weight * target_loss + latent_weight * MSE(z', z)
    + decoded_weight * mean_b MSE(D_b(z'_b), x) + physiological_weight * 0.
    b ranges over active encoder branches. Each branch decoder reconstructs
    the same original, preprocessed EEG trial; it never receives the other
    branch's features. No physiological constraint is currently enforced.

    target_probability is a desired softmax confidence, not a replacement
    for the predicted-class decision rule. The optimizer requires both the
    requested class and this confidence for success. Weights trade off the
    numerical penalties; they do not standardize EEG or encoder dimensions.
    Signal distances are in the model's input units, not necessarily microvolts.
    """

    target_weight: float = 1.0
    latent_weight: float = 0.1
    decoded_weight: float = 0.1
    physiological_weight: float = 0.0
    target_probability: float = 0.8

    def __post_init__(self):
        for name in (
            "target_weight",
            "latent_weight",
            "decoded_weight",
            "physiological_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.target_weight == 0:
            raise ValueError("target_weight must be positive.")
        if not 0 < self.target_probability < 1:
            raise ValueError("target_probability must be strictly between 0 and 1.")

    def target_loss(self, logits, target_class):
        """Return max(0, log(p_min) - log p(target_class | z_prime)).

        logits has shape (1, n_classes), before softmax; target_class is one
        integer class index. log_softmax avoids taking log of rounded or
        underflowed probabilities. The penalty is zero at/above p_min, so
        confidence beyond the requested threshold earns no further reward.
        Gradients flow from logits through the frozen BiGRU/VC head to z'.
        """
        log_probability = tf.nn.log_softmax(logits, axis=-1)[0, target_class]
        return tf.nn.relu(
            tf.math.log(tf.cast(self.target_probability, logits.dtype))
            - log_probability
        )

    @staticmethod
    def latent_distance(z_prime, z):
        """Return elementwise MSE between equally shaped tensors.

        For latent proximity, both tensors are (1, windows, timesteps,
        combined_features). All entries participate; there is no temporal
        pooling of the representation and no padded-window inference.
        The same exact MSE is also used for decoded EEG tensors. The second
        argument is a fixed reference: stop_gradient prevents accidental
        differentiation through its encoder or its original input.
        """
        tf.debugging.assert_equal(
            tf.shape(z_prime), tf.shape(z), message="MSE shapes must match."
        )
        return tf.reduce_mean(
            tf.square(z_prime - tf.stop_gradient(tf.cast(z, z_prime.dtype)))
        )

    def decoded_distance(self, z_prime, x, decoder):
        """Decode z' and return (mean MSE, reconstructions, branch MSEs).

        decoder is a differentiable callable accepting the complete latent
        trial and returning {branch_name: reconstructed_trial}. Each output
        must match x: (1, windows, timesteps, EEG_features). The optimizer
        supplies this small adapter to the saved, independent SIC decoders.

        Each D_b(z'_b) is compared to the ORIGINAL x, not to itself. Branch
        losses are averaged, so enabling a second decoder does not double
        the global decoded-loss scale. Returned reconstructions are reused
        for reporting/physiology without another decoder forward pass.
        No NumPy conversion or gradient stop is applied to decoder outputs.
        """
        reconstructions = decoder(z_prime)
        distances = {
            name: self.latent_distance(value, x)
            for name, value in reconstructions.items()
        }
        return (
            tf.add_n(list(distances.values())) / len(distances),
            reconstructions,
            distances,
        )

    @staticmethod
    def physiological_validity(x_prime):
        """Return scalar zero, with x_prime's dtype; this is a placeholder.

        No physiology is assessed or enforced. In particular, zero must not
        be interpreted as evidence that reconstructed EEG is physiologically
        valid. The argument is one decoded trial, retained so a future
        differentiable physiological penalty can replace this single method.
        """
        return tf.zeros((), dtype=x_prime.dtype)

    def central_loss(self, *, logits, target_class, z_prime, z, x, decoder):
        """Return (loss_terms, reconstructions) for one optimization step.

        loss_terms contains total, the four unweighted terms, their four
        weighted contributions, and decoded_<branch> distances. All are
        scalar tensors, allowing a single GradientTape to differentiate total.
        The caller handles optimization, finite checks, logging, and saving.
        """
        decoded, reconstructions, branch_distances = self.decoded_distance(
            z_prime, x, decoder
        )
        terms = {
            "target": self.target_loss(logits, target_class),
            "latent": self.latent_distance(z_prime, z),
            "decoded": decoded,
            "physiological": tf.add_n(
                [self.physiological_validity(v) for v in reconstructions.values()]
            )
            / len(reconstructions),
        }
        weighted = {
            f"weighted_{name}": getattr(self, f"{name}_weight") * value
            for name, value in terms.items()
        }
        return {
            "total": tf.add_n(list(weighted.values())),
            **terms,
            **weighted,
            **{f"decoded_{name}": value for name, value in branch_distances.items()},
        }, reconstructions
