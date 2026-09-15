"""Eq. (7): dimension-normalized diagonal KL(q_Z || learned VC class prior).

The learned Gaussian is copied from the frozen checkpoint, never estimated
again from source trials. Source class-1 trials calibrate only the threshold.
A sequence mapping must explicitly put Z into the VC prior's coordinates.
"""

from dataclasses import dataclass, field
import math

import numpy as np


def trial_representation(latent):
    """Concatenated [mean, variance], population moments (ddof=0) per trial."""
    z = np.asarray(latent, dtype=np.float64)
    if z.ndim < 3 or any(d == 0 for d in z.shape) or not np.isfinite(z).all():
        raise ValueError("latent must be finite with batch, sequence, and coordinate axes")
    axes = tuple(range(1, z.ndim - 1))
    return np.concatenate((z.mean(axis=axes), z.var(axis=axes)), axis=-1)


def diagonal_gaussian_kl(mean_q, variance_q, mean_p, variance_p, *, variance_floor=1e-6):
    """NumPy Eq. (7), averaged over dimensions; keep leading batch axes."""
    mq, vq, mp, vp = (np.asarray(v, dtype=np.float64) for v in (mean_q, variance_q, mean_p, variance_p))
    if mq.shape != vq.shape or mp.shape != vp.shape or mq.shape[-1] != mp.shape[-1]:
        raise ValueError("Posterior/prior mean and variance dimensions must agree")
    if any(not np.isfinite(v).all() for v in (mq, vq, mp, vp)) or np.any(vq < 0) or np.any(vp < 0):
        raise ValueError("Gaussian moments must be finite with nonnegative variances")
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    vq, vp = np.maximum(vq, variance_floor), np.maximum(vp, variance_floor)
    return np.maximum(0.5 * np.mean((vq + (mq - mp) ** 2) / vp - 1 + np.log(vp) - np.log(vq), axis=-1), 0.0)


@dataclass
class TypicalityRegion:
    prior_mean: np.ndarray
    prior_variance: np.ndarray
    tau: float
    metadata: dict
    variance_floor: float = 1e-6
    sequence_transform: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.prior_mean = np.asarray(self.prior_mean, dtype=np.float64).copy()
        self.prior_variance = np.asarray(self.prior_variance, dtype=np.float64).copy()
        if self.prior_mean.ndim != 1 or not len(self.prior_mean) or self.prior_variance.shape != self.prior_mean.shape:
            raise ValueError("Learned VC prior must contain matching coordinate vectors")
        if not math.isfinite(self.tau) or self.tau < 0:
            raise ValueError("tau must be finite and nonnegative")
        diagonal_gaussian_kl(self.prior_mean, self.prior_variance,
                             self.prior_mean, self.prior_variance,
                             variance_floor=self.variance_floor)

    @classmethod
    def calibrate(cls, representations, *, prior_mean, prior_variance,
                  subject_ids, trial_ids, labels, held_out_subject,
                  target_class=1, quantile=0.95, variance_floor=1e-6,
                  sequence_transform=None, sequence_definition="provided_VC_coordinate_sequence"):
        values = np.asarray(representations, dtype=np.float64)
        subjects, trials, labels = (np.asarray(v) for v in (subject_ids, trial_ids, labels))
        prior_mean, prior_variance = (np.asarray(v, dtype=np.float64).copy() for v in (prior_mean, prior_variance))
        if prior_mean.ndim != 1 or prior_variance.shape != prior_mean.shape or not len(prior_mean):
            raise ValueError("Learned VC prior must contain matching coordinate vectors")
        if values.ndim != 2 or values.shape[1] != 2 * len(prior_mean) or not np.isfinite(values).all():
            raise ValueError("Source moments must match the learned VC prior coordinates")
        if any(v.shape != (len(values),) for v in (subjects, trials, labels)):
            raise ValueError("Each representation needs a subject, trial, and label")
        if held_out_subject in subjects:
            raise ValueError("Held-out subject leaked into source threshold calibration")
        if len(set(zip(subjects.tolist(), trials.tolist()))) != len(values):
            raise ValueError("Duplicate source trial identifiers")
        if not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        target = labels == target_class
        if target.sum() < 2:
            raise ValueError("Need at least two source target-class calibration trials")
        region = cls(prior_mean, prior_variance, 1.0, {}, variance_floor, sequence_transform)
        scores = region.score(values)
        # No extra tau floor: the paper uses D - tau, not division by tau.
        region.tau = float(np.quantile(scores[target], quantile, method="higher"))
        region.metadata = {
            "schema_version": 1, "definition": "Eq7_diagonal_gaussian_KL_per_dimension",
            "sequence_definition": sequence_definition, "moments_ddof": 0,
            "prior": "frozen learned VC Gaussian; variance=exp(2*prior_log_sigma)",
            "target_class": int(target_class), "held_out_subject": int(held_out_subject),
            "quantile": quantile, "quantile_method": "higher",
            "source_subject_ids": np.unique(subjects).tolist(),
            "calibration_trial_keys": np.column_stack((subjects[target], trials[target])).tolist(),
            "variance_floor": variance_floor, "tau": region.tau,
            "n_calibration_trials": int(target.sum()), "dimension": int(len(prior_mean)),
            "coverage_claim": "empirical source quantile; no held-out coverage guarantee",
            "penalty": "lambda * max(0, D - tau)^2",
        }
        return region

    def score(self, representations):
        values = np.asarray(representations, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 * len(self.prior_mean):
            raise ValueError("Typicality representation dimensions do not match the VC prior")
        mean, variance = np.split(values, 2, axis=-1)
        return diagonal_gaussian_kl(mean, variance, self.prior_mean, self.prior_variance,
                                    variance_floor=self.variance_floor)

    def sequence(self, latent):
        import tensorflow as tf
        z = tf.cast(self.sequence_transform(latent) if self.sequence_transform else latent, tf.float32)
        if z.shape.rank is None or z.shape.rank < 3:
            raise ValueError("Typicality requires an ordered sequence, not one pooled vector")
        tf.debugging.assert_equal(tf.shape(z)[0], 1)
        tf.debugging.assert_equal(tf.shape(z)[-1], len(self.prior_mean),
                                  message="Z and the learned VC Gaussian must share coordinates")
        return z

    def discrepancy(self, latent, *, sequence=None):
        """Differentiable Eq. (7), with the checkpoint's prior held fixed."""
        import tensorflow as tf
        z = self.sequence(latent) if sequence is None else sequence
        mean, variance = tf.nn.moments(z, axes=tuple(range(1, z.shape.rank - 1)))
        variance = tf.maximum(variance, self.variance_floor)
        prior_mean = tf.constant(self.prior_mean, tf.float32)
        prior_variance = tf.maximum(tf.constant(self.prior_variance, tf.float32), self.variance_floor)
        terms = (variance + tf.square(mean - prior_mean)) / prior_variance - 1.0
        terms += tf.math.log(prior_variance) - tf.math.log(variance)
        return tf.maximum(0.5 * tf.reduce_mean(terms), 0.0)

    def save(self, directory):
        from pathlib import Path
        from .typicality_artifacts import write_json, write_npz
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_npz(directory / "region.npz", prior_mean=self.prior_mean,
                  prior_variance=self.prior_variance, tau=self.tau, variance_floor=self.variance_floor)
        write_json(directory / "region.json", self.metadata)

    @classmethod
    def load(cls, directory, *, sequence_transform=None):
        import json
        from pathlib import Path
        directory = Path(directory)
        with np.load(directory / "region.npz", allow_pickle=False) as data:
            region = cls(data["prior_mean"], data["prior_variance"], float(data["tau"]),
                         json.loads((directory / "region.json").read_text()), float(data["variance_floor"]), sequence_transform)
        if not np.isfinite(region.tau) or region.tau < 0:
            raise ValueError("Invalid saved typicality threshold")
        return region
