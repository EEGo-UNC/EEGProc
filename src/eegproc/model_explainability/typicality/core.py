"""Squared diagonal Mahalanobis distance per full-trial embedding coordinate.

The learned class Gaussian is copied from the frozen checkpoint. Source
class-1 trial embeddings calibrate only the acceptance threshold. Both the
Gaussian and the scored vector live in the classifier's terminal trial space.
"""

from dataclasses import dataclass
import math

import numpy as np


SCHEMA_VERSION = 2
SCORE_DEFINITION = "full_trial_diagonal_squared_mahalanobis_per_dimension"
REPRESENTATION = "vc_trial_embedding"


def diagonal_mahalanobis_squared(embedding, mean, variance, *, variance_floor=1e-6):
    """Return mean_j((embedding_j - mean_j)^2 / max(variance_j, epsilon)).

    This is squared Mahalanobis distance divided by dimension, not its square
    root. Leading embedding axes are retained. No within-trial moments, logit
    temperature, Gaussian refitting, or covariance estimation are involved.
    """
    e, mu, var = (np.asarray(value, dtype=np.float64) for value in (embedding, mean, variance))
    if mu.ndim != 1 or not mu.size or var.shape != mu.shape or e.ndim < 1 or e.shape[-1] != mu.size:
        raise ValueError("Embedding and Gaussian dimensions must agree with nonempty coordinate vectors")
    if any(not np.isfinite(value).all() for value in (e, mu, var)) or np.any(var < 0):
        raise ValueError("Embeddings and Gaussian parameters must be finite; variances must be nonnegative")
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    return np.mean(np.square(e - mu) / np.maximum(var, variance_floor), axis=-1)


@dataclass
class TypicalityRegion:
    prior_mean: np.ndarray
    prior_variance: np.ndarray
    tau: float
    metadata: dict
    variance_floor: float = 1e-6

    def __post_init__(self):
        self.prior_mean = np.asarray(self.prior_mean, dtype=np.float64).copy()
        self.prior_variance = np.asarray(self.prior_variance, dtype=np.float64).copy()
        diagonal_mahalanobis_squared(self.prior_mean, self.prior_mean, self.prior_variance,
                                     variance_floor=self.variance_floor)
        if not math.isfinite(self.tau) or self.tau < 0:
            raise ValueError("tau must be finite and nonnegative")
        self.metadata = dict(self.metadata)
        for key, value in (("schema_version", SCHEMA_VERSION), ("definition", SCORE_DEFINITION),
                           ("representation", REPRESENTATION)):
            if key in self.metadata and self.metadata[key] != value:
                raise ValueError("Incompatible typicality definition; recalibrate full-trial Mahalanobis scores")
            self.metadata[key] = value

    @classmethod
    def calibrate(cls, embeddings, *, prior_mean, prior_variance,
                  subject_ids, trial_ids, labels, held_out_subject,
                  target_class=1, quantile=0.95, variance_floor=1e-6):
        values = np.asarray(embeddings, dtype=np.float64)
        subjects, trials, labels = (np.asarray(v) for v in (subject_ids, trial_ids, labels))
        region = cls(prior_mean, prior_variance, 1.0, {}, variance_floor)
        scores = region.score(values)
        if any(v.shape != (len(values),) for v in (subjects, trials, labels)):
            raise ValueError("Each embedding needs a subject, trial, and label")
        if held_out_subject in subjects:
            raise ValueError("Held-out subject leaked into source threshold calibration")
        if len(set(zip(subjects.tolist(), trials.tolist()))) != len(values):
            raise ValueError("Duplicate source trial identifiers")
        if not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        target = labels == target_class
        if target.sum() < 2:
            raise ValueError("Need at least two source target-class calibration trials")
        # The cutoff is calibrated anew for this score. It also normalizes the
        # optimization penalty; normalization never changes acceptance.
        region.tau = float(np.quantile(scores[target], quantile, method="higher"))
        region.metadata.update({
            "prior": "frozen learned VC Gaussian; variance=exp(2*prior_log_sigma)",
            "target_class": int(target_class), "held_out_subject": int(held_out_subject),
            "quantile": quantile, "quantile_method": "higher",
            "source_subject_ids": np.unique(subjects).tolist(),
            "calibration_trial_keys": np.column_stack((subjects[target], trials[target])).tolist(),
            "variance_floor": variance_floor, "tau": region.tau,
            "n_calibration_trials": int(target.sum()), "dimension": int(len(region.prior_mean)),
            "coverage_claim": "empirical source quantile; no held-out coverage guarantee",
            "evaluation": "typical iff D <= tau; class-conditional embedding compatibility",
            "optimization_penalty": "lambda * D / max(tau, variance_floor), activated after target and physiology feasibility",
        })
        return region

    def score(self, embeddings):
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.prior_mean):
            raise ValueError("Expected full-trial embeddings shaped (trials, VC dimension), not window moments")
        return diagonal_mahalanobis_squared(values, self.prior_mean, self.prior_variance,
                                             variance_floor=self.variance_floor)

    def discrepancy(self, embedding):
        """Differentiable score of the SAME embedding used by the VC head."""
        import tensorflow as tf
        e = tf.cast(embedding, tf.float32)
        tf.debugging.assert_rank(e, 2, message="Supply one full-trial classification embedding")
        tf.debugging.assert_equal(tf.shape(e)[0], 1)
        tf.debugging.assert_equal(tf.shape(e)[1], len(self.prior_mean))
        tf.debugging.assert_all_finite(e, "Full-trial embedding must be finite")
        mean = tf.constant(self.prior_mean, tf.float32)
        variance = tf.maximum(tf.constant(self.prior_variance, tf.float32), self.variance_floor)
        return tf.reduce_mean(tf.square(e - mean) / variance)

    def save(self, directory):
        from pathlib import Path
        from .artifacts import write_json, write_npz
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_npz(directory / "region.npz", prior_mean=self.prior_mean,
                  prior_variance=self.prior_variance, tau=self.tau, variance_floor=self.variance_floor)
        write_json(directory / "region.json", self.metadata)

    @classmethod
    def load(cls, directory):
        import json
        from pathlib import Path
        directory = Path(directory)
        metadata = json.loads((directory / "region.json").read_text())
        if (metadata.get("schema_version") != SCHEMA_VERSION
                or metadata.get("definition") != SCORE_DEFINITION
                or metadata.get("representation") != REPRESENTATION):
            raise ValueError("Legacy or incompatible typicality region; recalibrate full-trial Mahalanobis scores in a new study")
        with np.load(directory / "region.npz", allow_pickle=False) as data:
            return cls(data["prior_mean"], data["prior_variance"], float(data["tau"]),
                       metadata, float(data["variance_floor"]))
