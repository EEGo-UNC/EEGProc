"""Evaluation-only k-nearest-neighbour support in frozen trial embeddings.

The score is the Euclidean distance to the kth neighbour, not the mean of
the k distances. Reference and calibration participants are disjoint and
both belong to the outer fold's source set. No Gaussian or predicted-label
filter is used. This evaluator must not select or optimize counterfactuals.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


KNN_SCHEMA_VERSION = 1
KNN_DEFINITION = "kth_nearest_real_target_trial_embedding_euclidean_distance"
KNN_REPRESENTATION = "vc_trial_embedding"


class InsufficientSourceDataError(ValueError):
    """An evaluation cannot be calibrated without inventing source support."""


def _matrix(values, *, dimension=None, name="embeddings", allow_empty=False):
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1] == 0
            or (not allow_empty and result.shape[0] == 0)
            or (dimension is not None and result.shape[1] != dimension)
            or not np.isfinite(result).all()):
        raise ValueError(f"{name} must contain finite full-trial embedding vectors of the expected dimension")
    return result


def kth_neighbor_distance(queries, reference, *, k=5, batch_size=32):
    """Return one kth-nearest Euclidean distance per query; no self removal.

    Calibration/query subjects must be absent from the reference bank.
    Reference points are never scored against a bank containing themselves.
    """
    bank = _matrix(reference, name="reference bank")
    queries = _matrix(queries, dimension=bank.shape[1], name="queries", allow_empty=True)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(bank):
        raise ValueError("k must be a positive integer no larger than the reference bank")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    scores = np.empty(len(queries), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, len(queries), batch_size):
            delta = queries[start:start + batch_size, None, :] - bank[None, :, :]
            squared = np.einsum("qrd,qrd->qr", delta, delta)
            scores[start:start + batch_size] = np.sqrt(np.partition(squared, k - 1, axis=1)[:, k - 1])
    if not np.isfinite(scores).all():
        raise ValueError("Neighbour distances overflowed; check the embedding scale")
    return scores


@dataclass
class EncodedKNNReference:
    bank: np.ndarray
    calibration_embeddings: np.ndarray
    calibration_scores: np.ndarray
    threshold: float
    metadata: dict

    def __post_init__(self):
        self.bank = _matrix(self.bank, name="reference bank").copy()
        self.calibration_embeddings = _matrix(
            self.calibration_embeddings, dimension=self.bank.shape[1], name="calibration embeddings").copy()
        self.calibration_scores = np.asarray(self.calibration_scores, dtype=np.float64).copy()
        self.metadata = dict(self.metadata)
        if self.calibration_scores.shape != (len(self.calibration_embeddings),):
            raise ValueError("Each calibration embedding requires one neighbour distance")
        if not np.isfinite(self.calibration_scores).all() or np.any(self.calibration_scores < 0):
            raise ValueError("Calibration distances must be finite and nonnegative")
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Neighbour cutoff must be finite and nonnegative")
        for key, expected in (("schema_version", KNN_SCHEMA_VERSION), ("definition", KNN_DEFINITION),
                              ("representation", KNN_REPRESENTATION)):
            if self.metadata.get(key) != expected:
                raise ValueError("Incompatible encoded-neighbour reference")
        k = self.metadata.get("k")
        # Validates k as well as the archived distances; no mutable/refitted bank.
        actual = kth_neighbor_distance(self.calibration_embeddings, self.bank, k=k)
        if not np.allclose(actual, self.calibration_scores, rtol=1e-12, atol=1e-12):
            raise ValueError("Archived calibration distances do not match the reference bank")
        quantile = self.metadata.get("quantile", math.nan)
        if not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("Neighbour quantile must be in (0, 1)")
        expected = float(np.quantile(actual, quantile, method="higher"))
        if not math.isclose(self.threshold, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Archived cutoff does not match source calibration")
        bank_subjects = set(self.metadata.get("bank_subject_ids", []))
        calibration_subjects = set(self.metadata.get("calibration_subject_ids", []))
        held_out = self.metadata.get("held_out_subject")
        if (not bank_subjects or not calibration_subjects or bank_subjects & calibration_subjects
                or held_out in bank_subjects | calibration_subjects):
            raise ValueError("Reference, calibration and held-out subjects must be disjoint")
        for values in (self.bank, self.calibration_embeddings, self.calibration_scores):
            values.setflags(write=False)

    @classmethod
    def fit_source(cls, embeddings, *, labels, subject_ids, trial_ids, held_out_subject,
                   target_class=1, k=5, quantile=0.10, calibration_fraction=0.25,
                   seed=42, calibration_subject_ids=None):
        """Split real true-target source trials by participant before fitting.

        The encoder is already frozen. The split separates density-reference
        and cutoff-calibration subjects; it is not a new classifier-training
        split and does not claim encoder-independent validation.
        """
        values = _matrix(embeddings)
        subjects, trials, labels = (np.asarray(v) for v in (subject_ids, trial_ids, labels))
        if any(v.shape != (len(values),) for v in (subjects, trials, labels)):
            raise ValueError("Every embedding needs a subject, trial and true label")
        if any(not np.issubdtype(v.dtype, np.integer) for v in (subjects, trials, labels)):
            raise ValueError("Subject IDs, trial IDs and true labels must be integers")
        if held_out_subject in subjects:
            raise ValueError("Held-out subject leaked into neighbour calibration")
        if len(set(zip(subjects.tolist(), trials.tolist()))) != len(values):
            raise ValueError("Duplicate source subject/trial identifiers")
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("Neighbour quantile must be in (0, 1)")
        if not math.isfinite(calibration_fraction) or not 0 < calibration_fraction < 1:
            raise ValueError("Calibration subject fraction must be in (0, 1)")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        target = labels == target_class
        eligible_subjects = np.unique(subjects[target])
        if len(eligible_subjects) < 2:
            raise InsufficientSourceDataError("Need target-class trials from at least two source participants for disjoint neighbour calibration")
        if calibration_subject_ids is None:
            n_calibration = min(len(eligible_subjects) - 1,
                                max(1, math.ceil(len(eligible_subjects) * calibration_fraction)))
            selected = np.random.default_rng(seed).permutation(eligible_subjects)[:n_calibration]
        else:
            selected = np.asarray(calibration_subject_ids)
            if (selected.ndim != 1 or not len(selected) or len(np.unique(selected)) != len(selected)
                    or not np.isin(selected, eligible_subjects).all()):
                raise ValueError("Calibration subjects must be unique source participants with true target-class trials")
        calibration_mask = target & np.isin(subjects, selected)
        bank_mask = target & ~np.isin(subjects, selected)
        bank, calibration = values[bank_mask], values[calibration_mask]
        if len(bank) < k:
            raise InsufficientSourceDataError(f"Reference bank has {len(bank)} target-class trials, fewer than k={k}; choose a source-only split with more reference trials")
        calibration_scores = kth_neighbor_distance(calibration, bank, k=k)
        threshold = float(np.quantile(calibration_scores, quantile, method="higher"))
        metadata = {
            "schema_version": KNN_SCHEMA_VERSION, "definition": KNN_DEFINITION,
            "representation": KNN_REPRESENTATION, "dimension": int(values.shape[1]),
            "k": int(k), "metric": "euclidean", "scaling": "unchanged frozen terminal embedding coordinates",
            "quantile": float(quantile), "quantile_method": "higher", "seed": int(seed),
            "calibration_fraction": float(calibration_fraction),
            "target_class": int(target_class), "held_out_subject": int(held_out_subject),
            "bank_subject_ids": np.unique(subjects[bank_mask]).tolist(),
            "calibration_subject_ids": np.unique(subjects[calibration_mask]).tolist(),
            "bank_trial_keys": np.column_stack((subjects[bank_mask], trials[bank_mask])).tolist(),
            "calibration_trial_keys": np.column_stack((subjects[calibration_mask], trials[calibration_mask])).tolist(),
            "n_bank_trials": int(bank_mask.sum()), "n_calibration_trials": int(calibration_mask.sum()),
            "selection": "real true-target source trials; no predicted-label or Gaussian-typicality filter",
            "role": "evaluation only; never used by optimization, early stopping or iterate selection",
            "acceptance": "kth neighbour distance <= source-calibration quantile",
            "coverage_claim": "empirical source-subject calibration; no guaranteed held-out coverage",
        }
        return cls(bank, calibration, calibration_scores, threshold, metadata)

    def score(self, embeddings):
        return kth_neighbor_distance(embeddings, self.bank, k=self.metadata["k"])

    def assess(self, embedding):
        values = np.asarray(embedding, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        scores = self.score(values)
        if len(scores) != 1:
            raise ValueError("assess requires exactly one full-trial embedding")
        score = float(scores[0])
        lower = np.count_nonzero(self.calibration_scores < score)
        ties = np.count_nonzero(self.calibration_scores == score)
        return {
            "distance": score, "threshold": float(self.threshold),
            "inside": bool(score <= self.threshold),
            "distance_over_threshold": score / self.threshold if self.threshold > 0 else None,
            "calibration_percentile": 100 * (lower + 0.5 * ties) / len(self.calibration_scores),
            "definition": KNN_DEFINITION, "representation": KNN_REPRESENTATION,
            "k": self.metadata["k"], "quantile": self.metadata["quantile"],
        }

    def to_dict(self):
        return {**self.metadata, "threshold": float(self.threshold),
                "bank_embeddings": self.bank.tolist(),
                "calibration_embeddings": self.calibration_embeddings.tolist(),
                "calibration_scores": self.calibration_scores.tolist()}

    def save(self, path):
        # Compact terminal vectors are intentionally persisted for independent
        # evaluators; window latents, EEG and gradients remain separate.
        from .artifacts import write_json
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        bank = payload.pop("bank_embeddings")
        calibration = payload.pop("calibration_embeddings")
        scores = payload.pop("calibration_scores")
        threshold = payload.pop("threshold")
        return cls(bank, calibration, scores, threshold, payload)
