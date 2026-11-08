"""Independent signal-space evaluation; no counterfactual selection occurs here."""

from dataclasses import dataclass

import numpy as np


def spectral_descriptor(diagnostics):
    power = np.asarray(diagnostics["spectral_power"], dtype=float)
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("Independent typicality needs finite nonnegative spectral power")
    return np.log(np.maximum(power, 1e-12)).ravel()


@dataclass
class IndependentTypicality:
    center: np.ndarray
    scale: np.ndarray
    reference: np.ndarray
    threshold: float
    neighbors: int
    metadata: dict

    @classmethod
    def fit(cls, descriptors, *, subject_ids, trial_ids, labels, held_out_subject,
            target_class, quantile=0.95, neighbors=5, seed=42):
        values = np.asarray(descriptors, dtype=float)
        subjects, trials, labels = map(np.asarray, (subject_ids, trial_ids, labels))
        if (values.ndim != 2 or not np.isfinite(values).all()
                or any(v.shape != (len(values),) for v in (subjects, trials, labels))):
            raise ValueError("Invalid independent calibration descriptors or trial metadata")
        if held_out_subject in subjects:
            raise ValueError("Held-out subject leaked into independent typicality calibration")
        if len(set(zip(subjects.tolist(), trials.tolist()))) != len(values):
            raise ValueError("Duplicate independent calibration trial keys")
        if not 0 < quantile < 1 or neighbors < 1:
            raise ValueError("Invalid independent typicality settings")
        target = labels == target_class
        source_subjects = np.unique(subjects[target])
        if len(source_subjects) < 2:
            raise ValueError("Independent typicality needs target-class trials from at least two source subjects")
        shuffled = np.random.RandomState(seed).permutation(source_subjects)
        calibration_subjects = shuffled[:max(1, len(shuffled) // 3)]
        calibration = target & np.isin(subjects, calibration_subjects)
        fit = target & ~calibration
        center = values[fit].mean(axis=0)
        scale = np.maximum(values[fit].std(axis=0), 1e-6)
        region = cls(center, scale, (values[fit] - center) / scale, 0.,
                     min(neighbors, int(fit.sum())), {})
        scores = region.score(values[calibration])
        region.threshold = float(np.quantile(scores, quantile, method="higher"))
        region.metadata = dict(
            definition="kth_neighbor_standardized_log_band_power_rmse",
            held_out_subject=int(held_out_subject), target_class=int(target_class),
            quantile=quantile, quantile_method="higher", neighbors=region.neighbors,
            fit_trial_keys=np.column_stack((subjects[fit], trials[fit])).tolist(),
            calibration_trial_keys=np.column_stack((subjects[calibration], trials[calibration])).tolist(),
            split="deterministic disjoint source subjects; no held-out tuning",
            seed=int(seed), interpretation="empirical source compatibility; no held-out coverage guarantee",
        )
        return region

    def score(self, descriptors):
        x = np.asarray(descriptors, dtype=float)
        if x.ndim != 2 or x.shape[1:] != self.center.shape or not np.isfinite(x).all():
            raise ValueError("Invalid independent typicality descriptor")
        distances = np.sqrt(np.mean((((x - self.center) / self.scale)[:, None] - self.reference[None]) ** 2, axis=-1))
        return np.partition(distances, self.neighbors - 1, axis=1)[:, self.neighbors - 1]

    def payload(self):
        return {**self.metadata, "center": self.center, "scale": self.scale,
                "standardized_reference_descriptors": self.reference, "threshold": self.threshold}


def conjunction(*values):
    if any(value is False for value in values):
        return False
    return None if any(value is None for value in values) else True


def signal_displacement(original, counterfactual, *, group_size=3, atol=1e-6, rtol=1e-3):
    x, cf = (np.asarray(v, dtype=np.float64) for v in (original, counterfactual))
    if (x.shape != cf.shape or x.ndim != 3 or x.shape[-1] % group_size
            or not np.isfinite(x).all() or not np.isfinite(cf).all()):
        raise ValueError("Displacement needs matching finite full trials")
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("Channel tolerances must be finite and nonnegative")
    delta = cf - x
    channel_delta = np.sqrt(np.mean(delta.reshape(*x.shape[:2], -1, group_size) ** 2, axis=(0, 1, 3)))
    channel_rms = np.sqrt(np.mean(x.reshape(*x.shape[:2], -1, group_size) ** 2, axis=(0, 1, 3)))
    changed = channel_delta > atol + rtol * channel_rms
    norm = float(np.linalg.norm(x))
    return dict(signal_rmse=float(np.sqrt(np.mean(delta ** 2))),
                signal_l2=float(np.linalg.norm(delta)),
                relative_signal_l2=float(np.linalg.norm(delta) / norm) if norm > 0 else None,
                changed_channels=int(changed.sum()), channel_sparsity=float(1 - changed.mean()),
                channel_rmse=channel_delta.tolist(), channel_atol=atol, channel_rtol=rtol)


def evaluate_signal(model, signal, original, *, target_class, target_probability,
                    vc_region, independent_region, diagnostics, physiology,
                    vcsc_penalty, vcsc_tolerance, atol=1e-6, rtol=1e-3):
    """Always call the frozen encoder on the delivered waveform, for BOTH methods."""
    x, cf = (np.asarray(v, dtype=np.float32) for v in (original, signal))
    if cf.shape != x.shape or cf.ndim != 3 or not np.isfinite(cf).all():
        raise ValueError("Counterfactual output is not a finite full trial")
    if not np.isfinite([vcsc_penalty, vcsc_tolerance]).all() or min(vcsc_penalty, vcsc_tolerance) < 0:
        raise ValueError("Physiological penalty and tolerance must be finite and nonnegative")
    features = model.get_encoder_features(cf[None])
    probabilities = np.asarray(features["probabilities"], dtype=float)
    embedding = np.asarray(features["classification_embedding"], dtype=float)
    if (probabilities.shape != (1, 2) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid full-pipeline output probabilities")
    predicted = int(probabilities[0].argmax())
    valid = bool(predicted == target_class and probabilities[0, target_class] >= target_probability)
    discrepancy = float(vc_region.score(embedding)[0])
    independent = float(independent_region.score(spectral_descriptor(diagnostics)[None])[0])
    typical = bool(independent <= independent_region.threshold)
    assessed = physiology.assess(diagnostics)
    vcsc_passed = bool(vcsc_penalty <= vcsc_tolerance)
    strict = conjunction(assessed["all_required_passed"], vcsc_passed)
    available = conjunction(assessed["available_checks_passed"], vcsc_passed)
    return dict(
        prediction_space="output_signal_reencoded_by_frozen_encoder_classifier",
        probabilities=probabilities[0].tolist(), predicted_class=predicted,
        target_probability=float(probabilities[0, target_class]), required_target_probability=target_probability,
        prediction_validity=valid, vc_discrepancy=discrepancy, vc_threshold=float(vc_region.tau),
        vc_typical=bool(discrepancy <= vc_region.tau), independent_score=independent,
        independent_threshold=independent_region.threshold, independent_typicality=typical,
        physiological_acceptance=strict, available_physiological_acceptance=available,
        physiology=assessed, vcsc_penalty=float(vcsc_penalty), vcsc_passed=vcsc_passed,
        joint_success=conjunction(valid, typical, strict),
        available_checks_joint_success=conjunction(valid, typical, available),
        **signal_displacement(x, cf, atol=atol, rtol=rtol),
    )
