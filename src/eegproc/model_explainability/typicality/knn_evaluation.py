"""Archive terminal embeddings and evaluate endpoints after optimization."""

import numpy as np


def _one_embedding(value, dimension=None):
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 1 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Endpoint must be one finite full-trial classification embedding")
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError("All endpoint embeddings must use the same frozen classifier coordinates")
    return values[0]


def evaluate_trial_embeddings(model, arrays, *, report_output, reference=None):
    """Called only after optimize() returns; cannot influence its iterates.

    Re-encoded decoded signals and optimized embeddings remain separately
    named. Terminal vectors are small and enable future independent scoring.
    """
    original = _one_embedding(arrays['classification_embedding'])
    dimension = len(original)
    endpoints = {
        'original': original,
        'optimized': _one_embedding(arrays['classification_embedding_prime'], dimension),
    }
    for name, array_name in (
        ('decoded_reconstruction', f'x_reconstructed_{report_output}'),
        ('decoded_counterfactual', f'x_prime_{report_output}'),
    ):
        features = model.get_encoder_features(arrays[array_name])
        endpoints[name] = _one_embedding(features['classification_embedding'], dimension)
    summary = {'classification_embeddings': {name: value.tolist() for name, value in endpoints.items()}}
    if reference is not None:
        scored = {name: reference.assess(value) for name, value in endpoints.items()}
        before, after = scored['original'], scored['optimized']
        scored['distance_change'] = after['distance'] - before['distance']
        scored['distance_over_threshold_change'] = (
            after['distance_over_threshold'] - before['distance_over_threshold']
            if reference.threshold > 0 else None)
        summary['encoded_neighbors'] = scored
    return summary
