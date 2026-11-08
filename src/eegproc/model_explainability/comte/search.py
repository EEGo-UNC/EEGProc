"""Array adapter of peaclab/CoMTE OptimizedSearch and its greedy fallback.

Derived from explainers.py at 6042b267928415bbae84448faf2c8b670434cc15.
Copyright (c) 2021, peaclab. BSD-3-Clause; see upstream/LICENSE.
The original source is retained in upstream/explainers.py.txt for auditing.

The binary loss, random hill climbing with five restarts, greedy pruning,
true-positive nearest distractors, and greedy fallback follow upstream.
Array indexing replaces the legacy pandas/pipeline interface. The confidence
criterion is configurable, the chosen mask stays paired with its donor, and
fallback checks the shared confidence criterion. Every finite endpoint,
including failures, is returned.
"""

from dataclasses import asdict, dataclass

import numpy as np

UPSTREAM_COMMIT = "6042b267928415bbae84448faf2c8b670434cc15"


@dataclass(frozen=True)
class SearchConfig:
    distractors: int = 2
    max_features: int = 3
    max_attempts: int = 1000
    max_iterations: int = 1000
    restarts: int = 5
    regularization: float = 0.8
    max_queries: int = 10000
    group_size: int = 3

    def __post_init__(self):
        for name in ("distractors", "max_features", "max_attempts", "max_iterations", "max_queries", "group_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.restarts, int) or self.restarts < 0:
            raise ValueError("restarts must be nonnegative")
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")


class QueryBudgetExceeded(RuntimeError):
    pass


def confidence_loss(probability, count, *, threshold, config):
    return (max(0.0, threshold - probability) ** 2
            + config.regularization * max(0, count - config.max_features))


def optimized_search(predict_proba, original, references, *, reference_labels,
                     reference_probabilities, reference_keys, held_out_subject,
                     target_class, target_probability, seed=42, config=None, progress=None):
    """Substitute complete electrode groups from one source donor at a time.

    Inputs use (windows, samples, features). No re-normalization, time
    truncation, or new classifier is introduced. Queries count complete
    one-trial pipeline predictions, including repeats; there is no cache.
    """
    config = config or SearchConfig()
    x = np.asarray(original, dtype=np.float32)
    refs = np.asarray(references, dtype=np.float32)
    labels = np.asarray(reference_labels)
    probabilities = np.asarray(reference_probabilities, dtype=float)
    keys = np.asarray(reference_keys)
    if (x.ndim != 3 or not x.size or refs.shape[1:] != x.shape or refs.ndim != 4
            or x.shape[-1] % config.group_size or not np.isfinite(x).all() or not np.isfinite(refs).all()):
        raise ValueError("Need finite complete trials with a divisible feature axis")
    if (labels.shape != (len(refs),) or keys.shape != (len(refs), 2)
            or probabilities.ndim != 2 or probabilities.shape[0] != len(refs)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Reference labels, keys and probabilities disagree")
    if held_out_subject in keys[:, 0]:
        raise ValueError("Held-out subject leaked into CoMTE references")
    if len(set(map(tuple, keys.tolist()))) != len(keys):
        raise ValueError("Duplicate CoMTE reference trial keys")
    if not 0 <= target_class < probabilities.shape[1] or not 0 < target_probability < 1:
        raise ValueError("Invalid target class or probability")
    eligible = np.flatnonzero((labels == target_class) & (probabilities.argmax(axis=1) == target_class))
    # Avoid duplicating the entire full-trial source bank in float64.
    distances = np.asarray([np.sum((refs[i].astype(np.float64) - x) ** 2) for i in eligible])
    donors = eligible[np.argsort(distances, kind="stable")[:config.distractors]]
    rng = np.random.RandomState(seed)
    n_groups = x.shape[-1] // config.group_size
    queries = 0
    best = None

    def evaluate(mask, donor, phase):
        nonlocal queries, best
        if queries >= config.max_queries:
            raise QueryBudgetExceeded()
        signal = x.copy()
        if donor is not None:
            columns = np.repeat(mask, config.group_size)
            signal[..., columns] = refs[donor][..., columns]
        p = np.asarray(predict_proba(signal[None]), dtype=float)
        if (p.shape != (1, probabilities.shape[1]) or not np.isfinite(p).all()
                or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Invalid frozen pipeline probabilities")
        queries += 1
        score = float(p[0, target_class])
        success = bool(p[0].argmax() == target_class and score >= target_probability)
        count = int(mask.sum())
        # Upstream selects the highest target probability among successful
        # pruned endpoints, then uses greedy search if no endpoint succeeds.
        # Do not promote unselected hill-climb neighbors into final outputs.
        rank = (not success, -score, count)
        row = dict(step=queries, phase=phase, target_probability=score,
                   success=success, changed_groups=count,
                   donor_key=keys[donor].tolist() if donor is not None else None)
        if phase in ("original", "prune_endpoint", "greedy_endpoint") and (best is None or rank < best[0]):
            best = (rank, signal, p[0], mask.copy(), donor)
        if progress:
            progress(row)
        return confidence_loss(score, count, threshold=target_probability, config=config), score, success

    stopped = "search_exhausted"
    try:
        evaluate(np.zeros(n_groups, dtype=bool), None, "original")
        for donor in donors:
            different = np.any((refs[donor] != x).reshape(*x.shape[:2], n_groups, config.group_size), axis=(0, 1, 3))
            active = np.flatnonzero(different)
            if not len(active):
                continue
            selected_mask, selected_loss = None, float("inf")
            for _ in range(config.restarts + 1):
                mask = np.zeros(n_groups, dtype=bool)
                loss, _, _ = evaluate(mask, donor, "hill_climb")
                attempts = 0
                for _ in range(config.max_iterations):
                    neighbor = mask.copy()
                    group = active[rng.randint(len(active))]
                    neighbor[group] = ~neighbor[group]
                    neighbor_loss, _, _ = evaluate(neighbor, donor, "hill_climb")
                    if neighbor_loss < loss:
                        mask, loss, attempts = neighbor, neighbor_loss, 0
                    else:
                        attempts += 1
                    if attempts >= config.max_attempts:
                        break
                if loss < selected_loss:
                    selected_mask, selected_loss = mask.copy(), loss
            # Upstream greedily orders the selected variables up to max_features.
            pruned = np.zeros(n_groups, dtype=bool)
            for _ in range(min(config.max_features, int(selected_mask.sum()))):
                _, previous, _ = evaluate(pruned, donor, "prune")
                chosen, score = None, previous
                for group in np.flatnonzero(selected_mask & ~pruned):
                    candidate = pruned.copy()
                    candidate[group] = True
                    _, probability, _ = evaluate(candidate, donor, "prune")
                    if probability > score:
                        chosen, score = group, probability
                if chosen is None:
                    break
                pruned[chosen] = True
            evaluate(pruned, donor, "prune_endpoint")
        if best is not None and best[0][0]:
            # Greedy search is not constrained by the soft max_features penalty,
            # matching upstream's BruteForceSearch fallback.
            for donor in donors:
                mask = np.zeros(n_groups, dtype=bool)
                _, previous, success = evaluate(mask, donor, "greedy")
                while not success:
                    chosen, score = None, previous
                    for group in np.flatnonzero(~mask):
                        candidate = mask.copy()
                        candidate[group] = True
                        _, probability, _ = evaluate(candidate, donor, "greedy")
                        if probability > score:
                            chosen, score = group, probability
                    if chosen is None:
                        break
                    mask[chosen] = True
                    _, previous, success = evaluate(mask, donor, "greedy_endpoint")
        if not len(donors):
            stopped = "no_source_target_true_positive"
    except QueryBudgetExceeded:
        stopped = "query_budget"
    _, signal, p, mask, donor = best
    return {"signal": signal, "probabilities": p, "changed_groups": np.flatnonzero(mask).tolist(),
            "donor_key": keys[donor].tolist() if donor is not None else None,
            "search_queries": queries, "stop_reason": stopped,
            "upstream_commit": UPSTREAM_COMMIT, "config": asdict(config)}
