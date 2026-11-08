"""Direction-specific eligibility decisions, including every held-out trial."""

import numpy as np

DIRECTIONS = {"0_to_1": (0, 1), "1_to_0": (1, 0)}


def eligibility_records(*, task, direction, subject, trial_ids, labels,
                        probabilities, trial_filter=None):
    source, target = DIRECTIONS[direction]
    trials, labels, probabilities = map(np.asarray, (trial_ids, labels, probabilities))
    if (trials.ndim != 1 or labels.shape != trials.shape
            or probabilities.shape != (len(trials), 2)
            or len(set(trials.tolist())) != len(trials)
            or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Eligibility needs unique trials, binary labels, and finite binary probabilities")
    selected = None if trial_filter is None else set(trial_filter)
    if selected is not None and not selected.issubset(set(trials.tolist())):
        raise ValueError("Trial filter contains absent trial IDs")
    rows = []
    for trial, label, probabilities in zip(trials, labels, probabilities):
        predicted = int(probabilities.argmax())
        if label != source:
            reason = "outside_source_class"
        elif predicted != source:
            reason = "originally_misclassified"
        elif selected is not None and trial not in selected:
            reason = "trial_filter"
        else:
            reason = None
        rows.append(dict(task=task, direction=direction, source_class=source,
                         target_class=target, subject_id=int(subject), trial_id=int(trial),
                         true_class=int(label), original_prediction=predicted,
                         eligible=reason is None, exclusion_reason=reason))
    return rows
