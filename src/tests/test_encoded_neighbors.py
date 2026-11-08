"""Scientific protocol checks for the independent encoded-neighbour evaluator."""

import copy
import json

import numpy as np
import pytest

from eegproc.model_explainability.typicality.neighbors import (
    EncodedKNNReference, kth_neighbor_distance,
)


def source_trials():
    subjects = np.repeat(np.arange(1, 9), 6)
    trials = np.tile(np.arange(6), 8)
    labels = (trials % 2).astype(int)
    rng = np.random.default_rng(12)
    return dict(embeddings=rng.normal(size=(48, 3)), labels=labels,
                subject_ids=subjects, trial_ids=trials, held_out_subject=0)


def fit(**kwargs):
    arguments = source_trials()
    arguments.update(kwargs)
    return EncodedKNNReference.fit_source(**arguments)


def test_distance_is_fifth_neighbor_not_average_or_nearest():
    # The five closest distances are 1, 2, 3, 4, 5: fifth = 5, mean = 3.
    bank = np.array([[7.], [2.], [6.], [1.], [5.], [3.], [4.]])
    np.testing.assert_allclose(kth_neighbor_distance([[0.]], bank, k=5), [5.])
    np.testing.assert_allclose(kth_neighbor_distance([[0.]], bank, k=1), [1.])
    np.testing.assert_allclose(kth_neighbor_distance([[0.]], bank, k=7), [7.])


def test_source_reference_and_calibration_are_disjoint_participants():
    reference = fit()
    metadata = reference.metadata
    bank = set(metadata['bank_subject_ids'])
    calibration = set(metadata['calibration_subject_ids'])
    assert bank and calibration and not bank & calibration
    assert 0 not in bank | calibration
    assert bank | calibration == set(range(1, 9))
    assert metadata['k'] == 5 and metadata['quantile'] == .10
    assert metadata['n_bank_trials'] == 18
    assert metadata['n_calibration_trials'] == 6
    assert all(trial % 2 == 1 for _, trial in metadata['bank_trial_keys'])
    assert all(trial % 2 == 1 for _, trial in metadata['calibration_trial_keys'])


def test_target_zero_uses_class_zero_source_trials_for_reverse_direction():
    reference = fit(target_class=0)
    assert reference.metadata['target_class'] == 0
    assert all(trial % 2 == 0 for _, trial in reference.metadata['bank_trial_keys'])


def test_non_target_source_outliers_do_not_change_reference_or_cutoff():
    arguments = source_trials()
    original = EncodedKNNReference.fit_source(**arguments)
    arguments['embeddings'][arguments['labels'] == 0] = 1e10
    changed = EncodedKNNReference.fit_source(**arguments)
    np.testing.assert_array_equal(original.bank, changed.bank)
    assert original.threshold == changed.threshold


def test_calibration_values_never_enter_reference_bank():
    arguments = source_trials()
    original = fit(calibration_subject_ids=[7, 8])
    arguments['embeddings'][np.isin(arguments['subject_ids'], [7, 8])] += 100
    changed = EncodedKNNReference.fit_source(**arguments, calibration_subject_ids=[7, 8])
    np.testing.assert_array_equal(original.bank, changed.bank)
    assert changed.threshold > original.threshold


def test_held_out_subject_leakage_and_duplicate_trials_are_rejected():
    with pytest.raises(ValueError, match='Held-out subject'):
        fit(held_out_subject=1)
    arguments = source_trials()
    arguments['trial_ids'][1] = arguments['trial_ids'][0]
    with pytest.raises(ValueError, match='Duplicate'):
        EncodedKNNReference.fit_source(**arguments)


@pytest.mark.parametrize('selected', [[0], [99], [1, 1], list(range(1, 9))])
def test_invalid_source_calibration_partition_is_rejected(selected):
    with pytest.raises(ValueError):
        fit(calibration_subject_ids=selected)


def test_tenth_percentile_is_stricter_than_95th_with_same_reference():
    strict = fit(quantile=.10)
    broad = fit(quantile=.95)
    np.testing.assert_array_equal(strict.bank, broad.bank)
    assert strict.threshold < broad.threshold
    query = strict.calibration_embeddings[np.argmax(strict.calibration_scores)]
    assert not strict.assess(query)['inside']
    assert broad.assess(query)['inside']
    assert strict.assess(query)['distance'] == broad.assess(query)['distance']


def test_zero_cutoff_reports_membership_without_dividing_by_zero():
    arguments = source_trials()
    arguments['embeddings'][:] = 0
    reference = EncodedKNNReference.fit_source(**arguments)
    result = reference.assess([0, 0, 0])
    assert result['inside'] and result['distance'] == 0 and result['threshold'] == 0
    assert result['distance_over_threshold'] is None
    assert not reference.assess([1, 0, 0])['inside']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bad', [0, -1, 1.5, True, 100])
def test_invalid_neighbor_counts_are_rejected(bad):
    with pytest.raises(ValueError):
        kth_neighbor_distance([[0.]], [[1.], [2.]], k=bad)


@pytest.mark.parametrize('bad', [0, 1, -.1, float('nan')])
def test_invalid_quantiles_are_rejected(bad):
    with pytest.raises(ValueError, match='quantile'):
        fit(quantile=bad)


def test_wrong_dimension_and_nonfinite_queries_are_rejected():
    reference = fit()
    with pytest.raises(ValueError, match='dimension'):
        reference.assess([1, 2])
    with pytest.raises(ValueError, match='finite'):
        reference.assess([float('nan'), 2, 3])
    assert reference.score(np.empty((0, 3))).shape == (0,)


def test_fixed_evaluator_survives_round_trip_and_detects_corrupt_cutoff(tmp_path):
    reference = fit()
    saved = tmp_path / 'neighbors.json'
    reference.save(saved)
    restored = EncodedKNNReference.load(saved)
    assert reference.assess([1, 2, 3]) == restored.assess([1, 2, 3])
    with pytest.raises(ValueError):
        restored.bank[0, 0] = 0
    payload = copy.deepcopy(reference.to_dict())
    payload['threshold'] += 1
    saved.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='cutoff'):
        EncodedKNNReference.load(saved)
