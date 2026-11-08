"""End-to-end archive checks for the evaluation-only neighbour metric."""

import hashlib
import json

import numpy as np
import pytest

from eegproc.model_explainability.typicality.knn_evaluation import evaluate_trial_embeddings
from eegproc.model_explainability.typicality.knn_report import build_knn_report, fit_archived_source
from eegproc.model_explainability.typicality.neighbors import EncodedKNNReference


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_archive(target=1):
    subjects = np.repeat(np.arange(1, 9), 6)
    trials = np.tile(np.arange(6), 8)
    return {'embeddings': np.random.default_rng(4).normal(size=(48, 3)).tolist(),
            'subject_ids': subjects.tolist(), 'trial_ids': trials.tolist(),
            'labels': (trials % 2).tolist()}


def study_archive(tmp_path, target=1):
    direction = f'{1-target}_to_{target}'
    root = tmp_path / direction
    write_json(root / 'study.json', {'task': 'valence', 'direction': direction,
        'target_class': target, 'typicality_representation': 'vc_trial_embedding',
        'folds': [{'subject_id': 0}, {'subject_id': 9}], 'objectives': ['base', 'typicality']})
    fold = root / 'subject_0'
    write_json(fold / 'fold.json', {'subject_id': 0, 'target_class': target,
        'status': 'completed', 'eligible_trial_ids': [0, 1, 2]})
    write_json(fold / 'calibration/source_trials.json', source_archive(target))
    write_json(fold / 'observations.json', {
        'trial_ids': [0, 1, 2, 3], 'labels': [1-target, 1-target, 1-target, target],
        'embeddings': [[3., 3., 3.], [2., 2., 2.], [1., 1., 1.], [0., 0., 0.]]})
    for objective in ('base', 'typicality'):
        path = fold / 'trial_0' / objective / 'attempt_0001' / 'result.json'
        result = {'status': 'completed'}
        if objective == 'base':
            result['classification_embeddings'] = {
                'original': [3., 3., 3.], 'optimized': [0., 0., 0.],
                'decoded_counterfactual': [1., 1., 1.]}
        write_json(path, result)
        write_json(path.with_name('complete.json'), {'schema_version': 1,
            'sha256': {'result.json': hashlib.sha256(path.read_bytes()).hexdigest()}})
        write_json(fold / 'trial_1' / objective / 'attempt_0001' / 'result.json', {'status': 'error'})
    return root


@pytest.mark.parametrize('target', [0, 1])
def test_report_retains_failed_pending_missing_vectors_and_folds(tmp_path, target):
    root = study_archive(tmp_path, target)
    output = tmp_path / 'report'
    report = build_knn_report([tmp_path], output)
    assert len(report['folds']) == 2
    assert report['folds'][1]['status'] == 'pending'
    assert all(row['provisional'] for row in report['population'])
    optimized = [row for row in report['population'] if row['role'] == 'optimized_counterfactual']
    assert len(optimized) == 2
    for row in optimized:
        assert row['n_total'] == 3 and row['n_attempt_errors'] == 1 and row['n_attempt_pending'] == 1
        assert row['target_class'] == target
        assert row['n_scored'] == (1 if row['objective'] == 'base' else 0)
    before = [row for row in report['population'] if row['role'] == 'original_before_optimization']
    assert len(before) == 2 and all(row['n_scored'] == 3 for row in before)
    assert (output / 'knn_trial_metrics.csv').is_file()
    saved = EncodedKNNReference.load(output / 'references/valence' / f'{1-target}_to_{target}' / 'subject_0.json')
    assert saved.metadata['target_class'] == target
    assert 0 not in saved.metadata['bank_subject_ids'] + saved.metadata['calibration_subject_ids']


def test_scalar_archive_cannot_silently_substitute_for_embeddings():
    with pytest.raises(ValueError, match='Scalar Gaussian discrepancies cannot recover'):
        fit_archived_source({'discrepancy': [.1, .2]}, held_out_subject=0, target_class=1)


def test_duplicate_roots_and_subject_shards_cannot_double_denominators(tmp_path):
    import shutil
    root = study_archive(tmp_path)
    with pytest.raises(ValueError, match='Duplicate study roots'):
        build_knn_report([root, root], tmp_path / 'duplicate_roots')
    other = tmp_path / 'copied_study'
    shutil.copytree(root, other)
    with pytest.raises(ValueError, match='Duplicate task/direction/subject'):
        build_knn_report([root, other], tmp_path / 'duplicate_shards')


def test_counterfactual_with_wrong_fold_identity_cannot_be_scored(tmp_path):
    root = study_archive(tmp_path)
    path = root / 'subject_0/trial_0/base/attempt_0001/result.json'
    payload = json.loads(path.read_text())
    payload['target_class'] = 0
    write_json(path, payload)
    write_json(path.with_name('complete.json'), {'schema_version': 1,
        'sha256': {'result.json': hashlib.sha256(path.read_bytes()).hexdigest()}})
    with pytest.raises(ValueError, match='identity disagrees'):
        build_knn_report([root], tmp_path / 'wrong_identity')


def test_endpoint_export_separates_optimized_and_reencoded_coordinates():
    class FrozenModel:
        def __init__(self):
            self.calls = []

        def get_encoder_features(self, signal):
            self.calls.append(signal)
            return {'classification_embedding': np.asarray(signal, dtype=float) + 10}

    model = FrozenModel()
    source = source_archive()
    reference = fit_archived_source(source, held_out_subject=0, target_class=1)
    arrays = {'classification_embedding': np.array([[1., 2., 3.]]),
              'classification_embedding_prime': np.array([[4., 5., 6.]]),
              'x_reconstructed_joint': np.array([[7., 8., 9.]]),
              'x_prime_joint': np.array([[10., 11., 12.]])}
    result = evaluate_trial_embeddings(model, arrays, report_output='joint', reference=reference)
    assert len(model.calls) == 2
    assert result['classification_embeddings']['optimized'] == [4., 5., 6.]
    assert result['classification_embeddings']['decoded_counterfactual'] == [20., 21., 22.]
    assert result['classification_embeddings']['decoded_reconstruction'] == [17., 18., 19.]
    scored = result['encoded_neighbors']
    assert scored['optimized']['distance'] == reference.assess([4., 5., 6.])['distance']
    assert scored['distance_change'] == scored['optimized']['distance'] - scored['original']['distance']


def test_embedding_export_remains_available_when_knn_evaluation_disabled():
    class FrozenModel:
        def get_encoder_features(self, signal):
            return {'classification_embedding': np.array([[1., 2., 3.]])}

    arrays = {'classification_embedding': np.array([[1., 2., 3.]]),
              'classification_embedding_prime': np.array([[4., 5., 6.]]),
              'x_reconstructed_joint': np.zeros((1, 3)), 'x_prime_joint': np.zeros((1, 3))}
    result = evaluate_trial_embeddings(FrozenModel(), arrays, report_output='joint')
    assert len(result['classification_embeddings']) == 4
    assert 'encoded_neighbors' not in result
