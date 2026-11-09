"""Bidirectional calibration, full-denominator reports, and restart regression tests."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from eegproc.model_explainability.typicality.core import TypicalityRegion
from eegproc.model_explainability.typicality.coverage import eligibility_records
from eegproc.model_explainability.typicality.results import build_report, population_summary


@pytest.mark.parametrize('target', [0, 1])
def test_tenth_percentile_uses_only_source_target_class(target, tmp_path):
    labels = np.repeat([target, 1 - target], 20)
    embeddings = np.arange(40, dtype=float)[:, None]
    region = TypicalityRegion.calibrate(embeddings, prior_mean=[0], prior_variance=[1],
        labels=labels, subject_ids=np.repeat([1, 2], 20), trial_ids=np.tile(np.arange(20), 2),
        target_class=target, held_out_subject=0)
    assert region.tau == np.quantile(np.arange(20) ** 2, 0.10, method='higher')
    assert region.metadata['target_class'] == target
    assert region.metadata['quantile'] == 0.10
    assert region.metadata['n_calibration_trials'] == 20
    region.save(tmp_path)
    assert TypicalityRegion.load(tmp_path).metadata == region.metadata
    with pytest.raises(ValueError, match='Held-out'):
        TypicalityRegion.calibrate(embeddings, prior_mean=[0], prior_variance=[1],
            labels=labels, subject_ids=np.repeat([0, 2], 20), trial_ids=np.tile(np.arange(20), 2),
            target_class=target, held_out_subject=0)


@pytest.mark.parametrize('direction,source', [('0_to_1', 0), ('1_to_0', 1)])
def test_exclusion_audit_is_exhaustive_and_direction_specific(direction, source):
    probabilities = np.eye(2)[[source, 1 - source, source, source]]
    rows = eligibility_records(task='valence', direction=direction, subject=0,
        trial_ids=[0, 1, 2, 3], labels=[source, source, 1 - source, source],
        probabilities=probabilities, trial_filter=[0])
    assert [r['exclusion_reason'] for r in rows] == [None, 'originally_misclassified', 'outside_source_class', 'trial_filter']
    assert sum(r['eligible'] for r in rows) == 1
    assert all(r['target_class'] == 1 - source for r in rows)


def test_failed_pending_and_error_trials_keep_denominator_and_finite_endpoints():
    rows = [dict(status='completed', latent_target_success=True, d_z=1),
            dict(status='completed', latent_target_success=False, d_z=9),
            dict(status='error'), dict(status='pending')]
    result = population_summary(rows)
    assert result['n_eligible'] == 4
    assert result['n_unsuccessful'] == result['n_error'] == result['n_pending'] == 1
    assert result['latent_target_success_percent'] == 25
    assert result['d_z_n'] == 2 and result['d_z_median'] == 5
    assert result['provisional']
    assert population_summary([])['latent_target_success_percent'] is None


def test_defaults_use_both_directions_four_arms_and_only_tenth_percentile():
    pytest.importorskip('tensorflow')
    from eegproc.model_explainability.typicality.runner import parse_args
    argv = ['--models-json', 'models.json', '--trials-npz', 'trials.npz', '--task', 'valence', '--out-dir', 'out']
    args = parse_args(argv)
    assert args.directions == ['0_to_1', '1_to_0']
    assert args.include_target_latent and args.include_typicality_no_physiology
    assert args.typicality_quantile == 0.10
    with pytest.raises(SystemExit):
        parse_args(argv + ['--typicality-quantile', '0.95'])
    with pytest.raises(SystemExit):
        parse_args(argv + ['--directions', '1_to_0', '1_to_0'])


@pytest.mark.parametrize('knn_k', [1, 5])
def test_paired_suite_errors_retries_resume_and_offline_reports(tmp_path, monkeypatch, knn_k):
    tf = pytest.importorskip('tensorflow')
    from eegproc.model_explainability.typicality import runner
    from eegproc.model_explainability.typicality.artifacts import completed_attempt

    class Head:
        prior_mu = tf.constant([[-2.], [2.]])
        prior_log_sigma = tf.zeros((2, 1))

        def __call__(self, embedding, training=False):
            return tf.concat([-embedding, embedding], axis=1)

        def vc_loss_components(self, **kwargs):
            zero = tf.reduce_sum(kwargs['mh']) * 0
            return {key: zero for key in ('weighted_focal_loss', 'weighted_latent_posterior_kl',
                    'weighted_discriminator_kl', 'weighted_class_prior_kl')}

    class Model:
        classification_level = 'trial'
        use_decoder = True
        use_gcn_gru_branch = use_bilstm_branch = True
        gcn_gru_decoder = bilstm_decoder = True
        gcn_gru_feature_dim = bilstm_feature_dim = 21
        vc_target = Head()

        def trial_recurrent_classifier(self, sequence, training=False):
            return tf.reduce_mean(sequence, axis=(1, 2))[:, None]

        def get_encoder_features(self, x):
            x = tf.convert_to_tensor(x)
            embedding = tf.reduce_mean(x, axis=(1, 2, 3))[:, None]
            return {'window_features': x, 'classification_embedding': embedding,
                    'probabilities': tf.nn.softmax(self.vc_target(embedding))}

        def decode_branch_feature_sequence(self, name, z):
            return tf.concat([z, z], axis=-1)

    model = Model()
    adapter = SimpleNamespace(model=model, metadata=lambda: {'fixture': True})
    monkeypatch.setattr(runner, 'create_sic_adapter', lambda **kwargs: adapter)
    x = np.random.default_rng(2).normal(0, .01, (12, 3, 32, 42)).astype('float32')
    labels = np.tile([0, 1, 0, 1], 3)
    x += np.where(labels == 1, 1., -1.)[:, None, None, None]
    # Two correctly predicted trials, one wrong prediction, and one filtered trial.
    labels[2] = 1
    data = tmp_path / 'trials.npz'
    np.savez(data, features=x, labels=labels, subject_ids=np.repeat([0, 1, 2], 4), trial_ids=np.tile(np.arange(4), 3))
    checkpoint = tmp_path / 'model.keras'
    checkpoint.write_text('mock checkpoint; factory supplies a deterministic differentiable model')
    manifest = tmp_path / 'models.json'
    manifest.write_text(json.dumps({'models': [{'path': str(checkpoint), 'target_subject': 0, 'stage': 'zero_shot_source_model'}]}))
    out = tmp_path / 'suite'
    args = runner.parse_args(['--models-json', str(manifest), '--task', 'valence', '--trials-npz', str(data),
        '--subjects', '0', '--trial-ids', '0', '1', '2', '--fixed-joint-alpha', '.5',
        '--max-steps', '1', '--log-every', '0', '--knn-k', str(knn_k), '--out-dir', str(out)])
    calls = []
    optimize = runner.CounterfactualOptimizer.optimize
    fail_once = [True]
    def capture(self, inputs, **kwargs):
        calls.append(kwargs['target_class'])
        if kwargs['target_class'] == 0 and fail_once[0]:
            fail_once[0] = False
            raise FloatingPointError('injected optimization failure')
        return optimize(self, inputs, **kwargs)
    monkeypatch.setattr(runner.CounterfactualOptimizer, 'optimize', capture)
    result = runner.run(args)
    assert result['complete']
    assert len(result['population']) == 8
    assert calls == [1] * 4 + [0] * 4
    assert sum(r['n_error'] for r in result['population']) == 1
    assert all(r['n_eligible'] == 1 for r in result['population'])
    assert sum(r['n_unsuccessful'] for r in result['population']) == 7
    assert all(r['latent_target_success_percent'] == 0 for r in result['population'])
    assert not list(out.rglob('*.npz'))
    for direction, target in [('0_to_1', 1), ('1_to_0', 0)]:
        region = TypicalityRegion.load(out / direction / 'subject_0/calibration')
        assert region.metadata['target_class'] == target
        assert region.metadata['quantile'] == 0.10
        assert region.metadata['source_subject_ids'] == [1, 2]
        fold = json.loads((out / direction / 'subject_0/fold.json').read_text())
        assert fold['eligible_trial_ids'] == [1 - target]
        assert fold['objective_losses']['target_latent']['physiological_weight'] == 0
        assert fold['objective_losses']['typicality_no_physiology']['physiological_weight'] == 0
        source_vectors = json.loads((out / direction / 'subject_0/calibration/source_trials.json').read_text())
        observed_vectors = json.loads((out / direction / 'subject_0/observations.json').read_text())
        assert np.asarray(source_vectors['embeddings']).shape == (8, 1)
        assert np.asarray(observed_vectors['embeddings']).shape == (4, 1)
        assert fold['encoded_neighbors']['status'] == ('calibrated' if knn_k == 1 else 'unavailable')
        for endpoint_path in (out / direction / 'subject_0').glob('trial_*/*/attempt_*/result.json'):
            endpoint = json.loads(endpoint_path.read_text())
            if endpoint['status'] != 'completed':
                continue
            assert set(endpoint['classification_embeddings']) == {
                'original', 'optimized', 'decoded_reconstruction', 'decoded_counterfactual'}
            assert endpoint['encoded_neighbors_status'] == ('calibrated' if knn_k == 1 else 'unavailable')
            for role in ('decoded_counterfactual', 'decoded_reconstruction'):
                scored = endpoint[role]
                assert scored['target_class'] == target
                assert scored['target_probability'] == scored['probabilities'][target]
                assert scored['success'] == (scored['predicted_class'] == target and
                                             scored['target_probability'] >= args.target_probability)
            if knn_k == 1:
                assert endpoint['encoded_neighbors']['optimized']['k'] == 1
                assert endpoint['encoded_neighbors']['optimized']['quantile'] == .10
    if knn_k == 1:
        independent = json.loads((out / 'knn_report/knn_report.json').read_text())
        optimized = [r for r in independent['population'] if r['role'] == 'optimized_counterfactual']
        assert len(optimized) == 8 and sum(r['n_attempt_errors'] for r in optimized) == 1
        assert {r['direction'] for r in optimized} == {'0_to_1', '1_to_0'}
    else:
        assert json.loads((out / 'knn_report/status.json').read_text())['status'] == 'unavailable'
    assert build_report([out], tmp_path / 'rebuilt') == result
    with (out / 'report/exclusions.csv').open() as stream:
        exclusions = list(csv.DictReader(stream))
    assert len(exclusions) == 6
    assert {r['exclusion_reason'] for r in exclusions} == {'outside_source_class', 'originally_misclassified', 'trial_filter'}
    # A failed trial is retried; completed endpoints are reused and counted once.
    args.resume = True
    retried = runner.run(args)
    assert len(calls) == 9
    assert sum(r['n_error'] for r in retried['population']) == 0
    assert all(r['n_eligible'] == 1 for r in retried['population'])
    assert build_report([out], tmp_path / 'rebuilt_after_retry') == retried
    def forbidden(**kwargs):
        raise AssertionError('Completed folds must resume without loading a model')
    monkeypatch.setattr(runner, 'create_sic_adapter', forbidden)
    assert runner.run(args) == retried
    # A missing declared fold cannot become a complete population result.
    missing = build_report([out], tmp_path / 'missing', expected_subjects=2,
                           expected_tasks=['valence'], expected_directions=['0_to_1', '1_to_0'])
    assert not missing['complete']
    assert all(r['n_expected_subjects'] == 2 and r['provisional'] for r in missing['population'])
    assert all(r['n_eligible'] is None for r in missing['coverage']
               if r.get('subject_id') == 1)
    with pytest.raises(ValueError, match='incomplete'):
        build_report([out], tmp_path / 'required', expected_subjects=2, require_complete=True)
    # Empty eligible cohorts retain subject rows with undefined rates.
    import shutil
    empty = tmp_path / 'empty'
    shutil.copytree(out, empty)
    for direction in ['0_to_1', '1_to_0']:
        directory = empty / direction / 'subject_0'
        observation = json.loads((directory / 'observations.json').read_text())
        observation['labels'] = (1 - np.argmax(observation['probabilities'], axis=1)).tolist()
        (directory / 'observations.json').write_text(json.dumps(observation))
        audit = eligibility_records(task='valence', direction=direction, subject=0,
            trial_ids=observation['trial_ids'], labels=observation['labels'],
            probabilities=observation['probabilities'], trial_filter=args.trial_ids)
        (directory / 'eligibility.json').write_text(json.dumps(audit))
        fold = json.loads((directory / 'fold.json').read_text())
        fold['eligible_trial_ids'] = []
        (directory / 'fold.json').write_text(json.dumps(fold))
    zero = build_report([empty], tmp_path / 'zero')
    assert zero['complete'] and len(zero['population']) == 8
    assert all(r['n_eligible'] == 0 and r['latent_target_success_percent'] is None for r in zero['population'])
    # Missing commit markers stay pending, with provisional rates in CSV and -- in TeX.
    attempt = completed_attempt(out / '1_to_0/subject_0/trial_1/base')
    marker = attempt / 'complete.json'
    marker.rename(attempt / 'marker.backup')
    pending = build_report([out], tmp_path / 'pending')
    assert not pending['complete']
    assert sum(r['n_pending'] for r in pending['population']) == 1
    assert '--' in (tmp_path / 'pending/tables.tex').read_text()
    with pytest.raises(ValueError, match='Duplicate'):
        build_report([out, out], tmp_path / 'duplicate')
    args.typicality_weight = 2
    with pytest.raises(ValueError, match='protocol changed'):
        runner.run(args)

    if knn_k == 1:
        # The incoming repair/temporal features must also work in the reverse
        # direction, with all four arms and independent neighbour evaluation.
        monkeypatch.setattr(runner, 'create_sic_adapter', lambda **kwargs: adapter)
        options = runner.parse_args(['--models-json', str(manifest), '--task', 'valence',
            '--trials-npz', str(data), '--subjects', '0', '--trial-ids', '0', '1', '2',
            '--fixed-joint-alpha', '.5', '--max-steps', '1', '--log-every', '0', '--knn-k', '1',
            '--out-dir', str(tmp_path / 'integrated'), '--include-repair-baseline',
            '--physiological-tolerance-quantile', '.5', '--amplitude-spectral-weight', '1',
            '--temporal-sparsity-weight', '.001', '--temporal-tv-weight', '.002',
            '--decoded-distance', 'envelope'])
        integrated = runner.run(options)
        assert integrated['complete'] and len(integrated['population']) == 8
        for direction, target in [('0_to_1', 1), ('1_to_0', 0)]:
            fold_root = options.out_dir / direction / 'subject_0'
            repair = completed_attempt(fold_root / f'trial_{1-target}/repair')
            repair_result = json.loads((repair / 'result.json').read_text())
            assert repair_result['target_class'] == target
            assert repair_result['source_class'] == 1-target
            assert repair_result['direction'] == direction
            assert repair_result['repair_only']
            for arm in ('target_latent', 'base', 'typicality', 'typicality_no_physiology'):
                endpoint = completed_attempt(fold_root / f'trial_{1-target}' / arm)
                saved = json.loads((endpoint / 'result.json').read_text())
                assert saved['decoded_repair']['target_class'] == target
                assert saved['decoded_repair']['target_probability'] == saved['decoded_repair']['probabilities'][target]
                assert saved['repair_baseline']['available']
                assert (endpoint / 'seams.json').is_file()
                assert saved['physiology']['components']['channel_names'] is None
        previous = len(calls)
        options.resume = True
        assert runner.run(options) == integrated
        assert len(calls) == previous


def test_saved_sic_model_supports_reverse_target_without_weight_changes(tmp_path):
    tf = pytest.importorskip('tensorflow')
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
    from eegproc.model_explainability.model_agnostic.sic_adapter import create_sic_adapter
    from eegproc.model_explainability.counterfactuals.optimizer import CounterfactualOptimizer
    from eegproc.model_explainability.typicality.physiology import vcsc_calibration, make_vcsc_loss
    tf.keras.utils.set_random_seed(17)
    model = build_sic_model(input_shape=(3, 32, 42), adjacency=np.eye(14, dtype=np.float32),
        classification_level='trial', n_channels=14, n_bands=3, gcn_units=(4,),
        spectral_gru_units=4, bilstm_units=2, classifier_rnn_units=4,
        use_decoder=True, use_subject_adversarial=False, decoder_dropout=0)
    x = np.random.default_rng(18).normal(size=(1, 3, 32, 42)).astype('float32')
    path = tmp_path / 'saved.keras'
    model.save(path)
    adapter = create_sic_adapter(model_path=path, config={}, sample_input=x[0])
    loaded = adapter.model
    before = [weight.numpy().copy() for weight in loaded.weights]
    flags = [weight.trainable for weight in loaded.weights]
    region = TypicalityRegion(loaded.vc_target.prior_mu[0].numpy(),
        np.exp(2 * loaded.vc_target.prior_log_sigma[0].numpy()), 1, {'target_class': 0})
    loss = make_vcsc_loss(vcsc_calibration(x), target_probability=.8)
    result = CounterfactualOptimizer(loaded, loss=loss, typicality=region,
                                    decoder_mode='joint', max_steps=1).optimize(x, target_class=0)
    assert result['summary']['target_class'] == 0
    from eegproc.model_explainability.typicality.knn_evaluation import evaluate_trial_embeddings
    exported = evaluate_trial_embeddings(loaded, result['arrays'], report_output='joint')
    assert set(exported['classification_embeddings']) == {
        'original', 'optimized', 'decoded_reconstruction', 'decoded_counterfactual'}
    dimension = int(loaded.vc_target.prior_mu.shape[-1])
    assert all(len(vector) == dimension for vector in exported['classification_embeddings'].values())
    probabilities = result['summary']['latent_counterfactual']['probabilities']
    assert result['summary']['latent_counterfactual']['target_probability'] == probabilities[0]
    for old, weight in zip(before, loaded.weights):
        np.testing.assert_array_equal(old, weight.numpy())
    assert flags == [weight.trainable for weight in loaded.weights]
    assert result['summary']['counterfactual_validity_prediction_space'] == 'latent'


def test_posthoc_decoded_rates_preserve_unknown_completed_measurements():
    scored = dict(status='completed', reencoded_target_success=True,
                  reencoded_typical=True, reencoded_success_and_typical=True)
    failed = population_summary([scored, dict(status='error')])
    assert failed['reencoded_n_evaluated'] == 1
    assert failed['reencoded_target_success_percent'] == 50
    unknown = population_summary([scored, dict(status='completed')])
    assert unknown['reencoded_n_evaluated'] == 1
    assert unknown['reencoded_target_success_percent'] is None
