"""Scientific-contract tests for the external signal-space comparator."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from eegproc.model_explainability.comte.evaluation import (
    IndependentTypicality, conjunction, evaluate_signal, signal_displacement,
)
from eegproc.model_explainability.comte.search import SearchConfig, confidence_loss, optimized_search
from eegproc.model_explainability.comte.report import summarize, build_report
from eegproc.model_explainability.typicality.core import TypicalityRegion


def _predict(x):
    score = x[..., :3].mean(axis=(1, 2, 3))
    p = 1 / (1 + np.exp(-score))
    return np.column_stack((1 - p, p))


def _search(**kwargs):
    x = np.zeros((2, 8, 6), dtype='float32')
    refs = np.stack([np.full_like(x, 3), np.full_like(x, 5)])
    inputs = dict(predict_proba=_predict, original=x, references=refs,
                  reference_labels=[1, 1], reference_probabilities=_predict(refs),
                  reference_keys=[[1, 4], [2, 9]], held_out_subject=0,
                  target_class=1, target_probability=.8, seed=4,
                  config=SearchConfig(max_attempts=3, max_iterations=8, restarts=1))
    inputs.update(kwargs)
    return optimized_search(**inputs)


def test_upstream_loss_and_grouped_donor_endpoint_are_consistent():
    config = SearchConfig()
    assert confidence_loss(.7, 4, threshold=.8, config=config) == pytest.approx(.81)
    result = _search()
    assert result['donor_key'] == [2, 9]
    assert result['changed_groups'] == [0]
    np.testing.assert_array_equal(result['signal'][..., :3], 5)
    np.testing.assert_array_equal(result['signal'][..., 3:], 0)
    np.testing.assert_allclose(_predict(result['signal'][None])[0], result['probabilities'])
    repeated = _search()
    np.testing.assert_array_equal(result['signal'], repeated['signal'])
    assert result['search_queries'] == repeated['search_queries']


def test_loss_matches_pinned_upstream_class_for_all_binary_masks():
    """Execute the original loss class, avoiding its obsolete optional imports."""
    import ast
    import hashlib
    from pathlib import Path
    import pandas as pd
    from eegproc.model_explainability.comte import search
    source = Path(search.__file__).parent / 'upstream/explainers.py.txt'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '42f15cc3aa193ef264d941178522c1921222e214206ca9ac65a040e0fafddca7'
    parsed = ast.parse(source.read_text())
    original_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'LossDiscreteState')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[original_class], type_ignores=[]), str(source), 'exec'), namespace)
    x = pd.DataFrame(np.zeros((16, 6)))
    donor = pd.DataFrame(np.ones((16, 6))*3)
    clf = SimpleNamespace(predict_proba=lambda frame: _predict(frame.to_numpy().reshape(1, 2, 8, 6)))
    original_loss = namespace['LossDiscreteState'](1, clf, x, donor, list(x.columns), reg=.8, max_features=3, maximize=False)
    for number in range(64):
        mask = np.array([(number >> bit) & 1 for bit in range(6)])
        candidate = x.copy()
        for col in np.flatnonzero(mask):
            candidate[col] = donor[col]
        probability = clf.predict_proba(candidate)[0, 1]
        assert confidence_loss(probability, mask.sum(), threshold=.95, config=SearchConfig()) == pytest.approx(original_loss(mask))


def test_queries_count_repeats_and_budget_failures_remain_finite():
    calls = []
    def prediction(x):
        calls.append(x.copy())
        return _predict(x)
    result = _search(predict_proba=prediction,
                     config=SearchConfig(max_queries=1, max_attempts=2, max_iterations=2))
    assert len(calls) == result['search_queries'] == 1
    assert result['stop_reason'] == 'query_budget'
    assert result['donor_key'] is None
    assert np.isfinite(result['signal']).all()
    assert result['probabilities'][1] < .8


def test_source_leakage_and_missing_target_true_positive_are_explicit():
    with pytest.raises(ValueError, match='Held-out'):
        _search(reference_keys=[[0, 4], [2, 9]])
    result = _search(reference_labels=[0, 0])
    assert result['stop_reason'] == 'no_source_target_true_positive'
    assert result['search_queries'] == 1


def test_greedy_fallback_can_cross_the_soft_feature_limit():
    x = np.zeros((2, 8, 6), dtype='float32')
    refs = np.full((1, *x.shape), 1, dtype='float32')
    def predictor(v):
        p = .1 + .4 * (v[..., :3].mean((1, 2, 3)) + v[..., 3:].mean((1, 2, 3)))
        return np.column_stack((1-p, p))
    result = _search(original=x, references=refs, reference_labels=[1],
        reference_probabilities=predictor(refs), reference_keys=[[1, 0]], predict_proba=predictor,
        config=SearchConfig(max_features=1, max_attempts=1, max_iterations=1, restarts=0))
    assert result['probabilities'][1] >= .8
    assert result['changed_groups'] == [0, 1]


def test_independent_reference_and_calibration_use_disjoint_source_subjects():
    values = np.arange(24).reshape(6, 4).astype(float)
    args = dict(subject_ids=[1, 1, 2, 2, 3, 3], trial_ids=[0, 1]*3, labels=[1]*6,
                held_out_subject=0, target_class=1)
    region = IndependentTypicality.fit(values, **args)
    fit_subjects = {key[0] for key in region.metadata['fit_trial_keys']}
    cal_subjects = {key[0] for key in region.metadata['calibration_trial_keys']}
    assert not fit_subjects & cal_subjects
    assert 0 not in fit_subjects | cal_subjects
    assert np.isfinite(region.score(values)).all()
    with pytest.raises(ValueError, match='Held-out'):
        IndependentTypicality.fit(values, **{**args, 'held_out_subject': 2})


def test_shared_evaluator_reencodes_delivered_signal_and_preserves_unknown_physiology():
    class Model:
        calls = 0
        def get_encoder_features(self, signal):
            self.calls += 1
            return {'probabilities': np.array([[.9, .1]]), 'classification_embedding': np.zeros((1, 1))}
    model = Model()
    typicality = TypicalityRegion(np.zeros(1), np.ones(1), 1., {'target_class': 1})
    independent = IndependentTypicality(np.zeros(1), np.ones(1), np.zeros((1, 1)), 1., 1, {})
    physiological = SimpleNamespace(assess=lambda d: {'all_required_passed': None, 'available_checks_passed': True})
    x = np.zeros((2, 8, 3), dtype='float32')
    result = evaluate_signal(model, x, x, target_class=1, target_probability=.8,
        vc_region=typicality, independent_region=independent, diagnostics={'spectral_power': np.ones(1)},
        physiology=physiological, vcsc_penalty=0., vcsc_tolerance=1e-8)
    assert model.calls == 1
    assert result['prediction_validity'] is False  # latent success cannot override the waveform
    assert result['physiological_acceptance'] is None
    assert result['joint_success'] is False
    assert conjunction(True, True, None) is None


def test_channel_sparsity_groups_all_bands_and_uses_observed_input():
    original = np.ones((2, 8, 6))
    cf = original.copy()
    cf[..., 1] += 1
    result = signal_displacement(original, cf)
    assert result['changed_channels'] == 1
    assert result['channel_sparsity'] == .5
    assert result['signal_rmse'] == pytest.approx(1 / np.sqrt(6))


def test_rates_include_errors_and_nonflips_distances_include_failed_endpoints():
    rows = [{'status': 'completed', 'prediction_validity': True, 'signal_rmse': 1},
            {'status': 'completed', 'prediction_validity': False, 'signal_rmse': 3},
            {'status': 'error'}]
    result = summarize(rows)
    assert result['prediction_validity_percent'] == pytest.approx(100/3)
    assert result['signal_rmse_median'] == 2
    assert result['signal_rmse_n'] == 2
    assert result['physiological_acceptance_percent'] is None
    assert summarize([])['prediction_validity_percent'] is None
    assert summarize([{'status': 'pending'}])['prediction_validity_percent'] is None


def _fixture_model(tf):
    class Head:
        prior_mu = tf.constant([[-1.], [1.]])
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
        weights = [tf.Variable([1.])]
        def trial_recurrent_classifier(self, sequence, training=False):
            return tf.reduce_mean(sequence, axis=(1, 2))[:, None]
        def get_encoder_features(self, x):
            x = tf.convert_to_tensor(x)
            embedding = tf.reduce_mean(x, axis=(1, 2, 3))[:, None]
            return {'window_features': x, 'classification_embedding': embedding,
                    'probabilities': tf.nn.softmax(self.vc_target(embedding))}
        def decode_branch_feature_sequence(self, name, z):
            # This deliberately biased decoder makes latent success uninformative.
            return tf.concat([z, z], axis=-1) - 2.
    return Model()


def test_paired_runner_both_directions_failures_resume_and_rebuild(tmp_path, monkeypatch):
    tf = pytest.importorskip('tensorflow')
    from eegproc.model_explainability.comte import runner
    model = _fixture_model(tf)
    monkeypatch.setattr(runner, 'create_sic_adapter', lambda **kwargs: SimpleNamespace(model=model))
    x = np.random.RandomState(7).normal(0, .1, (12, 3, 32, 42)).astype('float32')
    labels = np.tile([0, 1, 0, 1], 3)
    x += (2 * labels - 1)[:, None, None, None]
    trials = tmp_path / 'trials.npz'
    np.savez(trials, features=x, labels=labels, subject_ids=np.repeat([0, 1, 2], 4), trial_ids=np.tile(np.arange(4), 3))
    checkpoint = tmp_path / 'mock.keras'
    checkpoint.write_text('fixture')
    manifest = tmp_path / 'models.json'
    manifest.write_text(json.dumps({'models': [{'path': str(checkpoint), 'target_subject': 0,
        'stage': 'zero_shot_source_model', 'source_subject_ids': [1, 2]}]}))
    args = runner.parse_args(['--models-json', str(manifest), '--trials-npz', str(trials),
        '--task', 'valence', '--subjects', '0', '--out-dir', str(tmp_path / 'out'), '--max-steps', '0',
        '--fixed-joint-alpha', '.5', '--comte-max-queries', '15', '--comte-max-attempts', '2',
        '--comte-max-iterations', '3', '--comte-restarts', '0',
        '--physiological-tolerance-quantile', '.5', '--amplitude-spectral-weight', '.3',
        '--temporal-sparsity-weight', '.01', '--temporal-tv-weight', '.02', '--decoded-distance', 'envelope'])
    actual_search = runner.optimized_search
    failed_once = [False]
    def fail_once(*a, **kw):
        if not failed_once[0]:
            failed_once[0] = True
            raise FloatingPointError('injected failure')
        return actual_search(*a, **kw)
    monkeypatch.setattr(runner, 'optimized_search', fail_once)
    result = runner.run(args)
    assert result['complete']
    assert len(result['population']) == 4
    assert sum(r['n_error'] for r in result['population']) == 1
    assert all(r['n_eligible'] == 2 for r in result['population'])
    assert not list(args.out_dir.rglob('*.npz'))
    for result_path in args.out_dir.glob('*/subject_0/trial_*/*/attempt_*/result.json'):
        endpoint = json.loads(result_path.read_text())
        if endpoint['status'] == 'completed':
            assert endpoint['evaluation_signal_queries'] == 1
            assert endpoint['total_classifier_queries'] == endpoint['generation_classifier_queries'] + 1
            assert endpoint['physiological_acceptance'] in (False, None)
            fold_dir = result_path.parents[3]
            tolerance = json.loads((fold_dir / 'calibration/vcsc_tolerance.json').read_text())
            assert endpoint['vcsc_passed'] == (endpoint['vcsc_penalty'] <= tolerance['tolerance'])
            assert (fold_dir / 'calibration/amplitude_spectral.json').is_file()
            assert (fold_dir / 'calibration/latent_scale.json').is_file()
            if endpoint['method'] == 'cfo':
                assert endpoint['method_summary']['decoded_metric'] == 'envelope'
                assert endpoint['method_summary']['temporal_weights'] == {'sparsity': .01, 'tv': .02}
                assert endpoint['method_summary']['amplitude_spectral_constraint_enforced']
                assert endpoint['method_summary']['physiological_tolerance'] == tolerance['tolerance']
    roots = [args.out_dir / d for d in args.directions]
    assert build_report(roots, tmp_path / 'rebuilt') == result
    args.resume = True
    retried = runner.run(args)
    assert sum(r['n_error'] for r in retried['population']) == 0
    # Failed attempts remain in the ledger after retry; trial denominators stay paired.
    import csv
    with (args.out_dir / 'report/attempts.csv').open() as handle:
        ledger = list(csv.DictReader(handle))
    assert len(ledger) == 9 and sum(r['status'] == 'error' for r in ledger) == 1
    repeated = runner.run(args)
    assert repeated == retried
    args.channel_rtol *= 2
    with pytest.raises(ValueError, match='Resume refused'):
        runner.run(args)
    calibration = roots[0] / 'subject_0/calibration/independent.json'
    calibration.write_text(calibration.read_text() + '\n')
    with pytest.raises(ValueError, match='calibration changed'):
        build_report(roots, tmp_path / 'tampered')


def test_counted_optimizer_with_saved_sic_model_keeps_weights_frozen(tmp_path):
    tf = pytest.importorskip('tensorflow')
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
    from eegproc.model_explainability.model_agnostic.sic_adapter import create_sic_adapter
    from eegproc.model_explainability.comte.runner import CountingModel, CountedCFO
    from eegproc.model_explainability.typicality.physiology import make_vcsc_loss, vcsc_calibration
    tf.keras.utils.set_random_seed(17)
    real = build_sic_model(input_shape=(3, 32, 42), adjacency=np.eye(14, dtype=np.float32),
        classification_level='trial', n_channels=14, n_bands=3, gcn_units=(4,),
        spectral_gru_units=4, bilstm_units=2, classifier_rnn_units=4,
        use_decoder=True, use_subject_adversarial=False, decoder_dropout=0)
    x = np.random.RandomState(7).normal(size=(1, 3, 32, 42)).astype('float32')
    path = tmp_path / 'saved.keras'
    real.save(path)
    loaded = create_sic_adapter(model_path=path, config={}, sample_input=x[0]).model
    loaded.trainable = False
    before = [v.numpy().copy() for v in loaded.weights]
    model = CountingModel(loaded)
    result = CountedCFO(model, loss=make_vcsc_loss(vcsc_calibration(x)),
                        decoder_mode='joint', max_steps=1).optimize(x, target_class=1)
    assert model.signal_queries == 1
    assert model.latent_queries == len(result['history']) + 2
    endpoint = result['arrays']['x_prime_joint']
    features = model.get_encoder_features(endpoint)
    assert np.shape(features['probabilities']) == (1, 2)
    assert model.signal_queries == 2
    for initial, current in zip(before, loaded.weights):
        np.testing.assert_array_equal(initial, current.numpy())
