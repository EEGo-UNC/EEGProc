"""Common-epoch accuracy selection and weighted SIC checkpoint/calibration checks."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from oracle import (choose_epoch, prediction_accuracy, fingerprint, file_hash, write_json,
                    summarize, train_source, calibrate_selected, make_epoch_callback)
from train import training_config, main


def rows(scores):
    return [dict(target_subject=target, epoch=epoch, accuracy=score,
                 checkpoint=f'subject_{target:02}/epoch_{epoch:03}.keras')
            for target, values in enumerate(scores) for epoch, score in enumerate(values, 1)]


def test_one_shared_epoch_uses_equal_subject_accuracy():
    records = rows([[.9, .1], [.4, .9]])
    for row in records:
        row['n_windows'] = 10 if row['target_subject'] == 0 else 1000
    result = choose_epoch(records, [0, 1], 2)
    assert result['best_epoch'] == 1
    assert result['selection_score'] == pytest.approx(.65)
    assert result['selected_models'] == {'0': 'subject_00/epoch_001.keras', '1': 'subject_01/epoch_001.keras'}


def test_exact_ties_choose_earliest_epoch():
    assert choose_epoch(rows([[.8, .4], [.2, .6]]), [0, 1], 2)['best_epoch'] == 1


def test_missing_epoch_cannot_select_a_smaller_budget():
    result = choose_epoch(rows([[.9, .8], [.7]]), [0, 1], 2)
    assert result['status'] == 'incomplete'
    assert result['missing_subject_epochs'] == [[1, 2]]
    assert 'selected_models' not in result and 'best_epoch' not in result


@pytest.mark.parametrize('bad', ['duplicate', 'nan', 'wrong_subject'])
def test_invalid_epoch_records_rejected(bad):
    data = rows([[.5], [.5]])
    if bad == 'duplicate':
        data.append(data[0].copy())
    elif bad == 'nan':
        data[0]['accuracy'] = float('nan')
    else:
        data[0]['target_subject'] = 10
    with pytest.raises(ValueError):
        choose_epoch(data, [0, 1], 1)


def test_predictions_use_plain_argmax_and_unweighted_accuracy():
    y = np.array([0, 0, 0, 1, 2, 3])
    probabilities = np.eye(4)[[0, 0, 0, 0, 0, 3]]
    accuracy, correct = prediction_accuracy(y, probabilities)
    assert correct == 4 and accuracy == 4 / 6
    # Balanced accuracy would be .5 for this same prediction pattern.
    with pytest.raises(ValueError):
        prediction_accuracy(y, probabilities * 2)


def archive_fold(suite, target, *, protocol=None):
    folder = suite / f'subject_{target:02}'
    folder.mkdir(parents=True)
    protocol = protocol or dict(source_epochs=1, dataset_id='dataset', metric='accuracy')
    write_json(folder / 'protocol.json', dict(target_subject=target, protocol=protocol, protocol_id=fingerprint(protocol)))
    checkpoint = folder / 'epoch_001.keras'
    checkpoint.write_bytes(b'checkpoint fixture')
    probabilities = np.eye(4)
    archive = folder / 'epoch_001_predictions.npz'
    np.savez(archive, y_true=np.arange(4), probabilities=probabilities, trial_ids=np.arange(4))
    row = dict(target_subject=target, epoch=1, accuracy=1., n_correct=4, n_windows=4,
        checkpoint=str(checkpoint.relative_to(suite)), checkpoint_sha256=file_hash(checkpoint),
        predictions=str(archive.relative_to(suite)), protocol_id=fingerprint(protocol))
    write_json(folder / 'epoch_001.json', row)
    return folder


def test_selection_checks_prediction_archive_and_checkpoint_integrity(tmp_path):
    archive_fold(tmp_path, 0)
    folder = archive_fold(tmp_path, 1)
    selected = summarize(tmp_path, [0, 1], 1)
    assert selected['status'] == 'completed' and selected['selection_score'] == 1.
    assert (tmp_path / 'oracle_models.json').is_file()
    (folder / 'epoch_001.keras').write_bytes(b'modified checkpoint')
    with pytest.raises(ValueError, match='checkpoint'):
        summarize(tmp_path, [0, 1], 1)
    assert not (tmp_path / 'oracle_models.json').exists()


def test_mixed_protocols_and_falsified_scores_cannot_select(tmp_path):
    first = archive_fold(tmp_path, 0)
    archive_fold(tmp_path, 1, protocol=dict(source_epochs=1, dataset_id='different', metric='accuracy'))
    with pytest.raises(ValueError, match='different data or training'):
        summarize(tmp_path, [0, 1], 1)
    row = json.loads((first / 'epoch_001.json').read_text())
    row['accuracy'] = .5
    write_json(first / 'epoch_001.json', row)
    with pytest.raises(ValueError, match='archived predictions'):
        summarize(tmp_path, [0], 1)


def test_selection_cli_marks_incomplete_cohort_without_gpu(tmp_path):
    archive_fold(tmp_path, 0)
    assert main(['--phase', 'select', '--suite-dir', str(tmp_path),
                 '--source-epochs', '1', '--targets', '0', '1']) == 2
    assert not (tmp_path / 'oracle_models.json').exists()


def test_config_enables_both_weighting_flags_and_accuracy(tmp_path):
    model = json.loads(Path(__file__).with_name('model_config.json').read_text())
    config = training_config(tmp_path, 0, model)
    assert model['use_class_weight']
    assert config.source_use_class_weight and config.calibration_use_class_weight
    assert config.selection_metric == 'accuracy'
    assert config.early_stopping_patience is None and not config.restore_best_weights


def test_partial_epoch_does_not_write_a_checkpoint(tmp_path):
    callback = make_epoch_callback(tmp_path / 'subject_00', 0, {},
        np.zeros((4, 4, 310), np.float32), np.arange(4), np.arange(4), expected_batches=2, batch_size=4)
    callback.on_epoch_begin(0)
    callback.on_train_batch_end(0)
    with pytest.raises(RuntimeError, match='Incomplete/nonfinite'):
        callback.on_epoch_end(0, {'loss': 1.})
    assert not list(tmp_path.rglob('*.keras'))


def test_weighted_source_common_epoch_reload_and_calibration(tmp_path):
    import tensorflow as tf
    from streaming import install_streaming_fit
    from eegproc.deep_learning.joint_architectures.SICModelv15 import sic_model as v15
    tf.keras.utils.set_random_seed(42)
    subjects = np.repeat(np.arange(5), 12)
    trials = np.tile(np.arange(12), 5)
    labels = np.tile([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3], 5)
    x = np.random.default_rng(5).normal(size=(60, 4, 310)).astype(np.float32)
    x[:, 0, 0] = subjects
    settings = json.loads(Path(__file__).with_name('model_config.json').read_text())
    settings.update(gcn_units={'fixed': [2]}, spectral_gru_units=3, bilstm_units=2,
                    classifier_rnn_units={'fixed': [3]}, n_classifier_rnn_layers=1,
                    subject_hidden_units=3, gcn_dropout=0., spectral_gru_dropout=0.,
                    bilstm_dropout=0., classifier_rnn_dropout=0., decoder_dropout=0.)
    configs = []
    for target in (0, 1):
        config = training_config(tmp_path, target, settings, source_epochs=2)
        config.source_batch_size = 8
        config.calibration_epochs = 1
        config.calibration_levels = ((4, 1),)
        config.calibration_trials = config.calibration_selection_shots = 4
        config.calibration_folds = 1
        config.verbose = 0
        configs.append(config)
    estimator_inputs, fit_weights = [], []
    def graph(training_features, **kwargs):
        estimator_inputs.append(np.unique(training_features[:, 0, 0]).astype(int).tolist())
        return np.eye(62, dtype=np.float32)
    original = tf.keras.Model.fit
    def capture(self, *args, **kwargs):
        fit_weights.append(kwargs.get('class_weight'))
        return original(self, *args, **kwargs)
    try:
        tf.keras.Model.fit = capture
        install_streaming_fit()
        with patch.object(v15, '_source_only_mi_adjacency', side_effect=graph):
            for config in configs:
                train_source(x, labels, subjects, trials, config, tmp_path, dataset_id='synthetic-generation')
        for target, source_subjects in enumerate(estimator_inputs):
            meta = json.loads((tmp_path / f'subject_{target:02}/protocol.json').read_text())
            assert source_subjects == meta['source_subjects']
            assert target not in source_subjects
            assert not set(source_subjects) & set(meta['validation_subjects'])
            assert meta['class_weights'] == {'0': .5, '1': 1.5, '2': 1.5, '3': 1.5}
        result = summarize(tmp_path, [0, 1], 2)
        assert result['status'] == 'completed'
        assert all(f"epoch_{result['best_epoch']:03}.keras" in path for path in result['selected_models'].values())
        for config in configs:
            calibrated = calibrate_selected(x, labels, subjects, trials, config, tmp_path,
                                            dataset_id='synthetic-generation')
            assert calibrated['n_source_model_fits'] == 0
            assert calibrated['n_calibration_fits'] == 1
            assert calibrated['oracle_selection']['epoch'] == result['best_epoch']
        assert len(fit_weights) == 4  # Two source fits, two independent calibration fits.
        assert all(weights is not None for weights in fit_weights)
        assert fit_weights[0][0] == .5 and fit_weights[0][1] == 1.5
        with pytest.raises(FileExistsError):
            train_source(x, labels, subjects, trials, configs[0], tmp_path, dataset_id='synthetic-generation')
    finally:
        tf.keras.Model.fit = original
