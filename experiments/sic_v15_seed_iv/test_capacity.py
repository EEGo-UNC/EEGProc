"""Measured per-hidden-layer capacity, serialization, and matched run preparation."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest

from capacity import doubled_hidden_config, audit_configs, inventory, load_model_config, resolved_model_config
from oracle import summarize, calibrate_selected, code_identity, write_json
from train import training_config

HERE = Path(__file__).resolve().parent


def configs():
    return (load_model_config(HERE / 'model_config.json'),
            load_model_config(HERE / 'model_config_hidden_params_x2.json'))


def test_comparison_derives_only_architecture_width_changes():
    baseline, comparison = configs()
    assert comparison == doubled_hidden_config(baseline)
    architecture_keys = {'gcn_units', 'spectral_gru_units', 'bilstm_units',
        'classifier_rnn_units', 'subject_hidden_units', 'gcn_gru_decoder_gcn_units',
        'gcn_gru_decoder_projection_units', 'bilstm_decoder_gcn_units', 'bilstm_decoder_projection_units'}
    differences = {key for key in baseline.keys() | comparison.keys() if baseline.get(key) != comparison.get(key)}
    assert differences == architecture_keys
    for key in baseline.keys() - architecture_keys:
        assert baseline[key] == comparison[key]
    assert comparison['gcn_units']['fixed'] == [256, 64]


def test_every_hidden_layer_has_twice_the_measured_parameters():
    baseline, comparison = configs()
    actual = audit_configs(baseline, comparison)
    assert actual == json.loads((HERE / 'capacity_report.json').read_text())
    assert len(actual['hidden_layers']) == 16
    assert all(abs(row['ratio'] / 2 - 1) <= .01 for row in actual['hidden_layers'])
    assert actual['baseline_total_parameters'] == 6430477
    assert actual['comparison_total_parameters'] == 12836084


def test_comparison_reconstructs_reloads_and_calibrates(tmp_path):
    import tensorflow as tf
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
    _, comparison = configs()
    model = build_sic_model((4, 310), adjacency=np.eye(62, dtype=np.float32),
                            n_subject_classes=12, **resolved_model_config(comparison))
    x = np.random.default_rng(12).normal(size=(2, 4, 310)).astype(np.float32)
    logits = model(x, training=False).numpy()
    reconstructions = {key: value.numpy() for key, value in model.reconstruct_branches(x).items()}
    assert logits.shape == (2, 4)
    assert all(value.shape == x.shape for value in reconstructions.values())
    counts = inventory(model)
    checkpoint = tmp_path / 'capacity.keras'
    model.save(checkpoint)
    del model
    tf.keras.backend.clear_session()
    restored = tf.keras.models.load_model(checkpoint, compile=False)
    assert inventory(restored) == counts
    np.testing.assert_allclose(restored(x, training=False).numpy(), logits, atol=1e-6)
    for key, value in restored.reconstruct_branches(x).items():
        np.testing.assert_allclose(value.numpy(), reconstructions[key], atol=1e-6)
    assert restored.gcn_gru_decoder.input_projection.filters == 69
    assert restored.bilstm_decoder.input_projection.filters == 72
    restored.prepare_for_subject_calibration(learning_rate=1e-4, optimizer_name='adamw', weight_decay=5e-5)
    losses = restored.train_on_batch(x, np.array([0, 1]), sample_weight=np.array([.5, 1.5]), return_dict=True)
    assert all(np.isfinite(value) for value in losses.values())
    tf.keras.backend.clear_session()


def test_selection_and_calibration_refuse_wrong_capacity(tmp_path):
    baseline, comparison = configs()
    protocol = dict(source_epochs=1, dataset_id='generation', code_id=code_identity(),
                    model_config=resolved_model_config(comparison))
    fold = tmp_path / 'subject_00'
    fold.mkdir()
    from oracle import fingerprint
    write_json(fold / 'protocol.json', dict(target_subject=0, protocol=protocol, protocol_id=fingerprint(protocol)))
    with pytest.raises(ValueError, match='requested model configuration'):
        summarize(tmp_path, [0], 1, expected_model_config=resolved_model_config(baseline))
    write_json(tmp_path / 'oracle_models.json', dict(status='completed', metric='accuracy', protocol=protocol,
               selected_models={'0': 'subject_00/epoch_001.keras'}))
    cfg = training_config(tmp_path, 0, baseline)
    with pytest.raises(ValueError, match='requested model configuration'):
        calibrate_selected(None, None, None, None, cfg, tmp_path, dataset_id='generation')


def test_submission_preview_keeps_profiles_separate_and_gates_full_first_fold(tmp_path):
    env = {**os.environ, 'ROOT': str(tmp_path)}
    result = subprocess.run(['bash', str(HERE / 'submit_comparison.sh'), '--dry-run', '--id', 'audit'],
                            env=env, capture_output=True, text=True, check=True)
    commands = [shlex.split(line) for line in result.stderr.splitlines()]
    assert len(commands) == 8
    for offset, variant in ((0, 'baseline'), (4, 'hidden_params_x2')):
        first, remaining, selection, calibration = commands[offset:offset+4]
        assert '--array=0' in first
        assert not any(arg.startswith('--dependency=') for arg in first)
        assert '--array=1-14%2' in remaining
        assert f'--dependency=afterok:DRY_{variant}_first' in remaining
        assert f'--dependency=afterok:DRY_{variant}_remaining' in selection
        assert f'--dependency=afterok:DRY_{variant}_select' in calibration
        for command in (first, remaining, selection, calibration):
            export = next(arg for arg in command if arg.startswith('--export='))
            assert f'SUITE_DIR={tmp_path}/runs/audit/{variant}/full' in export
            config_name = 'model_config.json' if variant == 'baseline' else 'model_config_hidden_params_x2.json'
            assert f'MODEL_CONFIG={HERE / config_name}' in export
            assert 'SOURCE_EPOCHS=12' in export
    assert not (tmp_path / 'runs').exists()
