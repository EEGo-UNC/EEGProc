"""Validate the three-scale grid, source shards, common-epoch selection and launch."""
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SIC = ROOT / 'src/eegproc/deep_learning/joint_architectures/SICModelv15'
MANIFEST = SIC / 'sic_v15_valence_scale_configs.json'
SCRIPT = SIC / 'SLURM_scripts/full_run_v15_valence_focal_scale_recon_sweep.sh'
sys.path.insert(0, str(SIC))
from sic_v15_valence_sweep import configurations, select_oracle, write_json
from collect_valence_sweep_shards import collect, SHARD_TARGETS


def clean_environment():
    removed = {'PROJECT_DIR', 'SLURM_SUBMIT_DIR', 'SLURM_ARRAY_TASK_ID', 'SWEEP_SUITE_ID',
               'SOURCE_EPOCHS', 'SOURCE_BATCH_SIZE', 'CALIBRATION_EPOCHS', 'CALIBRATION_BATCH_SIZE',
               'TIME_BUDGET_SECONDS', 'OUTPUT_ROOT', 'MAX_CONCURRENT', 'RUN_CALIBRATION',
               'DRY_RUN', 'EBROOTCUDA', 'CUDA_HOME', 'GPU_PREFLIGHT', 'METRICS_ONLY'}
    removed.update({'SWEEP_STORAGE_ROOT', 'LOG_DIR', 'MIN_FREE_GIB', 'MIN_FREE_INODES',
                    'SBATCH_OUTPUT', 'SBATCH_ERROR', 'SBATCH_EXPORT'})
    return {key: value for key, value in os.environ.items() if key not in removed}


def test_three_scale_grid_fixes_gamma_reconstruction_and_episode_budget():
    records = configurations(MANIFEST)
    baseline = configurations(SIC / 'sic_v15_valence_run_5168991_12.json')[0]['hyperparameters']
    axes = ('focal_gamma', 'vc_logit_scale', 'reconstruction_loss_weight', 'mldg_steps_per_epoch')
    expected = list(itertools.product([.5], [8., 16., 32.], [.4], [8]))
    assert len(records) == 3
    assert [tuple(row['hyperparameters'][key] for key in axes) for row in records] == expected
    for row in records:
        config = row['hyperparameters']
        assert {key: value for key, value in config.items() if key not in axes} == {
            key: value for key, value in baseline.items() if key not in axes}
        assert config['subject_loss_weight'] == .3
    assert records[1]['hyperparameters'] == {**baseline, 'reconstruction_loss_weight': .4, 'mldg_steps_per_epoch': 8}
    legacy = configurations(SIC / 'sic_v15_valence_focal_scale_recon_configs.json')
    assert len(legacy) == 24
    assert [row['hyperparameters'] for row in records] == [legacy[i]['hyperparameters'] for i in (2, 6, 10)]
    assert sorted(user for group in SHARD_TARGETS for user in group) == list(range(23))


def test_dry_run_submits_nothing_and_shows_both_arrays(tmp_path):
    result = subprocess.run(['bash', str(SCRIPT)], cwd=tmp_path,
        env={**clean_environment(), 'DRY_RUN': '1', 'SUBMIT_PYTHON': sys.executable,
             'OUTPUT_ROOT': str(tmp_path / 'outputs'), 'MIN_FREE_GIB': '0', 'MIN_FREE_INODES': '0'},
        capture_output=True, text=True, check=True)
    assert '--array=0-8%2' in result.stdout
    assert '--array=0-2%2' in result.stdout
    assert '--dependency=afterany:SOURCE_ARRAY_ID' in result.stdout
    assert '3 configurations; 9 source tasks' in result.stdout
    assert 'Gamma 0.5; scale 8/16/32; reconstruction 0.4; MLDG episodes 8; subject loss 0.3' in result.stdout
    assert 'Outer/source epochs: 48' in result.stdout
    assert 'calibration enabled: 0' in result.stdout
    assert 'Metrics only: 1' in result.stdout
    assert '24:00:00 per task' in result.stdout
    assert str(tmp_path / 'outputs/logs/v15_valence_focal_scale_recon_%A_%a.out') in result.stdout
    assert str(tmp_path / 'outputs/logs/v15_valence_collect_%A_%a.err') in result.stdout


def test_submission_connects_collector_to_source_array(tmp_path):
    binaries = tmp_path / 'bin'; binaries.mkdir()
    capture = tmp_path / 'submission.jsonl'
    sbatch = binaries / 'sbatch'
    sbatch.write_text(f'#!{sys.executable}\n' + """
import json, os, pathlib, sys
with pathlib.Path(os.environ['SUBMISSION_CAPTURE']).open('a') as handle:
    handle.write(json.dumps({'args': sys.argv[1:], 'project': os.environ['PROJECT_DIR'],
                             'epochs': os.environ['SOURCE_EPOCHS'], 'metrics_only': os.environ['METRICS_ONLY'],
                             'calibration': os.environ['RUN_CALIBRATION']}) + '\\n')
print('7001;longleaf' if any(arg.startswith('--dependency=') for arg in sys.argv) else '7000;longleaf')
""")
    sbatch.chmod(0o755)
    result = subprocess.run(['bash', str(SCRIPT)], cwd=tmp_path, env={
        **clean_environment(), 'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
        'SUBMISSION_CAPTURE': str(capture), 'SUBMIT_PYTHON': sys.executable,
        'OUTPUT_ROOT': str(tmp_path / 'outputs'), 'LOG_DIR': str(tmp_path / 'logs'),
        'MIN_FREE_GIB': '0', 'MIN_FREE_INODES': '0'},
        capture_output=True, text=True, check=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 2
    assert '--array=0-8%2' in calls[0]['args']
    assert '--dependency=afterany:7000' in calls[1]['args']
    assert '--array=0-2%2' in calls[1]['args']
    assert '--export=ALL,SWEEP_SUITE_ID=7000' in calls[1]['args']
    assert '--gres=gpu:2' in calls[1]['args']
    assert calls[1]['args'][-1] == '--collect'
    assert all(call['epochs'] == '48' and call['project'] == str(ROOT) for call in calls)
    assert all(call['metrics_only'] == '1' and call['calibration'] == '0' for call in calls)
    for call, name in zip(calls, ('v15_valence_focal_scale_recon', 'v15_valence_collect')):
        assert f'--output={tmp_path}/logs/{name}_%A_%a.out' in call['args']
        assert f'--error={tmp_path}/logs/{name}_%A_%a.err' in call['args']
        assert (tmp_path / 'logs').is_dir()


@pytest.mark.parametrize('phase,index', [('--task', '-1'), ('--task', '9'), ('--task', 'abc'),
                                       ('--task', '01'), ('--collect', '3')])
def test_invalid_indices_fail_before_cluster_setup(phase, index):
    result = subprocess.run(['bash', str(SCRIPT), phase], env={
        **clean_environment(), 'SLURM_ARRAY_TASK_ID': index, 'SWEEP_SUITE_ID': '7000'},
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'module: command not found' not in result.stderr


def cluster_environment(tmp_path):
    project = tmp_path / 'project checkout'
    sic = project / SIC.relative_to(ROOT); sic.mkdir(parents=True)
    (sic / MANIFEST.name).write_bytes(MANIFEST.read_bytes())
    (sic / 'valence_sweep_storage.py').write_bytes((SIC / 'valence_sweep_storage.py').read_bytes())
    venv = project / 'venv312'; binaries = venv / 'bin'; binaries.mkdir(parents=True)
    python = binaries / 'python'
    python.write_text(f'#!{sys.executable}\n' + """
import json, os, pathlib, sys
if len(sys.argv) > 1 and sys.argv[1].endswith(('sic_v15_valence_sweep.py', 'collect_valence_sweep_shards.py')):
    with pathlib.Path(os.environ['TRAINING_CAPTURE']).open('a') as handle:
        handle.write(json.dumps(sys.argv[1:]) + '\\n')
    if sys.argv[1].endswith('collect_valence_sweep_shards.py'):
        sys.exit(int(os.environ.get('COLLECT_RETURN_CODE', '0')))
    if sys.argv[2] == 'run':
        sys.exit(int(os.environ.get('SOURCE_RETURN_CODE', '0')))
elif len(sys.argv) > 1 and sys.argv[1] == '-':
    code = sys.stdin.read()
    if 'import tensorflow as tf' not in code:
        sys.argv = sys.argv[1:]
        exec(compile(code, '<launcher>', 'exec'), {'__name__': '__main__'})
elif sys.argv[1:3] == ['-m', 'src.tests.test_sic_v15_multi_gpu']:
    pass
else:
    os.execv(os.environ['REAL_TEST_PYTHON'], [os.environ['REAL_TEST_PYTHON']] + sys.argv[1:])
""")
    python.chmod(0o755)
    for name in ('module', 'nvidia-smi', 'git'):
        command = binaries / name; command.write_text('#!/bin/bash\nexit 0\n'); command.chmod(0o755)
    (binaries / 'activate').write_text(f'export VIRTUAL_ENV={shlex.quote(str(venv))}\n'
                                      f'export PATH={shlex.quote(str(binaries))}:"$PATH"\n')
    libdevice = venv / 'lib/python3.12/site-packages/nvidia/nvvm/libdevice/libdevice.10.bc'
    libdevice.parent.mkdir(parents=True); libdevice.touch()
    return {**clean_environment(), 'PROJECT_DIR': str(project), 'VENV_DIR': str(venv),
            'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
            'SLURM_ARRAY_JOB_ID': '7000', 'SLURM_JOB_ID': '7000',
            'TRAINING_SEED': '99', 'REAL_TEST_PYTHON': sys.executable,
            'LOG_DIR': str(project / 'logs'), 'MIN_FREE_GIB': '0', 'MIN_FREE_INODES': '0',
            'OUTPUT_ROOT': str(project / 'outputs'), 'TIME_BUDGET_SECONDS': '70000'}


@pytest.mark.parametrize('index', [0, 1, 2, 3, 8])
def test_source_tasks_route_to_exact_configuration_and_user_shard(tmp_path, index):
    capture = tmp_path / 'training.jsonl'
    env = {**cluster_environment(tmp_path), 'SLURM_ARRAY_TASK_ID': str(index),
           'TRAINING_CAPTURE': str(capture)}
    result = subprocess.run(['bash', str(SCRIPT), '--task'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1
    args = calls[0]
    value = lambda flag: args[args.index(flag) + 1]
    assert args[1] == 'run'
    assert value('--config-index') == str(index // 3)
    assert value('--output-dir').endswith(f'/suite_7000/configuration_{index // 3}/shard_{index % 3}')
    assert [int(user) for user in args[args.index('--target-subjects') + 1:]] == list(SHARD_TARGETS[index % 3])
    assert value('--source-epochs') == '48'
    assert value('--source-batch-size') == '64'
    assert value('--time-budget-seconds') == '70000'
    assert value('--seed') == '99' and '--deterministic-training' in args
    assert '--metrics-only' in args
    config = json.loads(next(line.split(': ', 1)[1] for line in result.stdout.splitlines()
                             if line.startswith('Selected configuration: ')))['hyperparameters']
    assert config == configurations(MANIFEST, seed=99)[index // 3]['hyperparameters']


@pytest.mark.parametrize('collector_code,calibrate', [(0, '1'), (0, '0'), (2, '1')])
def test_collector_routes_only_successful_global_selection_to_calibration(tmp_path, collector_code, calibrate):
    capture = tmp_path / 'collect.jsonl'
    env = {**cluster_environment(tmp_path), 'SLURM_ARRAY_TASK_ID': '1', 'SWEEP_SUITE_ID': '7000',
           'TRAINING_CAPTURE': str(capture), 'COLLECT_RETURN_CODE': str(collector_code),
           'RUN_CALIBRATION': calibrate, 'METRICS_ONLY': '0'}
    result = subprocess.run(['bash', str(SCRIPT), '--collect'], env=env, capture_output=True, text=True)
    assert result.returncode == collector_code, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == (2 if collector_code == 0 and calibrate == '1' else 1)
    assert calls[0][0].endswith('collect_valence_sweep_shards.py')
    assert calls[0][1].endswith('/suite_7000/configuration_1')
    if len(calls) == 2:
        assert calls[1][1:3] == ['calibrate', calls[0][1]]
        assert calls[1][-4:] == ['--calibration-epochs', '1', '--calibration-batch-size', '64']


def test_full_data_flag_retains_original_source_mode(tmp_path):
    capture = tmp_path / 'training.jsonl'
    result = subprocess.run(['bash', str(SCRIPT), '--task'], env={
        **cluster_environment(tmp_path), 'SLURM_ARRAY_TASK_ID': '0', 'METRICS_ONLY': '0',
        'TRAINING_CAPTURE': str(capture)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(capture.read_text())
    assert '--metrics-only' not in args
    assert 'calibration enabled: 1' in result.stdout


def test_metrics_only_collector_skips_models_and_calibration(tmp_path):
    capture = tmp_path / 'collect.jsonl'
    env = {**cluster_environment(tmp_path), 'SLURM_ARRAY_TASK_ID': '1', 'SWEEP_SUITE_ID': '7000',
           'TRAINING_CAPTURE': str(capture)}
    # Metrics-only selection needs neither libdevice nor TensorFlow.
    next(Path(env['VENV_DIR']).rglob('libdevice.10.bc')).unlink()
    result = subprocess.run(['bash', str(SCRIPT), '--collect'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1 and calls[0][0].endswith('collect_valence_sweep_shards.py')


def test_metrics_only_cannot_request_calibration(tmp_path):
    result = subprocess.run(['bash', str(SCRIPT)], env={
        **clean_environment(), 'METRICS_ONLY': '1', 'RUN_CALIBRATION': '1'},
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'cannot calibrate unsaved models' in result.stderr


def make_shards(tmp_path, metrics_only=False):
    config = configurations(MANIFEST)[1]
    for shard_number, targets in enumerate(SHARD_TARGETS):
        shard = tmp_path / f'shard_{shard_number}'
        write_json(shard / 'task.json', {'configuration': config, 'source_epochs': 3,
            'source_batch_size': 64, 'seed': 42, 'gpus_per_fold': 2, 'deterministic_training': True,
            'target_subjects': list(targets), 'eeg': '/data/eeg.npy', 'labels': '/data/labels.npy',
            'preprocessing': 'same full run', 'metrics_only': metrics_only})
        write_json(shard / 'execution.json', {'status': 'completed', 'failures': [], 'interrupted': False})
        scores = (.9, .5, .8) if shard_number == 0 else (.4, .8, .5)
        for user in targets:
            for epoch, score in enumerate(scores, 1):
                checkpoint = shard / f'subject_{user:02d}/epoch_{epoch:03d}.keras'
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                record = {'target_subject': user, 'epoch': epoch, 'balanced_accuracy': score,
                          'accuracy': .4 if epoch == 2 else .99, 'decision_threshold': .5}
                if not metrics_only:
                    checkpoint.touch()
                    record['checkpoint'] = str(checkpoint.relative_to(shard))
                write_json(checkpoint.with_suffix('.json'), record)
    return tmp_path


def test_global_selection_uses_all_23_users_not_local_shard_winners(tmp_path):
    output = make_shards(tmp_path)
    result = collect(output)
    assert result['best_epoch'] == 2
    assert result['selection_score'] == pytest.approx((8 * .5 + 15 * .8) / 23)
    assert result['target_subjects'] == list(range(23))
    assert len(result['selected_models']) == 23
    for user, checkpoint in result['selected_models'].items():
        assert checkpoint == f'shard_{int(user) // 8}/subject_{int(user):02d}/epoch_002.keras'
        assert (output / checkpoint).is_file()
    assert len(list(output.glob('subject_*/epoch_*.json'))) == 23 * 3
    assert not list(output.glob('subject_*/*.keras'))


def test_metrics_only_collection_uses_balanced_accuracy_at_one_common_epoch(tmp_path):
    from audit_valence_sweep import audit_suite
    output = make_shards(tmp_path / 'configuration_1', metrics_only=True)
    result = collect(output)
    assert result['best_epoch'] == 2
    assert result['selection_score'] == pytest.approx((8 * .5 + 15 * .8) / 23)
    assert result['selected_mean_accuracy'] == pytest.approx(.4)
    assert len(result['selected_subject_metrics']) == 23
    assert all(row['epoch'] == 2 for row in result['selected_subject_metrics'].values())
    assert 'selected_models' not in result
    assert not (output / 'oracle_models.json').exists()
    assert not list(output.rglob('*.keras'))
    final = json.loads((output / 'final_metrics.json').read_text())
    assert final['selected_epoch'] == 2 and final['metrics_only']
    assert final['mean_balanced_accuracy'] == result['selection_score']
    assert final['mean_accuracy'] == pytest.approx(.4)
    assert final['subject_metrics'] == result['selected_subject_metrics']
    audit = audit_suite(tmp_path, MANIFEST, verify_checkpoints=True)
    assert all(row['ready_for_collector'] and row['metrics_only'] for row in audit['shards'])


def test_audit_preserves_results_and_distinguishes_missing_execution_status(tmp_path):
    from audit_valence_sweep import audit_suite
    configuration = make_shards(tmp_path / 'configuration_1')
    (configuration / 'shard_1/execution.json').unlink()
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = audit_suite(tmp_path, MANIFEST)
    rows = {row['task_id']: row for row in result['shards']}
    assert rows[3]['ready_for_collector']
    assert rows[4]['complete_epoch_files']
    assert rows[4]['execution_status'] == 'missing'
    assert not rows[4]['ready_for_collector']
    assert len(result['missing_tasks']) == 6
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_audit_full_archive_check_rejects_truncated_checkpoints(tmp_path):
    import zipfile
    from audit_valence_sweep import audit_suite
    configuration = make_shards(tmp_path / 'configuration_1')
    for path in configuration.rglob('*.keras'):
        with zipfile.ZipFile(path, 'w') as archive:
            for member in ('config.json', 'metadata.json', 'model.weights.h5'):
                archive.writestr(member, 'test fixture')
    checkpoint = configuration / 'shard_1/subject_08/epoch_001.keras'
    checkpoint.write_bytes(checkpoint.read_bytes()[:20])
    rows = {row['task_id']: row for row in audit_suite(tmp_path, MANIFEST, True)['shards']}
    assert rows[3]['ready_for_collector']
    assert not rows[4]['ready_for_collector']
    assert rows[4]['errors']


@pytest.mark.parametrize('failure', ['missing_shard', 'missing_epoch', 'missing_model',
                                    'failed_worker', 'budget_mismatch', 'configuration_mismatch', 'mode_mismatch'])
def test_partial_or_mixed_shards_cannot_publish_a_winner(tmp_path, failure):
    output = make_shards(tmp_path)
    shard = output / 'shard_1'
    if failure == 'missing_shard':
        (shard / 'execution.json').unlink()
    elif failure == 'missing_epoch':
        (shard / 'subject_08/epoch_003.json').unlink()
    elif failure == 'missing_model':
        (shard / 'subject_08/epoch_003.keras').unlink()
    elif failure == 'failed_worker':
        write_json(shard / 'execution.json', {'status': 'completed', 'failures': [{'returncode': 1}]})
    else:
        path = shard / 'task.json'; spec = json.loads(path.read_text())
        if failure == 'budget_mismatch':
            spec['source_epochs'] = 2
        elif failure == 'mode_mismatch':
            spec['metrics_only'] = True
        else:
            spec['configuration']['hyperparameters']['mldg_steps_per_epoch'] = 16
        write_json(path, spec)
    with pytest.raises(ValueError):
        collect(output)
    assert not (output / 'oracle_models.json').exists()
