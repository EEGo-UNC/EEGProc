"""Validate the extended grid, source shards, common-epoch selection and launch."""
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
MANIFEST = SIC / 'sic_v15_valence_focal_scale_recon_configs.json'
SCRIPT = SIC / 'SLURM_scripts/full_run_v15_valence_focal_scale_recon_sweep.sh'
sys.path.insert(0, str(SIC))
from sic_v15_valence_sweep import configurations, select_oracle, write_json
from collect_valence_sweep_shards import collect, SHARD_TARGETS


def clean_environment():
    removed = {'PROJECT_DIR', 'SLURM_SUBMIT_DIR', 'SLURM_ARRAY_TASK_ID', 'SWEEP_SUITE_ID',
               'SOURCE_EPOCHS', 'SOURCE_BATCH_SIZE', 'CALIBRATION_EPOCHS', 'CALIBRATION_BATCH_SIZE',
               'TIME_BUDGET_SECONDS', 'OUTPUT_ROOT', 'MAX_CONCURRENT', 'RUN_CALIBRATION',
               'DRY_RUN', 'EBROOTCUDA', 'CUDA_HOME', 'GPU_PREFLIGHT'}
    return {key: value for key, value in os.environ.items() if key not in removed}


def test_complete_cartesian_grid_keeps_run12_baseline():
    records = configurations(MANIFEST)
    baseline = configurations(SIC / 'sic_v15_valence_run_5168991_12.json')[0]['hyperparameters']
    axes = ('focal_gamma', 'vc_logit_scale', 'reconstruction_loss_weight', 'mldg_steps_per_epoch')
    expected = list(itertools.product([.5, 1.], [8., 16., 32.], [0., .4], [5, 10, 20]))
    assert len(records) == 36
    assert [tuple(row['hyperparameters'][key] for key in axes) for row in records] == expected
    for row in records:
        config = row['hyperparameters']
        assert {key: value for key, value in config.items() if key not in axes} == {
            key: value for key, value in baseline.items() if key not in axes}
        assert config['subject_loss_weight'] == .3
    assert records[7]['hyperparameters'] == baseline
    assert records[25]['hyperparameters'] == {**baseline, 'focal_gamma': 1.}
    assert sorted(user for group in SHARD_TARGETS for user in group) == list(range(23))


def test_dry_run_submits_nothing_and_shows_both_arrays(tmp_path):
    result = subprocess.run(['bash', str(SCRIPT)], cwd=tmp_path,
        env={**clean_environment(), 'DRY_RUN': '1', 'SUBMIT_PYTHON': sys.executable},
        capture_output=True, text=True, check=True)
    assert '--array=0-107%2' in result.stdout
    assert '--array=0-35%2' in result.stdout
    assert '--dependency=afterany:SOURCE_ARRAY_ID' in result.stdout
    assert '36 configurations; 108 source tasks' in result.stdout
    assert 'Outer/source epochs: 24' in result.stdout
    assert '24:00:00 per task' in result.stdout


def test_submission_connects_collector_to_source_array(tmp_path):
    binaries = tmp_path / 'bin'; binaries.mkdir()
    capture = tmp_path / 'submission.jsonl'
    sbatch = binaries / 'sbatch'
    sbatch.write_text(f'#!{sys.executable}\n' + """
import json, os, pathlib, sys
with pathlib.Path(os.environ['SUBMISSION_CAPTURE']).open('a') as handle:
    handle.write(json.dumps({'args': sys.argv[1:], 'project': os.environ['PROJECT_DIR'],
                             'epochs': os.environ['SOURCE_EPOCHS']}) + '\\n')
print('7001;longleaf' if any(arg.startswith('--dependency=') for arg in sys.argv) else '7000;longleaf')
""")
    sbatch.chmod(0o755)
    result = subprocess.run(['bash', str(SCRIPT)], cwd=tmp_path, env={
        **clean_environment(), 'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
        'SUBMISSION_CAPTURE': str(capture), 'SUBMIT_PYTHON': sys.executable},
        capture_output=True, text=True, check=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 2
    assert '--array=0-107%2' in calls[0]['args']
    assert '--dependency=afterany:7000' in calls[1]['args']
    assert '--export=ALL,SWEEP_SUITE_ID=7000' in calls[1]['args']
    assert '--gres=gpu:2' in calls[1]['args']
    assert calls[1]['args'][-1] == '--collect'
    assert all(call['epochs'] == '24' and call['project'] == str(ROOT) for call in calls)


@pytest.mark.parametrize('phase,index', [('--task', '-1'), ('--task', '108'), ('--task', 'abc'),
                                       ('--task', '01'), ('--collect', '36')])
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
            'OUTPUT_ROOT': str(project / 'outputs'), 'TIME_BUDGET_SECONDS': '70000'}


@pytest.mark.parametrize('index', [0, 1, 2, 21, 107])
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
    assert value('--source-epochs') == '24'
    assert value('--source-batch-size') == '64'
    assert value('--time-budget-seconds') == '70000'
    assert value('--seed') == '99' and '--deterministic-training' in args
    config = json.loads(next(line.split(': ', 1)[1] for line in result.stdout.splitlines()
                             if line.startswith('Selected configuration: ')))['hyperparameters']
    assert config == configurations(MANIFEST, seed=99)[index // 3]['hyperparameters']


@pytest.mark.parametrize('collector_code,calibrate', [(0, '1'), (0, '0'), (2, '1')])
def test_collector_routes_only_successful_global_selection_to_calibration(tmp_path, collector_code, calibrate):
    capture = tmp_path / 'collect.jsonl'
    env = {**cluster_environment(tmp_path), 'SLURM_ARRAY_TASK_ID': '7', 'SWEEP_SUITE_ID': '7000',
           'TRAINING_CAPTURE': str(capture), 'COLLECT_RETURN_CODE': str(collector_code),
           'RUN_CALIBRATION': calibrate}
    result = subprocess.run(['bash', str(SCRIPT), '--collect'], env=env, capture_output=True, text=True)
    assert result.returncode == collector_code, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == (2 if collector_code == 0 and calibrate == '1' else 1)
    assert calls[0][0].endswith('collect_valence_sweep_shards.py')
    assert calls[0][1].endswith('/suite_7000/configuration_7')
    if len(calls) == 2:
        assert calls[1][1:3] == ['calibrate', calls[0][1]]
        assert calls[1][-4:] == ['--calibration-epochs', '1', '--calibration-batch-size', '64']


def make_shards(tmp_path):
    config = configurations(MANIFEST)[7]
    for shard_number, targets in enumerate(SHARD_TARGETS):
        shard = tmp_path / f'shard_{shard_number}'
        write_json(shard / 'task.json', {'configuration': config, 'source_epochs': 3,
            'source_batch_size': 64, 'seed': 42, 'gpus_per_fold': 2, 'deterministic_training': True,
            'target_subjects': list(targets), 'eeg': '/data/eeg.npy', 'labels': '/data/labels.npy',
            'preprocessing': 'same full run'})
        write_json(shard / 'execution.json', {'status': 'completed', 'failures': [], 'interrupted': False})
        scores = (.9, .5, .8) if shard_number == 0 else (.4, .8, .5)
        for user in targets:
            for epoch, score in enumerate(scores, 1):
                checkpoint = shard / f'subject_{user:02d}/epoch_{epoch:03d}.keras'
                checkpoint.parent.mkdir(parents=True, exist_ok=True); checkpoint.touch()
                write_json(checkpoint.with_suffix('.json'), {'target_subject': user, 'epoch': epoch,
                    'balanced_accuracy': score, 'decision_threshold': .5,
                    'checkpoint': str(checkpoint.relative_to(shard))})
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


@pytest.mark.parametrize('failure', ['missing_shard', 'missing_epoch', 'missing_model',
                                    'failed_worker', 'budget_mismatch', 'configuration_mismatch'])
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
        else:
            spec['configuration']['hyperparameters']['mldg_steps_per_epoch'] = 20
        write_json(path, spec)
    with pytest.raises(ValueError):
        collect(output)
    assert not (output / 'oracle_models.json').exists()
