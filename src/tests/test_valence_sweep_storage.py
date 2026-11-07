"""Storage guards must stop submission before any GPU jobs are allocated."""
import errno
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SIC = Path(__file__).resolve().parents[1] / 'eegproc/deep_learning/joint_architectures/SICModelv15'
sys.path.insert(0, str(SIC))
import valence_sweep_storage as storage


def test_home_and_symlink_to_home_are_rejected_before_writing(tmp_path, monkeypatch):
    home = tmp_path / 'home'; home.mkdir()
    monkeypatch.setattr(storage.Path, 'home', lambda: home)
    alias = tmp_path / 'alias'; alias.symlink_to(home, target_is_directory=True)
    for path in (home / 'new', alias / 'new'):
        with pytest.raises(ValueError, match='outside home'):
            storage.check_storage(path, tmp_path / 'logs', 0, 0)
    assert not (home / 'new').exists()


def test_explicit_storage_root_and_output_override(tmp_path):
    output, logs = storage.storage_paths(tmp_path / 'work')
    assert output == tmp_path / 'work/runs/sweeps/sic_v15_valence_focal_scale_recon'
    assert logs == output / 'logs'
    assert storage.storage_paths('/unused', tmp_path / 'results', tmp_path / 'logs') == (
        tmp_path / 'results', tmp_path / 'logs')


def test_default_work_path_never_falls_back_to_home(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.getpass, 'getuser', lambda: 'inserra')
    original = storage.Path.is_dir
    monkeypatch.setattr(storage.Path, 'is_dir', lambda p: False if str(p).startswith('/work/') else original(p))
    with pytest.raises(ValueError, match='/work/users/i/n/inserra'):
        storage.storage_paths()


def test_write_probe_catches_quota_failure_and_cleans_up(tmp_path, monkeypatch):
    def quota_exceeded(_):
        raise OSError(errno.EDQUOT, 'Disk quota exceeded')
    monkeypatch.setattr(storage.os, 'fsync', quota_exceeded)
    with pytest.raises(OSError) as failure:
        storage.check_storage(tmp_path / 'outputs', tmp_path / 'logs', 0, 0)
    assert failure.value.errno == errno.EDQUOT
    assert not list(tmp_path.rglob('.sweep-storage-*'))


def test_filesystem_capacity_and_inodes_are_independent_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1024**3))
    with pytest.raises(ValueError, match='filesystem space'):
        storage.check_storage(tmp_path / 'outputs', tmp_path / 'logs', 2, 0)
    monkeypatch.setattr(storage.os, 'statvfs', lambda p: SimpleNamespace(f_files=1000, f_favail=2))
    with pytest.raises(ValueError, match='filesystem inodes'):
        storage.check_storage(tmp_path / 'outputs', tmp_path / 'logs', 0, 3)


@pytest.mark.parametrize('gib,inodes', [(-1, 1), (float('nan'), 1), (float('inf'), 1), (0, -1)])
def test_invalid_minimums_are_rejected_without_creating_directories(tmp_path, gib, inodes):
    with pytest.raises(ValueError, match='minimums'):
        storage.check_storage(tmp_path / 'outputs', tmp_path / 'logs', gib, inodes)
    assert not (tmp_path / 'outputs').exists()


def test_log_quota_is_checked_even_when_output_is_writable(tmp_path, monkeypatch):
    original = storage.tempfile.NamedTemporaryFile
    def probe(*args, **kwargs):
        if kwargs['dir'] == tmp_path / 'logs':
            raise OSError(errno.EDQUOT, 'Disk quota exceeded')
        return original(*args, **kwargs)
    monkeypatch.setattr(storage.tempfile, 'NamedTemporaryFile', probe)
    with pytest.raises(OSError):
        storage.check_storage(tmp_path / 'outputs', tmp_path / 'logs', 0, 0)


def test_failed_preflight_submits_no_arrays(tmp_path):
    script = SIC / 'SLURM_scripts/full_run_v15_valence_focal_scale_recon_sweep.sh'
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    capture = tmp_path / 'submitted'
    sbatch = bin_dir / 'sbatch'
    sbatch.write_text(f'#!/bin/bash\ntouch "{capture}"\n')
    sbatch.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('SLURM_', 'SBATCH_')) and key not in ('PROJECT_DIR', 'LOG_DIR')}
    env.update(PATH=str(bin_dir) + os.pathsep + env['PATH'], OUTPUT_ROOT=str(tmp_path / 'outputs'),
               SUBMIT_PYTHON=sys.executable, MIN_FREE_GIB='1e30', MIN_FREE_INODES='0')
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Insufficient filesystem space' in result.stderr
    assert not capture.exists()
