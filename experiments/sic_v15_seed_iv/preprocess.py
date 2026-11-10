"""Convert SEED-IV MATLAB recordings using EEGProc's signal preprocessor.

One output shard per real participant; sessions never become fake subjects.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

from eegproc.preprocessing import bandpass_filter

LABELS = (
    (1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3),
    (2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1),
    (1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0),
)
CLASSES = ('neutral', 'sad', 'fear', 'happy')
BANDS = dict(delta=(1,4), theta=(4,8), alpha=(8,14), beta=(14,31), gamma=(31,50))
FS = 200
WINDOW = 800


def trial_fields(path):
    fields = []
    for name, shape, _ in whosmat(path):
        match = re.fullmatch(r'.*_eeg(\d+)', name)
        if match:
            assert len(shape) == 2 and shape[0] == 62, (name, shape)
            fields.append((int(match[1]), name, shape[1]))
    fields.sort()
    assert [x[0] for x in fields] == list(range(1,25)), path
    return fields


def preprocess_trial(signal):
    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[0] != 62 or not np.isfinite(signal).all():
        raise ValueError('Expected finite EEG with 62 rows in original channel order')
    frame = pd.DataFrame(signal.T, columns=[f'ch{i:02}' for i in range(62)])
    filtered = bandpass_filter(frame, FS, bands=BANDS, notch_hz=None,
                              reref=True, detrend=True, order=4)
    # EEGProc orders columns channel-major, band-minor, matching the GCN.
    values = filtered.to_numpy(dtype=np.float32)
    count = len(values)//WINDOW
    if not count:
        raise ValueError('Trial shorter than four seconds')
    windows = values[:count*WINDOW].reshape(count, WINDOW, 310)
    rms = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=(1,2), keepdims=True))
    windows = (windows / np.maximum(rms, 1e-6)).astype(np.float32)
    assert np.isfinite(windows).all()
    return windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--subject', type=int, required=True, choices=range(1,16))
    a = p.parse_args()
    dest = a.out_dir/f'subject_{a.subject:02}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        print('Already complete:', dest, flush=True)
        return
    records = []
    for session in range(1,4):
        paths = list((a.raw_dir/str(session)).glob(f'{a.subject}_*.mat'))
        assert len(paths) == 1, paths
        records.append((session, paths[0], trial_fields(paths[0])))
    total = sum(n//WINDOW for _,_,fields in records for _,_,n in fields)
    x = np.lib.format.open_memmap(dest/'X.npy', mode='w+', dtype='float32', shape=(total, WINDOW,310))
    y = np.empty(total, dtype=np.int64)
    trials = np.empty(total, dtype=np.int64)
    sessions = np.empty(total, dtype=np.int64)
    manifest = []
    offset = 0
    for session, path, fields in records:
        mat = loadmat(path, variable_names=[name for _,name,_ in fields])
        for trial, name, n in fields:
            w = preprocess_trial(mat.pop(name))
            count = len(w)
            label = LABELS[session-1][trial-1]
            sl = slice(offset, offset+count)
            x[sl] = w
            y[sl] = label
            trials[sl] = (session-1)*24+trial-1
            sessions[sl] = session
            manifest.append(dict(session=session, trial=trial, class_id=label,
                                 source=str(path), field=name, samples=n,
                                 windows=count, dropped_tail_samples=n%WINDOW))
            offset += count
            print(f'subject={a.subject} session={session} trial={trial} windows={count}', flush=True)
    assert offset == total
    x.flush()
    np.save(dest/'y.npy', y)
    np.save(dest/'trials.npy', trials)
    np.save(dest/'sessions.npy', sessions)
    counts = np.bincount([r['class_id'] for r in manifest], minlength=4)
    assert counts.tolist() == [18]*4
    meta = dict(subject=a.subject, subject_index=a.subject-1, fs=FS, window_samples=WINDOW,
                window_seconds=4, overlap=0, channels=62, bands=BANDS, classes=CLASSES,
                normalization='per-window global RMS', reference='common average',
                notch_hz=None, filter_order=4, filter='EEGProc zero-phase Butterworth SOS',
                shape=list(x.shape), trial_class_counts=counts.tolist(), trials=manifest,
                label_source='Kaggle phhasian0710/seed-iv ReadMe.txt',
                dataset_source='https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html')
    (dest/'complete.json').write_text(json.dumps(meta, indent=2))
    print(f'COMPLETE {dest} {x.shape}', flush=True)


if __name__ == '__main__':
    main()
