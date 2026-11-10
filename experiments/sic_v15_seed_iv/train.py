"""Class-weighted SEED-IV source training, common-epoch accuracy selection, and calibration."""
import argparse
import json
import time
import threading
from pathlib import Path
import numpy as np


def load_data(root, smoke=False):
    shards = []
    for sid in range(15):
        folder = root/f'subject_{sid+1:02}'
        meta = json.loads((folder/'complete.json').read_text())
        assert meta['subject_index'] == sid and meta['shape'][1:] == [800,310]
        x = np.load(folder/'X.npy', mmap_mode='r')
        y = np.load(folder/'y.npy')
        trials = np.load(folder/'trials.npy')
        assert len(x) == len(y) == len(trials)
        assert len(np.unique(trials)) == 72
        if smoke:
            # Three complete trial identities/class, one window each; only a
            # wiring/optimization check, never a reportable accuracy estimate.
            chosen = []
            for c in range(4):
                for t in np.unique(trials[y==c])[:3]:
                    chosen.append(np.flatnonzero(trials==t)[0])
            idx = np.asarray(sorted(chosen))
        else:
            idx = None
        shards.append((x,y,trials,idx))
    total = sum(len(x) if idx is None else len(idx) for x,_,_,idx in shards)
    if smoke:
        X = np.empty((total,800,310), dtype=np.float32)
    else:
        X = np.load(root/'all_windows.npy', mmap_mode='r')
        assert X.shape == (total,800,310) and X.dtype == np.float32
    Y = np.empty(total, dtype=np.int64)
    S = np.empty(total, dtype=np.int64)
    T = np.empty(total, dtype=np.int64)
    offset=0
    for sid,(x,y,t,idx) in enumerate(shards):
        n = len(x) if idx is None else len(idx)
        sl=slice(offset,offset+n)
        if smoke:
            X[sl]=x[idx]
        Y[sl]=y if idx is None else y[idx]
        T[sl]=t if idx is None else t[idx]
        S[sl]=sid
        offset+=n
    return X,Y,S,T


def training_config(out_dir, target, model, *, smoke=False, source_epochs=12):
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_args import SICTrainingConfig
    return SICTrainingConfig(
        output_dir=out_dir, run_name=f'sic_v15_seed_iv_oracle_subject_{target+1:02}',
        dataset='seed_iv', n_channels=62, n_bands=5, classification_level='window',
        source_epochs=1 if smoke else source_epochs, source_batch_size=4 if smoke else 8,
        validation_subjects=2, early_stopping_patience=None, restore_best_weights=False,
        selection_metric='accuracy', hyperparameter_selection_level='losocv',
        source_use_class_weight=True, calibration_use_class_weight=True,
        calibration_epochs=1 if smoke else 5, calibration_batch_size=8,
        calibration_levels=((4,1),) if smoke else ((4,3),(8,3),(12,3)),
        calibration_trials=4 if smoke else 12, calibration_folds=1 if smoke else 3,
        calibration_selection_shots=4 if smoke else 12,
        calibration_learning_rate=1e-4, calibration_weight_decay=5e-5,
        target_subjects=(target,), n_jobs=1, gpus_per_fold=1, cpus_per_worker=4,
        verbose=2, deterministic_training=True, seed=42, model_config=model)


def main(argv=None):
    from oracle import data_identity, train_source, calibrate_selected, summarize
    from capacity import load_model_config, resolved_model_config
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('source','select','calibrate'), default='source')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--suite-dir', type=Path, required=True,
                   help='new shared output root for the entire oracle cohort')
    p.add_argument('--target', type=int, choices=range(15))
    p.add_argument('--targets', type=int, nargs='+', choices=range(15), default=list(range(15)),
                   help='complete cohort expected by the selection phase')
    p.add_argument('--source-epochs', type=int, default=12)
    p.add_argument('--model-config', type=Path,
                   default=Path(__file__).with_name('model_config.json'),
                   help='fixed baseline or hidden_params_x2 configuration')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(argv)
    if a.source_epochs < 1:
        p.error('--source-epochs must be positive')
    model = load_model_config(a.model_config)
    if a.smoke:
        model['mi_max_observations'] = 256
    if a.phase == 'select':
        result = summarize(a.suite_dir, a.targets, 1 if a.smoke else a.source_epochs,
                           expected_model_config=resolved_model_config(model))
        print(json.dumps({key: result.get(key) for key in
              ('status','best_epoch','selection_score','selection_rule')}, indent=2))
        return 0 if result['status'] == 'completed' else 2
    if a.data_dir is None or a.target is None:
        p.error('--data-dir and --target are required for source/calibrate')
    import tensorflow as tf
    from streaming import install_streaming_fit
    import eegproc.deep_learning.joint_architectures.SICModelv15.sic_model as v15
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        raise RuntimeError('Training/calibration must run inside a GPU allocation')
    for gpu in devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    install_streaming_fit()
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    def memory_log():
        while True:
            status = Path('/proc/self/status').read_text().splitlines()
            print('MEMORY', *[s for s in status if s.startswith(('VmRSS:','VmHWM:'))], flush=True)
            time.sleep(300)
    threading.Thread(target=memory_log, daemon=True).start()
    original_graph = v15._source_only_mi_adjacency
    def timed_source_graph(*args, **kwargs):
        started = time.monotonic()
        print('SOURCE_MI_START', {k:v for k,v in kwargs.items() if k != 'training_features'}, flush=True)
        graph = original_graph(*args, **kwargs)
        print('SOURCE_MI_DONE seconds=', round(time.monotonic()-started, 2), flush=True)
        return graph
    v15._source_only_mi_adjacency = timed_source_graph
    identity = data_identity(a.data_dir, smoke=a.smoke)
    x, y, subjects, trials = load_data(a.data_dir, a.smoke)
    if set(np.unique(y)) != set(range(4)):
        raise ValueError('SEED-IV requires all four emotion classes')
    config = training_config(a.suite_dir, a.target, model, smoke=a.smoke, source_epochs=a.source_epochs)
    print('SICv15 source:', v15.__file__, 'phase:', a.phase, 'data:', x.shape, flush=True)
    try:
        if a.phase == 'source':
            train_source(x, y, subjects, trials, config, a.suite_dir, dataset_id=identity, smoke=a.smoke)
        else:
            calibrate_selected(x, y, subjects, trials, config, a.suite_dir, dataset_id=identity)
    finally:
        v15._source_only_mi_adjacency = original_graph
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
