"""SEED-IV adapter to the unchanged SICModelv15 calibration/LOSO pipeline."""
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


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--target',type=int,choices=range(15),required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    import tensorflow as tf
    from streaming import install_streaming_fit
    install_streaming_fit()
    # Keep the source snapshot first on PYTHONPATH; never import SICv16.
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_args import SICTrainingConfig
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_train import train_sic
    import eegproc.deep_learning.joint_architectures.SICModelv15.sic_model as v15
    # Avoid asynchronous faulthandler stack-dump signals: previous failures
    # aligned with their ten-minute timer. Read process memory from a daemon
    # thread instead, without traversing interpreter stacks.
    def memory_log():
        while True:
            status=Path('/proc/self/status').read_text().splitlines()
            print('MEMORY',*[s for s in status if s.startswith(('VmRSS:','VmHWM:'))],flush=True)
            time.sleep(300)
    threading.Thread(target=memory_log,daemon=True).start()
    original_graph = v15._source_only_mi_adjacency
    def timed_source_graph(*args, **kwargs):
        started = time.monotonic()
        print('SOURCE_MI_START', {k:v for k,v in kwargs.items() if k != 'training_features'}, flush=True)
        graph = original_graph(*args, **kwargs)
        print('SOURCE_MI_DONE seconds=', round(time.monotonic()-started,2), flush=True)
        return graph
    v15._source_only_mi_adjacency = timed_source_graph
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu,True)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Training must run inside a GPU allocation')
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    print('SICv15 source:',v15.__file__,flush=True)
    X,y,subjects,trials=load_data(a.data_dir,a.smoke)
    print('DATA',X.shape,'classes',np.bincount(y).tolist(),flush=True)
    assert set(np.unique(y))==set(range(4))
    assert all(len(np.unique(trials[subjects==s]))==(12 if a.smoke else 72) for s in range(15))
    config_path=Path(__file__).with_name('model_config.json')
    model=json.loads(config_path.read_text())
    if a.smoke:
        # Exercise the same estimator and graph shape without spending most
        # of a wiring test fitting 19,220 kNN MI estimates to 15,000 rows.
        model['mi_max_observations'] = 256
    config=SICTrainingConfig(
        output_dir=a.out_dir,run_name=f'sic_v15_seed_iv_subject_{a.target+1:02}',
        dataset='seed_iv',n_channels=62,n_bands=5,classification_level='window',
        source_epochs=1 if a.smoke else 12,source_batch_size=4 if a.smoke else 8,
        validation_subjects=2,early_stopping_patience=3,
        early_stopping_monitor='val_loss',early_stopping_mode='min',
        calibration_epochs=1 if a.smoke else 5,calibration_batch_size=8,
        calibration_levels=((4,1),) if a.smoke else ((4,3),(8,3),(12,3)),
        calibration_trials=4 if a.smoke else 12,calibration_folds=1 if a.smoke else 3,
        calibration_selection_shots=4 if a.smoke else 12,
        calibration_learning_rate=1e-4,calibration_weight_decay=5e-5,
        source_use_class_weight=True,selection_metric='balanced_accuracy',
        hyperparameter_selection_level='losocv',target_subjects=(a.target,),
        n_jobs=1,gpus_per_fold=1,cpus_per_worker=4,verbose=2,
        deterministic_training=True,seed=42,model_config=model)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/'adapter_protocol.json').write_text(json.dumps(dict(
        target_subject=a.target+1,source_module=v15.__file__,smoke=a.smoke,
        classes=['neutral','sad','fear','happy'],evaluation='4-second window',
        subject_split='LOSO across all three sessions; two source-only validation subjects',
        calibration_split='whole original movie trials; no shared windows across splits',
        hyperparameter_search=False,input_shape=list(X.shape),
        training_method=model['training_method']),indent=2))
    train_sic(X,y,subjects,trials,config)
    (a.out_dir/'SUCCESS').write_text('SICModelv15 SEED-IV completed\n')


if __name__=='__main__':
    main()
