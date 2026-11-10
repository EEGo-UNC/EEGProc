"""Accuracy-based common-epoch oracle selection for SEED-IV SICv15."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ORACLE_NOTE = 'Held-out labels select one common source epoch; scores are oracle estimates.'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_identity():
    from eegproc.deep_learning.joint_architectures.SICModelv15 import sic_model as v15
    model_path = Path(v15.__file__)
    deep_learning = model_path.parents[2]
    paths = [Path(__file__), Path(__file__).with_name('train.py'), Path(__file__).with_name('streaming.py'),
             Path(__file__).with_name('capacity.py'),
             model_path, model_path.parents[1] / 'sic_common/model.py',
             deep_learning / 'unsupervised/GNN/GCNMTL.py',
             deep_learning / 'unsupervised/GNN/GraphConvMTL.py',
             deep_learning / 'supervised/variational_classifier.py',
             deep_learning / 'cross_validation/calibration_subject.py']
    return fingerprint([file_hash(path) for path in paths])


def data_identity(root, *, smoke=False):
    """Bind all fold protocols to the same preprocessing generation/cache."""
    root = Path(root).resolve()
    records = []
    for sid in range(1, 16):
        folder = root / f'subject_{sid:02}'
        meta_path = folder / 'complete.json'
        meta = json.loads(meta_path.read_text())
        expected_bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 14], beta=[14, 31], gamma=[31, 50])
        if (meta.get('fs') != 200 or meta.get('channels') != 62
                or meta.get('bands') != expected_bands or meta.get('window_samples') != 800):
            raise ValueError(f'Incompatible SEED-IV preprocessing metadata: {meta_path}')
        for name in ('X.npy', 'y.npy', 'trials.npy'):
            path = folder / name
            stat = path.stat()
            records.append(dict(path=str(path.relative_to(root)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        records.append(dict(path=str(meta_path.relative_to(root)), sha256=file_hash(meta_path)))
    if not smoke:
        cache = root / 'all_windows.npy'
        stat = cache.stat()
        records.append(dict(path=cache.name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return fingerprint(records)


def prediction_accuracy(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if (labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or np.any(labels > 3)):
        raise ValueError('Expected nonempty integer SEED-IV labels in 0..3')
    if (probabilities.shape != (len(labels), 4) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('Expected finite normalized four-class probabilities')
    correct = int(np.sum(probabilities.argmax(axis=1) == labels))
    return correct / len(labels), correct


def choose_epoch(rows, targets, epochs):
    """Equal participant means at one epoch; exact ties choose the earliest."""
    targets = [int(value) for value in targets]
    if not targets or len(set(targets)) != len(targets) or epochs < 1:
        raise ValueError('Expected distinct targets and a positive epoch budget')
    indexed = {}
    for row in rows:
        key = (int(row['target_subject']), int(row['epoch']))
        accuracy = float(row['accuracy'])
        if key[0] not in targets or not 1 <= key[1] <= epochs or key in indexed:
            raise ValueError(f'Duplicate or unexpected subject/epoch: {key}')
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f'Invalid accuracy: {accuracy}')
        indexed[key] = row
    missing = [[target, epoch] for target in targets for epoch in range(1, epochs + 1)
               if (target, epoch) not in indexed]
    means = [dict(epoch=epoch, mean_accuracy=math.fsum(
        float(indexed[target, epoch]['accuracy']) for target in targets) / len(targets))
        for epoch in range(1, epochs + 1)
        if all((target, epoch) in indexed for target in targets)]
    result = dict(status='incomplete' if missing else 'completed', oracle=True,
                  metric='accuracy', evaluation_level='window', target_subjects=targets,
                  source_epochs=epochs, selection_rule='max_epoch(mean_subject(window_accuracy)); earliest tie',
                  evaluation_note=ORACLE_NOTE, missing_subject_epochs=missing, epoch_means=means)
    if not missing:
        best = max(means, key=lambda row: (row['mean_accuracy'], -row['epoch']))
        result.update(best_epoch=best['epoch'], selection_score=best['mean_accuracy'],
            selected_models={str(target): indexed[target, best['epoch']]['checkpoint'] for target in targets},
            selected_accuracies={str(target): float(indexed[target, best['epoch']]['accuracy']) for target in targets})
    return result


def summarize(suite, targets=tuple(range(15)), epochs=12, *, expected_model_config=None):
    suite = Path(suite).resolve()
    suite.mkdir(parents=True, exist_ok=True)
    targets = tuple(targets)
    rows, common, partitions = [], None, {}
    models_path = suite / 'oracle_models.json'
    if models_path.exists():
        models_path.unlink()
    for target in targets:
        folder = suite / f'subject_{int(target):02}'
        if not folder.exists():
            continue
        meta = json.loads((folder / 'protocol.json').read_text())
        protocol = meta['protocol']
        if expected_model_config is not None and protocol.get('model_config') != expected_model_config:
            raise ValueError('Source fold does not match the requested model configuration')
        if (meta['target_subject'] != int(target) or protocol['source_epochs'] != epochs
                or fingerprint(protocol) != meta['protocol_id']):
            raise ValueError(f'Unexpected source protocol: {folder}')
        if common is not None and protocol != common:
            raise ValueError('Cannot combine folds from different data or training protocols')
        common = protocol
        partitions[str(target)] = {key: meta.get(key, []) for key in ('source_subjects', 'validation_subjects')}
        for path in sorted(folder.glob('epoch_*.json')):
            row = json.loads(path.read_text())
            if row['target_subject'] != int(target) or row['protocol_id'] != meta['protocol_id']:
                raise ValueError(f'Epoch protocol mismatch: {path}')
            checkpoint = suite / row['checkpoint']
            if (checkpoint.parent != folder or not checkpoint.is_file()
                    or file_hash(checkpoint) != row['checkpoint_sha256']):
                raise ValueError(f'Missing or changed checkpoint: {checkpoint}')
            archive = suite / row['predictions']
            if archive.parent != folder:
                raise ValueError('Prediction archive must belong to its source fold')
            with np.load(archive, allow_pickle=False) as saved:
                accuracy, correct = prediction_accuracy(saved['y_true'], saved['probabilities'])
                n_windows = len(saved['y_true'])
                if len(saved['trial_ids']) != n_windows:
                    raise ValueError('Prediction trial IDs do not align')
            if (accuracy != row['accuracy'] or correct != row['n_correct']
                    or n_windows != row['n_windows']):
                raise ValueError(f'Accuracy does not match archived predictions: {path}')
            rows.append(row)
    result = choose_epoch(rows, targets, epochs)
    result['protocol'] = common
    result['protocol_id'] = None if common is None else fingerprint(common)
    result['source_partitions'] = partitions
    if result['status'] == 'completed':
        selected = {row['target_subject']: row for row in rows if row['epoch'] == result['best_epoch']}
        result['selected_checkpoint_hashes'] = {str(target): selected[int(target)]['checkpoint_sha256'] for target in targets}
    write_json(suite / 'oracle_summary.json', result)
    with (suite / 'oracle_epoch_means.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'mean_accuracy'])
        writer.writeheader()
        writer.writerows(result['epoch_means'])
    if result['status'] == 'completed':
        write_json(models_path, result)
    return result


def make_epoch_callback(folder, target, protocol, x_target, y_target, trials, *, expected_batches, batch_size):
    import tensorflow as tf
    folder = Path(folder)
    suite = folder.parent
    class SaveOracleEpoch(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.batches = 0

        def on_train_batch_end(self, batch, logs=None):
            self.batches += 1

        def on_epoch_end(self, epoch, logs=None):
            if self.batches != expected_batches or not math.isfinite(float((logs or {}).get('loss', float('nan')))):
                raise RuntimeError('Incomplete/nonfinite source epoch cannot become an oracle checkpoint')
            pieces = [tf.nn.softmax(self.model(x_target[start:start + batch_size], training=False), axis=-1).numpy()
                      for start in range(0, len(y_target), batch_size)]
            probabilities = np.concatenate(pieces)
            accuracy, correct = prediction_accuracy(y_target, probabilities)
            number = epoch + 1
            checkpoint = folder / f'epoch_{number:03}.keras'
            temporary = folder / f'epoch_{number:03}.tmp.keras'
            self.model.save(temporary)
            temporary.replace(checkpoint)
            archive = folder / f'epoch_{number:03}_predictions.npz'
            temporary_archive = archive.with_suffix('.tmp')
            with temporary_archive.open('wb') as handle:
                np.savez_compressed(handle, y_true=y_target, probabilities=probabilities, trial_ids=trials)
            temporary_archive.replace(archive)
            row = dict(target_subject=int(target), subject_number=int(target) + 1,
                       epoch=number, accuracy=accuracy, n_correct=correct, n_windows=len(y_target),
                       checkpoint=str(checkpoint.relative_to(suite)), checkpoint_sha256=file_hash(checkpoint),
                       predictions=str(archive.relative_to(suite)), protocol_id=fingerprint(protocol))
            write_json(folder / f'epoch_{number:03}.json', row)
            print(f'ORACLE subject={target} epoch={number} accuracy={accuracy:.6f}', flush=True)
    return SaveOracleEpoch()


def train_source(x, y, subjects, trials, config, suite, *, dataset_id, smoke=False):
    import tensorflow as tf
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
    from eegproc.deep_learning.joint_architectures.sic_common.config import expand_cartesian_grid, split_data_hyperparameters
    from eegproc.deep_learning.cross_validation.splits import _class_weight_from_labels
    target, = config.target_subjects
    folder = Path(suite) / f'subject_{target:02}'
    folder.mkdir(parents=True, exist_ok=False)
    candidates, dimensions = expand_cartesian_grid(config.model_config)
    if len(candidates) != 1 or dimensions:
        raise ValueError('Oracle source run requires one fixed configuration')
    model_config, data_config = split_data_hyperparameters(candidates[0])
    if data_config['remove_median_label'] or not model_config.get('use_class_weight'):
        raise ValueError('SEED-IV oracle requires all four classes and enabled class weights')
    for key, expected in dict(n_channels=62, n_bands=5, n_classes=4, classification_level='window').items():
        if key in model_config and model_config[key] != expected:
            raise ValueError(f'Incompatible SEED-IV model setting: {key}')
        model_config[key] = expected
    outer_sources = np.unique(subjects[subjects != target])
    # Match the previous adapter: each invocation evaluated one target, whose
    # source-validation split used subject_number=1 in the shared runner.
    rng = np.random.default_rng(np.random.SeedSequence([config.validation_seed, 1]))
    validation = np.sort(rng.choice(outer_sources, config.validation_subjects, replace=False))
    source_mask = (subjects != target) & ~np.isin(subjects, validation)
    validation_mask = np.isin(subjects, validation)
    target_mask = subjects == target
    if not source_mask.any() or not target_mask.any():
        raise ValueError('Empty source or target partition')
    tf.keras.utils.set_random_seed(config.seed + int(target))
    x_source, y_source = x[source_mask], y[source_mask]
    class_weights = _class_weight_from_labels(y_source)
    protocol = dict(schema_version=1, dataset='SEED-IV', dataset_id=dataset_id,
                    code_id=code_identity(),
                    oracle=True, metric='accuracy', evaluation_level='window', smoke=smoke,
                    source_epochs=config.source_epochs, source_batch_size=config.source_batch_size,
                    seed=config.seed, validation_seed=config.validation_seed,
                    validation_subjects=config.validation_subjects,
                    input_shape=list(x.shape[1:]), n_channels=62, n_bands=5, n_classes=4,
                    source_use_class_weight=True, calibration_use_class_weight=config.calibration_use_class_weight,
                    model_config=model_config)
    write_json(folder / 'protocol.json', dict(target_subject=int(target),
        protocol=protocol, protocol_id=fingerprint(protocol),
        source_subjects=np.unique(subjects[source_mask]).tolist(), validation_subjects=validation.tolist(),
        class_weights=class_weights, evaluation_note=ORACLE_NOTE))
    model = build_sic_model(tuple(x.shape[1:]), training_features=x_source,
        training_labels=y_source, training_subject_ids=subjects[source_mask],
        training_trial_ids=trials[source_mask], **model_config)
    callback = make_epoch_callback(folder, target, protocol, x[target_mask], y[target_mask], trials[target_mask],
        expected_batches=math.ceil(len(y_source) / config.source_batch_size), batch_size=config.source_batch_size)
    model.fit(model.prepare_fit_inputs(x_source, subjects[source_mask]), y_source,
        epochs=config.source_epochs, batch_size=config.source_batch_size, verbose=config.verbose,
        class_weight=class_weights,
        validation_data=(x[validation_mask], y[validation_mask]) if validation_mask.any() else None,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(), callback])
    if len(list(folder.glob('epoch_*.json'))) != config.source_epochs:
        raise RuntimeError('Source training did not complete its full epoch budget')
    write_json(folder / 'SUCCESS', dict(stage='source', target_subject=int(target),
        protocol_id=fingerprint(protocol), completed_epochs=config.source_epochs))
    return folder


def calibrate_selected(x, y, subjects, trials, config, suite, *, dataset_id):
    import tensorflow as tf
    from eegproc.deep_learning.cross_validation import subject_calibration_cv
    from eegproc.deep_learning.joint_architectures.sic_common.config import SIC_CLASSIFICATION_METRICS
    from eegproc.deep_learning.joint_architectures.sic_common.training import _save_calibration_artifacts
    from capacity import resolved_model_config
    suite = Path(suite)
    selection = json.loads((suite / 'oracle_models.json').read_text())
    target, = config.target_subjects
    if (selection['status'] != 'completed' or selection['protocol']['dataset_id'] != dataset_id
            or selection['metric'] != 'accuracy' or str(target) not in selection['selected_models']):
        raise ValueError('A completed, matching common-epoch accuracy selection is required')
    if selection['protocol']['code_id'] != code_identity():
        raise ValueError('Source code changed after source training; use the same reviewed snapshot')
    if selection['protocol']['model_config'] != resolved_model_config(config.model_config):
        raise ValueError('Selected checkpoint does not match the requested model configuration')
    checkpoint = suite / selection['selected_models'][str(target)]
    if file_hash(checkpoint) != selection['selected_checkpoint_hashes'][str(target)]:
        raise ValueError('Selected checkpoint changed after oracle selection')
    folder = suite / 'calibration' / f'subject_{target:02}'
    folder.mkdir(parents=True, exist_ok=False)
    model = tf.keras.models.load_model(checkpoint, compile=False)
    # Loading compile=False does not retain an optimizer; the calibration hook
    # installs a fresh one before each independent target-only fit.
    results = subject_calibration_cv(model_builder_function=None, pretrained_model=model,
        feature_array=x, label_array=y, subject_id_array=subjects, trial_id_array=trials,
        target_subjects=(target,), n_jobs=1, gpus_per_fold=1,
        validation_subjects_per_fold=0, training_seed=config.seed, deterministic_training=True,
        source_batch_size=config.source_batch_size,
        calibration_epochs=config.calibration_epochs, calibration_batch_size=config.calibration_batch_size,
        calibration_levels=config.calibration_levels, calibration_selection_shots=config.calibration_selection_shots,
        calibration_learning_rate=config.calibration_learning_rate,
        calibration_optimizer=config.calibration_optimizer, calibration_weight_decay=config.calibration_weight_decay,
        calibration_seed=config.calibration_seed, stratify_calibration=config.stratify_calibration,
        calibration_use_class_weight=config.calibration_use_class_weight,
        evaluation_level='window', metrics=SIC_CLASSIFICATION_METRICS, verbose=config.verbose)
    expected = selection['selected_accuracies'][str(target)]
    actual = results['overall']['zero_shot_all_trials_mean_scores']['accuracy']
    if not np.isclose(actual, expected, atol=1e-8, rtol=0):
        raise RuntimeError('Reloaded oracle checkpoint does not reproduce its selected accuracy')
    results['oracle_selection'] = dict(epoch=selection['best_epoch'], metric='accuracy',
        cohort_mean_accuracy=selection['selection_score'], checkpoint=str(checkpoint),
        evaluation_note=ORACLE_NOTE, protocol_id=selection['protocol_id'],
        source_partition=selection['source_partitions'][str(target)])
    _save_calibration_artifacts(folder, model_config=selection['protocol']['model_config'], results=results)
    write_json(folder / 'oracle_selection.json', results['oracle_selection'])
    write_json(folder / 'SUCCESS', dict(stage='calibration', target_subject=int(target),
        oracle_epoch=selection['best_epoch'], protocol_id=selection['protocol_id']))
    return results
