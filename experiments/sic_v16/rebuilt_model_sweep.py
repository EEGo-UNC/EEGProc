"""Verified rebuilt data: four memorization gates, then validation-selected LOSO."""
import argparse
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DIAGNOSTICS = ("reference", "no_dropout", "leaky_relu", "class_weighted")
VARIANTS = ("cnn", "cnn_reconstruction", "cnn_transformer", "cnn_transformer_reconstruction")


def configuration(index, check):
    return dict(classification_level="trial", use_decoder=index in (1, 3),
        use_transformer=index in (2, 3), conv_filters=[32, 64], spatial_pool_sizes=[2, 2],
        temporal_pool_sizes=[], temporal_kernel_size=7, spatial_kernel_size=3,
        emb_dim=128, classifier_units=128, dropout=0.0 if check else 0.1,
        classifier_dropout=0.0 if check else 0.1, weight_decay=0.0,
        learning_rate=0.001, label_smoothing=0.0, focal_gamma=0.0,
        use_recurrent_classifier=False, use_variational_head=False,
        use_subject_adversarial=False, training_method="erm", reconstruction_loss_weight=0.1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "check", "train", "summarize", "collect", "diagnose", "diagnostic_summary"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--rebuild-dir", type=Path)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--check-epochs", type=int, default=1000)
    p.add_argument("--lr-diagnostic", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.check_epochs) < 1:
        p.error("Epoch budgets must be positive.")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    def write(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)
    if args.mode == "prepare":
        from permutation_audit import digest
        from eegproc.deep_learning.joint_architectures.SICModelv16.data import prepare_data
        rebuild = args.rebuild_dir
        if rebuild is None:
            candidates = list((Path(os.environ["SIC_V16_WORK_ROOT"]) / "crop-rebuild-comparison").glob("job-*/comparison.json"))
            if not candidates:
                raise ValueError("No completed rebuild; set REBUILD_DIR.")
            rebuild = max(candidates, key=lambda path: path.stat().st_mtime).parent
        report = json.loads((rebuild / "crop_verification.json").read_text())
        if report["summary"]["identified"] != 414:
            raise ValueError("Incomplete crop verification.")
        data = rebuild / "centred_full_filter"
        hashes = {name: digest(data / name) for name in ("dreamer_eeg.npy", "dreamer_labels.npy")}
        if hashes["dreamer_eeg.npy"] != report["centred_full_filter"]["eeg_sha256"] or hashes["dreamer_labels.npy"] != report["centred_full_filter"]["labels_sha256"]:
            raise ValueError("Rebuilt dataset hash mismatch.")
        if (root / "provenance.json").exists():
            raise ValueError("Sweep already prepared; use a new sweep directory.")
        prepare_data(data / "dreamer_eeg.npy", data / "dreamer_labels.npy", root / "data", data_subjects=list(range(23)))
        write(root / "provenance.json", dict(data_dir=str(data.resolve()), hashes=hashes,
            epochs=args.epochs, check_epochs=args.check_epochs, seed=42,
            validation_subjects=2, source_subjects_per_fold=20,
            selection="highest mean validation-subject balanced accuracy; lower validation cross-entropy on ties; earliest exact tie",
            variants=list(VARIANTS), checkpoint_policy="retain best validation checkpoint only; evaluate test once after selection",
            early_stopping="after at least 200 epochs, stop after 100 epochs without validation selection improvement"))
        return
    if args.mode == "diagnostic_summary":
        results = [json.loads((root / f"diagnostic-{i}" / "result.json").read_text()) for i in range(2 if args.lr_diagnostic else 4)]
        write(root / "diagnostic_comparison.json", results)
        archive = root / ("learning-rate-diagnostics.tar.gz" if args.lr_diagnostic else "classifier-collapse-diagnostics.tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in (".json", ".txt", ".log"):
                    bundle.add(path, arcname=str(path.relative_to(root)))
        print(f"Download: {archive}", flush=True)
        return
    if args.mode == "collect":
        archive = root / "rebuilt-four-model-partial-diagnostics.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in (".json", ".txt", ".log"):
                    bundle.add(path, arcname=str(path.relative_to(root)))
        print(f"Download: {archive}", flush=True)
        return
    if args.mode == "summarize":
        results = []
        for index in range(92):
            result = json.loads((root / f"fold-{index:02d}" / "result.json").read_text())
            if result["status"] != "completed":
                raise ValueError("Incomplete fold.")
            results.append(result)
        import statistics
        write(root / "comparison.json", {name: dict(
            balanced_accuracy=statistics.mean(r["test"]["balanced_accuracy"] for r in results if r["variant"] == name),
            roc_auc=statistics.mean(r["test"]["roc_auc"] for r in results if r["variant"] == name)) for name in VARIANTS})
        archive = root / "rebuilt-four-model-diagnostics.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in (".json", ".txt", ".log"):
                    bundle.add(path, arcname=str(path.relative_to(root)))
        print(f"Download: {archive}", flush=True)
        return
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
    from eegproc.deep_learning.joint_architectures.SICModelv16.run_experiments import fold_subjects
    from eegproc.deep_learning.joint_architectures.SICModelv16.sic_model import build_sic_model
    from overfit_eight import select_trials
    diagnostic = args.mode == "diagnose"
    check = args.mode == "check"
    if not 0 <= args.index < (4 if check or diagnostic else 92):
        raise ValueError("Invalid task index")
    variant = 0 if diagnostic else args.index if check else args.index // 23
    target = 0 if check or diagnostic else args.index % 23
    spec = json.loads((root / "provenance.json").read_text())
    if not check and not diagnostic:
        for index in range(4):
            if json.loads((root / f"check-{index}" / "result.json").read_text())["status"] != "passed":
                raise ValueError("All four learning gates must pass before full training.")
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise RuntimeError("Exactly one GPU required")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(42 + target)
    arrays = {name: np.load(root / "data" / f"{name}.npy", mmap_mode="r")
              for name in ("features", "labels", "subjects", "trials")}
    sources, validation = fold_subjects(arrays["subjects"], target, 2, 42)
    train = np.flatnonzero(np.isin(arrays["subjects"], sources))
    val = np.flatnonzero(np.isin(arrays["subjects"], validation))
    test = np.flatnonzero(arrays["subjects"] == target)
    if check:
        train = np.asarray(select_trials(arrays["labels"], arrays["subjects"], arrays["trials"], [2, 3]))
        val = train  # Memorization gate, explicitly not a validation estimate.
    folder = root / (f"diagnostic-{args.index}" if diagnostic else f"check-{args.index}" if check else f"fold-{args.index:02d}")
    folder.mkdir(exist_ok=False)
    config = configuration(variant, check)
    weights = None
    if diagnostic and not args.lr_diagnostic:
        if args.index == 1:
            config.update(dropout=0.0, classifier_dropout=0.0)
        if args.index == 2:
            config["classifier_activation"] = "leaky_relu"
        if args.index == 3:
            config["use_class_weight"] = True
            config["class_weight_batch_mean"] = True
            counts = np.bincount(arrays["labels"][train].astype(int), minlength=2)
            weights = {c: float(len(train)/(2*counts[c])) for c in (0, 1)}
    arm_name = DIAGNOSTICS[args.index] if diagnostic else None
    if diagnostic and args.lr_diagnostic:
        if args.index not in (0, 1):
            raise ValueError("Learning-rate diagnostic index must be 0 or 1")
        config["learning_rate"] = (0.0001, 0.00001)[args.index]
        arm_name = f"learning_rate_{config['learning_rate']}"
    write(folder / "config.json", dict(model=config, variant=VARIANTS[variant], target_subject=target,
        train_subjects=sorted(set(arrays["subjects"][train].tolist())), validation_subjects=validation if not check else [],
        class_weight=weights, diagnostic_arm=arm_name,
        train_trials=[dict(subject=int(arrays["subjects"][i]), trial=int(arrays["trials"][i])) for i in train]))
    model = build_sic_model(input_shape=tuple(arrays["features"].shape[1:]),
        training_subject_ids=arrays["subjects"][train], training_trial_ids=arrays["trials"][train], **config)
    def evaluate(indices):
        probabilities_list, embeddings, preactivations = [], [], []
        for i in indices:
            out = model._encode(arrays["features"][i:i+1], training=False)
            probabilities_list.append(out["probabilities"].numpy())
            if diagnostic:
                pooled = model.trial_recurrent_classifier.layers[0](out["classifier_sequence"])
                dense = model.trial_recurrent_classifier.layers[1]
                preactivations.append((tf.matmul(pooled, dense.kernel)+dense.bias).numpy())
                embeddings.append(out["classification_embedding"].numpy())
        probabilities = np.concatenate(probabilities_list)
        activity = {}
        if diagnostic:
            emb, pre = np.concatenate(embeddings), np.concatenate(preactivations)
            activity = dict(mean_positive_preactivation_units=float(np.mean(np.sum(pre>0,axis=1))),
                positive_unit_union=int(np.sum(np.any(pre>0,axis=0))),
                mean_nonzero_embedding_units=float(np.mean(np.sum(np.abs(emb)>1e-8,axis=1))),
                embedding_between_trial_std=float(np.mean(np.std(emb,axis=0))),
                prediction_std=float(np.std(probabilities[:,1])))
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite inference probabilities")
        labels = arrays["labels"][indices]
        prediction = (probabilities[:, 1] >= 0.5).astype(int)
        return dict(classifier_activity=activity, accuracy=float(np.mean(prediction == labels)),
            balanced_accuracy=float(balanced_accuracy_score(labels, prediction)),
            roc_auc=float(roc_auc_score(labels, probabilities[:, 1])),
            cross_entropy=float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels.astype(int)], 1e-12)).mean()),
            single_class_predictions=bool(len(set(prediction.tolist())) == 1),
            probability_min=float(probabilities[:, 1].min()), probability_max=float(probabilities[:, 1].max()),
            confusion_matrix=confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
            labels=labels.tolist(), probabilities=probabilities.tolist(),
            subjects=arrays["subjects"][indices].tolist(), trials=arrays["trials"][indices].tolist())
    if diagnostic and args.lr_diagnostic:
        from layer_measurements import representation_summary, weight_changes
        # Two labels from each of four source subjects; fixed across both runs.
        probes = []
        for subject in sources[:4]:
            for label in (0, 1):
                probes.append(int(next(i for i in train if arrays["subjects"][i] == subject and arrays["labels"][i] == label)))
        write(folder / "probe_trials.json", [dict(subject=int(arrays["subjects"][i]), trial=int(arrays["trials"][i]), label=int(arrays["labels"][i])) for i in probes])
        def layer_activity():
            stages = {}
            for i in probes:
                out = model._encode(arrays["features"][i:i+1], training=False)
                pool = model.trial_recurrent_classifier.layers[0](out["classifier_sequence"])
                dense = model.trial_recurrent_classifier.layers[1]
                tensors = dict(encoder_sequence=out["graph_sequence"], pooled_encoder=pool,
                    classifier_preactivation=tf.matmul(pool, dense.kernel)+dense.bias,
                    classifier_embedding=out["classification_embedding"], logits=out["logits"])
                for band_index, band in enumerate(("theta", "alpha", "beta")):
                    tensors[band+"_encoder"] = out["graph_sequence"][..., band_index*128:(band_index+1)*128]
                for name, tensor in tensors.items():
                    stages.setdefault(name, []).append(tensor.numpy())
            return {name: representation_summary(values) for name, values in stages.items()}
        write(folder / "initial_layer_activity.json", layer_activity())
        update_history = []
    if diagnostic:
        write(folder / "initial_metrics.json", dict(training=evaluate(train), validation=evaluate(val)))
    history = []
    class Monitor(tf.keras.callbacks.Callback):
        best = None
        best_epoch = 0
        passed = False
        def on_epoch_begin(self, epoch, logs=None):
            if diagnostic and args.lr_diagnostic:
                self.epoch_before = [v.numpy().copy() for v in model.trainable_variables]
        def on_train_batch_begin(self, batch, logs=None):
            if diagnostic and args.lr_diagnostic:
                self.step = int(model.main_optimizer.iterations.numpy())
                self.batch_before = ([v.numpy().copy() for v in model.trainable_variables]
                    if self.step < 20 or self.step in (49,99,199,499) or self.step % 1000 == 0 else None)
        def on_train_batch_end(self, batch, logs=None):
            if diagnostic and args.lr_diagnostic and self.batch_before is not None:
                update_history.append(dict(step=self.step+1, layers=weight_changes(
                    [getattr(v,"path",v.name) for v in model.trainable_variables],
                    self.batch_before, [v.numpy() for v in model.trainable_variables])))
                write(folder / "sampled_batch_updates.json", update_history)
        def on_epoch_end(self, epoch, logs=None):
            values = {k: float(v) for k, v in (logs or {}).items()}
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError("Nonfinite training metrics")
            training = evaluate(train)
            valid = training if check else evaluate(val)
            if check:
                score = valid["balanced_accuracy"]
            else:
                labels = np.asarray(valid["labels"])
                probabilities = np.asarray(valid["probabilities"])
                groups = np.asarray(valid["subjects"])
                score = float(np.mean([balanced_accuracy_score(labels[groups == s],
                    (probabilities[groups == s, 1] >= 0.5).astype(int)) for s in validation]))
            key = (score, -valid["cross_entropy"])
            if self.best is None or key > self.best:
                self.best, self.best_epoch = key, epoch + 1
                if not check:
                    temporary = folder / "best.tmp.keras"
                    model.save(temporary)
                    temporary.replace(folder / "best.keras")
            history.append(dict(epoch=epoch + 1, training_metrics=values, training=training,
                                validation=None if check else valid, validation_subject_mean_BA=score))
            if diagnostic and args.lr_diagnostic:
                history[-1]["layer_activity"] = layer_activity()
                history[-1]["epoch_net_weight_changes"] = weight_changes(
                    [getattr(v,"path",v.name) for v in model.trainable_variables],
                    self.epoch_before, [v.numpy() for v in model.trainable_variables])
            write(folder / "history.json", history)
            self.passed = check and training["accuracy"] == 1.0 and training["cross_entropy"] <= 0.1
            print(f"{VARIANTS[variant]} epoch={epoch+1} train_BA={training['balanced_accuracy']:.3f} selection_BA={score:.3f} best_epoch={self.best_epoch}", flush=True)
            if self.passed or (not diagnostic and not check and epoch + 1 >= 200 and epoch + 1 - self.best_epoch >= 100):
                model.stop_training = True
    monitor = Monitor()
    start = time.monotonic()
    model.fit(model.prepare_fit_inputs(arrays["features"][train], arrays["subjects"][train]),
              arrays["labels"][train], batch_size=1, epochs=spec["check_epochs"] if check else spec["epochs"],
              verbose=2, callbacks=[monitor], class_weight=weights)
    if diagnostic and args.lr_diagnostic:
        model.save(folder / "last.keras")
    result = dict(status="passed" if check and monitor.passed else "failed" if check else "completed",
                  variant=VARIANTS[variant], target_subject=target, elapsed_seconds=time.monotonic()-start,
                  completed_epochs=len(history), best_epoch=monitor.best_epoch, best_validation_score=monitor.best)
    if diagnostic:
        result.update(diagnostic_arm=arm_name, final_training=history[-1]["training"],
                      final_validation=history[-1]["validation"], class_weight=weights)
    if not check and not diagnostic:
        model = tf.keras.models.load_model(folder / "best.keras", compile=False)
        result["test"] = evaluate(test)
        result["selected_validation"] = history[monitor.best_epoch-1]["validation"]
    write(folder / "result.json", result)
    if check and not monitor.passed:
        raise RuntimeError("Learning check did not memorize eight trials; full sweep remains blocked.")


if __name__ == "__main__":
    main()
