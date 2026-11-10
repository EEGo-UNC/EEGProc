"""36-trial source memorization diagnostic with separate source-validation tracking."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("Epochs and learning rate must be positive.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import tensorflow as tf
    from eegproc.deep_learning.joint_architectures.SICModelv16.run_experiments import fold_subjects
    from eegproc.deep_learning.joint_architectures.SICModelv16.data import prepare_data
    from eegproc.deep_learning.joint_architectures.SICModelv16.sic_model import build_sic_model

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise RuntimeError(f"Exactly one allocated GPU required; found {len(gpus)}.")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(args.seed)
    cache = output / "data"
    prepare_data(args.data_dir / "dreamer_eeg.npy", args.data_dir / "dreamer_labels.npy",
                 cache, data_subjects=[0, 1, 2, 3])
    arrays = {key: np.load(cache / f"{key}.npy", mmap_mode="r")
              for key in ("features", "labels", "subjects", "trials")}
    source, validation = fold_subjects(arrays["subjects"], 0, 1, args.seed)
    indices = np.flatnonzero(np.isin(arrays["subjects"], source))
    validation_indices = np.flatnonzero(np.isin(arrays["subjects"], validation))
    if len(indices) != 36 or len(validation_indices) != 18:
        raise ValueError("Expected 36 source-training and 18 validation trials.")
    vx = np.asarray(arrays["features"][validation_indices])
    vy = np.asarray(arrays["labels"][validation_indices])
    if set(np.unique(arrays["labels"][indices])) != {0, 1} or set(np.unique(vy)) != {0, 1}:
        raise ValueError("Training and validation must each contain both classes.")
    x = np.asarray(arrays["features"][indices])
    y = np.asarray(arrays["labels"][indices])
    subjects = np.asarray(arrays["subjects"][indices])
    trials = np.asarray(arrays["trials"][indices])
    config = dict(use_decoder=False, use_transformer=False, dropout=0.0,
                  classifier_dropout=0.0, weight_decay=0.0, label_smoothing=0.0,
                  focal_gamma=0.0, use_subject_adversarial=False,
                  use_variational_head=False, use_recurrent_classifier=False,
                  training_method="erm", learning_rate=args.learning_rate,
                  classification_level="trial", conv_filters=[32, 64],
                  spatial_pool_sizes=[2, 2], temporal_pool_sizes=[],
                  classifier_units=128, emb_dim=128)
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    write("config.json", dict(model=config, seed=args.seed, epochs=args.epochs,
          batch_size=1, source_subjects=source, validation_subjects=validation,
          excluded_test_subject=0, evaluation="36 training trials and 18 separate validation trials; training=False",
          selected_trials=[dict(subject=int(s), trial=int(t), label=int(label))
                           for s, t, label in zip(subjects, trials, y)]))
    model = build_sic_model(input_shape=x.shape[1:], training_subject_ids=subjects,
                            training_trial_ids=trials, **config)
    history = []

    class Measure(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            probabilities = np.concatenate([
                tf.nn.softmax(model(x[i:i+1], training=False), axis=-1).numpy()
                for i in range(len(x))])
            if not np.isfinite(probabilities).all():
                raise ValueError("Nonfinite inference probabilities.")
            predicted = (probabilities[:, 1] >= 0.5).astype(int)
            accuracy = float(np.mean(predicted == y))
            loss = float(-np.log(np.maximum(probabilities[np.arange(len(y)), y.astype(int)], 1e-12)).mean())
            passed = accuracy == 1.0 and loss <= 0.1
            row = dict(epoch=epoch + 1, inference_accuracy=accuracy,
                       inference_balanced_accuracy=float(np.mean([np.mean(predicted[y == c] == c) for c in (0, 1)])), inference_cross_entropy=loss,
                       probabilities=probabilities.tolist(), labels=y.tolist(),
                       training_metrics={k: float(v) for k, v in (logs or {}).items()},
                       passed=passed)
            vp = np.concatenate([tf.nn.softmax(model(vx[i:i+1], training=False), axis=-1).numpy()
                                 for i in range(len(vx))])
            if not np.isfinite(vp).all():
                raise ValueError("Nonfinite validation probabilities.")
            vpred = (vp[:, 1] >= 0.5).astype(int)
            row["validation"] = dict(
                accuracy=float(np.mean(vpred == vy)),
                balanced_accuracy=float(np.mean([np.mean(vpred[vy == c] == c) for c in (0, 1)])),
                cross_entropy=float(-np.log(np.maximum(vp[np.arange(len(vy)), vy.astype(int)], 1e-12)).mean()),
                probabilities=vp.tolist(), labels=vy.tolist(),
                subjects=arrays["subjects"][validation_indices].tolist(),
                trials=arrays["trials"][validation_indices].tolist())
            history.append(row)
            write("history.json", history)
            print(f"MEMORIZATION epoch={epoch+1} accuracy={accuracy:.3f} CE={loss:.4f} passed={passed} validation_BA={row['validation']['balanced_accuracy']:.3f}", flush=True)
            if passed:
                model.stop_training = True

    started = time.monotonic()
    model.fit(model.prepare_fit_inputs(x, subjects), y, batch_size=1,
              epochs=args.epochs, shuffle=True, verbose=2,
              callbacks=[tf.keras.callbacks.TerminateOnNaN(), Measure()])
    if not history:
        raise RuntimeError("No completed epochs.")
    write("result.json", dict(status="passed" if history[-1]["passed"] else "not_memorized",
          elapsed_seconds=time.monotonic()-started, final=history[-1],
          tensorflow=tf.__version__, numpy=np.__version__,
          interpretation="Memorization pass uses training trials only. Validation is tracked independently; test subject 0 is never evaluated."))
    archive = output / "overfit-36-diagnostics.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in [output / "config.json", output / "history.json", output / "result.json",
                     cache / "preprocessing.json"]:
            bundle.add(path, arcname=str(path.relative_to(output)))
    print(f"Download: {archive}", flush=True)


if __name__ == "__main__":
    main()
