"""Training-only memorization diagnostic; never produces a generalization score."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def select_trials(labels, subjects, trials, source_subjects):
    """Choose four trials per class, in stable subject/trial order."""
    eligible = sorted((i for i, subject in enumerate(subjects)
                       if int(subject) in source_subjects),
                      key=lambda i: (int(subjects[i]), int(trials[i])))
    selected = []
    for label in (0, 1):
        candidates = [i for i in eligible if int(labels[i]) == label]
        if len(candidates) < 4:
            raise ValueError(f"Source pool needs four trials of class {label}.")
        selected.extend(candidates[:4])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
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
    indices = select_trials(arrays["labels"], arrays["subjects"], arrays["trials"], source)
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
          batch_size=1, source_subjects=source, excluded_validation_subjects=validation,
          excluded_test_subject=0, evaluation="same eight training trials; training=False",
          selected_trials=[dict(subject=int(s), trial=int(t), label=int(label))
                           for s, t, label in zip(subjects, trials, y)]))
    model = build_sic_model(input_shape=x.shape[1:], training_subject_ids=subjects,
                            training_trial_ids=trials, **config)
    history = []

    class Measure(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            probabilities = np.concatenate([
                tf.nn.softmax(model(x[i:i+1], training=False), axis=-1).numpy()
                for i in range(8)])
            if not np.isfinite(probabilities).all():
                raise ValueError("Nonfinite inference probabilities.")
            predicted = (probabilities[:, 1] >= 0.5).astype(int)
            accuracy = float(np.mean(predicted == y))
            loss = float(-np.log(np.maximum(probabilities[np.arange(8), y.astype(int)], 1e-12)).mean())
            passed = accuracy == 1.0 and loss <= 0.1
            row = dict(epoch=epoch + 1, inference_accuracy=accuracy,
                       inference_balanced_accuracy=accuracy, inference_cross_entropy=loss,
                       probabilities=probabilities.tolist(), labels=y.tolist(),
                       training_metrics={k: float(v) for k, v in (logs or {}).items()},
                       passed=passed)
            history.append(row)
            write("history.json", history)
            print(f"MEMORIZATION epoch={epoch+1} accuracy={accuracy:.3f} CE={loss:.4f} passed={passed}", flush=True)
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
          interpretation="Training-only diagnostic, not validation or test performance."))
    archive = output / "overfit-eight-diagnostics.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in [output / "config.json", output / "history.json", output / "result.json",
                     cache / "preprocessing.json"]:
            bundle.add(path, arcname=str(path.relative_to(output)))
    print(f"Download: {archive}", flush=True)


if __name__ == "__main__":
    main()
