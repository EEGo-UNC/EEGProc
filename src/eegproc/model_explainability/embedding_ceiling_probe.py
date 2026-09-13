"""Ask what confidence the classifier can express, independent of the search.

Every counterfactual run so far plateaus at target_p ~= 0.541 regardless of
learning rate (0.1 to 5.0), target weight (1 or 50), physiological weight
(0 or 0.1), or removing every constraint. Two explanations remain, and they
imply completely different fixes:

  (a) The classifier genuinely cannot express more confidence, so no search
      method will ever do better and the plateau is a result to report.
  (b) The classifier can, but gradient descent cannot get there through the
      trial BiGRU, which processes W*T = 7,680 timesteps. Backpropagation
      through that many recurrent steps attenuates the gradient severely.
      Then the plateau is an optimization artifact and a different search
      method (evolutionary, or a two-stage embedding target) would fix it.

This script separates the two by optimizing the TRIAL EMBEDDING directly.
The embedding is the 128-dim BiGRU output that feeds the variational
classifier head, so optimizing it involves no recurrent backpropagation at
all -- the bottleneck in (b) is bypassed entirely. Whatever confidence is
reachable here is an upper bound on what any latent-space search could
achieve, because every latent maps to some embedding.

Two bounds are reported:

  unconstrained -- embedding free to take any value. The head's logits are
  log-Gaussian likelihoods with near-identical per-class sigmas, so the
  logit gap is close to linear and this bound is loose; it mainly confirms
  whether the head is the limiting factor. Measured at 0.9562 on subject 0
  trial 0, versus 0.5413 from latent-space search, so it is not.

  norm-matched -- embedding constrained to the same L2 norm as real trial
  embeddings (~11.33 here). An earlier version of this script clipped to
  [-1, 1] on the assumption that GRU tanh outputs bound the embedding; that
  was wrong, real embeddings reach +-1.83, and the resulting bound came out
  BELOW the observed value. A norm ball matched to real embeddings is the
  honest constraint: it asks what is reachable without leaving the scale
  the recurrent stack actually produces.

Also reported is the spread between real trial embeddings. If different
trials map to nearly the same embedding, the recurrent summarizer has
collapsed and is largely ignoring its input, which would explain the
model's ~0.50 confidence on every trial independently of any counterfactual
concern.

Usage:
    PYTHONPATH=src python -m eegproc.model_explainability.embedding_ceiling_probe \\
        --model runs/.../loso_fold_0001_target_0_zero_shot.keras \\
        --raw-eeg-npy datasets/dreamer_eeg.npy \\
        --raw-labels-npy datasets/dreamer_labels.npy \\
        --subject-id 0 --trial-id 0
"""

import argparse
import importlib
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_parser():
    parser = argparse.ArgumentParser(
        description="Upper-bound the confidence reachable by any latent-space search."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-module",
        default="eegproc.deep_learning.joint_architectures.SICModelv11.sic_model",
    )
    parser.add_argument("--raw-eeg-npy", type=Path, required=True)
    parser.add_argument("--raw-labels-npy", type=Path, required=True)
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--dataset", default="dreamer")
    parser.add_argument("--label-dimension", default="valence")
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--window-overlap", type=float, default=0.0)
    parser.add_argument("--median-label", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser


def load_one_trial(args):
    """Rebuild a single trial exactly as the counterfactual runner does."""
    data = importlib.import_module(
        "eegproc.deep_learning.joint_architectures.joint_models_data"
    )
    features = np.asarray(
        data.build_dataset(
            dataset=args.dataset,
            eeg_path=args.raw_eeg_npy,
            labels_path=args.raw_labels_npy,
            label_dimension=args.label_dimension,
            window_size_sec=args.window_sec,
            fs=args.fs,
            overlap=args.window_overlap,
            median_label=args.median_label,
            zscore=False,
        )[0],
        dtype=np.float32,
    )
    # global_rms normalization, matching _normalize_each_window
    rms = np.sqrt(
        np.mean(np.square(features, dtype=np.float64), axis=(1, 2), keepdims=True)
    )
    features = (features.astype(np.float64) / np.maximum(rms, 1e-6)).astype(np.float32)

    raw = np.load(args.raw_eeg_npy, mmap_mode="r")
    _, n_trials, _, n_samples = raw.shape
    window = int(round(args.window_sec * args.fs))
    hop = max(1, int(round(window * (1.0 - args.window_overlap))))
    n_windows = 1 + (n_samples - window) // hop

    start = (args.subject_id * n_trials + args.trial_id) * n_windows
    trial = features[start : start + n_windows]
    return tf.constant(trial[None, ...]), n_windows


def ascend(head, start, target_class, steps, learning_rate, max_norm=None):
    """Maximize the target-class logit over the embedding.

    max_norm, if given, projects the embedding back onto that L2 ball after
    every step, restricting the search to the scale real embeddings occupy.
    """
    embedding = tf.Variable(start)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            logits = tf.cast(head(embedding, training=False), tf.float32)
            loss = -tf.nn.log_softmax(logits, axis=-1)[0, target_class]
        optimizer.apply_gradients([(tape.gradient(loss, embedding), embedding)])
        if max_norm is not None:
            norm = tf.norm(embedding)
            if float(norm) > max_norm:
                embedding.assign(embedding * (max_norm / norm))
    probabilities = tf.nn.softmax(
        tf.cast(head(embedding, training=False), tf.float32), axis=-1
    ).numpy()[0]
    return embedding, float(probabilities[target_class])


def embed(model, trial):
    """Encoder features -> flattened window sequence -> trial embedding."""
    latent = tf.cast(model.get_encoder_features(trial)["window_features"], tf.float32)
    sequence = tf.reshape(latent, [1, -1, latent.shape[-1]])
    return model.trial_recurrent_classifier(sequence, training=False), sequence


def main(argv=None):
    args = build_parser().parse_args(argv)
    importlib.import_module(args.model_module)
    model = tf.keras.models.load_model(args.model, compile=False, safe_mode=True)

    trial, n_windows = load_one_trial(args)
    print(f"trial shape {tuple(trial.shape)} | windows={n_windows}", flush=True)

    embedding, sequence = embed(model, trial)
    print(f"BiGRU sequence length = {int(sequence.shape[1])} timesteps", flush=True)
    head = model.vc_target
    probabilities = tf.nn.softmax(
        tf.cast(head(embedding, training=False), tf.float32), axis=-1
    ).numpy()[0]
    original = int(np.argmax(probabilities))
    target = 1 - original
    print(
        f"\noriginal: class={original} p={probabilities[original]:.4f} "
        f"-> target class {target} (currently p={probabilities[target]:.4f})",
        flush=True,
    )
    print(
        f"embedding: dim={embedding.shape[-1]} norm={float(tf.norm(embedding)):.4f} "
        f"range=[{float(tf.reduce_min(embedding)):.3f}, {float(tf.reduce_max(embedding)):.3f}]",
        flush=True,
    )

    reference_norm = float(tf.norm(embedding))

    print("\noptimizing the embedding directly (no recurrent backprop)...", flush=True)
    _, free = ascend(head, embedding, target, args.steps, args.learning_rate)
    _, matched = ascend(
        head,
        embedding,
        target,
        args.steps,
        args.learning_rate,
        max_norm=reference_norm,
    )

    print(f"\n{'bound':<38} {'max p(target)':>14}")
    print("-" * 54)
    print(f"{'observed in latent-space runs':<38} {0.5413:>14.4f}")
    print(f"{f'norm-matched embedding (|e|<={reference_norm:.2f})':<38} {matched:>14.4f}")
    print(f"{'unconstrained embedding':<38} {free:>14.4f}")

    # How much do real trials differ in embedding space? If barely at all, the
    # recurrent summarizer is ignoring its input and no search can help.
    other = args.trial_id + 1 if args.trial_id == 0 else args.trial_id - 1
    print(f"\ncomparing against trial {other} for embedding spread...", flush=True)
    args_other = argparse.Namespace(**{**vars(args), "trial_id": other})
    other_trial, _ = load_one_trial(args_other)
    other_embedding, _ = embed(model, other_trial)
    delta = float(tf.norm(embedding - other_embedding))
    cosine = float(
        tf.reduce_sum(embedding * other_embedding)
        / (tf.norm(embedding) * tf.norm(other_embedding))
    )
    print(
        f"  trial {args.trial_id} vs {other}: L2={delta:.5f} cosine={cosine:.6f} "
        f"({100.0 * delta / reference_norm:.3f}% of embedding norm)"
    )

    print("\ninterpretation:")
    if matched < 0.60:
        print(
            "  Even at the scale real embeddings occupy, the head cannot express\n"
            "  high confidence. The plateau is a property of the classifier and no\n"
            "  search method will beat it. Report it as a measured limitation."
        )
    else:
        print(
            f"  The head reaches p={matched:.4f} at realistic embedding scale, while\n"
            f"  latent-space search only achieves 0.5413. That gap is an optimization\n"
            f"  failure, not a model limit -- most likely gradient attenuation through\n"
            f"  {int(sequence.shape[1])} recurrent steps. A gradient-free search over the\n"
            "  latent, or a two-stage objective targeting this embedding, should close it."
        )
    if cosine > 0.999:
        print(
            f"\n  WARNING: two different trials embed almost identically "
            f"(cosine={cosine:.6f}).\n"
            "  The recurrent summarizer is largely ignoring its input, which would\n"
            "  explain ~0.50 confidence on every trial regardless of counterfactuals."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
