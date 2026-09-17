"""Measure where a finished counterfactual actually landed in embedding space.

Every latent-space run plateaus at target_p ~= 0.541, while optimizing the
trial embedding directly reaches 0.99. So the classifier head is not the
limit. What is unknown is WHY the latent search stops short, and there are
two very different possibilities:

  distance -- the search moves the embedding along the right direction but
  not far enough. Then the fix is a stronger or better-conditioned search
  (gradient-free, or a two-stage objective that targets the embedding).

  direction -- the search moves the embedding a long way, but mostly
  perpendicular to the axis that separates the classes, so the motion is
  wasted. Then the fix is to change what the objective points at, not how
  hard it pushes.

This script settles it by decomposing the displacement. It reads z and
z_prime from a completed run's counterfactual.npz -- no optimization, just
two forward passes through the recurrent summarizer -- and splits the
embedding displacement into the component along the discriminative axis and
the component orthogonal to it.

The discriminative axis comes from the variational head itself. Its logits
are log-Gaussian likelihoods, so the class gap is

    gap(e) = sum_d [ a_d e_d^2 + b_d e_d ] + const,
    a_d = 0.5 (1/sigma0_d^2 - 1/sigma1_d^2),
    b_d = mu1_d/sigma1_d^2 - mu0_d/sigma0_d^2.

The two per-class sigmas are near-identical in the trained checkpoint
(1.00204 vs 1.00203), so a is negligible and the gap is effectively linear
with gradient b. Motion along b/||b|| is the only motion that changes the
prediction; everything orthogonal to it is wasted effort.

Usage:
    PYTHONPATH=src python -m eegproc.model_explainability.embedding_displacement \\
        --model runs/.../loso_fold_0001_target_0_zero_shot.keras \\
        --run-dir runs/counterfactuals/s0_tw50_t0/subject_0_trial_0
"""

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_parser():
    parser = argparse.ArgumentParser(
        description="Decompose a counterfactual's embedding displacement."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-module",
        default="eegproc.deep_learning.joint_architectures.SICModelv11.sic_model",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        nargs="+",
        help="One or more subject_N_trial_M directories containing counterfactual.npz.",
    )
    return parser


def embed(model, latent):
    """Flatten windows and timesteps, then summarize, as _classify does."""
    latent = tf.cast(latent, tf.float32)
    sequence = tf.reshape(latent, [1, -1, latent.shape[-1]])
    return model.trial_recurrent_classifier(sequence, training=False)


def head_geometry(model):
    """Return (b, quadratic_scale) describing the class-separating direction."""
    head = model.vc_target
    layer = getattr(head, "logits_layer", head)
    mu = layer.prior_mu.numpy()
    sigma = np.exp(layer.prior_log_sigma.numpy())
    b = mu[1] / sigma[1] ** 2 - mu[0] / sigma[0] ** 2
    a = 0.5 * (1.0 / sigma[0] ** 2 - 1.0 / sigma[1] ** 2)
    return b, a, mu, sigma


def mahalanobis(embedding, mu, sigma):
    """Distance to one class Gaussian, in units of that class's own sigma."""
    return float(np.sqrt(np.sum(((embedding - mu) / sigma) ** 2)))


def crossing_verdict(distances_original, distances_counterfactual, target_class):
    """Did the counterfactual cross to the other class, or just leave the data?

    In a binary problem, leaving the source class's distribution while
    remaining inside the data domain necessarily means entering the target's.
    The failure mode is leaving BOTH -- moving somewhere no real trial of
    either class lives. The two are distinguishable: a genuine crossing moves
    TOWARD the target Gaussian and AWAY from the source, whereas an escape
    moves away from both.

    distances_* are indexed by class, so the target must be supplied rather
    than assumed. An earlier version hardcoded target=1 and consequently
    reported genuine crossings as "moved toward the source class".
    """
    source_class = 1 - target_class
    toward_target = distances_counterfactual[target_class] < distances_original[target_class]
    away_from_source = (
        distances_counterfactual[source_class] > distances_original[source_class]
    )
    if toward_target and away_from_source:
        return "CROSSED: moved toward the target class and away from the source"
    if away_from_source and not toward_target:
        return "ESCAPED: left the source class but did NOT approach the target"
    if toward_target and not away_from_source:
        return "approached the target class without leaving the source"
    return "moved toward the source class (unexpected)"


def gap_of(model, embedding):
    logits = tf.cast(model.vc_target(embedding, training=False), tf.float32).numpy()[0]
    return float(logits[1] - logits[0])


def main(argv=None):
    args = build_parser().parse_args(argv)
    importlib.import_module(args.model_module)
    model = tf.keras.models.load_model(args.model, compile=False, safe_mode=True)

    b, a, mu, sigma = head_geometry(model)
    axis = b / np.linalg.norm(b)
    print(
        f"head geometry: ||b||={np.linalg.norm(b):.4f} "
        f"|a|max={np.abs(a).max():.2e} (negligible => gap is ~linear)"
    )
    print(f"||mu1-mu0||={np.linalg.norm(mu[1] - mu[0]):.4f} sigma~{sigma.mean():.5f}\n")

    real_reference = []
    for run_dir in args.run_dir:
        npz = Path(run_dir) / "counterfactual.npz"
        if not npz.is_file():
            print(f"SKIP {run_dir}: no counterfactual.npz")
            continue
        with np.load(npz, allow_pickle=False) as data:
            if "z" not in data.files or "z_prime" not in data.files:
                print(f"SKIP {run_dir}: keys are {sorted(data.files)}")
                continue
            z, z_prime = data["z"], data["z_prime"]

        result_path = Path(run_dir) / "result.json"
        if not result_path.is_file():
            print(f"SKIP {run_dir}: no result.json, cannot determine target class")
            continue
        with result_path.open(encoding="utf-8") as handle:
            target_class = int(json.load(handle)["target_class"])

        e0 = embed(model, z).numpy()[0]
        e1 = embed(model, z_prime).numpy()[0]
        displacement = e1 - e0
        along = float(np.dot(displacement, axis))
        orthogonal = float(np.linalg.norm(displacement - along * axis))

        gap0, gap1 = gap_of(model, e0[None]), gap_of(model, e1[None])
        p1 = 1.0 / (1.0 + math.exp(-abs(gap1)))

        print(f"=== {run_dir} ===")
        print(
            f"  embedding norm: original {np.linalg.norm(e0):8.4f} "
            f"-> counterfactual {np.linalg.norm(e1):8.4f}"
        )
        print(f"  logit gap:      {gap0:+8.5f} -> {gap1:+8.5f}   p(target)={p1:.4f}")
        print(f"  displacement:   ||de|| = {np.linalg.norm(displacement):.4f}")
        print(
            f"    along discriminative axis : {along:+8.4f}  "
            f"({100.0 * abs(along) / max(np.linalg.norm(displacement), 1e-12):5.1f}% of motion)"
        )
        print(
            f"    orthogonal (wasted)       : {orthogonal:8.4f}  "
            f"({100.0 * orthogonal / max(np.linalg.norm(displacement), 1e-12):5.1f}% of motion)"
        )
        # Crossed to the other class, or left the data domain entirely?
        d_orig = [mahalanobis(e0, mu[c], sigma[c]) for c in range(len(mu))]
        d_cf = [mahalanobis(e1, mu[c], sigma[c]) for c in range(len(mu))]
        source_class = 1 - target_class
        print(
            f"  Mahalanobis distance (target=class {target_class}, "
            f"source=class {source_class}):"
        )
        print(
            f"    original      : target {d_orig[target_class]:8.3f}  "
            f"source {d_orig[source_class]:8.3f}  "
            f"(margin {d_orig[source_class] - d_orig[target_class]:+.3f})"
        )
        print(
            f"    counterfactual: target {d_cf[target_class]:8.3f}  "
            f"source {d_cf[source_class]:8.3f}  "
            f"(margin {d_cf[source_class] - d_cf[target_class]:+.3f})"
        )
        print(f"    -> {crossing_verdict(d_orig, d_cf, target_class)}")
        real_reference.append((d_orig[0], d_orig[1]))
        print()

    if real_reference:
        d0s = [d for d, _ in real_reference]
        d1s = [d for _, d in real_reference]
        print(
            f"real-trial reference ({len(real_reference)} originals): "
            f"class0 distance {min(d0s):.3f}-{max(d0s):.3f}, "
            f"class1 distance {min(d1s):.3f}-{max(d1s):.3f}"
        )
        print(
            "A counterfactual whose distances sit far outside these ranges has left\n"
            "the data domain rather than crossing between classes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
