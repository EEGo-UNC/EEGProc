"""Smoke-test training script for DREAMER joined data using project CNN1D classes.

This intentionally does NOT wrap or redefine the CNN1D architecture.
It simply imports:

    CNN1DEncoder
    CNN1DDecoder
    VariationalAutoencoderLoss

and trains:

    z = encoder(x)
    x_hat = decoder(z)
    loss = reconstruction_loss(x, x_hat) + beta * KL(N(z, I) || N(0, I))

Because CNN1DEncoder currently returns one latent tensor, this script treats that
latent tensor as z_mean and uses z_log_var = 0. That makes this a deterministic
autoencoder with a variational/KL target on the latent representation. A full VAE
would require the encoder to output both z_mean and z_log_var, plus sampling.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

# Keep these direct and boring. Adjust the import paths only if this script lives
# somewhere else in your package.
from ..Convolutions.CNN1D import CNN1DEncoder, CNN1DDecoder
from ..VariationalAutoencoderLoss import VariationalAutoencoderLoss


EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


PARAM_GRID: dict[str, list[Any]] = {
    # Extremely tiny values on purpose: this is only a smoke test.
    "conv_filters": [(4,)],
    "kernel_sizes": [(3,)],
    "pool_after_layers": [(0,)],
    "pool_sizes": [(2,)],
    "t_down": [2],
    "emb_dim": [4],
    "dropout": [0.0],
    "use_batch_norm": [False],
    "activation": ["relu"],
    "reconstruction": ["mse", "huber"],
    "beta": [0.0, 1e-3],
    "huber_delta": [1.0],
    "learning_rate": [1e-3],
    "batch_size": [8],
}


def iter_grid(param_grid: dict[str, list[Any]]):
    keys = list(param_grid.keys())
    for values in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, values))


def load_dreamer_windows(
    csv_path: str | Path,
    window_size: int,
    stride: int,
    max_windows: int | None,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Load DREAMER joined CSV and convert rows into EEG windows.

    Expected columns include:
        subject_id, trial_id, segment, sample_idx, EEG channels...

    Output shape:
        X: (n_windows, window_size, 14)
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in EEG_CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing EEG channel columns: {missing}")

    group_cols = [c for c in ["subject_id", "trial_id", "segment"] if c in df.columns]
    sort_cols = group_cols + (["sample_idx"] if "sample_idx" in df.columns else [])

    if sort_cols:
        df = df.sort_values(sort_cols)

    windows: list[np.ndarray] = []

    if group_cols:
        groups = df.groupby(group_cols, sort=False)
    else:
        groups = [(None, df)]

    for _, group in groups:
        values = group[EEG_CHANNELS].to_numpy(dtype=np.float32)

        if len(values) < window_size:
            continue

        for start in range(0, len(values) - window_size + 1, stride):
            windows.append(values[start : start + window_size])

            if max_windows is not None and len(windows) >= max_windows:
                break

        if max_windows is not None and len(windows) >= max_windows:
            break

    if not windows:
        raise ValueError(
            "No windows were created. Check window_size, stride, and CSV format."
        )

    X = np.stack(windows).astype(np.float32)

    rng = np.random.default_rng(seed)
    rng.shuffle(X)

    return X, EEG_CHANNELS


def train_val_split(
    X: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    val_size = max(1, int(len(X) * val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    if len(train_idx) == 0:
        raise ValueError("Not enough windows for train/validation split.")

    return X[train_idx], X[val_idx]


def normalize_from_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    return X_train_norm.astype(np.float32), X_val_norm.astype(np.float32), mean, std


def make_dataset(
    X: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(X)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def flatten_latent_for_kl(z: tf.Tensor) -> tf.Tensor:
    """Convert latent sequence (batch, latent_steps, emb_dim) to (batch, latent_dim)."""
    return tf.reshape(z, [tf.shape(z)[0], -1])


def build_models(
    hp: dict[str, Any],
    window_size: int,
    n_features: int,
) -> tuple[CNN1DEncoder, CNN1DDecoder]:
    encoder = CNN1DEncoder(
        timesteps=window_size,
        n_features=n_features,
        t_down=hp["t_down"],
        conv_filters=hp["conv_filters"],
        kernel_sizes=hp["kernel_sizes"],
        pool_after_layers=hp["pool_after_layers"],
        pool_sizes=hp["pool_sizes"],
        emb_dim=hp["emb_dim"],
        dropout=hp["dropout"],
        activation=hp["activation"],
        use_batch_norm=hp["use_batch_norm"],
        name="cnn1d_encoder",
    )

    decoder = CNN1DDecoder.from_encoder(encoder, name="cnn1d_decoder")

    return encoder, decoder


def train_one_epoch(
    encoder: CNN1DEncoder,
    decoder: CNN1DDecoder,
    loss_fn: VariationalAutoencoderLoss,
    optimizer: tf.keras.optimizers.Optimizer,
    train_ds: tf.data.Dataset,
) -> dict[str, float]:
    totals = []
    recons = []
    kls = []

    for x in train_ds:
        with tf.GradientTape() as tape:
            z = encoder(x, training=True)
            x_hat = decoder(z, training=True)

            z_mean = flatten_latent_for_kl(z)
            z_log_var = tf.zeros_like(z_mean)

            losses = loss_fn(
                x_true=x,
                x_pred=x_hat,
                z_mean=z_mean,
                z_log_var=z_log_var,
            )

        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(losses["total_loss"], variables)
        optimizer.apply_gradients(zip(gradients, variables))

        totals.append(float(losses["total_loss"].numpy()))
        recons.append(float(losses["reconstruction_loss"].numpy()))
        kls.append(float(losses["kl_loss"].numpy()))

    return {
        "total_loss": float(np.mean(totals)),
        "reconstruction_loss": float(np.mean(recons)),
        "kl_loss": float(np.mean(kls)),
    }


def evaluate(
    encoder: CNN1DEncoder,
    decoder: CNN1DDecoder,
    loss_fn: VariationalAutoencoderLoss,
    val_ds: tf.data.Dataset,
) -> dict[str, float]:
    totals = []
    recons = []
    kls = []

    for x in val_ds:
        z = encoder(x, training=False)
        x_hat = decoder(z, training=False)

        z_mean = flatten_latent_for_kl(z)
        z_log_var = tf.zeros_like(z_mean)

        losses = loss_fn(
            x_true=x,
            x_pred=x_hat,
            z_mean=z_mean,
            z_log_var=z_log_var,
        )

        totals.append(float(losses["total_loss"].numpy()))
        recons.append(float(losses["reconstruction_loss"].numpy()))
        kls.append(float(losses["kl_loss"].numpy()))

    return {
        "total_loss": float(np.mean(totals)),
        "reconstruction_loss": float(np.mean(recons)),
        "kl_loss": float(np.mean(kls)),
    }


def run_grid_search(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, channels = load_dreamer_windows(
        csv_path=args.csv_path,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        seed=args.seed,
    )

    X_train, X_val = train_val_split(
        X,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    X_train, X_val, mean, std = normalize_from_train(X_train, X_val)

    np.save(out_dir / "channel_mean.npy", mean)
    np.save(out_dir / "channel_std.npy", std)

    results = []
    best_val_loss = float("inf")
    best_hp = None

    for run_idx, hp in enumerate(iter_grid(PARAM_GRID), start=1):
        tf.keras.backend.clear_session()
        tf.random.set_seed(args.seed + run_idx)

        train_ds = make_dataset(
            X_train,
            batch_size=hp["batch_size"],
            shuffle=True,
            seed=args.seed,
        )
        val_ds = make_dataset(
            X_val,
            batch_size=hp["batch_size"],
            shuffle=False,
            seed=args.seed,
        )

        encoder, decoder = build_models(
            hp=hp,
            window_size=args.window_size,
            n_features=len(channels),
        )

        loss_fn = VariationalAutoencoderLoss(
            reconstruction=hp["reconstruction"],
            beta=hp["beta"],
            feature_reduction="mean",
            huber_delta=hp["huber_delta"],
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"])

        last_train_metrics = None
        for epoch in range(args.epochs):
            last_train_metrics = train_one_epoch(
                encoder=encoder,
                decoder=decoder,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_ds=train_ds,
            )

        val_metrics = evaluate(
            encoder=encoder,
            decoder=decoder,
            loss_fn=loss_fn,
            val_ds=val_ds,
        )

        row = {
            "run_idx": run_idx,
            "epochs": args.epochs,
            **hp,
            "train_total_loss": last_train_metrics["total_loss"],
            "train_reconstruction_loss": last_train_metrics["reconstruction_loss"],
            "train_kl_loss": last_train_metrics["kl_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_reconstruction_loss": val_metrics["reconstruction_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
        }
        results.append(row)

        print(
            f"run={run_idx} "
            f"reconstruction={hp['reconstruction']} "
            f"beta={hp['beta']} "
            f"val_total={val_metrics['total_loss']:.6f} "
            f"val_recon={val_metrics['reconstruction_loss']:.6f} "
            f"val_kl={val_metrics['kl_loss']:.6f}"
        )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_hp = row
            encoder.save_weights(out_dir / "best_encoder.weights.h5")
            decoder.save_weights(out_dir / "best_decoder.weights.h5")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "grid_results.csv", index=False)

    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_hp, f, indent=2, default=str)

    print(f"\nSaved results to: {out_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_dir", default="./runs/dreamer_cnn1d_variational_smoke")
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max_windows", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    run_grid_search(parse_args())
