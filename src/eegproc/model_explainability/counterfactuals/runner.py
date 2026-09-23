"""Load a saved SIC model, optimize selected trials, and write diagnostics.

Use --help for arguments (defined separately in arguments.py).
No model is rebuilt or trained. Prepared NPZ trials are used as-is; raw mode
reuses the existing SIC data preparation functions with explicit settings.
"""

import csv
import importlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

if __package__:
    from .arguments import parse_args
    from .loss import CounterfactualLoss
    from .optimizer import CounterfactualOptimizer
else:
    from arguments import parse_args
    from loss import CounterfactualLoss
    from optimizer import CounterfactualOptimizer


OBJECTIVE_TERMS = ("target", "latent", "decoded", "physiological")


def format_optimization_diagnostics(row):
    """Format raw terms, weighted contributions, and total-loss shares."""
    total = float(row["total"])
    raw = " ".join(f"{name}={row[name]:.6g}" for name in OBJECTIVE_TERMS)
    weighted = " ".join(
        f"{name}={row[f'weighted_{name}']:.6g}" for name in OBJECTIVE_TERMS
    )
    if total > np.finfo(float).eps:
        shares = " ".join(
            f"{name}={100.0 * row[f'weighted_{name}'] / total:.1f}%"
            for name in OBJECTIVE_TERMS
        )
    else:
        shares = " ".join(f"{name}=n/a" for name in OBJECTIVE_TERMS)
    gradient = row["gradient_norm"]
    gradient_text = "n/a" if gradient is None else f"{gradient:.6g}"
    learning_rate = row.get("learning_rate")
    learning_rate_text = (
        "" if learning_rate is None else f"lr={learning_rate:.6g} "
    )
    return (
        f"step={row['step']} total={total:.6g} "
        f"{learning_rate_text}"
        f"RAW[{raw}] WEIGHTED[{weighted}] SHARE[{shares}] "
        f"target_p={row['target_probability']:.4f} "
        f"predicted={row['predicted_class']} grad={gradient_text} "
        f"success={row['success']}"
    )


def load_model(path, module):
    """Import custom class registrations, then load .keras with compile=False.

    Keras safe mode remains enabled. The saved architecture, weights, fixed MI
    adjacency, recurrent head, and decoders are restored; no builder or fit
    function is called. A decoder's presence does not prove it was trained:
    the caller must select a checkpoint trained with reconstruction enabled.
    """
    path = Path(path)
    if not path.is_file() or path.suffix != ".keras":
        raise ValueError(f"Expected an existing full .keras model: {path}")
    importlib.import_module(module)
    return tf.keras.models.load_model(path, compile=False, safe_mode=True)


def load_trials(args):
    """Return features, subject IDs, trial IDs, and optional integer labels.

    Prepared NPZ keys: features (N,W,T,F), subject_ids (N,), trial_ids (N,),
    optional labels (N,). Features must already have the training window order,
    preprocessing, normalization, and label filtering. Object/pickled arrays,
    duplicate subject/trial pairs, and padded trials are rejected. Real EEG
    zeros are not treated as padding. No time-axis pooling is performed.

    Raw mode uses SIC's existing loader and grouping helper, not a copy of
    their implementation. That loader preserves chronological window order
    and requires equal window counts across trials. Removing median ratings
    is done before grouping. The checkpoint alone cannot verify preprocessing.
    """
    if args.trials_npz:
        with np.load(args.trials_npz, allow_pickle=False) as data:
            required = {"features", "subject_ids", "trial_ids"}
            if not required.issubset(data.files):
                raise ValueError(f"NPZ requires keys {sorted(required)}.")
            x, subjects, trials = (
                data[key] for key in ("features", "subject_ids", "trial_ids")
            )
            y = data["labels"] if "labels" in data.files else None
            if "window_mask" in data.files:
                mask = data["window_mask"]
                if mask.shape != x.shape[:2] or not np.all(mask == 1):
                    raise ValueError(
                        "Padded/masked trials are unsupported by this SIC full-sequence path."
                    )
    else:
        training = importlib.import_module(
            args.model_module.rsplit(".", 1)[0] + ".sic_model_train"
        )
        x, y, subjects, trials, ratings = training.load_sic_training_data(
            eeg_path=args.raw_eeg_npy,
            labels_path=args.raw_labels_npy,
            dataset=args.dataset,
            label_dimension=args.label_dimension,
            window_size_sec=args.window_sec,
            fs=args.fs,
            overlap=args.window_overlap,
            window_normalization=args.window_normalization,
            label_threshold_mode=args.label_threshold_mode,
            median_label=args.median_label,
            return_original_ratings=True,
        )
        if args.remove_median_label:
            keep = ~np.isclose(ratings, args.median_label)
            x, y, subjects, trials = (
                np.asarray(v)[keep] for v in (x, y, subjects, trials)
            )
        x, y, subjects, trials = training._group_windows_into_trials(
            x, y, subjects, trials
        )
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4 or any(d < 1 for d in x.shape) or not np.isfinite(x).all():
        raise ValueError("features must be finite, nonempty, and shaped (N,W,T,F).")
    for name, values in (
        ("subject_ids", subjects),
        ("trial_ids", trials),
        ("labels", y),
    ):
        if values is not None and (
            values.shape != (len(x),) or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError(f"{name} must contain one integer per trial.")
    if len(set(zip(subjects.tolist(), trials.tolist()))) != len(x):
        raise ValueError(
            "Each (subject_id, trial_id) must identify exactly one complete trial."
        )
    return x, subjects, trials, y


def _write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run(args):
    """Orchestrate a run; step optimization remains entirely in the optimizer.

    Existing nonempty output directories are refused. Each completed trial is
    written immediately, so results survive a later trial's failure. Step
    metrics are optimization diagnostics, not estimates of classifier accuracy.
    Counterfactual success is measured directly in the optimized latent state.
    Reconstructions are reported only for signal distance and physiology.
    """
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {out}")
    tf.keras.utils.set_random_seed(args.seed)
    model = load_model(args.model, args.model_module)
    loss = CounterfactualLoss(
        target_weight=args.target_weight,
        latent_weight=args.latent_weight,
        decoded_weight=args.decoded_weight,
        physiological_weight=args.physiological_weight,
        target_probability=args.target_probability,
        vcsc_distance_cm=args.vcsc_distance_cm,
        vcsc_tau_cm=args.vcsc_tau_cm,
        vcsc_z0=args.vcsc_z0,
        vcsc_z_max=args.vcsc_z_max,
    )
    optimizer = CounterfactualOptimizer(
        model,
        loss=loss,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        target_loss_component=args.target_loss_component,
        max_steps=args.max_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        stop_on_success=args.stop_on_success,
        decoder_mode=args.decoder_mode,
    )
    x, subjects, trials, labels = load_trials(args)
    selected = subjects == args.subject_id
    if args.trial_ids is not None:
        missing = set(args.trial_ids) - set(trials[selected].tolist())
        if missing:
            raise ValueError(
                f"Requested trials not found for subject {args.subject_id}: {sorted(missing)}"
            )
        selected &= np.isin(trials, args.trial_ids)
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError(f"No trials found for subject {args.subject_id}.")
    out.mkdir(parents=True, exist_ok=True)
    _write_json(
        out / "settings.json",
        {
            "round_trip_evaluation": "latent_only",
            "counterfactual_validity_prediction_space": "latent",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "loss": asdict(loss),
            "tensorflow_version": tf.__version__,
            "trial_input_shape": list(x.shape[1:]),
            "selected_trial_ids": trials[indices].tolist(),
            "physiological_constraint_enforced": loss.physiological_weight > 0,
            "preprocessing_match": "caller must use the settings used to train this checkpoint",
            "checkpoint_subject_match": "caller must select the intended LOSO checkpoint",
        },
    )
    print(
        f"Model: {args.model}\nTrials: {len(indices)} | input per trial: {x.shape[1:]}",
        flush=True,
    )
    print(
        f"Decoder mode: {optimizer.decoder_mode} | "
        f"reconstruction paths: {', '.join(optimizer.decoded_names)}",
        flush=True,
    )
    print(
        f"Target loss component: {optimizer.target_loss_component}",
        flush=True,
    )
    if optimizer.decoder_mode == "joint":
        alpha = float(model.joint_reconstruction_fusion.alpha.numpy())
        print(
            "Frozen joint reconstruction weights: "
            f"GCN-GRU alpha={alpha:.6g} | BiLSTM 1-alpha={1.0 - alpha:.6g}",
            flush=True,
        )
    physiological_status = (
        "computed and logged, but diagnostic only"
        if loss.physiological_weight == 0
        else "active in the objective"
    )
    print(
        "Physiological validity: VCSC | "
        f"weight={loss.physiological_weight:.6g} ({physiological_status})",
        flush=True,
    )
    summaries = []
    for index in indices:
        subject_id, trial_id = int(subjects[index]), int(trials[index])
        print(
            f"\n{'-' * 72}\nSubject={subject_id} trial={trial_id} | counterfactual optimization",
            flush=True,
        )

        def report(row):
            if args.log_every and row["step"] % args.log_every == 0:
                print(format_optimization_diagnostics(row), flush=True)

        result = optimizer.optimize(
            x[index : index + 1], target_class=args.target_class, progress=report
        )
        summary = {"subject_id": subject_id, "trial_id": trial_id, **result["summary"]}
        if labels is not None:
            summary["true_class"] = int(labels[index])
        trial_dir = out / f"subject_{subject_id}_trial_{trial_id}"
        trial_dir.mkdir()
        with (trial_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["history"][0]))
            writer.writeheader()
            writer.writerows(result["history"])
        _write_json(trial_dir / "result.json", summary)
        summaries.append(summary)
        _write_json(out / "results.json", summaries)
        before, after = summary["original"], summary["latent_counterfactual"]
        print(
            f"Original: class={before['predicted_class']} target_p={before['target_probability']:.4f}",
            flush=True,
        )
        print(
            f"Selected step={summary['selected_step']} of {summary['steps_completed']} | latent class={after['predicted_class']} target_p={after['target_probability']:.4f} success={after['success']} | stop={summary['stop_reason']}",
            flush=True,
        )
        for branch, details in summary["decoded_trials"].items():
            print(
                f"Decoded {branch}: MSE_to_x={details['counterfactual_to_original_mse']:.6g} "
                f"change_MSE={details['decoded_change_mse']:.6g}",
                flush=True,
            )
            print(
                f"VCSC {branch}: "
                f"reconstructed_original={details['vcsc_original_reconstruction']:.6g} "
                f"counterfactual={details['vcsc_counterfactual']:.6g} "
                f"delta={details['vcsc_delta']:+.6g}",
                flush=True,
            )
    aggregate = {
        "n_trials": len(summaries),
        "round_trip_evaluation": summaries[0]["round_trip_evaluation"],
        "latent_success_rate": float(
            np.mean([s["latent_counterfactual"]["success"] for s in summaries])
        ),
        "mean_selected_latent_mse": float(
            np.mean([s["selected_losses"]["latent"] for s in summaries])
        ),
        "mean_selected_decoded_mse": float(
            np.mean([s["selected_losses"]["decoded"] for s in summaries])
        ),
        "mean_selected_physiological": float(
            np.mean([s["selected_losses"]["physiological"] for s in summaries])
        ),
        "mean_vcsc_reconstructed_original": {
            name: float(
                np.mean(
                    [
                        s["decoded_trials"][name][
                            "vcsc_original_reconstruction"
                        ]
                        for s in summaries
                    ]
                )
            )
            for name in optimizer.decoded_names
        },
        "mean_vcsc_counterfactual": {
            name: float(
                np.mean(
                    [
                        s["decoded_trials"][name]["vcsc_counterfactual"]
                        for s in summaries
                    ]
                )
            )
            for name in optimizer.decoded_names
        },
        "mean_vcsc_delta": {
            name: float(
                np.mean(
                    [s["decoded_trials"][name]["vcsc_delta"] for s in summaries]
                )
            )
            for name in optimizer.decoded_names
        },
        "latent_class_flip_rate": float(
            np.mean(
                [
                    s["latent_counterfactual"]["predicted_class"]
                    != s["original"]["predicted_class"]
                    for s in summaries
                ]
            )
        ),
        "mean_target_probability_delta": float(
            np.mean(
                [
                    s["latent_counterfactual"]["target_probability"]
                    - s["original"]["target_probability"]
                    for s in summaries
                ]
            )
        ),
        "physiological_constraint_enforced": loss.physiological_weight > 0,
    }
    _write_json(out / "summary.json", aggregate)
    print(f"\nRun summary:\n{json.dumps(aggregate, indent=2)}", flush=True)
    return aggregate


def main(argv=None):
    """Parse arguments separately, then invoke the runtime orchestration."""
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
