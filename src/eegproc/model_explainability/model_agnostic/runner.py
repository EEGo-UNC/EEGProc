"""CLI for adapter-based latent- or input-space counterfactual optimization."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from .adapter import (
    create_adapter,
    load_json_mapping,
    load_trial_dataset,
)
from .optimizer import (
    ModelAgnosticCounterfactualOptimizer,
)


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize counterfactuals through a model adapter. Existing SIC-specific "
            "commands remain available in counterfactuals.runner."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        required=True,
        help="Adapter factory as package.module:function.",
    )
    parser.add_argument(
        "--adapter-config",
        help="Inline JSON object or path to a JSON object passed to the adapter.",
    )
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--trials-npz", type=Path)
    data.add_argument(
        "--data-loader",
        help="Dataset loader as package.module:function; it must return TrialDataset.",
    )
    parser.add_argument(
        "--data-config",
        help="Inline JSON object or path to a JSON object passed to the dataset loader.",
    )
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument("--trial-ids", "--trial-id", type=int, nargs="+")
    parser.add_argument("--target-class", type=_nonnegative_int)
    parser.add_argument("--target-probability", type=_positive_float, default=0.8)
    parser.add_argument("--learning-rate", type=_positive_float, default=0.01)
    parser.add_argument("--max-steps", type=_nonnegative_int, default=200)
    parser.add_argument("--gradient-clip-norm", type=_positive_float, default=5.0)
    parser.add_argument("--target-weight", type=_positive_float, default=1.0)
    parser.add_argument(
        "--state-weight",
        type=_nonnegative_float,
        help="Default comes from the selected adapter.",
    )
    parser.add_argument(
        "--signal-weight",
        type=_nonnegative_float,
        help="Default comes from the selected adapter.",
    )
    parser.add_argument(
        "--constraint-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help="Activate an adapter-provided constraint; repeat for multiple metrics.",
    )
    parser.add_argument(
        "--report-constraint",
        action="append",
        default=[],
        metavar="NAME",
        help="Evaluate a diagnostic only on the final reference and counterfactual.",
    )
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--log-every", type=_nonnegative_int, default=1)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def _parse_constraint_weights(entries):
    weights = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("Constraint weights must use NAME=WEIGHT.")
        name, value = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Constraint names cannot be empty.")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Constraint weight for {name!r} must be nonnegative.")
        if name in weights:
            raise ValueError(f"Constraint {name!r} was supplied more than once.")
        weights[name] = weight
    return weights


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target_probability >= 1:
        parser.error("--target-probability must be strictly below 1")
    try:
        args.adapter_config = load_json_mapping(args.adapter_config)
        args.data_config = load_json_mapping(args.data_config)
        args.constraint_weights = _parse_constraint_weights(args.constraint_weight)
    except (ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return args


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _json_arguments(args):
    excluded = {"constraint_weight"}
    output = {}
    for name, value in vars(args).items():
        if name in excluded:
            continue
        output[name] = str(value) if isinstance(value, Path) else value
    return output


def _metadata_arrays(dataset, index):
    arrays = {}
    if dataset.normalization_offset is not None:
        arrays["normalization_offset"] = dataset.normalization_offset[index]
    if dataset.normalization_scale is not None:
        arrays["normalization_scale"] = dataset.normalization_scale[index]
    if dataset.channel_names is not None:
        arrays["channel_names"] = np.asarray(dataset.channel_names, dtype=str)
    if dataset.band_names is not None:
        arrays["band_names"] = np.asarray(dataset.band_names, dtype=str)
    if dataset.channel_positions is not None:
        arrays["channel_positions"] = np.asarray(dataset.channel_positions, dtype=float)
    if dataset.feature_order is not None:
        arrays["feature_order"] = np.asarray(dataset.feature_order)
    if dataset.signal_unit is not None:
        arrays["signal_unit"] = np.asarray(dataset.signal_unit)
    return arrays


def _format_progress(row):
    extras = " ".join(
        f"{name.removeprefix('constraint_')}={value:.6g}"
        for name, value in row.items()
        if name.startswith("constraint_")
        and not name.startswith("constraint_weight")
    )
    extras = f" constraints[{extras}]" if extras else ""
    return (
        f"step={row['step']} total={row['total']:.6g} "
        f"target={row['target']:.6g} state={row['state']:.6g} "
        f"signal={row['signal']:.6g} target_p={row['target_probability']:.4f} "
        f"predicted={row['predicted_class']} grad={row['gradient_norm']} "
        f"success={row['success']}{extras}"
    )


def run(args):
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {out}")
    dataset = load_trial_dataset(
        args.trials_npz,
        loader_spec=args.data_loader,
        loader_config=args.data_config,
    )
    selected = np.asarray(dataset.subject_ids) == args.subject_id
    if args.trial_ids is not None:
        available = set(np.asarray(dataset.trial_ids)[selected].tolist())
        missing = set(args.trial_ids) - available
        if missing:
            raise ValueError(
                f"Requested trials not found for subject {args.subject_id}: {sorted(missing)}"
            )
        selected &= np.isin(dataset.trial_ids, args.trial_ids)
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError(f"No trials found for subject {args.subject_id}.")

    adapter = create_adapter(
        args.adapter,
        model_path=args.model,
        config=args.adapter_config,
        sample_input=dataset.features[indices[0]],
    )
    optimizer = ModelAgnosticCounterfactualOptimizer(
        adapter,
        target_probability=args.target_probability,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        target_weight=args.target_weight,
        state_weight=args.state_weight,
        signal_weight=args.signal_weight,
        constraint_weights=args.constraint_weights,
        report_constraints=args.report_constraint,
        stop_on_success=args.stop_on_success,
    )
    tf.keras.utils.set_random_seed(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": _json_arguments(args),
        "adapter": adapter.metadata(),
        "dataset_metadata": dataset.metadata,
        "selected_trial_ids": dataset.trial_ids[indices].tolist(),
        "physical_unit_available": bool(
            dataset.normalization_scale is not None and dataset.signal_unit
        ),
    }
    _write_json(out / "settings.json", settings)
    summaries = []
    for index in indices:
        subject_id = int(dataset.subject_ids[index])
        trial_id = int(dataset.trial_ids[index])
        print(f"Subject={subject_id} trial={trial_id} | adapter={adapter.name}", flush=True)

        def progress(row):
            if args.log_every and row["step"] % args.log_every == 0:
                print(_format_progress(row), flush=True)

        result = optimizer.optimize(
            dataset.features[index : index + 1],
            target_class=args.target_class,
            progress=progress,
        )
        summary = {"subject_id": subject_id, "trial_id": trial_id, **result["summary"]}
        if dataset.labels is not None:
            summary["true_class"] = int(dataset.labels[index])
        trial_dir = out / f"subject_{subject_id}_trial_{trial_id}"
        trial_dir.mkdir()
        with (trial_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["history"][0]))
            writer.writeheader()
            writer.writerows(result["history"])
        _write_json(trial_dir / "result.json", summary)
        summaries.append(summary)
        _write_json(out / "results.json", summaries)
        print(
            f"Selected step={summary['selected_step']} | "
            f"target_p={summary['counterfactual']['target_probability']:.4f} "
            f"success={summary['counterfactual']['success']} "
            f"stop={summary['stop_reason']}",
            flush=True,
        )

    aggregate = {
        "n_trials": len(summaries),
        "counterfactual_success_rate": float(
            np.mean([summary["counterfactual"]["success"] for summary in summaries])
        ),
        "class_flip_rate": float(
            np.mean(
                [
                    summary["counterfactual"]["predicted_class"]
                    != summary["original"]["predicted_class"]
                    for summary in summaries
                ]
            )
        ),
    }
    _write_json(out / "summary.json", aggregate)
    print(f"Run summary:\n{json.dumps(aggregate, indent=2)}", flush=True)
    return aggregate


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
