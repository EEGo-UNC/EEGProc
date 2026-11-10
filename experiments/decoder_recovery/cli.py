"""Runnable source-fold training and decode/re-encode evaluation commands."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from eegproc.model_explainability.counterfactuals.loss import CounterfactualLoss
from eegproc.model_explainability.counterfactuals.optimizer import CounterfactualOptimizer
from .data import TrialData, new_output, prepare_fold, scalar, write_json
from .identity import reconstruction_metrics
from .latent_coords import LatentCoordinatesDecoder, LatentCoordsOptimizer
from .training import load_sic, train_identifier, train_idea1, train_latent_coords


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def nonnegative_float(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-fold")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Prepared EEGProc arrays in an NPZ archive.")
    source.add_argument("--loader-config", type=Path,
                        help="JSON kwargs for EEGProc's existing load_sic_training_data.")
    prepare.add_argument("--test-subjects", nargs="+", required=True)
    prepare.add_argument("--validation-subjects", nargs="+", required=True)
    prepare.add_argument("--identity-validation-trials", type=positive_int, default=1)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("pretrain-identifier", "idea1", "latent-coords"):
        command = commands.add_parser(name)
        command.add_argument("--train", type=Path, required=True)
        command.add_argument("--validation", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--batch-size", type=positive_int, default=8)
        command.add_argument("--seed", type=int, default=42)
        if name != "latent-coords":
            command.add_argument("--epochs", type=positive_int, default=30)
            command.add_argument("--learning-rate", type=positive_float,
                                 default=1e-3 if name == "pretrain-identifier" else 1e-4)
        if name in ("idea1", "latent-coords"):
            command.add_argument(
                "--reconstruction-seam-weight", type=nonnegative_float,
                help="Override the seam weight from the model config or checkpoint. Use 0 to ablate.",
            )
            command.add_argument(
                "--reconstruction-derivative-weight", type=nonnegative_float,
                help="Override the derivative weight from the model config or checkpoint. Use 0 to ablate.",
            )
        if name == "idea1":
            command.add_argument("--model", type=Path, required=True)
            command.add_argument("--identifier", type=Path, required=True)
            command.add_argument("--identifier-metadata", type=Path, required=True)
            command.add_argument("--identity-weight", type=nonnegative_float, default=0.1)
            command.add_argument("--reconstruction-weight", type=positive_float, default=1.0)
            command.add_argument("--identity-loss", choices=("cross_entropy", "kl"), default="cross_entropy")
        if name == "latent-coords":
            command.add_argument("--model-config", type=Path, required=True,
                                 help="JSON kwargs for the existing v15 build_sic_model.")
            command.add_argument("--phase1-epochs", type=positive_int, default=30)
            command.add_argument("--phase2-epochs", type=positive_int, default=30)
            command.add_argument("--isa-weight", type=positive_float, default=0.8)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--reference-model", type=Path)
    evaluate.add_argument("--identifier", type=Path)
    evaluate.add_argument("--identifier-metadata", type=Path)
    evaluate.add_argument("--batch-size", type=positive_int, default=8)
    counterfactual = commands.add_parser("counterfactual")
    counterfactual.add_argument("--model", type=Path, required=True)
    counterfactual.add_argument("--reference-model", type=Path)
    counterfactual.add_argument("--data", type=Path, required=True)
    counterfactual.add_argument("--output", type=Path, required=True)
    counterfactual.add_argument("--max-steps", type=positive_int, default=200)
    counterfactual.add_argument("--limit", type=positive_int)
    counterfactual.add_argument("--target-class", type=int)
    counterfactual.add_argument("--target-probability", type=positive_float, default=0.8)
    counterfactual.add_argument("--learning-rate", type=positive_float, default=0.01)
    counterfactual.add_argument("--latent-weight", type=nonnegative_float, default=0.1)
    counterfactual.add_argument("--decoded-weight", type=nonnegative_float, default=0.1)
    counterfactual.add_argument("--physiological-weight", type=nonnegative_float, default=0.0)
    counterfactual.add_argument("--decoder-mode", choices=("branches", "joint"), default="joint")
    return root


def _identifier(args):
    if args.identifier is None:
        if getattr(args, "identifier_metadata", None):
            raise ValueError("--identifier-metadata requires --identifier.")
        return None, None
    if args.identifier_metadata is None:
        raise ValueError("--identifier requires --identifier-metadata.")
    model = tf.keras.models.load_model(args.identifier, compile=False)
    metadata = json.loads(args.identifier_metadata.read_text())
    if metadata.get("outputs") != "logits":
        raise ValueError("Identifier must provide logits, not probabilities.")
    classes = metadata["subject_ids"]
    if model.output_shape[-1] != len(classes):
        raise ValueError("Identifier output width and vocabulary disagree.")
    return model, classes


def _counterfactual(args, data, output):
    model = load_sic(args.model)
    reference = load_sic(args.reference_model) if args.reference_model else None
    kwargs = dict(loss=CounterfactualLoss(target_probability=args.target_probability,
        latent_weight=args.latent_weight, decoded_weight=args.decoded_weight,
        physiological_weight=args.physiological_weight),
        max_steps=args.max_steps, learning_rate=args.learning_rate,
        decoder_mode=args.decoder_mode)
    optimizer = (LatentCoordsOptimizer(model, reference_model=reference, **kwargs)
                 if reference is not None else CounterfactualOptimizer(model, **kwargs))
    summaries = []
    for index, eeg in enumerate(data.eeg[:args.limit]):
        result = optimizer.optimize(eeg, target_class=args.target_class)
        summary, arrays = result["summary"], result["arrays"]
        target = summary["target_class"]
        round_trip = {}
        for name in optimizer.decoded_names:
            row = {}
            for key in ("x_reconstructed", "x_prime"):
                probabilities = tf.nn.softmax(model(arrays[f"{key}_{name}"], training=False)).numpy()[0]
                row[key] = {"probabilities": probabilities.tolist(),
                            "predicted_class": int(np.argmax(probabilities)),
                            "target_probability": float(probabilities[target]),
                            "target_attained": bool(np.argmax(probabilities) == target
                                and probabilities[target] >= args.target_probability)}
            round_trip[name] = row
        # Latent optimization/selection remains unchanged. Decoded validity is
        # measured separately, never inferred from latent target attainment.
        summary.update(subject_id=scalar(data.subject_ids[index]),
                       trial_id=scalar(data.trial_ids[index]),
                       true_label=int(data.labels[index]), decoded_reencoded=round_trip,
                       round_trip_evaluation="latent_and_decoded_reencoded",
                       counterfactual_validity_prediction_space="decoded_reencoded")
        np.savez_compressed(output / f"trial_{index:05d}.npz", **arrays)
        write_json(output / f"trial_{index:05d}.json", summary)
        write_json(output / f"trial_{index:05d}_history.json", result["history"])
        summaries.append(summary)
        print(json.dumps({"trial": index, "latent_success": summary["latent_counterfactual"]["success"],
                          "decoded_reencoded": round_trip}), flush=True)
    write_json(output / "summary.json", summaries)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare-fold":
        if args.data:
            data = TrialData.load(args.data)
        else:
            from eegproc.deep_learning.joint_architectures.sic_common.training import (
                load_sic_training_data, _group_windows_into_trials)
            config = json.loads(args.loader_config.read_text())
            arrays = load_sic_training_data(**config)
            data = TrialData(*_group_windows_into_trials(*arrays))
        prepare_fold(data, args.output, test_subjects=args.test_subjects,
            validation_subjects=args.validation_subjects,
            identity_validation_trials=args.identity_validation_trials, seed=args.seed)
    elif args.command in {"pretrain-identifier", "idea1", "latent-coords"}:
        train, validation = TrialData.load(args.train), TrialData.load(args.validation)
        common = dict(batch_size=args.batch_size, seed=args.seed)
        temporal_weights = {
            key: getattr(args, key)
            for key in ("reconstruction_seam_weight", "reconstruction_derivative_weight")
            if getattr(args, key, None) is not None
        }
        tf.keras.utils.set_random_seed(args.seed)
        if args.command == "pretrain-identifier":
            train_identifier(train, validation, args.output, epochs=args.epochs,
                             learning_rate=args.learning_rate, **common)
        elif args.command == "idea1":
            identifier, classes = _identifier(args)
            train_idea1(load_sic(args.model, **temporal_weights), identifier, classes, train, validation,
                args.output, epochs=args.epochs, learning_rate=args.learning_rate,
                identity_weight=args.identity_weight, reconstruction_weight=args.reconstruction_weight,
                loss_mode=args.identity_loss, **common)
        else:
            config = json.loads(args.model_config.read_text())
            config.update(temporal_weights)
            train_latent_coords(train, validation, config, args.output,
                phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
                isa_weight=args.isa_weight, **common)
    else:
        data, output = TrialData.load(args.data), new_output(args.output)
        if args.command == "evaluate":
            model = load_sic(args.model)
            identifier, classes = _identifier(args)
            reconstruct = (LatentCoordinatesDecoder(load_sic(args.reference_model), model).reconstruct
                           if args.reference_model else None)
            report = reconstruction_metrics(model, data, identifier=identifier,
                classes=classes, reconstruction_fn=reconstruct, batch_size=args.batch_size)
            write_json(output / "report.json", report)
            print(json.dumps(report), flush=True)
        else:
            _counterfactual(args, data, output)
    # Record exact checkpoints as well as split fingerprints in stage reports.
    provenance = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
    def checksum(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    provenance["input_sha256"] = {key: checksum(value) for key, value in vars(args).items()
        if isinstance(value, Path) and value.is_file()}
    write_json(args.output / "command.json", provenance)


if __name__ == "__main__":
    main()
