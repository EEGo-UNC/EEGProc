"""Paired base/typicality SIC study with source-only calibration and archives.

Run --help for the CLI. No training or held-out hyperparameter selection is
performed. Use a frozen LOSO model manifest; tune all protocol choices on
source data before the final study. Reports can be rebuilt independently.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import tensorflow as tf

from ..model_agnostic.adapter import load_trial_dataset, load_json_mapping
from ..counterfactuals.arguments import _positive_float, _nonnegative_float, _nonnegative_int, _decay_float
from ..counterfactuals.optimizer import CounterfactualOptimizer
from ..counterfactuals.loss import _VCSC_CHANNELS
from ..model_agnostic.runner import _metadata_arrays
from ..model_agnostic.sic_adapter import create_sic_adapter
from .sic_sequence import SICVCSequence
from .core import TypicalityRegion, trial_representation
from .artifacts import (write_json, write_npz, write_csv, file_sha256,
                                  array_sha256, TrialRecorder, completed_attempt, next_attempt)
from .physiology import (signal_diagnostics, PhysiologicalReference,
                                   source_vcsc_calibration, make_source_loss, DEFAULT_BANDS, FAMILIES)
from .results import recognition_metrics, build_report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-json", required=True, type=Path, help="Existing loso_zero_shot_models.json or {models:[{target_subject,path,stage}]}.")
    parser.add_argument("--model-dir", type=Path, help="Relocated directory containing manifest model filenames.")
    parser.add_argument("--model-module", default="eegproc.deep_learning.joint_architectures.SICModelv15.sic_model")
    parser.add_argument("--task", choices=("valence", "arousal"), required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--trials-npz", type=Path)
    data.add_argument("--data-loader", help="Existing TrialDataset loader as package.module:function.")
    parser.add_argument("--data-config", help="Inline JSON or JSON file passed to the data loader.")
    parser.add_argument("--subjects", type=int, nargs="+", help="Optional fold shard; all eligible trials remain included within each fold.")
    parser.add_argument("--trial-ids", type=int, nargs="+", help="Optional trial filter applied only to the selected held-out fold(s); source calibration still uses the complete dataset.")
    parser.add_argument("--typicality-sequence", required=True, choices=("vc_window_embeddings", "vc_hidden_sequence"), help="Explicit mapping into the learned VC coordinates; see typicality/README.md.")
    parser.add_argument("--typicality-weight", type=_positive_float, default=1.0)
    parser.add_argument("--typicality-quantile", type=_positive_float, default=0.95)
    parser.add_argument("--variance-floor", type=_positive_float, default=1e-6)
    parser.add_argument("--decoder-mode", choices=("branches", "joint"), default="joint")
    parser.add_argument("--report-output", choices=("joint", "gcn_gru", "bilstm"), help="Mandatory in branch mode; joint is the default in joint mode.")
    parser.add_argument("--target-probability", type=_positive_float, default=0.8)
    parser.add_argument("--target-loss-component", choices=("confidence", "focal", "vc", "focal_vc"), default="confidence")
    parser.add_argument("--target-weight", type=_positive_float, default=1.0)
    parser.add_argument("--latent-weight", type=_nonnegative_float, default=0.1)
    parser.add_argument("--decoded-weight", type=_nonnegative_float, default=0.1)
    parser.add_argument("--physiological-weight", type=_nonnegative_float, default=0.0)
    parser.add_argument("--learning-rate", type=_positive_float, default=0.01)
    parser.add_argument("--learning-rate-decay", type=_decay_float, default=1.0)
    parser.add_argument("--max-steps", type=_nonnegative_int, default=200)
    parser.add_argument("--gradient-clip-norm", type=_positive_float, default=5.0)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--snapshot-every", type=int, default=1, help="Full tensors every N steps; scalars every step. Default saves every iterate.")
    parser.add_argument("--log-every", type=_nonnegative_int, default=10)
    parser.add_argument("--fs", type=_positive_float, default=128.0)
    parser.add_argument("--physiology-quantile", type=_positive_float, default=0.95)
    parser.add_argument("--physiology-required-fraction", type=_positive_float, default=0.95)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed trials; preserve attempts for interrupted trials.")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("target_probability", "typicality_quantile", "physiology_quantile"):
        if getattr(args, name) >= 1:
            parser.error(f"--{name.replace('_', '-')} must be below 1")
    if args.physiology_required_fraction > 1 or args.snapshot_every < 1 or args.ece_bins < 1:
        parser.error("Invalid physiology fraction, snapshot interval, or ECE bin count")
    if args.decoder_mode == "branches" and args.report_output not in ("gcn_gru", "bilstm"):
        parser.error("Branch mode requires an explicit --report-output gcn_gru or bilstm")
    if args.decoder_mode == "joint" and args.report_output not in (None, "joint"):
        parser.error("Joint mode reports the joint output")
    args.report_output = args.report_output or "joint"
    args.data_config = load_json_mapping(args.data_config)
    return args


def _manifest(args, dataset):
    payload = json.loads(args.models_json.read_text())
    entries = payload["models"] if isinstance(payload, dict) else payload
    all_subjects = set(dataset.subject_ids.tolist())
    folds, seen = [], set()
    for entry in entries:
        subject = int(entry["target_subject"])
        if subject in seen:
            raise ValueError(f"Duplicate LOSO checkpoint for subject {subject}")
        seen.add(subject)
        if args.subjects is not None and subject not in args.subjects:
            continue
        if subject not in all_subjects:
            raise ValueError(f"Subject {subject} has no prepared trials")
        if entry.get("stage") != "zero_shot_source_model":
            raise ValueError("Only explicitly marked zero_shot_source_model checkpoints are accepted")
        if "source_subject_ids" in entry and subject in entry["source_subject_ids"]:
            raise ValueError("Held-out subject appears in checkpoint's source subjects")
        path = Path(entry["path"])
        if args.model_dir:
            path = args.model_dir / Path(entry.get("filename", path.name)).name
        elif not path.is_absolute():
            path = args.models_json.parent / path
        if not path.is_file() or path.suffix != ".keras":
            raise ValueError(f"Missing .keras checkpoint: {path}")
        source_ids = sorted(set(entry.get("source_subject_ids", all_subjects - {subject})))
        if not source_ids or not set(source_ids).issubset(all_subjects):
            raise ValueError("Manifest source subjects are absent from the trial dataset")
        folds.append({"subject_id": subject, "path": str(path.resolve()), "sha256": file_sha256(path),
                      "stage": entry["stage"], "source_subject_ids": source_ids,
                      "source_subject_ids_origin": "checkpoint_manifest" if "source_subject_ids" in entry else "LOSO complement of dataset"})
    if args.subjects is not None and set(args.subjects) - {f["subject_id"] for f in folds}:
        raise ValueError("Requested subjects are absent from the checkpoint manifest")
    if not folds:
        raise ValueError("No fold selected")
    return sorted(folds, key=lambda fold: fold["subject_id"])


def _physical_signal(dataset, index, values):
    if dataset.normalization_scale is None:
        return np.asarray(values)
    return np.asarray(values) * dataset.normalization_scale[index] + dataset.normalization_offset[index]


def _diagnostics(dataset, index, values, args):
    return signal_diagnostics(_physical_signal(dataset, index, values), fs=args.fs,
                              n_channels=14, band_edges=DEFAULT_BANDS,
                              feature_order=dataset.feature_order or "channel-major")


def _protocol(args, dataset, folds):
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                 if key not in ("resume", "out_dir", "log_every")}
    arrays = {"features": dataset.features, "subject_ids": dataset.subject_ids,
              "trial_ids": dataset.trial_ids, "labels": dataset.labels,
              **_metadata_arrays(dataset, 0)}
    # Fingerprint complete normalization arrays, not just the first trial.
    for name in ("normalization_offset", "normalization_scale"):
        if getattr(dataset, name) is not None:
            arrays[name] = getattr(dataset, name)
    code_dir = Path(__file__).parent
    package_dir = code_dir.parent
    code_files = [
        *code_dir.glob("*.py"),
        package_dir / "counterfactuals" / "optimizer.py",
        package_dir / "counterfactuals" / "loss.py",
        package_dir / "model_agnostic" / "adapter.py",
        package_dir / "model_agnostic" / "runner.py",
        package_dir / "model_agnostic" / "sic_adapter.py",
    ]
    return {"schema_version": 1, "task": args.task, "arguments": arguments, "folds": folds,
            "dataset_sha256": {name: array_sha256(value) for name, value in arrays.items()},
            "source_sha256": {
                p.relative_to(package_dir).as_posix(): file_sha256(p)
                for p in sorted(set(code_files))
            },
            "dataset_metadata": dataset.metadata,
            "eligibility": "true_class == 0 and original argmax prediction == 0",
            "target_class": 1, "prediction_rule": "argmax; confidence threshold separately recorded",
            "typicality_formula": "0.5 * mean((var_q + (mu_q-mu_p)^2)/var_p - 1 + log(var_p/var_q))",
            "penalty": "lambda * max(0,D-tau)^2", "prior": "frozen learned VC parameters",
            "distance_definition": "d_z=RMSE(Zcf,Z) in mapped VC sequence; delta_dec=RMSE(dec(Zcf),dec(Z)); e_rec=RMSE(dec(Z),x)",
            "physiology_unit": dataset.signal_unit if dataset.normalization_scale is not None else "model_input_units",
            "physiology_families": list(FAMILIES), "band_edges_hz": DEFAULT_BANDS,
            "physiology_reference": "all source classes, empirical per-component central intervals",
            "aperiodic_status": "unavailable from band-filtered decoder; never counted as passed"}


def run_fold(args, dataset, entry, out):
    subject = entry["subject_id"]
    directory = out / f"subject_{subject}"
    directory.mkdir(parents=True, exist_ok=True)
    held = np.flatnonzero(dataset.subject_ids == subject)
    if args.trial_ids is not None:
        held = held[np.isin(dataset.trial_ids[held], args.trial_ids)]
        if not len(held):
            raise ValueError(f"Requested trial IDs {sorted(set(args.trial_ids))} are absent for subject {subject}")
    source = np.flatnonzero(np.isin(dataset.subject_ids, entry["source_subject_ids"]))
    adapter = create_sic_adapter(model_path=Path(entry["path"]),
                                 config={"model_module": args.model_module, "decoder_mode": args.decoder_mode},
                                 sample_input=dataset.features[held[0]])
    model = adapter.model
    sequence = SICVCSequence(model, mode=args.typicality_sequence)
    prior_mean, prior_variance = sequence.learned_prior(1)
    source_moments = []
    source_probabilities = []
    print(f"Subject {subject}: encoding {len(source)} source trials for calibration", flush=True)
    for index in source:
        features = model.get_encoder_features(dataset.features[index:index + 1])
        source_moments.append(trial_representation(sequence(features["window_features"]).numpy())[0])
        source_probabilities.append(features["probabilities"].numpy()[0])
    source_moments = np.stack(source_moments)
    region = TypicalityRegion.calibrate(
        source_moments, prior_mean=prior_mean, prior_variance=prior_variance,
        subject_ids=dataset.subject_ids[source], trial_ids=dataset.trial_ids[source], labels=dataset.labels[source],
        held_out_subject=subject, quantile=args.typicality_quantile, variance_floor=args.variance_floor,
        sequence_transform=sequence, sequence_definition=args.typicality_sequence,
    )
    region.save(directory / "calibration")
    write_json(directory / "calibration" / "sequence.json", sequence.metadata())
    write_npz(directory / "calibration" / "source_trials.npz", moments=source_moments,
              probabilities=np.stack(source_probabilities), discrepancy=region.score(source_moments),
              subject_ids=dataset.subject_ids[source], trial_ids=dataset.trial_ids[source], labels=dataset.labels[source],
              learned_prior_log_sigma=model.vc_target.prior_log_sigma.numpy(), learned_prior_mu=model.vc_target.prior_mu.numpy())
    vcsc = source_vcsc_calibration(dataset.features[source])
    write_npz(directory / "calibration" / "vcsc.npz", **vcsc, subject_ids=dataset.subject_ids[source], trial_ids=dataset.trial_ids[source])
    source_diagnostics = [_diagnostics(dataset, i, dataset.features[i], args) for i in source]
    physiological = PhysiologicalReference.fit(source_diagnostics, quantile=args.physiology_quantile,
                                               required_fraction=args.physiology_required_fraction)
    write_npz(directory / "calibration" / "physiology.npz", **physiological.arrays(),
              **{f"source_{name}": np.stack([d[name] for d in source_diagnostics]) for name in FAMILIES})
    loss = make_source_loss(vcsc, target_weight=args.target_weight, latent_weight=args.latent_weight,
                            decoded_weight=args.decoded_weight, physiological_weight=args.physiological_weight,
                            target_probability=args.target_probability)
    common = dict(loss=loss, learning_rate=args.learning_rate, learning_rate_decay=args.learning_rate_decay,
                  target_loss_component=args.target_loss_component, max_steps=args.max_steps,
                  gradient_clip_norm=args.gradient_clip_norm, stop_on_success=args.stop_on_success,
                  decoder_mode=args.decoder_mode, typicality=region)
    # Both arms use identical seeds, frozen models, starts, budgets, and losses.
    optimizers = {name: CounterfactualOptimizer(model, typicality_weight=weight, **common)
                  for name, weight in (("base", 0.0), ("typicality", args.typicality_weight))}
    if args.report_output not in optimizers["base"].decoded_names:
        raise ValueError("The selected report output is not present in this checkpoint")
    predictions, discrepancies, moments, eligible = [], [], [], []
    for index in held:
        features = model.get_encoder_features(dataset.features[index:index + 1])
        z = features["window_features"]
        vc_sequence = sequence(z)
        moment = trial_representation(vc_sequence.numpy())[0]
        probabilities = features["probabilities"].numpy()[0]
        if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
            raise ValueError("The study requires finite binary trial probabilities")
        d = float(region.discrepancy(z).numpy())
        predictions.append(probabilities)
        discrepancies.append(d)
        moments.append(moment)
        trial = int(dataset.trial_ids[index])
        trial_dir = directory / f"trial_{trial}"
        trial_dir.mkdir(exist_ok=True)
        write_npz(trial_dir / "observed.npz", x=dataset.features[index:index + 1], z=z.numpy(),
                  typicality_sequence=vc_sequence.numpy(), moments=moment, probabilities=probabilities,
                  classification_embedding=features["classification_embedding"].numpy(),
                  discrepancy=d, **_metadata_arrays(dataset, index))
        if dataset.labels[index] == 0 and probabilities.argmax() == 0:
            eligible.append(index)
    observations = dict(trial_ids=dataset.trial_ids[held], labels=dataset.labels[held],
                        probabilities=np.stack(predictions), discrepancy=np.asarray(discrepancies), moments=np.stack(moments))
    write_npz(directory / "observations.npz", **observations)
    fold_info = {"subject_id": subject, "status": "running", "checkpoint": entry,
                 "threshold": region.tau, "sequence": sequence.metadata(),
                 "eligible_trial_ids": dataset.trial_ids[eligible].tolist(),
                 "all_trial_ids": dataset.trial_ids[held].tolist(), "loss": asdict(loss),
                 "recognition": recognition_metrics(dataset.labels[held], np.stack(predictions), ece_bins=args.ece_bins)}
    write_json(directory / "fold.json", fold_info)
    print(f"Subject {subject}: tau={region.tau:.6g}; {len(eligible)} eligible trials, both objectives", flush=True)
    errors = 0
    for index in eligible:
        trial = int(dataset.trial_ids[index])
        for objective, optimizer in optimizers.items():
            arm_dir = directory / f"trial_{trial}" / objective
            if args.resume and completed_attempt(arm_dir):
                continue
            seed = int(np.random.SeedSequence([args.seed, subject, trial]).generate_state(1)[0])
            tf.keras.utils.set_random_seed(seed)
            recorder = TrialRecorder(next_attempt(arm_dir), snapshot_every=args.snapshot_every)
            metadata = {"schema_version": 1, "task": args.task, "subject_id": subject, "trial_id": trial,
                        "true_class": 0, "objective": objective, "seed": seed, "report_output": args.report_output,
                        "checkpoint_sha256": entry["sha256"], "sequence_definition": args.typicality_sequence}

            def progress(row):
                if args.log_every and row["step"] % args.log_every == 0:
                    print(f"  s={subject} trial={trial} {objective} step={row['step']} "
                          f"p1={row['target_probability']:.4f} D={row['discrepancy']:.5g} tau={region.tau:.5g}", flush=True)

            try:
                result = optimizer.optimize(dataset.features[index:index + 1], target_class=1,
                                             progress=progress, state_progress=recorder.record)
                arrays, summary = result["arrays"], result["summary"]
                original, baseline, counterfactual = (arrays[name] for name in
                    ("x", f"x_reconstructed_{args.report_output}", f"x_prime_{args.report_output}"))
                diagnostic_sets = {name: _diagnostics(dataset, index, values, args)
                                   for name, values in (("original", original), ("reconstruction", baseline), ("counterfactual", counterfactual))}
                phys = physiological.assess(diagnostic_sets["counterfactual"])
                summary.update(physiology=phys,
                               vcsc_original_input=float(loss.physiological_validity(tf.convert_to_tensor(original)).numpy()),
                               d_z=float(np.sqrt(np.mean((arrays["typicality_sequence_prime"] - arrays["typicality_sequence"]) ** 2))),
                               decoder_latent_rmse=float(np.sqrt(summary["selected_losses"]["latent"])),
                               report_output=args.report_output)
                for prefix, diagnostics in diagnostic_sets.items():
                    write_npz(recorder.directory / f"physiology_{prefix}.npz", **diagnostics)
                write_json(recorder.directory / "physiology.json", {name: physiological.assess(d) for name, d in diagnostic_sets.items()})
                recorder.finish(result, metadata, _metadata_arrays(dataset, index))
            except (FloatingPointError, RuntimeError, ValueError, tf.errors.OpError) as error:
                recorder.fail(error, metadata)
                errors += 1
                print(f"  FAILED s={subject} trial={trial} {objective}: {error}", flush=True)
            finally:
                recorder.close()
    fold_info.update(status="completed", n_optimization_errors=errors)
    write_json(directory / "fold.json", fold_info)
    tf.keras.backend.clear_session()


def run(args):
    out = args.out_dir
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Choose a new output directory, or --resume: {out}")
    tf.keras.utils.set_random_seed(args.seed)
    dataset = load_trial_dataset(args.trials_npz, loader_spec=args.data_loader, loader_config=args.data_config)
    if args.trials_npz:
        with np.load(args.trials_npz, allow_pickle=False) as prepared:
            if "window_mask" in prepared.files:
                mask = prepared["window_mask"]
                if mask.shape != dataset.features.shape[:2] or not np.all(mask == 1):
                    raise ValueError("Padded/masked trials cannot be used in the full-sequence study")
    if dataset.labels is None or not np.isin(dataset.labels, [0, 1]).all():
        raise ValueError("True binary labels are required for trial eligibility and source calibration")
    if dataset.features.ndim != 4 or dataset.features.shape[-1] != 42:
        raise ValueError("This study runner expects SIC DREAMER trials (N,W,T,14*3)")
    if np.any(dataset.subject_ids < 0) or np.any(dataset.trial_ids < 0):
        raise ValueError("Subject and trial IDs must be nonnegative")
    if (dataset.normalization_scale is None) != (dataset.normalization_offset is None):
        raise ValueError("Provide both normalization scale and offset, or neither")
    if dataset.feature_order not in (None, "channel-major"):
        raise ValueError("SIC VCSC and the learned decoder expect channel-major features")
    if dataset.channel_names is not None and tuple(dataset.channel_names) != tuple(_VCSC_CHANNELS):
        raise ValueError("Channel order must match the checkpoint's DREAMER montage")
    if dataset.band_names is not None and tuple(n.lower() for n in dataset.band_names) != ("theta", "alpha", "beta"):
        raise ValueError("Band order must be theta, alpha, beta")
    dataset.channel_names = tuple(_VCSC_CHANNELS)
    dataset.band_names = ("theta", "alpha", "beta")
    dataset.feature_order = "channel-major"
    if dataset.metadata.get("fs", args.fs) != args.fs:
        raise ValueError("Sampling rate disagrees with loader metadata")
    # Validate the spectral configuration before loading any checkpoint or
    # creating an output directory.
    _diagnostics(dataset, 0, dataset.features[0], args)
    folds = _manifest(args, dataset)
    protocol = _protocol(args, dataset, folds)
    # Normalize tuples to JSON arrays before comparison on resumed runs.
    protocol = json.loads(json.dumps(protocol))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "study.json").exists():
        saved = json.loads((out / "study.json").read_text())
        if saved != protocol:
            raise ValueError("Resume refused: data, checkpoint, source code, or protocol changed")
    else:
        if args.resume and any(out.iterdir()):
            raise ValueError("Cannot resume an output directory with no study manifest")
        write_json(out / "study.json", protocol)
        try:
            git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            git_revision = None
        write_json(out / "environment.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "python": sys.version, "numpy": np.__version__, "tensorflow": tf.__version__,
                   "git_revision": git_revision, "command": sys.argv})
        metadata = _metadata_arrays(dataset, 0)
        for name in ("normalization_offset", "normalization_scale"):
            if getattr(dataset, name) is not None:
                metadata[name] = getattr(dataset, name)
        write_npz(out / "inputs.npz", features=dataset.features, subject_ids=dataset.subject_ids,
                  trial_ids=dataset.trial_ids, labels=dataset.labels, **metadata)
    for fold in folds:
        info = out / f"subject_{fold['subject_id']}" / "fold.json"
        if args.resume and info.exists():
            previous = json.loads(info.read_text())
            if previous["status"] == "completed" and all(
                completed_attempt(info.parent / f"trial_{trial}" / objective)
                for trial in previous["eligible_trial_ids"] for objective in ("base", "typicality")
            ):
                print(f"Subject {fold['subject_id']}: verified cached fold", flush=True)
                continue
        run_fold(args, dataset, fold, out)
        build_report([out], out / "report")
    return build_report([out], out / "report")


def main(argv=None):
    result = run(parse_args(argv))
    return 1 if any(row["n_error"] for row in result["population"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
