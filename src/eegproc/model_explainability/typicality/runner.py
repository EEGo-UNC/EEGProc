"""Paired base/typicality SIC study with fold-isolated calibration and archives.

Run --help for the CLI. No training or held-out hyperparameter selection is
performed. Use a frozen LOSO model manifest; tune all protocol choices on
source data before the final study. Reports can be rebuilt independently.
"""

import argparse
from copy import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import tensorflow as tf

from ..model_agnostic.adapter import load_trial_dataset, load_json_mapping
from ..counterfactuals.arguments import _positive_float, _nonnegative_float, _nonnegative_int, _decay_float
from ..counterfactuals.optimizer import CounterfactualOptimizer, EVALUATION_PROTOCOL
from ..counterfactuals.fusion import validate_fixed_joint_alpha
from ..counterfactuals.loss import _VCSC_CHANNELS
from ..counterfactuals.amplitude_spectral import AmplitudeSpectralReference
from ..counterfactuals.loss import DECODED_METRICS
from ..counterfactuals.temporal_phase import envelope_distance_np, latent_scale
from ..model_agnostic.sic_adapter import create_sic_adapter
from .sic_embedding import SICTrialEmbedding
from .core import TypicalityRegion, SCHEMA_VERSION, SCORE_DEFINITION, REPRESENTATION, TYPICALITY_QUANTILE
from .coverage import eligibility_records
from .neighbors import EncodedKNNReference, InsufficientSourceDataError, KNN_DEFINITION
from .knn_evaluation import evaluate_trial_embeddings
from .knn_report import build_knn_report
from .artifacts import (write_json, write_csv, file_sha256,
                                  array_sha256, TrialRecorder, completed_attempt, next_attempt)
from .physiology import (signal_diagnostics, seam_diagnostics, PhysiologicalReference,
                                   vcsc_calibration, make_vcsc_loss, DEFAULT_BANDS, FAMILIES)
from .results import recognition_metrics, build_report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-json", required=True, type=Path, help="Existing loso_zero_shot_models.json or {models:[{target_subject,path,stage}]}.")
    parser.add_argument("--model-dir", type=Path, help="Relocated directory containing manifest model filenames.")
    parser.add_argument("--model-module", default="eegproc.deep_learning.joint_architectures.SICModelv15.sic_model")
    parser.add_argument("--task", choices=("valence", "arousal"), required=True)
    parser.add_argument("--directions", nargs="+", choices=("0_to_1", "1_to_0"),
                        default=["0_to_1", "1_to_0"], help="Run both class directions by default; each gets an isolated study directory.")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--trials-npz", type=Path)
    data.add_argument("--data-loader", help="Existing TrialDataset loader as package.module:function.")
    parser.add_argument("--data-config", help="Inline JSON or JSON file passed to the data loader.")
    parser.add_argument("--subjects", type=int, nargs="+", help="Optional fold shard; all eligible trials remain included within each fold.")
    parser.add_argument("--include-target-latent", action=argparse.BooleanOptionalAction, default=True,
                        help="Add a matched target+latent-only arm, with decoded, physiological, and typicality optimization weights zero. All diagnostics remain enabled.")
    parser.add_argument("--include-typicality-no-physiology", action=argparse.BooleanOptionalAction, default=True,
                        help="Add a matched full-CFO typicality arm with physiological weight zero. Target, latent, decoded, and typicality settings and all diagnostics are retained.")
    parser.add_argument("--trial-ids", type=int, nargs="+", help="Optional optimization filter; every held-out eligibility decision is still archived and all source calibration trials are retained.")
    parser.add_argument("--typicality-representation", "--typicality-sequence", dest="typicality_representation",
                        default=REPRESENTATION, choices=(REPRESENTATION,),
                        help="Full-trial terminal classification embedding. Legacy window/hidden-sequence KL modes are no longer supported.")
    parser.add_argument("--typicality-weight", type=_positive_float, default=1.0)
    parser.add_argument("--typicality-quantile", type=_positive_float, default=TYPICALITY_QUANTILE,
                        choices=(TYPICALITY_QUANTILE,), help="Closest 10%% of source target-class discrepancies only (0.10).")
    parser.add_argument("--knn-evaluation", action=argparse.BooleanOptionalAction, default=True,
                        help="Evaluate fixed real-target neighbours after optimization; never an optimization or selection criterion.")
    parser.add_argument("--knn-k", type=int, default=5,
                        help="Distance to this kth real target-class neighbour (not an average).")
    parser.add_argument("--knn-quantile", type=_positive_float, default=0.10,
                        help="Source-participant-calibrated neighbour-distance cutoff; lower is stricter.")
    parser.add_argument("--knn-calibration-fraction", type=_positive_float, default=0.25,
                        help="Fraction of source participants reserved for neighbour cutoff calibration, disjoint from the bank.")
    parser.add_argument("--variance-floor", type=_positive_float, default=1e-6)
    parser.add_argument("--decoder-mode", choices=("branches", "joint"), default="joint")
    parser.add_argument("--fixed-joint-alpha", type=float,
                        help="Fixed GCN-GRU reconstruction weight in [0,1]; BiLSTM gets 1-alpha. Joint mode only. Omit to use the saved learned mixer.")
    parser.add_argument("--report-output", choices=("joint", "gcn_gru", "bilstm"), help="Mandatory in branch mode; joint is the default in joint mode.")
    parser.add_argument("--target-probability", type=_positive_float, default=0.8)
    parser.add_argument("--target-loss-component", choices=("confidence", "focal", "vc", "focal_vc"), default="confidence")
    parser.add_argument("--target-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument("--latent-weight", type=_nonnegative_float, default=0.1)
    parser.add_argument("--decoded-weight", type=_nonnegative_float, default=0.1)
    parser.add_argument("--physiological-weight", type=_nonnegative_float, default=1.0,
                        help="VCSC optimization weight; positive by default so physiology is enforced.")
    parser.add_argument("--learning-rate", type=_positive_float, default=0.01)
    parser.add_argument("--learning-rate-decay", type=_decay_float, default=1.0)
    parser.add_argument("--max-steps", type=_nonnegative_int, default=200)
    parser.add_argument("--gradient-clip-norm", type=_positive_float, default=5.0)
    parser.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=True,
                        help="Stop as soon as the target criterion is met (default: enabled).")
    parser.add_argument("--min-gradient-norm", type=_nonnegative_float, default=1e-6,
                        help="Raw global gradient norm at or below which a step is considered stalled; 0 disables with patience 0.")
    parser.add_argument("--low-gradient-patience", type=_nonnegative_int, default=5,
                        help="Consecutive low-gradient evaluations required before stopping; 0 disables with threshold 0.")
    parser.add_argument("--typicality-improvement-patience", type=_nonnegative_int, default=10,
                        help="Stop the typicality phase after this many feasible evaluations without a meaningful D improvement; 0 disables.")
    parser.add_argument("--typicality-min-delta", type=_nonnegative_float, default=1e-6,
                        help="Minimum decrease in D that resets typicality improvement patience.")
    parser.add_argument("--physiological-tolerance", type=_nonnegative_float, default=1e-8,
                        help="Largest raw VCSC penalty considered physiologically feasible when its weight is positive.")
    parser.add_argument("--physiological-tolerance-quantile", type=_positive_float,
                        help="Replace --physiological-tolerance with this quantile (0,1) of VCSC on the fold's "
                             "source-subject REAL trials, so feasible means 'as plausible as real EEG' rather "
                             "than 'more typical than almost all real EEG'. Recorded in calibration/vcsc_tolerance.json.")
    parser.add_argument("--amplitude-spectral-weight", type=_nonnegative_float, default=0.0,
                        help="Weight of the amplitude/spectral-power penalty (graded exceedance of the source "
                             "intervals used by the reported physiology checks). 0 disables it.")
    parser.add_argument("--amplitude-spectral-margin", type=_nonnegative_float, default=0.0,
                        help="Shrink each source interval by this fraction of its half-width inside the penalty, in [0,1).")
    parser.add_argument("--temporal-sparsity-weight", type=_nonnegative_float, default=0.0,
                        help="Fused-lasso sparsity: weight of mean |delta| on the latent edit standardized by the "
                             "fold's source-trial latent std. 0 disables it.")
    parser.add_argument("--temporal-tv-weight", type=_nonnegative_float, default=0.0,
                        help="Fused-lasso smoothness: weight of mean |delta_{t+1} - delta_t| (total variation over "
                             "time) on the same standardized edit. 0 disables it.")
    parser.add_argument("--decoded-distance", choices=DECODED_METRICS, default="mse",
                        help="Decoded proximity term: sample-wise mse (historical) or envelope, a phase-invariant "
                             "distance between log band-power envelopes. Both are always reported.")
    parser.add_argument("--include-repair-baseline", action="store_true",
                        help="Per eligible trial, first find the smallest latent edit that makes the decoded trial "
                             "physiologically feasible with NO target term (z_rep). Every arm then also reports its "
                             "decoded change relative to R(z_rep), separating explanation from decoder repair.")
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
    if len(args.directions) != len(set(args.directions)):
        parser.error("--directions must not contain duplicates")
    try:
        args.fixed_joint_alpha = validate_fixed_joint_alpha(args.fixed_joint_alpha, args.decoder_mode)
    except ValueError as error:
        parser.error(str(error))
    for name in ("target_probability", "typicality_quantile", "physiology_quantile"):
        if getattr(args, name) >= 1:
            parser.error(f"--{name.replace('_', '-')} must be below 1")
    if args.knn_k < 1 or args.knn_quantile >= 1 or args.knn_calibration_fraction >= 1:
        parser.error("--knn-k must be positive; neighbour quantile and calibration fraction must be in (0, 1)")
    if args.physiology_required_fraction > 1 or args.ece_bins < 1:
        parser.error("Invalid physiology fraction or ECE bin count")
    if args.physiological_tolerance_quantile is not None and not 0 < args.physiological_tolerance_quantile < 1:
        parser.error("--physiological-tolerance-quantile must be in (0, 1)")
    if not 0 <= args.amplitude_spectral_margin < 1:
        parser.error("--amplitude-spectral-margin must be in [0, 1)")
    if args.include_repair_baseline and args.physiological_weight == 0 and args.amplitude_spectral_weight == 0:
        parser.error("--include-repair-baseline needs a positive physiological or amplitude/spectral weight")
    if (args.min_gradient_norm == 0) != (args.low_gradient_patience == 0):
        parser.error("--min-gradient-norm and --low-gradient-patience must both be zero to disable stopping, or both be positive")
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
    if args.subjects is None and seen != all_subjects:
        raise ValueError("Full studies require a zero-shot checkpoint for every dataset subject; use --subjects for an explicit shard")
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


def _seams(dataset, index, values):
    """Window-seam continuity of one trial in physical units, JSON-safe."""
    seams = seam_diagnostics(_physical_signal(dataset, index, values), n_channels=14,
                             band_edges=DEFAULT_BANDS, feature_order=dataset.feature_order or "channel-major")
    for key in ("phase_residual_median_rad", "amplitude_jump_ratio_median"):
        if seams[key] is not None:
            seams[key] = [float(v) if np.isfinite(v) else None for v in np.asarray(seams[key], dtype=float)]
    return seams


def _objective_names(args):
    names = ["base", "typicality"]
    if args.include_target_latent:
        names.insert(0, "target_latent")
    if args.include_typicality_no_physiology:
        names.append("typicality_no_physiology")
    return names


def _metadata_arrays(dataset, index):
    arrays = {}
    for name in ("normalization_offset", "normalization_scale"):
        value = getattr(dataset, name)
        if value is not None:
            arrays[name] = value[index]
    for name in ("channel_names", "band_names", "feature_order", "signal_unit"):
        value = getattr(dataset, name)
        if value is not None:
            arrays[name] = np.asarray(value)
    return arrays


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
        package_dir / "counterfactuals" / "fusion.py",
        package_dir / "counterfactuals" / "loss.py",
        package_dir / "counterfactuals" / "amplitude_spectral.py",
        package_dir / "counterfactuals" / "temporal_phase.py",
        package_dir / "model_agnostic" / "adapter.py",
        package_dir / "model_agnostic" / "optimizer.py",
        package_dir / "model_agnostic" / "sic_adapter.py",
    ]
    return {"schema_version": SCHEMA_VERSION, "task": args.task, "arguments": arguments, "folds": folds,
            "artifact_format": "json_csv_summaries",
            "objectives": _objective_names(args),
            "repair_baseline": bool(args.include_repair_baseline),
            "repair_baseline_definition": ("z_rep = feasible latent edit of z under the physiological constraints and "
                                           "latent/decoded proximity, with no target term; explanation change = "
                                           "RMSE(dec(Hcf), dec(H_rep)), repair change = RMSE(dec(H_rep), dec(H))"
                                           if args.include_repair_baseline else None),
            "physiological_tolerance_rule": (
                f"quantile {args.physiological_tolerance_quantile} of VCSC on source-subject real trials"
                if args.physiological_tolerance_quantile is not None
                else f"fixed {args.physiological_tolerance}"),
            "amplitude_spectral_objective": (
                "weight * mean relu(|v-m|/h - (1-margin))^2 over amplitude and spectral-power source intervals; "
                "feasible iff each family meets the physiology required fraction"
                if args.amplitude_spectral_weight > 0 else None),
            "temporal_smoothing_objective": (
                "fused lasso on delta = (z' - z) / sigma_source flattened over windows x timesteps: "
                "sparsity_weight * mean smooth|delta| + tv_weight * mean smooth|delta_{t+1} - delta_t|, "
                "smooth|x| = sqrt(x^2 + 1e-6) - 1e-3; sigma_source = per-feature latent std over the "
                "fold's source trials"
                if args.temporal_sparsity_weight > 0 or args.temporal_tv_weight > 0 else None),
            "decoded_distance": (
                "mean squared difference of log(pooled Hann-tapered Hilbert envelope + 1e-3), "
                "pooled over up to 32 samples per window"
                if args.decoded_distance == "envelope" else "sample-wise mse"),
            "dataset_sha256": {name: array_sha256(value) for name, value in arrays.items()},
            "source_sha256": {
                p.relative_to(package_dir).as_posix(): file_sha256(p)
                for p in sorted(set(code_files))
            },
            "dataset_metadata": dataset.metadata,
            "eligibility": f"true_class == {args.source_class} and original argmax prediction == {args.source_class}",
            "direction": args.direction, "source_class": args.source_class,
            "target_class": args.target_class, "typicality_quantile": args.typicality_quantile,
            "prediction_rule": "argmax; confidence threshold separately recorded",
            "round_trip_evaluation": EVALUATION_PROTOCOL,
            "counterfactual_validity": "target argmax and confidence on the optimized full-trial classification embedding; all eligible trials retained",
            "counterfactual_validity_prediction_space": "latent",
            "decoded_evaluation": "post-hoc frozen re-encoding: target argmax/confidence, Gaussian typicality, and independent neighbours for every arm; optimization success remains latent-only",
            "posthoc_decoded_evaluation": True,
            "encoded_neighbors": {"enabled": args.knn_evaluation, "definition": KNN_DEFINITION,
                "representation": REPRESENTATION, "k": args.knn_k, "quantile": args.knn_quantile,
                "calibration_fraction": args.knn_calibration_fraction, "seed": args.seed,
                "selection": "true target-class source trials; disjoint bank/calibration participants",
                "role": "evaluation only after optimize returns; never a loss, early-stop, or iterate-selection criterion"},
            "embedding_archival": "full terminal vectors in source_trials, observations and per-attempt classification_embeddings; no window latents or decoded signals",
            "typicality_definition": SCORE_DEFINITION,
            "typicality_representation": REPRESENTATION,
            "typicality_formula": "mean((classification_embedding - prior_mean)^2 / max(prior_variance, variance_floor))",
            "evaluation": "typical iff D <= tau",
            "optimization_penalty": "lambda * D / max(tau,variance_floor), activated after target and physiology feasibility",
            "selection_priority": "target and physiology feasible, then lowest D, then proximity",
            "prior": "frozen learned VC parameters",
            "distance_definition": "d_z=RMSE(e(Hcf),e(H)) in full-trial classification embeddings; delta_dec=RMSE(dec(Hcf),dec(H)); e_rec=RMSE(dec(H),x)",
            "physiology_unit": dataset.signal_unit if dataset.normalization_scale is not None else "model_input_units",
            "physiology_families": list(FAMILIES), "band_edges_hz": DEFAULT_BANDS,
            "physiology_reference": "all source classes, empirical per-component central intervals",
            "vcsc_reference": "LOSO complement source subjects' real EEG",
            "vcsc_label_usage": "none",
            "aperiodic_status": "unavailable from band-filtered decoder; never counted as passed"}


def _score_decoded_counterfactual(model, region, decoded, *, report_output, target_class=None, target_probability=None,
                                  input_name="decoded_counterfactual_eeg_reencoded_by_frozen_classifier"):
    """Score the generated EEG after a fresh pass through the frozen encoder.

    This differs from the optimizer's D(Zcf): the decoder and encoder need not
    be an exact inverse, so only this score describes the delivered waveform.
    With target_probability, also apply the latent success rule (argmax is the
    target and its probability >= target_probability) and the typicality rule
    (D <= tau) to the re-encoded trial.
    """
    signal = np.asarray(decoded, dtype=np.float32)
    if signal.ndim != 4 or signal.shape[0] != 1 or not np.isfinite(signal).all():
        raise ValueError("Decoded counterfactual must be one finite full EEG trial")
    features = model.get_encoder_features(signal)
    target = int(region.metadata["target_class"])
    if target_class is not None and target_class != target:
        raise ValueError("Decoded evaluation target disagrees with calibrated region")
    embedding = np.asarray(features["classification_embedding"], dtype=np.float64)
    probabilities = np.asarray(features["probabilities"], dtype=np.float64)
    if embedding.shape != (1, len(region.prior_mean)) or not np.isfinite(embedding).all():
        raise ValueError("Re-encoded decoded counterfactual has an invalid embedding")
    if (probabilities.shape != (1, 2) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Re-encoded decoded counterfactual has invalid probabilities")
    score = {
        "input": input_name,
        "report_output": report_output,
        "target_class": target,
        "classification_embedding": embedding[0].tolist(),
        "probabilities": probabilities[0].tolist(),
        "predicted_class": int(probabilities[0].argmax()),
        "discrepancy": float(region.score(embedding)[0]),
        "source_threshold": float(region.tau),
    }
    if target_probability is not None:
        score.update(target_probability=float(probabilities[0, target]),
                     success=bool(score["predicted_class"] == target and probabilities[0, target] >= target_probability),
                     typical=bool(score["discrepancy"] <= score["source_threshold"]))
    return score


def _normalization(dataset, index):
    """This trial's normalization arrays (model units -> physical units)."""
    if dataset.normalization_scale is None:
        return None, None
    return dataset.normalization_scale[index], dataset.normalization_offset[index]


def _repair_baseline(args, dataset, index, subject, trial, entry, optimizer, loss,
                     physiological, scale, offset, directory):
    """Run (or reuse) the repair-only optimization for one trial.

    Returns ({reconstruction, diagnostics, summary} or None, n_errors).
    Its scalar metrics go in result.json like every arm. The repaired
    reconstruction R(z_rep) itself is the one array this runner archives
    (repair_reconstruction.npz, one per eligible trial, only with
    --include-repair-baseline): the other arms are measured against it,
    including after --resume, and it cannot be rebuilt from scalars.
    """
    arm_dir = directory / f"trial_{trial}" / "repair"
    done = completed_attempt(arm_dir) if args.resume else None
    if done is not None:
        with np.load(done / "repair_reconstruction.npz", allow_pickle=False) as saved:
            reconstruction = saved["x_repair"]
        summary = json.loads((done / "result.json").read_text())
    else:
        seed = int(np.random.SeedSequence([args.seed, subject, trial]).generate_state(1)[0])
        tf.keras.utils.set_random_seed(seed)
        recorder = TrialRecorder(next_attempt(arm_dir))
        metadata = {"schema_version": SCHEMA_VERSION, "task": args.task, "subject_id": subject, "trial_id": trial,
                    "true_class": args.source_class, "source_class": args.source_class,
                    "target_class": args.target_class, "direction": args.direction,
                    "objective": "repair", "seed": seed, "report_output": args.report_output,
                    "checkpoint_sha256": entry["sha256"], "round_trip_evaluation": EVALUATION_PROTOCOL}
        try:
            result = optimizer.optimize(dataset.features[index:index + 1], target_class=args.target_class,
                                        progress=recorder.record, signal_scale=scale, signal_offset=offset)
            arrays, summary = result["arrays"], result["summary"]
            reconstruction = arrays[f"x_prime_{args.report_output}"]
            baseline = arrays[f"x_reconstructed_{args.report_output}"]
            repair_change = float(np.mean((reconstruction - baseline) ** 2))
            summary.update(
                repair_change_mse=repair_change,
                delta_dec_repair=float(np.sqrt(repair_change)),
                repair_change_env=envelope_distance_np(reconstruction, baseline),
                vcsc_repair=float(loss.physiological_validity(tf.convert_to_tensor(reconstruction)).numpy()),
                vcsc_original_reconstruction=float(loss.physiological_validity(tf.convert_to_tensor(baseline)).numpy()),
                physiology=physiological.assess(_diagnostics(dataset, index, reconstruction, args)),
            )
            np.savez_compressed(recorder.directory / "repair_reconstruction.npz",
                                x_repair=reconstruction.astype(np.float32))
            summary = recorder.finish(result, metadata)
        except (FloatingPointError, RuntimeError, ValueError, tf.errors.OpError) as error:
            recorder.fail(error, metadata)
            print(f"  FAILED s={subject} trial={trial} repair: {error}", flush=True)
            return None, 1
        finally:
            recorder.close()
    return {
        "reconstruction": reconstruction,
        "diagnostics": _diagnostics(dataset, index, reconstruction, args),
        "summary": {
            "available": True,
            "feasible": bool(summary["selected_feasible"]),
            "preserves_prediction": bool(summary.get("repair_preserves_prediction")),
            "selected_step": summary["selected_step"],
            "stop_reason": summary["stop_reason"],
            "repair_change_mse": summary["repair_change_mse"],
            "delta_dec_repair": summary["delta_dec_repair"],
            "repair_change_env": summary["repair_change_env"],
            "vcsc_repair": summary["vcsc_repair"],
        },
    }, 0


def run_fold(args, dataset, entry, out):
    subject = entry["subject_id"]
    directory = out / f"subject_{subject}"
    directory.mkdir(parents=True, exist_ok=True)
    held_all = np.flatnonzero(dataset.subject_ids == subject)
    held = held_all
    if args.trial_ids is not None:
        if not set(args.trial_ids).issubset(set(dataset.trial_ids[held].tolist())):
            raise ValueError(f"Requested trial IDs {sorted(set(args.trial_ids))} are absent for subject {subject}")
    source = np.flatnonzero(np.isin(dataset.subject_ids, entry["source_subject_ids"]))
    adapter = create_sic_adapter(model_path=Path(entry["path"]),
                                 config={"model_module": args.model_module, "decoder_mode": args.decoder_mode,
                                         "fixed_joint_alpha": args.fixed_joint_alpha},
                                 sample_input=dataset.features[held[0]])
    model = adapter.model
    embedding_map = SICTrialEmbedding(model)
    prior_mean, prior_variance = embedding_map.learned_prior(args.target_class)
    source_embeddings = []
    source_probabilities = []
    print(f"Subject {subject}: encoding {len(source)} source trials for calibration", flush=True)

    def encode_source():
        # Streams each source trial's latent sequence into latent_scale, so
        # source latents are never all held in memory.
        for index in source:
            features = model.get_encoder_features(dataset.features[index:index + 1])
            source_embeddings.append(features["classification_embedding"].numpy()[0])
            source_probabilities.append(features["probabilities"].numpy()[0])
            yield features["window_features"].numpy()
    # Per-feature latent std over the source trials: the unit in which the
    # temporal smoothing terms and the reported latent-edit structure are
    # measured. Source-derived, like every other calibration constant here.
    temporal_scale = latent_scale(encode_source())
    source_embeddings = np.stack(source_embeddings)
    region = TypicalityRegion.calibrate(
        source_embeddings, prior_mean=prior_mean, prior_variance=prior_variance,
        subject_ids=dataset.subject_ids[source], trial_ids=dataset.trial_ids[source], labels=dataset.labels[source],
        held_out_subject=subject, target_class=args.target_class,
        quantile=args.typicality_quantile, variance_floor=args.variance_floor,
    )
    region.save(directory / "calibration")
    write_json(directory / "calibration" / "representation.json", embedding_map.metadata())
    write_json(directory / "calibration" / "latent_scale.json", {
        "latent_std": temporal_scale, "n_source_trials": int(len(source)),
        "definition": "per-feature std of window_features over source trials, windows and timesteps; "
                      "floored at 1e-3 x median"})
    write_json(directory / "calibration" / "source_trials.json", dict(
        embeddings=source_embeddings, probabilities=np.stack(source_probabilities), discrepancy=region.score(source_embeddings),
        subject_ids=dataset.subject_ids[source], trial_ids=dataset.trial_ids[source], labels=dataset.labels[source]))
    neighbors = None
    neighbor_info = {"enabled": args.knn_evaluation, "status": "disabled"}
    if args.knn_evaluation:
        try:
            neighbors = EncodedKNNReference.fit_source(
                source_embeddings, labels=dataset.labels[source], subject_ids=dataset.subject_ids[source],
                trial_ids=dataset.trial_ids[source], held_out_subject=subject, target_class=args.target_class,
                k=args.knn_k, quantile=args.knn_quantile,
                calibration_fraction=args.knn_calibration_fraction, seed=args.seed)
        except InsufficientSourceDataError as error:
            neighbor_info = {"enabled": True, "status": "unavailable", "reason": str(error)}
            print(f"Subject {subject}: neighbour evaluation unavailable: {error}", flush=True)
        else:
            neighbors.save(directory / "calibration" / "knn_reference.json")
            neighbor_info = {"enabled": True, "status": "calibrated", **neighbors.metadata,
                             "threshold": neighbors.threshold}
    write_json(directory / "calibration" / "knn_status.json", neighbor_info)
    #Real source-subject EEG, matching the typicality region and the physiology
    #reference above. Calibrating on the held-out subject's own decoder
    #reconstructions instead leaks that subject into a reference the paper
    #describes as source-derived, and is circular: any decoder output scores
    #near zero against a decoder-output reference, which is why VCSC read
    #0.000 on every counterfactual while the independent checks all failed.
    vcsc = vcsc_calibration(dataset.features[source])
    write_json(directory / "calibration" / "vcsc.json", {
        **{name: vcsc[name] for name in ("c_hat", "w_hat", "sigma_raw", "sigma_spec")},
        "subject_ids": dataset.subject_ids[source], "trial_ids": dataset.trial_ids[source],
        "reference": "source_subject_real_eeg"})
    source_diagnostics = [_diagnostics(dataset, i, dataset.features[i], args) for i in source]
    physiological = PhysiologicalReference.fit(source_diagnostics, quantile=args.physiology_quantile,
                                               required_fraction=args.physiology_required_fraction)
    # Bounds are small; per-source descriptors, PSDs, and connectivity matrices
    # are used only in memory and are not archived.
    write_json(directory / "calibration" / "physiology.json", {
        name: np.where(np.isfinite(value), value, None).tolist()
        for name, value in physiological.arrays().items()})
    write_json(directory / "calibration" / "storage.json", {
        "artifact_format": "json_csv_summaries",
        "source_trial_ids": dataset.trial_ids[source].tolist(),
        "source_subject_ids": dataset.subject_ids[source].tolist(),
        "physiology_quantile": args.physiology_quantile,
        "physiology_required_fraction": args.physiology_required_fraction,
        "source_component_arrays_saved": False,
    })
    amplitude_spectral = (
        AmplitudeSpectralReference.from_physiological_reference(physiological, margin=args.amplitude_spectral_margin)
        if args.amplitude_spectral_weight > 0 else None)
    loss = make_vcsc_loss(vcsc, amplitude_spectral=amplitude_spectral, temporal_scale=temporal_scale,
                          target_weight=args.target_weight, latent_weight=args.latent_weight,
                          decoded_weight=args.decoded_weight, physiological_weight=args.physiological_weight,
                          amplitude_spectral_weight=args.amplitude_spectral_weight,
                          temporal_sparsity_weight=args.temporal_sparsity_weight,
                          temporal_tv_weight=args.temporal_tv_weight,
                          decoded_metric=args.decoded_distance,
                          target_probability=args.target_probability)
    physiological_tolerance = args.physiological_tolerance
    if args.physiological_tolerance_quantile is not None:
        # The same calibrated VCSC the optimizer uses, on the same real
        # source trials it is calibrated from. In-sample, so slightly
        # optimistic; held-out real-trial VCSC is archived per trial as
        # vcsc_original_input for comparison.
        score = tf.function(loss.physiological_validity)
        source_vcsc = np.asarray([float(score(tf.convert_to_tensor(dataset.features[i][None], tf.float32)).numpy())
                                  for i in source])
        physiological_tolerance = float(np.quantile(source_vcsc, args.physiological_tolerance_quantile))
        write_json(directory / "calibration" / "vcsc_tolerance.json", {
            "rule": "quantile of VCSC on source-subject real trials",
            "quantile": args.physiological_tolerance_quantile, "tolerance": physiological_tolerance,
            "n_source_trials": len(source_vcsc),
            "source_vcsc_quantiles": {str(q): float(np.quantile(source_vcsc, q))
                                      for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "source_vcsc_fraction_zero": float(np.mean(source_vcsc == 0)),
            "subject_ids": dataset.subject_ids[source], "trial_ids": dataset.trial_ids[source]})
        print(f"Subject {subject}: VCSC feasibility tolerance {physiological_tolerance:.6g} "
              f"(source real-trial quantile {args.physiological_tolerance_quantile})", flush=True)
    if amplitude_spectral is not None:
        write_json(directory / "calibration" / "amplitude_spectral.json", amplitude_spectral.metadata())
    common = dict(loss=loss, learning_rate=args.learning_rate, learning_rate_decay=args.learning_rate_decay,
                  target_loss_component=args.target_loss_component, max_steps=args.max_steps,
                  gradient_clip_norm=args.gradient_clip_norm,
                  min_gradient_norm=args.min_gradient_norm,
                  low_gradient_patience=args.low_gradient_patience,
                  physiological_tolerance=physiological_tolerance,
                  decoder_mode=args.decoder_mode, fixed_joint_alpha=args.fixed_joint_alpha, typicality=region)
    # Both arms use identical seeds, frozen models, starts, budgets, and central
    # losses. Base stops at first feasible success; typicality then minimizes D.
    optimizers = {
        "base": CounterfactualOptimizer(
            model, typicality_weight=0.0, stop_on_success=args.stop_on_success,
            **common,
        ),
        "typicality": CounterfactualOptimizer(
            model, typicality_weight=args.typicality_weight,
            stop_on_success=False,
            typicality_improvement_patience=args.typicality_improvement_patience,
            typicality_min_delta=args.typicality_min_delta,
            **common,
        ),
    }
    if args.include_target_latent:
        # Preserve the calibrated diagnostic while removing its gradient
        # contribution and the decoded proximity term for this arm only.
        target_latent = CounterfactualOptimizer(
            model, typicality_weight=0.0, stop_on_success=args.stop_on_success,
            **{**common, "loss": replace(loss, decoded_weight=0.0, physiological_weight=0.0,
                                         amplitude_spectral_weight=0.0, temporal_sparsity_weight=0.0,
                                         temporal_tv_weight=0.0)},
        )
        optimizers = {"target_latent": target_latent, **optimizers}
    if args.include_typicality_no_physiology:
        optimizers["typicality_no_physiology"] = CounterfactualOptimizer(
            model, typicality_weight=args.typicality_weight,
            stop_on_success=False,
            typicality_improvement_patience=args.typicality_improvement_patience,
            typicality_min_delta=args.typicality_min_delta,
            **{**common, "loss": replace(loss, physiological_weight=0.0, amplitude_spectral_weight=0.0)},
        )
    if args.report_output not in optimizers["base"].decoded_names:
        raise ValueError("The selected report output is not present in this checkpoint")
    # Not an objective: a per-trial baseline the objectives are measured against.
    # Stops at the first feasible iterate, so it is the smallest repair found.
    repair_optimizer = (CounterfactualOptimizer(model, typicality_weight=0.0, stop_on_success=True,
                                                repair_only=True, **common)
                        if args.include_repair_baseline else None)
    predictions, discrepancies, held_embeddings, eligible = [], [], [], []
    for index in held:
        features = model.get_encoder_features(dataset.features[index:index + 1])
        embedding = features["classification_embedding"]
        probabilities = features["probabilities"].numpy()[0]
        if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
            raise ValueError("The study requires finite binary trial probabilities")
        d = float(region.discrepancy(embedding).numpy())
        predictions.append(probabilities)
        discrepancies.append(d)
        held_embeddings.append(embedding.numpy()[0])
    audit = eligibility_records(task=args.task, direction=args.direction, subject=subject,
                                trial_ids=dataset.trial_ids[held], labels=dataset.labels[held],
                                probabilities=np.stack(predictions), trial_filter=args.trial_ids)
    eligible = [index for index, record in zip(held, audit) if record["eligible"]]
    write_json(directory / "eligibility.json", audit)
    observations = dict(trial_ids=dataset.trial_ids[held], labels=dataset.labels[held],
                        embeddings=np.stack(held_embeddings),
                        probabilities=np.stack(predictions), discrepancy=np.asarray(discrepancies))
    write_json(directory / "observations.json", observations)
    fold_info = {"subject_id": subject, "status": "running", "checkpoint": entry,
                 "direction": args.direction, "source_class": args.source_class,
                 "target_class": args.target_class, "typicality_quantile": args.typicality_quantile,
                 "reconstruction": adapter.metadata(),
                 "threshold": region.tau, "representation": embedding_map.metadata(),
                 "typicality_definition": SCORE_DEFINITION, "encoded_neighbors": neighbor_info,
                 "eligible_trial_ids": dataset.trial_ids[eligible].tolist(),
                 "all_trial_ids": dataset.trial_ids[held].tolist(), "loss": asdict(loss),
                 "objective_losses": {name: asdict(optimizer.loss) for name, optimizer in optimizers.items()},
                 "physiological_tolerance": physiological_tolerance,
                 "repair_baseline": bool(repair_optimizer),
                 "recognition": recognition_metrics(dataset.labels[held], np.stack(predictions), ece_bins=args.ece_bins)}
    write_json(directory / "fold.json", fold_info)
    print(f"Subject {subject}: tau={region.tau:.6g}; {len(eligible)} eligible trials, objectives={','.join(optimizers)}", flush=True)
    errors = 0
    for index in eligible:
        trial = int(dataset.trial_ids[index])
        scale, offset = _normalization(dataset, index)
        repair = None
        if repair_optimizer is not None:
            repair, failed = _repair_baseline(args, dataset, index, subject, trial, entry, repair_optimizer,
                                              loss, physiological, scale, offset, directory)
            errors += failed
        for objective, optimizer in optimizers.items():
            arm_dir = directory / f"trial_{trial}" / objective
            if args.resume and completed_attempt(arm_dir):
                continue
            seed = int(np.random.SeedSequence([args.seed, subject, trial]).generate_state(1)[0])
            tf.keras.utils.set_random_seed(seed)
            recorder = TrialRecorder(next_attempt(arm_dir))
            metadata = {"schema_version": SCHEMA_VERSION, "task": args.task, "subject_id": subject, "trial_id": trial,
                        "true_class": args.source_class, "source_class": args.source_class,
                        "target_class": args.target_class, "direction": args.direction,
                        "objective": objective, "seed": seed, "report_output": args.report_output,
                        "checkpoint_sha256": entry["sha256"], "typicality_definition": SCORE_DEFINITION,
                        "typicality_representation": REPRESENTATION,
                        "round_trip_evaluation": EVALUATION_PROTOCOL}

            def progress(row):
                recorder.record(row)
                if args.log_every and row["step"] % args.log_every == 0:
                    print(f"  s={subject} trial={trial} {objective} step={row['step']} "
                          f"p_target={row['target_probability']:.4f} D={row['discrepancy']:.5g} tau={region.tau:.5g}", flush=True)

            try:
                result = optimizer.optimize(dataset.features[index:index + 1], target_class=args.target_class,
                                             progress=progress, signal_scale=scale, signal_offset=offset)
                arrays, summary = result["arrays"], result["summary"]
                original, baseline, counterfactual = (arrays[name] for name in
                    ("x", f"x_reconstructed_{args.report_output}", f"x_prime_{args.report_output}"))
                diagnostic_sets = {name: _diagnostics(dataset, index, values, args)
                                   for name, values in (("original", original), ("reconstruction", baseline), ("counterfactual", counterfactual))}
                phys = physiological.assess(diagnostic_sets["counterfactual"])
                summary.update(physiology=phys,
                               vcsc_original_input=float(loss.physiological_validity(tf.convert_to_tensor(original)).numpy()),
                               d_z=float(np.sqrt(np.mean((arrays["classification_embedding_prime"] - arrays["classification_embedding"]) ** 2))),
                               decoder_latent_rmse=float(np.sqrt(summary["selected_losses"]["latent"])),
                               report_output=args.report_output)
                # Evaluation starts after optimization has selected its endpoint.
                # These vectors support independent scorers without retaining EEG.
                summary.update(evaluate_trial_embeddings(model, arrays,
                    report_output=args.report_output, reference=neighbors))
                summary["encoded_neighbors_status"] = neighbor_info["status"]
                # Every arm's delivered EEG is re-encoded by the frozen classifier
                # (reviewer request): does the decoded counterfactual itself flip
                # and land in the region? The reconstruction R(z) and, when run,
                # the repair R(z_rep) are scored the same way as controls.
                summary["decoded_counterfactual"] = _score_decoded_counterfactual(
                    model, region, counterfactual, report_output=args.report_output,
                    target_class=args.target_class, target_probability=args.target_probability,
                )
                summary["decoded_reconstruction"] = _score_decoded_counterfactual(
                    model, region, baseline, report_output=args.report_output,
                    target_class=args.target_class, target_probability=args.target_probability,
                    input_name="decoded_reconstruction_eeg_reencoded_by_frozen_classifier",
                )
                if repair is not None:
                    summary["decoded_repair"] = _score_decoded_counterfactual(
                        model, region, repair["reconstruction"], report_output=args.report_output,
                        target_class=args.target_class, target_probability=args.target_probability,
                        input_name="decoded_repair_eeg_reencoded_by_frozen_classifier",
                    )
                # A channel-by-band summary supports the report's scalp maps
                # without retaining decoded signals or full spectral arrays.
                band_power = {
                    "channel_names": dataset.channel_names, "band_names": dataset.band_names,
                    "power_change": (diagnostic_sets["counterfactual"]["spectral_power"]
                                     - diagnostic_sets["reconstruction"]["spectral_power"]),
                }
                if repair is not None:
                    # Explanation = change relative to the repaired decoder output.
                    diagnostic_sets["repair"] = repair["diagnostics"]
                    band_power["power_change_vs_repair"] = (diagnostic_sets["counterfactual"]["spectral_power"]
                                                            - repair["diagnostics"]["spectral_power"])
                    explanation = float(np.mean((counterfactual - repair["reconstruction"]) ** 2))
                    summary["repair_baseline"] = {
                        **repair["summary"],
                        "explanation_change_mse": explanation,
                        "delta_dec_explanation": float(np.sqrt(explanation)),
                        "explanation_change_env": envelope_distance_np(counterfactual, repair["reconstruction"]),
                    }
                elif repair_optimizer is not None:
                    summary["repair_baseline"] = {"available": False}
                write_json(recorder.directory / "band_power.json", band_power)
                write_json(recorder.directory / "physiology.json", {
                    name: physiological.assess(d, channel_names=dataset.channel_names,
                        band_names=dataset.band_names, signal_unit=dataset.signal_unit)
                    for name, d in diagnostic_sets.items()})
                # Kept out of physiology.json, whose schema downstream graders read.
                seam_signals = {"original": original, "reconstruction": baseline, "counterfactual": counterfactual}
                if repair is not None:
                    seam_signals["repair"] = repair["reconstruction"]
                write_json(recorder.directory / "seams.json", {
                    "band_names": dataset.band_names,
                    **{name: _seams(dataset, index, values) for name, values in seam_signals.items()}})
                recorder.finish(result, metadata)
            except (FloatingPointError, RuntimeError, ValueError, tf.errors.OpError) as error:
                recorder.fail(error, metadata)
                errors += 1
                print(f"  FAILED s={subject} trial={trial} {objective}: {error}", flush=True)
            finally:
                recorder.close()
    fold_info.update(status="completed", n_optimization_errors=errors)
    write_json(directory / "fold.json", fold_info)
    tf.keras.backend.clear_session()


def run_study(args):
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
    for fold in folds:
        info = out / f"subject_{fold['subject_id']}" / "fold.json"
        if args.resume and info.exists():
            previous = json.loads(info.read_text())
            if previous["status"] == "completed" and all(
                completed_attempt(info.parent / f"trial_{trial}" / objective)
                for trial in previous["eligible_trial_ids"]
                for objective in protocol["objectives"] + (["repair"] if protocol.get("repair_baseline") else [])
            ):
                print(f"Subject {fold['subject_id']}: verified cached fold", flush=True)
                continue
        run_fold(args, dataset, fold, out)
        build_report([out], out / "report")
    return build_report([out], out / "report")


def run(args):
    """Keep the two directions isolated, then report without pooling them."""
    roots = []
    for direction in args.directions:
        study_args = copy(args)
        study_args.direction = direction
        study_args.source_class, study_args.target_class = map(int, direction.split("_to_"))
        study_args.out_dir = args.out_dir / direction
        run_study(study_args)
        roots.append(study_args.out_dir)
    result = build_report(roots, args.out_dir / "report")
    if args.knn_evaluation:
        try:
            build_knn_report(roots, args.out_dir / "knn_report", k=args.knn_k,
                quantile=args.knn_quantile, calibration_fraction=args.knn_calibration_fraction, seed=args.seed)
        except InsufficientSourceDataError as error:
            # A small development shard can be valid for optimization while
            # lacking enough independent real target references for k=5.
            write_json(args.out_dir / "knn_report" / "status.json",
                       {"status": "unavailable", "reason": str(error)})
    return result


def main(argv=None):
    result = run(parse_args(argv))
    return 1 if any(row["n_error"] for row in result["population"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
