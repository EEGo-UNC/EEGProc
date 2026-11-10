"""Paired CoMTE / CFO LOSO comparison on delivered full-trial EEG signals.

Uses the existing typicality study's data, frozen checkpoints, eligibility,
directions, target probability, VC threshold, losses, and physiology. Every
endpoint is independently re-encoded, irrespective of optimizer success.
"""

from copy import copy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

from ..typicality import runner as common
from ..typicality.artifacts import (array_sha256, completed_attempt, file_sha256,
                                   next_attempt, TrialRecorder, write_json)
from ..typicality.core import TypicalityRegion
from ..typicality.coverage import eligibility_records
from ..typicality.physiology import PhysiologicalReference, make_vcsc_loss, vcsc_calibration
from ..typicality.sic_embedding import SICTrialEmbedding
from ..counterfactuals.optimizer import CounterfactualOptimizer
from ..model_agnostic.adapter import load_trial_dataset
from ..model_agnostic.sic_adapter import create_sic_adapter
from .evaluation import IndependentTypicality, evaluate_signal, spectral_descriptor, signal_displacement
from .report import build_report
from .search import SearchConfig, UPSTREAM_COMMIT, optimized_search


def parse_args(argv=None):
    parser = common.build_parser()
    parser.description = __doc__
    parser.set_defaults(include_target_latent=False, include_typicality_no_physiology=False)
    parser.add_argument("--cfo-objective", choices=("base", "typicality"), default="typicality")
    for flag, default in (("distractors", 2), ("max-features", 3), ("max-attempts", 1000),
                          ("max-iterations", 1000), ("restarts", 5), ("max-queries", 10000)):
        parser.add_argument("--comte-" + flag, type=int, default=default)
    parser.add_argument("--comte-group-size", type=int, choices=(1, 3), default=3,
                        help="3 replaces all bands of one electrode; 1 reproduces feature-wise substitutions.")
    parser.add_argument("--independent-quantile", type=float, default=.95)
    parser.add_argument("--independent-neighbors", type=int, default=5)
    parser.add_argument("--channel-atol", type=float, default=1e-6)
    parser.add_argument("--channel-rtol", type=float, default=1e-3)
    args = common.parse_args(argv, parser=parser)
    if args.include_target_latent or args.include_typicality_no_physiology:
        parser.error("This paired comparator uses --cfo-objective; run other ablations separately")
    try:
        search_config(args)
    except ValueError as error:
        parser.error(str(error))
    if (not np.isfinite(args.independent_quantile) or not 0 < args.independent_quantile < 1
            or args.independent_neighbors < 1):
        parser.error("Independent quantile must be in (0,1) and neighbors positive")
    if not np.isfinite([args.channel_atol, args.channel_rtol]).all() or min(args.channel_atol, args.channel_rtol) < 0:
        parser.error("Channel tolerances must be finite and nonnegative")
    return args


def search_config(args):
    return SearchConfig(**{name: getattr(args, "comte_" + name) for name in
                          ("distractors", "max_features", "max_attempts", "max_iterations",
                           "restarts", "max_queries", "group_size")})


class CountingModel:
    """Count actual full-signal model invocations without changing the model."""
    def __init__(self, model):
        self.wrapped = model
        self.signal_queries = 0
        self.latent_queries = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def get_encoder_features(self, inputs):
        self.signal_queries += int(np.shape(inputs)[0])
        return self.wrapped.get_encoder_features(inputs)


class CountedCFO(CounterfactualOptimizer):
    def _classification_state(self, latent):
        self.model.latent_queries += 1
        return super()._classification_state(latent)


def _protocol(args, dataset, folds):
    protocol = common._protocol(args, dataset, folds)
    code_files = [*Path(__file__).parent.rglob("*.py"), *Path(__file__).parent.joinpath("upstream").glob("*")]
    protocol.update(
        comparison_schema=1, methods=["cfo", "comte"], objectives=["cfo", "comte"],
        comparison_settings={
            "seed": args.seed,
            "target_probability": args.target_probability, "vc_quantile": args.typicality_quantile,
            "independent_quantile": args.independent_quantile, "independent_neighbors": args.independent_neighbors,
            "physiology_quantile": args.physiology_quantile, "physiology_required_fraction": args.physiology_required_fraction,
            "vcsc_tolerance": args.physiological_tolerance, "fs": args.fs,
            "channel_atol": args.channel_atol, "channel_rtol": args.channel_rtol,
            "cfo_objective": args.cfo_objective, "search": asdict(search_config(args)),
            "cfo": {key: getattr(args, key) for key in (
                "max_steps", "learning_rate", "learning_rate_decay", "target_loss_component",
                "target_weight", "latent_weight", "decoded_weight", "physiological_weight",
                "typicality_weight", "variance_floor", "decoder_mode", "fixed_joint_alpha", "report_output",
                "gradient_clip_norm", "stop_on_success", "min_gradient_norm", "low_gradient_patience",
                "typicality_improvement_patience", "typicality_min_delta")},
        },
        upstream=dict(repository="https://github.com/peaclab/CoMTE", commit=UPSTREAM_COMMIT,
                      variant="OptimizedSearch with greedy fallback; array/group/confidence and endpoint fixes documented"),
        counterfactual_validity_prediction_space="delivered_signal_full_pipeline",
        round_trip_evaluation="both_method_output_signals_reencoded",
        counterfactual_validity="target argmax AND shared target probability on output EEG",
        independent_typicality="source-subject-split standardized log-band-power kNN, never used for generation or selection",
        joint_success="signal prediction validity AND independent typicality AND all required physiology checks AND VCSC",
        reference_policy="only checkpoint-declared source subjects; no held-out tuning",
        query_definition="one classifier prediction per trial; distinguish complete signal forwards, latent-only forwards, gradients, and shared evaluation",
        budget_policy="CFO max_steps and CoMTE max_queries are separate predeclared budgets; no claim of equal computational budgets",
        decoded_evaluation="every selected endpoint is re-encoded by the frozen complete classifier; latent metrics are secondary",
        distance_definition="RMSE and relative L2 from delivered output to observed original input; electrode sparsity groups all three bands",
    )
    protocol["source_sha256"].update({"comte/" + p.relative_to(Path(__file__).parent).as_posix(): file_sha256(p)
                                      for p in code_files if p.is_file()})
    return json.loads(json.dumps(protocol))


def _validate_dataset(args, dataset):
    if (dataset.features.ndim != 4 or dataset.features.shape[-1] != 42
            or dataset.labels is None or not np.isin(dataset.labels, [0, 1]).all()):
        raise ValueError("Use the same labeled full SIC trials (N,W,T,42) as the CFO study")
    if dataset.feature_order not in (None, "channel-major"):
        raise ValueError("The frozen SIC model requires channel-major theta/alpha/beta inputs")
    if dataset.band_names is not None and tuple(str(v).lower() for v in dataset.band_names) != ("theta", "alpha", "beta"):
        raise ValueError("Band order must be theta, alpha, beta")
    if dataset.channel_names is not None and tuple(dataset.channel_names) != tuple(common._VCSC_CHANNELS):
        raise ValueError("Channel names must match the frozen DREAMER montage")
    if (dataset.normalization_scale is None) != (dataset.normalization_offset is None):
        raise ValueError("Provide both normalization scale and offset, or neither")
    if dataset.metadata.get("fs", args.fs) != args.fs:
        raise ValueError("Sampling rate differs from the training preprocessing")
    if np.any(dataset.subject_ids < 0) or np.any(dataset.trial_ids < 0):
        raise ValueError("Subject/trial IDs must be nonnegative")
    if args.trials_npz:
        with np.load(args.trials_npz, allow_pickle=False) as prepared:
            if "window_mask" in prepared and (prepared["window_mask"].shape != dataset.features.shape[:2]
                                               or not np.all(prepared["window_mask"] == 1)):
                raise ValueError("Padded/masked trials are not supported")
    dataset.channel_names = tuple(common._VCSC_CHANNELS)
    dataset.band_names = ("theta", "alpha", "beta")
    dataset.feature_order = "channel-major"
    common._diagnostics(dataset, 0, dataset.features[0], args)


def run_fold(args, dataset, entry, out):
    started = time.perf_counter()
    tf.keras.utils.set_random_seed(args.seed)
    subject = entry["subject_id"]
    directory = out / f"subject_{subject}"
    directory.mkdir(parents=True, exist_ok=True)
    held = np.flatnonzero(dataset.subject_ids == subject)
    source = np.flatnonzero(np.isin(dataset.subject_ids, entry["source_subject_ids"]))
    if subject in dataset.subject_ids[source] or not len(source):
        raise ValueError("CoMTE references/calibration must contain source subjects only")
    source_signals = dataset.features[source]
    if args.trial_ids and not set(args.trial_ids).issubset(set(dataset.trial_ids[held].tolist())):
        raise ValueError("Requested trial IDs are absent from the held-out subject")
    adapter = create_sic_adapter(model_path=Path(entry["path"]),
        config=dict(model_module=args.model_module, decoder_mode=args.decoder_mode,
                    fixed_joint_alpha=args.fixed_joint_alpha), sample_input=dataset.features[held[0]])
    adapter.model.trainable = False
    model = CountingModel(adapter.model)
    initial_weights = [value.numpy().copy() for value in model.weights]
    embeddings, probabilities = [], []
    for i in source:
        features = model.get_encoder_features(dataset.features[i:i+1])
        embeddings.append(np.asarray(features["classification_embedding"])[0])
        probabilities.append(np.asarray(features["probabilities"])[0])
    embeddings, probabilities = np.stack(embeddings), np.stack(probabilities)
    prior_mean, prior_variance = SICTrialEmbedding(adapter.model).learned_prior(args.target_class)
    region = TypicalityRegion.calibrate(embeddings, prior_mean=prior_mean, prior_variance=prior_variance,
        subject_ids=dataset.subject_ids[source], trial_ids=dataset.trial_ids[source], labels=dataset.labels[source],
        held_out_subject=subject, target_class=args.target_class, quantile=args.typicality_quantile,
        variance_floor=args.variance_floor)
    region.save(directory / "calibration")
    diagnostics = [common._diagnostics(dataset, i, dataset.features[i], args) for i in source]
    physiological = PhysiologicalReference.fit(diagnostics, quantile=args.physiology_quantile,
                                               required_fraction=args.physiology_required_fraction)
    independent = IndependentTypicality.fit(np.stack([spectral_descriptor(d) for d in diagnostics]),
        subject_ids=dataset.subject_ids[source], trial_ids=dataset.trial_ids[source], labels=dataset.labels[source],
        held_out_subject=subject, target_class=args.target_class, quantile=args.independent_quantile,
        neighbors=args.independent_neighbors, seed=args.seed)
    vcsc = vcsc_calibration(source_signals)
    loss = make_vcsc_loss(vcsc, target_weight=args.target_weight, latent_weight=args.latent_weight,
        decoded_weight=args.decoded_weight, physiological_weight=args.physiological_weight,
        target_probability=args.target_probability)
    optimizer = CountedCFO(model, loss=loss, learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay, target_loss_component=args.target_loss_component,
        max_steps=args.max_steps, gradient_clip_norm=args.gradient_clip_norm,
        stop_on_success=args.stop_on_success if args.cfo_objective == "base" else False,
        min_gradient_norm=args.min_gradient_norm, low_gradient_patience=args.low_gradient_patience,
        physiological_tolerance=args.physiological_tolerance, decoder_mode=args.decoder_mode,
        fixed_joint_alpha=args.fixed_joint_alpha, typicality=region,
        typicality_weight=args.typicality_weight if args.cfo_objective == "typicality" else 0.,
        typicality_improvement_patience=args.typicality_improvement_patience,
        typicality_min_delta=args.typicality_min_delta)
    if args.report_output not in optimizer.decoded_names:
        raise ValueError("Selected decoder output does not exist in this checkpoint")
    keys = np.column_stack((dataset.subject_ids[source], dataset.trial_ids[source]))
    write_json(directory / "calibration" / "independent.json", independent.payload())
    write_json(directory / "calibration" / "source_trials.json", dict(keys=keys, labels=dataset.labels[source], probabilities=probabilities))
    write_json(directory / "calibration" / "physiology.json", {
        name: np.where(np.isfinite(value), value, None).tolist() for name, value in physiological.arrays().items()})
    write_json(directory / "calibration" / "vcsc.json", {
        **{name: vcsc[name] for name in ("c_hat", "w_hat", "sigma_raw", "sigma_spec")}, "source_trial_keys": keys})
    held_probabilities = np.stack([np.asarray(model.get_encoder_features(dataset.features[i:i+1])["probabilities"])[0] for i in held])
    audit = eligibility_records(task=args.task, direction=args.direction, subject=subject,
        trial_ids=dataset.trial_ids[held], labels=dataset.labels[held], probabilities=held_probabilities,
        trial_filter=args.trial_ids)
    eligible = [i for i, row in zip(held, audit) if row["eligible"]]
    write_json(directory / "eligibility.json", audit)
    calibration_hashes = {p.name: file_sha256(p) for p in (directory / "calibration").glob("*.json")}
    fold = dict(subject_id=subject, status="running", checkpoint=entry, direction=args.direction,
        eligible_trial_ids=dataset.trial_ids[eligible], calibration_sha256=calibration_hashes,
        setup_seconds=time.perf_counter()-started, setup_signal_queries=model.signal_queries,
        source_reference_keys=keys, thresholds=dict(target_probability=args.target_probability,
        vc=region.tau, independent=independent.threshold, vcsc=args.physiological_tolerance))
    write_json(directory / "fold.json", fold)
    print(f"Subject {subject}, {args.direction}: {len(eligible)} paired trials; source-only references={len(source)}", flush=True)
    study_hash = file_sha256(out / "study.json")
    errors = 0
    for index in eligible:
        trial = int(dataset.trial_ids[index])
        seed = int(np.random.SeedSequence([args.seed, subject, trial]).generate_state(1)[0])
        # Alternate ordering deterministically to reduce systematic warmup bias.
        methods = ("cfo", "comte") if seed % 2 else ("comte", "cfo")
        for method in methods:
            arm = directory / f"trial_{trial}" / method
            if args.resume and completed_attempt(arm):
                continue
            tf.keras.utils.set_random_seed(seed)
            recorder = TrialRecorder(next_attempt(arm))
            metadata = dict(task=args.task, direction=args.direction, subject_id=subject, trial_id=trial,
                target_class=args.target_class, method=method, seed=seed,
                checkpoint_sha256=entry["sha256"], study_sha256=study_hash,
                calibration_sha256=calibration_hashes, execution_order=list(methods))
            signal_before, latent_before = model.signal_queries, model.latent_queries
            generation_started = time.perf_counter()
            evaluation_started = None
            try:
                original = dataset.features[index]
                if method == "cfo":
                    result = optimizer.optimize(original[None], target_class=args.target_class,
                        progress=recorder.record, record_state_arrays=False)
                    endpoint = result["arrays"][f"x_prime_{args.report_output}"][0]
                    method_summary = result["summary"]
                    baseline = result["arrays"][f"x_reconstructed_{args.report_output}"][0]
                    extra = dict(latent_prediction_validity=method_summary["latent_counterfactual"]["success"],
                                 reconstruction_rmse=signal_displacement(original, baseline)["signal_rmse"])
                else:
                    method_summary = optimized_search(
                        lambda x: model.get_encoder_features(x)["probabilities"], original, source_signals,
                        reference_labels=dataset.labels[source], reference_probabilities=probabilities,
                        reference_keys=keys, held_out_subject=subject, target_class=args.target_class,
                        target_probability=args.target_probability, seed=seed, config=search_config(args), progress=recorder.record)
                    endpoint = method_summary.pop("signal")
                    extra = {}
                generation_seconds = time.perf_counter() - generation_started
                generation_signal = model.signal_queries - signal_before
                generation_classifier = generation_signal + model.latent_queries - latent_before
                evaluation_started = time.perf_counter()
                endpoint_diagnostics = common._diagnostics(dataset, index, endpoint, args)
                penalty = float(loss.physiological_validity(tf.convert_to_tensor(endpoint[None])).numpy())
                evaluated = evaluate_signal(model, endpoint, original, target_class=args.target_class,
                    target_probability=args.target_probability, vc_region=region, independent_region=independent,
                    diagnostics=endpoint_diagnostics, physiology=physiological,
                    vcsc_penalty=penalty, vcsc_tolerance=args.physiological_tolerance,
                    atol=args.channel_atol, rtol=args.channel_rtol)
                evaluation_seconds = time.perf_counter() - evaluation_started
                if len(initial_weights) != len(model.weights) or any(
                    not np.array_equal(initial, current.numpy()) for initial, current in zip(initial_weights, model.weights)):
                    raise RuntimeError("Frozen model weights changed during comparison")
                # Preserve model-unit displacement for a common comparison, plus
                # physical-unit displacement only when the input affine map exists.
                if dataset.normalization_scale is not None:
                    physical = signal_displacement(common._physical_signal(dataset, index, original),
                        common._physical_signal(dataset, index, endpoint), atol=args.channel_atol, rtol=args.channel_rtol)
                    evaluated["physical_signal_rmse"] = physical["signal_rmse"]
                    evaluated["physical_signal_unit"] = dataset.signal_unit
                summary = {**evaluated, **extra, "method_summary": method_summary,
                    "stop_reason": method_summary["stop_reason"], "generation_seconds": generation_seconds,
                    "evaluation_seconds": evaluation_seconds, "total_seconds": generation_seconds + evaluation_seconds,
                    "generation_signal_queries": generation_signal,
                    "generation_classifier_queries": generation_classifier,
                    "gradient_evaluations": method_summary["gradient_evaluations"] if method == "cfo" else 0,
                    "evaluation_signal_queries": model.signal_queries - signal_before - generation_signal,
                    "total_classifier_queries": model.signal_queries - signal_before + model.latent_queries - latent_before}
                recorder.finish({"summary": summary}, metadata)
                print(f"  trial={trial} {method}: valid={evaluated['prediction_validity']} independent={evaluated['independent_typicality']} queries={summary['total_classifier_queries']}", flush=True)
            except (FloatingPointError, RuntimeError, ValueError, tf.errors.OpError) as error:
                errors += 1
                failed_metadata = {**metadata, "generation_seconds": time.perf_counter() - generation_started
                    if evaluation_started is None else generation_seconds,
                    "generation_signal_queries": model.signal_queries - signal_before if evaluation_started is None else generation_signal,
                    "generation_classifier_queries": model.signal_queries - signal_before + model.latent_queries - latent_before
                    if evaluation_started is None else generation_classifier,
                    "evaluation_seconds": time.perf_counter() - evaluation_started if evaluation_started is not None else 0.,
                    "total_seconds": time.perf_counter() - generation_started,
                    "evaluation_signal_queries": model.signal_queries - signal_before - generation_signal if evaluation_started is not None else 0,
                    "total_classifier_queries": model.signal_queries - signal_before + model.latent_queries - latent_before}
                recorder.fail(error, failed_metadata)
                print(f"  FAILED trial={trial} {method}: {error}", flush=True)
            finally:
                recorder.close()
    fold.update(status="completed", n_errors=errors)
    write_json(directory / "fold.json", fold)
    tf.keras.backend.clear_session()


def run(args):
    dataset = load_trial_dataset(args.trials_npz, loader_spec=args.data_loader, loader_config=args.data_config)
    _validate_dataset(args, dataset)
    roots = []
    for direction in args.directions:
        current = copy(args)
        current.direction = direction
        current.source_class, current.target_class = map(int, direction.split("_to_"))
        current.out_dir = args.out_dir / direction
        out = current.out_dir
        folds = common._manifest(current, dataset)
        protocol = _protocol(current, dataset, folds)
        if out.exists() and any(out.iterdir()) and not args.resume:
            raise FileExistsError(f"Choose a new output directory or use --resume: {out}")
        out.mkdir(parents=True, exist_ok=True)
        if (out / "study.json").exists():
            if json.loads((out / "study.json").read_text()) != protocol:
                raise ValueError("Resume refused: comparison data, code, model, or settings changed")
        else:
            if args.resume and any(out.iterdir()):
                raise ValueError("Cannot resume a comparison without study.json")
            write_json(out / "study.json", protocol)
            try:
                revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = None
            write_json(out / "environment.json", dict(python=sys.version, tensorflow=tf.__version__,
                numpy=np.__version__, platform=platform.platform(), devices=[d.name for d in tf.config.list_physical_devices()],
                git_revision=revision, command=sys.argv))
        for entry in folds:
            info = out / f"subject_{entry['subject_id']}" / "fold.json"
            if args.resume and info.exists():
                previous = json.loads(info.read_text())
                if previous["status"] == "completed" and all(completed_attempt(info.parent / f"trial_{trial}" / method)
                    for trial in previous["eligible_trial_ids"] for method in ("cfo", "comte")):
                    continue
            run_fold(current, dataset, entry, out)
            build_report([out], out / "report")
        roots.append(out)
    return build_report(roots, args.out_dir / "report")


def main(argv=None):
    result = run(parse_args(argv))
    return int(any(row["n_error"] for row in result["population"]))


if __name__ == "__main__":
    raise SystemExit(main())
