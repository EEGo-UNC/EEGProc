# External CoMTE comparison

This is a paired signal-space evaluation of CFO and CoMTE using the **same
frozen SIC encoder–classifier**. The default CFO arm is the existing typicality
objective; `--cfo-objective base` selects its base objective. Neither method
trains a classifier. Both class directions are evaluated separately by default.

## Provenance and adaptations

CoMTE is from [peaclab/CoMTE](https://github.com/peaclab/CoMTE), accompanying
[Ates et al., ICAPAI 2021](https://www.bu.edu/peaclab/files/2021/05/CoMTE___ICAPAI.pdf).
The upstream revision is `6042b267928415bbae84448faf2c8b670434cc15`.
The unmodified `explainers.py` and its BSD-3-Clause license are retained under
`upstream/`; the source SHA-256 is
`42f15cc3aa193ef264d941178522c1921222e214206ca9ac65a040e0fafddca7`.

`search.py` adapts upstream `OptimizedSearch`, `LossDiscreteState`, and
`BruteForceSearch` to arrays and full-trial SIC probabilities. It preserves:

- Nearest Euclidean source distractors with the requested true class and
  correctly predicted class; a counterfactual uses one distractor.
- Binary whole-series substitution and the relaxed objective
  `max(0, p_target - p)^2 + 0.8 * max(0, n_substitutions - max_features)`.
- Random hill climbing with five additional restarts by default, greedy
  pruning, and the greedy fallback when no pruned endpoint succeeds.
- Highest target-probability successful endpoint selection across distractors.
  Typicality and physiological evaluation never select CoMTE endpoints.

Explicit adaptations for this study:

- The classifier callback runs the saved encoder **and** classifier on all
  windows of the supplied trial. No classifier is refitted for CoMTE.
- Default substitution groups are electrodes: all three theta/alpha/beta
  streams of one electrode are replaced together. `--comte-group-size 1`
  reproduces upstream feature-wise substitutions; the report still measures
  sparsity over electrodes. Label this choice in any publication.
- Upstream's hard-coded `0.95` loss target and argmax-only acceptance are
  replaced by the same configured probability **and** target argmax criterion
  used by CFO. The default probability is `0.8`.
- A local NumPy random hill climber replaces the obsolete mlrose/scikit-learn
  dependency interface. Model calls are not cached. This is an adapted CoMTE
  implementation, not a bit-for-bit runtime reproduction of its old software.
- A selected substitution mask is always paired with its actual distractor.
  Fallback checks the shared confidence criterion. Unsuccessful finite
  endpoints and missing-distractor outcomes are
  returned explicitly. A query budget returns the best completed endpoint,
  including the original if pruning did not finish.
- `max_features` remains a **soft** hill-climb limit; the upstream greedy
  fallback may replace more groups. The independent output sparsity is measured
  from the signal, not inferred from that parameter.

## Run

Install the repository with its `deep-learning` dependencies, then run from the
repository root. Supply exactly the frozen checkpoint manifest and prepared
trials used for the corresponding CFO study:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.comte.runner \
  --models-json /path/to/loso_zero_shot_models.json \
  --model-dir /path/to/loso_zero_shot_models \
  --trials-npz /path/to/valence_trials.npz \
  --task valence --directions 0_to_1 1_to_0 \
  --cfo-objective typicality --target-probability 0.8 \
  --decoder-mode joint --report-output joint \
  --typicality-quantile 0.10 --max-steps 200 \
  --physiological-weight 1 --physiological-tolerance 1e-8 \
  --comte-distractors 2 --comte-max-features 3 --comte-group-size 3 \
  --comte-max-attempts 1000 --comte-max-iterations 1000 \
  --comte-restarts 5 --comte-max-queries 10000 \
  --independent-quantile 0.95 --independent-neighbors 5 \
  --out-dir runs/comte/valence
```

Use the arousal manifest/data and `--task arousal` for arousal. Copy **all**
loss weights, optimizer settings, decoder mixing, sampling rate, normalization,
and stopping settings from the CFO protocol being compared; the example uses
this branch's defaults. The current CFO branch uses the **10th percentile** VC
cutoff, recorded separately from independent typicality's 95th percentile.
Do not substitute a different historical cutoff without recalibrating the
entire paired study. Joint fusion can use the saved mixer or the same
`--fixed-joint-alpha` override as the original CFO study.

The existing `--data-loader`/`--data-config` route is also supported. It must
produce the same preprocessing as training: finite chronological full trials
`(N,W,T,42)`, channel-major theta/alpha/beta, binary true labels and unique
subject/trial IDs. Padding/masking and band/channel reordering are rejected.
Outputs are band-filtered model-input EEG, not reconstructed unfiltered raw
EEG. Both methods preserve the complete supplied window and time axes.

`--subjects 0 1` is an explicit fold shard; each selected subject needs its
zero-shot source checkpoint. Without it, every dataset subject needs a
checkpoint. `--trial-ids` is only an evaluation filter: all held-out eligibility
decisions are still saved and all declared source trials remain in calibration.
It must not be used to cherry-pick successful trials for the reported study.

Hyperparameters are fixed before held-out evaluation; this runner performs no
held-out tuning or model selection. If parameters need tuning, do that on
nested source-subject folds, then lock them for the outer fold. CoMTE's donor
bank, VC cutoff, independent fit/calibration sets, VCSC constants, and
physiological bounds all exclude the held-out subject and use only the
checkpoint's declared source subjects (or the explicitly recorded LOSO
complement when the manifest omits them). Checkpoint training provenance is
declared by the manifest and cannot be proven by this evaluation script.

## Shared evaluation

Eligibility matches the CFO study: true source class and original prediction
equal to source class. Each eligible trial has exactly one primary endpoint
per method, regardless of success. The target class, trial, source references,
frozen checkpoint, seed, and thresholds are shared within each pair.

Every delivered endpoint is passed through `get_encoder_features` again.
For CFO this means **decoding and then re-encoding**, even if the optimized
latent prediction succeeded. CoMTE's final signal is also independently
re-evaluated, even if it was queried during search. Latent success is a
secondary CFO diagnostic and never overrides signal-space validity.

The shared metrics are:

| Metric | Definition |
|---|---|
| Prediction validity | Delivered signal has target argmax and probability at least the shared threshold. |
| VC typicality | Re-encoded delivered signal has `D <= tau` against the same learned target Gaussian and source VC cutoff. |
| Independent typicality | k-th-neighbor RMSE in standardized log electrode/band powers is at most its source calibration cutoff. It uses neither the VC Gaussian nor optimized latents. |
| Physiological acceptance | All five existing physiological checks pass **and** source-calibrated VCSC penalty is within the shared tolerance. |
| Joint success | Prediction validity AND independent typicality AND physiological acceptance. |
| Available-check outcomes | Separate physiology and joint outcomes using only available checks plus VCSC; these do not establish full physiological acceptance. |
| Displacement | RMSE, L2 and relative L2 between delivered signal and observed original input, in model-input units. CFO reconstruction error is recorded separately. Physical RMSE is additionally recorded when the inverse normalization is supplied. |
| Channel sparsity | One minus the fraction of electrodes whose RMS change over all samples, windows, and bands exceeds `channel_atol + channel_rtol * original_channel_RMS`. Defaults: `1e-6`, `1e-3`. |
| Cost | Separate generation/evaluation/total wall time; full-signal forwards, total classifier predictions, and CFO gradient evaluations. |

Independent typicality uses deterministic **disjoint source subjects** for
its reference/scaling fit and cutoff calibration. Only true target-class real
source trials enter either set; at least two source subjects must have target
trials. `k` is clipped to the reference size. Scaling and reference descriptors,
the split keys, and the quantile cutoff are saved. No counterfactuals or held-out
trials calibrate it. It is an operational independent assessment, not a proof
of distributional or physiological realism, and its spectral descriptors may
correlate with CFO's physiological terms.

The existing aperiodic-exponent check is unavailable from these band-filtered
signals. It remains `null`, never a pass. Strict physiological/joint rates are
therefore unavailable when the corresponding decision is unknown; a known
failed condition still makes the conjunction false. The available-check metrics
are supplied separately rather than relabeled as complete acceptance.

Setup/calibration cost and queries are saved per fold, outside per-attempt
generation cost. CFO counts its actual latent-only classifier calls separately
from complete encoder–classifier forwards. A complete signal forward counts
once as a classifier prediction. The selected endpoint evaluation adds one
complete forward to each method. Forward-query counts do not represent the
cost of CFO's gradients; gradient evaluations and wall time are also reported.
CoMTE uses a predeclared query cap and CFO uses its predeclared step cap; these
are **not equal computational budgets**. Method order alternates deterministically
across trial seeds. Use the same device for all pairs and record the environment.

## Artifacts and reports

Only compact JSON/CSV metrics and histories are written. Generated signals,
latents, PSDs, and connectivity tensors are not persisted. Stored source
descriptors are small log-power vectors used to audit independent calibration.

```text
<out>/<direction>/study.json       paired protocol, dataset/code/checkpoint hashes
<out>/<direction>/environment.json software/device information
<out>/<direction>/subject_<id>/
  fold.json                       thresholds, source keys, setup cost, eligibility
  eligibility.json                every held-out inclusion/exclusion
  calibration/*.json              source references and frozen scoring settings
  trial_<id>/{cfo,comte}/attempt_<n>/
    history.jsonl + history.csv
    result.json                   shared output-signal metrics and method diagnostics
    complete.json                 hashes of committed attempt artifacts
<out>/report/
  trials.csv                      one primary row per eligible trial/method
  attempts.csv                    all attempts, including errors before retries
  subjects.csv                    held-out-subject summaries, including zero-eligible folds
  population.csv                  micro rates and equal-subject macro rates
  paired_subject_differences.csv   CFO minus CoMTE within each held-out subject
  comparison.json + README.md
```

Success-rate denominators include unsuccessful endpoints and errors. Pending
outcomes remain provisional. Unavailable scientific checks remain unknown.
Distance, sparsity, runtime and query summaries include finite unsuccessful
endpoints, and runtime/query counts are also retained for errors; each summary
includes its observed count. Undefined distances are not filled with zero.
Rates are separated by task and class direction. Error retries remain in the
attempt ledger, while primary per-trial denominators count the paired trial once.

`--resume` verifies committed results and refuses changed data, code, checkpoints,
or protocol. Interrupted attempts are retained. Reports verify calibration and
completion hashes and refuse incompatible settings or duplicate fold shards.
Rebuild reports without loading TensorFlow or a model:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.comte.report \
  runs/comte/valence/0_to_1 runs/comte/valence/1_to_0 \
  --out-dir runs/comte/valence/report
```

This implementation and its synthetic checks are not empirical evidence that
either method outperforms the other. Final results require running the complete
paired protocol on the actual frozen LOSO checkpoints and prepared trial data.
