# Physiological reports from JSON

Both grading commands read `physiology.json`. The report type is explicit:

- `--report binary` reads the saved per-family pass/fail and in-range fractions.
  It needs no raw EEG, checkpoints, component arrays, or TensorFlow.
- `--report graded` (default) reports graded distance when the archive has
  saved graded values or enough component values to recompute it. Binary
  pass fractions alone cannot recover graded distances.

## Existing summary-only suites

For the saved arousal/valence suites containing only binary JSON summaries:

```bash
python src/eegproc/model_explainability/typicality/grade_physiology_table.py \
  --suite runs/path/to/suite --task valence --which counterfactual --report binary
```

The table reports each objective and physiological family separately: number
of evaluable attempts / total attempts, percent passing the saved binary
check, and median [Q1, Q3] in-range fraction. Fractions are on the 0–1 scale.
Undefined families show `N/A`; they are excluded from that family's statistics.
These values are binary-check summaries, not graded-distance substitutes.

For one attempt:

```bash
python src/eegproc/model_explainability/typicality/grade_physiology.py \
  --attempt runs/path/to/subject_0/trial_7/base/attempt_0001 \
  --held-out 0 --task valence --which counterfactual --report binary
```

`--which` accepts `counterfactual`, `original`, or `reconstruction`. The table
also accepts `all` (its default): real EEG and reconstruction are counted once
per trial, followed by each objective's counterfactual and its available-family
pass rate. Explicit single-signal reports retain grouping by objective.

## Saving component values for new runs

`PhysiologicalReference.assess()` now includes a JSON-safe `components` object
alongside the existing checks and graded scores. A runner that saves the three
state assessments in `physiology.json` therefore retains actual component
values for `original`, `reconstruction`, and `counterfactual` automatically.
This applies equally to `target_latent`, `base`, `typicality`, and
`typicality_no_physiology`; exporting diagnostics does not depend on an
objective's physiological optimization weight.

For DREAMER, every state contains `amplitude` and `spectral_power` arrays shaped
`(14, 3)` (42 values each), plus `coherence` and signed
`debiased_wpli_squared` arrays shaped `(91, 3)` (273 each): 630 numeric
components per state. Values retain their diagnostic precision; undefined
values, including the 14 unavailable aperiodic exponents, are JSON `null`.
Band edges and zero-based channel-pair indices describe the axes. Callers can
pass `channel_names`, `band_names`, and `signal_unit` to `assess()`; these fields
remain null when unknown. Signals, full PSDs, and other large tensors are not
included.

For example, a suite runner can save the compact assessments with:

```python
assessments = {
    state: reference.assess(diagnostics, channel_names=channel_names,
                            band_names=band_names, signal_unit=signal_unit)
    for state, diagnostics in diagnostic_sets.items()
}
with open(attempt_dir / "physiology.json", "w") as handle:
    json.dump(assessments, handle, allow_nan=False)
```

The `CFX-update` branch also includes the bidirectional suite runner with
source-only 10th-percentile target-class typicality. See
[the coverage guide](../../../../docs/CFX_COVERAGE.md) for commands and
exclusion/failure accounting. Use this updated shared assessment code when
generating new archives. `diagnostic_components()` is also exported
for callers that write separate `physiology_<state>.json` files. Existing
summary-only archives cannot recover missing component values from their pass
fractions and need regeneration from checkpoints/data or historical arrays.

## Graded inputs

Graded mode accepts any of these archives:

1. `physiology.json` with saved `worst_family_exceedance` and
   `mean_family_exceedance`, or per-family `deviation.exceedance_mean` for every
   evaluable family. These are reported as saved, with an explicit notice that
   the reference bounds were not independently recomputed.
2. `physiology.json` with per-state `components` objects produced by `assess()`.
   The grader recomputes grades from these component values even when the
   assessment also contains saved graded scores.
3. `physiology_<which>.json` containing the component arrays `amplitude`,
   `spectral_power`, `aperiodic_exponent`, `coherence`, and
   `debiased_wpli_squared`. Undefined array entries can be JSON `null`.
4. Legacy `physiology_<which>.npz` component arrays.

Component archives also require the saved binary checks in `physiology.json`
and raw EEG/labels (`--eeg`/`--labels`, or the default DREAMER dataset paths).
The grader refits the source-only reference and requires all recomputed
in-range fractions to match the saved fractions. Embedded JSON values,
separate JSON component files, and legacy NPZ use the same bounds gate.
Separate component files take precedence when both forms are present.
The original float32 normalization and
inverse transform are preserved.

For graded tables, `--statistic worst` (default) reports the worst-family
exceedance; `--statistic mean` reports the mean-family exceedance. The original
units and definition remain: `max(z - 1, 0)`, where `z` is distance from the
source interval's center in half-interval widths.

Missing graded values are reported as unavailable and cause exit status 2.
A bounds mismatch causes status 1. Unavailable/mismatched attempts are counted
and excluded from reported quantiles. The grader never fills missing grades
with zero or infers them from binary fractions.

## CPU Slurm launcher

The six-task launcher runs both tasks and all three signal types:

```bash
REPORT=binary sbatch src/eegproc/model_explainability/slurm/run_grade_physiology_table.sh
```

`REPORT=graded` is the default and runs both worst/mean statistics. Binary
mode writes one report per array task. The launcher accepts JSON summaries
without requiring NPZ files, raw EEG, or checkpoints up front. Raw EEG is
loaded only when component arrays need rescoring. `GRADE_REPO` defaults to
the current project checkout and can select another checkout explicitly.

## Repaired baseline, real-trial VCSC tolerance, and amplitude/spectral term

Three opt-in options address what the VCSC diagnostic showed on the
1599318 / 65452590 ablation suites: decoder reconstructions R(z) score
thousands of times above real EEG on VCSC; when VCSC is in the objective
nearly all of the decoded change R(z') - R(z) (99.9%) is the optimizer
repairing that, not class-relevant change; counterfactuals end up more
typical than 95-100% of real trials because of the fixed 1e-8 feasibility
tolerance; and amplitude and spectral power, which VCSC cannot see, fail in
every arm. These additions are opt-in. The CFX-update defaults remain both
class directions and the source target-class 10th-percentile typicality cutoff.

`--include-repair-baseline`. For each eligible
trial, first optimize z with the physiological constraints and the
latent/decoded proximity terms but no target term, stopping at the first
feasible iterate. Its decoded output R(z_rep) is the baseline:

* explanation change: `delta_dec_explanation = RMSE(R(z'), R(z_rep))`
* repair change: `delta_dec_repair = RMSE(R(z_rep), R(z))`
* `band_power.json` adds `power_change_vs_repair`; `physiology.json` adds a `repair` set.

Every arm's `result.json` gets a `repair_baseline` block, `trial_metrics.csv`
gets the two columns, and population summaries report their quantiles plus
`repair_feasible_percent` and `repair_preserves_prediction_percent`. The
repair is not an objective; it lives in `trial_<id>/repair/` with its own
`result.json` and the one archived array, `repair_reconstruction.npz`
(`x_repair`), needed to measure the other arms and to resume. Report
explanation topographies from `power_change_vs_repair`, not `power_change`.

`--physiological-tolerance-quantile Q`. Replaces the fixed VCSC tolerance with
the Q-quantile of the same calibrated VCSC on the fold's source-subject real
trials, so feasible means "as plausible as real EEG". The tolerance and the
source distribution are saved in `calibration/vcsc_tolerance.json`. The
source trials are also the calibration set, so the quantile is in-sample and
slightly optimistic; held-out real-trial VCSC is archived per trial as
`vcsc_original_input`.

`--amplitude-spectral-weight W` and `--amplitude-spectral-margin M`. Adds
`W * mean relu(|v-m|/h - (1-M))^2` over the amplitude and spectral-power
components, where m and h are the midpoint and half-width of each source
interval used by the reported checks. The TensorFlow descriptors in
`counterfactuals/amplitude_spectral.py` reproduce `signal_diagnostics`
(tested to 1e-4) in physical units via each trial's normalization. With
W > 0, feasibility also requires each family to meet
`--physiology-required-fraction`, the reported pass rule. Band power mixes
oscillatory and aperiodic power, and the aperiodic exponent cannot be
estimated from the band-filtered decoder, so the term can be met by a
broadband shift; report that limitation. `target_latent` and
`typicality_no_physiology` drop this term along with VCSC.

Suggested ablation suite:

```bash
sbatch src/eegproc/model_explainability/slurm/run_cfx_coverage.sh \
  --include-repair-baseline --physiological-tolerance-quantile 0.5 \
  --amplitude-spectral-weight 1.0
```

Compare against the original suite on: amplitude/spectral in-range
fractions, coherence/wPLI in range, counterfactual VCSC relative to
`vcsc_original_input`, `delta_dec_explanation` vs `delta_dec`, and latent
target success.

## Temporal smoothing and phase-invariant distance

Implemented in `counterfactuals/temporal_phase.py`. All three options are
off by default and the defaults reproduce the earlier runs.

`--temporal-sparsity-weight A` and `--temporal-tv-weight B`. A fused lasso
(Tibshirani et al. 2005) on the latent edit `delta = (z' - z) / sigma`,
where sigma is each latent feature's standard deviation over the fold's
source trials (`calibration/latent_scale.json`). The edit is flattened over
windows x timesteps in time order and the objective adds
`A * mean|delta| + B * mean|delta_{t+1} - delta_t|`, with `|x|` smoothed
as `sqrt(x^2 + 1e-6) - 1e-3`. Sparsity alone favours edits that are zero
for most of the trial; TV alone favours piecewise-constant edits; together
they favour a few constant episodes. The flattening assumes contiguous
windows (overlap 0). `target_latent` drops these terms (it stays target +
latent MSE); `typicality_no_physiology` keeps them.

`--decoded-distance envelope`.
Replaces the sample-wise MSE decoded proximity with the mean squared
difference of log Hilbert envelopes (Hann-tapered per window, pooled over up
to 32 samples = 250 ms, +1e-3 before the log). A phase shift with unchanged
band power costs almost nothing; a power change does.

Reported for every run, whatever the options:

* `result.json` → `decoded_trials.<path>`: `original_reconstruction_env`,
  `counterfactual_to_original_env`, `decoded_change_env` (envelope versions
  of the three MSEs, in model-input units);
* `result.json` → `latent_edit`: `l1_mean`, `tv_mean`, `active_fraction`
  and `n_episodes` of the standardized edit (a time step is active when its
  mean |delta| exceeds 0.1 x the trial's maximum);
* `repair_baseline` adds `explanation_change_env` and `repair_change_env`;
* `seams.json` per arm: window-seam continuity of the original,
  reconstruction, counterfactual (and repair) in physical units. Per band,
  the median over boundaries and channels of the phase residual (radians;
  phase advance over 17 samples centred on each boundary minus the advance
  expected from nearby increments) and of the amplitude-jump ratio (step at
  the boundary / median nearby step). Compare against the `original` row:
  real band-limited EEG is not a pure sinusoid, so its residual is not 0.
  `physiology.json` is unchanged;
* `trial_metrics.csv`: `delta_env`, `e_rec_env` (RMS of the envelope
  distances), `delta_env_explanation`, `delta_env_repair`, and
  `latent_edit_l1/_tv/_active_fraction/_episodes`, with medians/IQRs in
  the population tables.

Suggested ablation on top of the repair suite:

```bash
sbatch src/eegproc/model_explainability/slurm/run_cfx_coverage.sh \
  --include-repair-baseline --physiological-tolerance-quantile 0.5 \
  --amplitude-spectral-weight 1.0 --temporal-sparsity-weight 0.003 \
  --temporal-tv-weight 0.003 --decoded-distance envelope
```

The weights are starting points, not tuned values; choose them on source
subjects. Scale matters: on earlier v15 DREAMER counterfactuals the
standardized edit had mean |delta| of 12-50 and mean |delta_{t+1} -
delta_t| of 16-65, against a raw latent MSE of 0.7-9 (weighted by 0.1). A
weight of 0.003 puts each smoothing term at roughly the latent term's
size; 0.1 would make it dominate and block the target. Sweep 0.001, 0.003
and 0.01. Compare `latent_edit_episodes`/`latent_edit_active_fraction`,
`delta_env_explanation`, seam residuals relative to `original`, and latent
target success against the run without these options.


These options regularize counterfactual generation, not AE training. All are
optional: the default four-objective study keeps its existing objective
settings. See [the bidirectional run protocol](../../../../docs/CFX_COVERAGE.md)
for the required task-specific environment and output layout. Numerical
examples above are development settings, not held-out-selected recommendations.
