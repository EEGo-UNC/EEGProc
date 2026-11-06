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

`--which` accepts `counterfactual`, `original`, or `reconstruction`.

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

The `valence-enhancement` branch contains the diagnostics and grading tools;
the suite runner remains on `SIC-model`. Use this updated shared assessment
code when generating new archives. `diagnostic_components()` is also exported
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
