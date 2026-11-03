# Archived physiology grading

The physiological diagnostics and graded deviation from
`graded-physiological-checks` are available on `valence-enhancement`.
The grading tools use the existing NumPy dataset preparation and do not
import TensorFlow, load checkpoints, or require the removed SIC adapter.
The original float32 normalization and inverse transform are preserved.

From the repository root, grade one archived attempt:

```bash
python src/eegproc/model_explainability/typicality/grade_physiology.py \
  --attempt runs/path/to/subject_0/trial_16/typicality/attempt_0001 \
  --held-out 0 --task arousal
```

Grade a suite, aggregating separately for each objective:

```bash
python src/eegproc/model_explainability/typicality/grade_physiology_table.py \
  --suite runs/path/to/suite --task arousal --statistic worst
```

Both tools accept `--which counterfactual`, `original`, or `reconstruction`,
and optional `--eeg` and `--labels` paths. They read `physiology_<which>.npz`
and `physiology.json` from each archived attempt, reconstruct the held-out
user's source-only reference, and require its fractions to match the archive.
Mismatch exits nonzero; failed suite attempts are excluded from quantiles.
`--statistic mean` reports the alternative mean-family exceedance.

The six-task CPU Slurm launcher runs all three signal types for both tasks
and prints both statistics. It uses the current project checkout by default;
`GRADE_REPO` can select another checkout explicitly.

```bash
sbatch src/eegproc/model_explainability/slurm/run_grade_physiology_table.sh
```

Graded exceedance is `max(z - 1, 0)`, where `z` is the distance from the source
interval's center in half-interval widths. The binary pass checks are
unchanged. Undefined and zero-width intervals remain unscorable; an aperiodic
exponent is unavailable for the band-filtered decoder.
