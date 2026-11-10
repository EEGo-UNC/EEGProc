# Trial identity and EEG comparison on the verified rebuilt dataset

```bash
sbatch experiments/sic_v16/compare_trial_eeg.slurm
```

The default discovers the most recently completed crop-rebuild-comparison job
under SIC_V16_WORK_ROOT. It selects centred_full_filter and verifies both
array hashes against crop_verification.json. The chosen absolute path and
hashes are printed and saved. To select a specific rebuild explicitly:

```bash
REBUILD_DIR=/absolute/path/to/crop-rebuild-comparison/job-JOBID sbatch experiments/sic_v16/compare_trial_eeg.slurm
```

The 24-hour job computes RMS log-power features once, then evaluates trial ID
alone (18 features), EEG alone (42), and their concatenation (60). All arms
use identical 23 outer/22 inner subject folds, the same C and threshold grids,
training-only scaling, balanced logistic regression and final refitting on
22 non-test subjects. Each arm selects its own C and threshold using only
inner validation predictions. Output includes fixed-0.5 and selected-threshold
scores, per-subject predictions, provenance and combined-minus-trial-ID mean
differences. Fold and label alignment is checked before reporting completion.

This evaluates added EEG information under shared trial identities, not
unseen-stimulus generalization. Trial identity remains a stimulus proxy until
source film mappings are independently verified. Original arrays are preserved.

Send trial-eeg-comparison-diagnostics.tar.gz from the printed path.
