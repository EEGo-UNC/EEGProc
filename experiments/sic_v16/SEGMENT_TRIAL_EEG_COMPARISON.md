# Six-segment trial identity and EEG comparison

From the Longleaf repository root:

```bash
sbatch experiments/sic_v16/compare_segment_trial_eeg.slurm
```

Uses the newest completed verified crop rebuild by default. To pin the known
rebuild, set REBUILD_DIR to its crop-rebuild-comparison/job-JOBID directory.
Input is centred_full_filter; source dataset hashes are verified.

Each 60-second trial is represented by six ordered nonoverlapping 10-second
segments. Within each segment, mean natural log power is computed across ten
one-second globally RMS-normalized windows, separately for 14 channels and
three bands. Features are segment-major, channel-major, then band-minor.
All segments remain together as one trial: no window or segment splitting.

Arms: trial identity alone (18 features); segment EEG alone (252); concatenated
identity and EEG (270). The existing 23 outer and 22 inner whole-subject folds,
C grid, threshold grid, training-only scaling and refitting are identical.
Each arm tunes independently. Both selected and fixed-0.5 thresholds are
reported. Fold alignment is checked. The original one-segment launcher and
results remain available for comparison.

This tests incremental predictive value under shared trial identities, not
unseen-stimulus generalization. The same regularization strength applies to
both feature groups in the combined model. Larger feature count can increase
overfitting; this experiment does not separately regularize those groups.

Send segment-trial-eeg-comparison-diagnostics.tar.gz from the printed path.
