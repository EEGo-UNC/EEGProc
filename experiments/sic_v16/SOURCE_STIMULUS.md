# Source correspondence and trial-ID baseline

From the Longleaf repository root:

```bash
sbatch experiments/sic_v16/source_stimulus.slurm
```

By default the source check looks for datasets/dreamer_joined.csv. To use an
original export elsewhere:

```bash
SOURCE_CSV=/absolute/path/to/dreamer_joined.csv sbatch experiments/sic_v16/source_stimulus.slurm
```

The audit compares all 414 labels and stimulus-only central 7680-sample crops
against the arrays, preserving sorted source subject/trial IDs. Raw 14-channel
arrays are compared with unfiltered source crops; 42-feature arrays are
compared with full-stimulus filtering followed by cropping. It reports exact
source mappings, crop boundaries, mismatches and the source CSV hash.

Missing CSV files yield source_unavailable while the baseline still runs.
A numerical mismatch yields mismatch and remains in the diagnostics. Neither
status permits a claim that source alignment has been established. Matching
the CSV does not establish correspondence to the original recording/MAT files,
film identity, or actual acquisition sampling rate.

The trial-index-only baseline uses one-hot trial IDs with the same 23 outer
subject folds, 22 inner subject folds, C grid and threshold selection as the
EEG baseline. No EEG features enter the classifier. Trial indices are proxies
for common stimuli until film mapping is verified. Trial-ID performance can
reflect shared stimulus or trial-order label structure; it cannot establish
EEG-specific information or generalization to unseen stimuli.

Send source-stimulus-diagnostics.tar.gz from the path printed in the log.
