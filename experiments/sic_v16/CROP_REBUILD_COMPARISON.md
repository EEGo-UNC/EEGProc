# Verify crop rules and compare rebuilt datasets

From the Longleaf repository root:

```bash
SOURCE_CSV=/absolute/path/to/dreamer_joined.csv sbatch experiments/sic_v16/verify_rebuild_crops.slurm
```

The source CSV is mandatory; its default location is datasets/dreamer_joined.csv.
The job requests 24 hours and creates a new work directory for its job ID.
Original source files and datasets are preserved.

All 414 raw 14-channel saved trials are checked for exact correspondence with
stimulus start/centre/end crops and the centre of baseline+stimulus. When these
rules fail, additional raw-stimulus candidates are checked using channel-0
correlation. Unmatched trials remain unresolved; the fallback is not an
exhaustive proof of absence. Exact baseline+stimulus matches report whether
the actual retained window contains any baseline samples.

Rebuilt arrays are stored separately under centred_raw and centred_full_filter.
Both use the central 7680 samples of stimulus and matching [valence, arousal]
ratings. Invalid source labels or discontinuous stimulus samples halt the job.
The provenance report records source IDs, baseline/stimulus lengths, crop
boundaries and file hashes. Sampling rate remains assumed 128 Hz.

Three full nested log-power evaluations run sequentially with identical seeds,
folds and tuning: original arrays; centred raw arrays filtered by the existing
loader after cropping; and full-stimulus filtered 42-feature arrays cropped
after filtering. This separates crop effects from filtering-boundary effects.
The dataset files remain on work storage; the small diagnostics archive includes
provenance and all baseline results, excluding EEG arrays and derived caches.

Send crop-rebuild-comparison-diagnostics.tar.gz from the printed path. If the
job fails, send the job .out/.err logs and crop_verification.json if available.
