# Trace saved EEG against source recordings

Submit from the repository root:

```bash
SOURCE_CSV=/absolute/path/to/dreamer_joined.csv sbatch experiments/sic_v16/trace_source_mismatch.slurm
```

The default CSV is datasets/dreamer_joined.csv. Unlike the earlier baseline,
this diagnostic requires the CSV. It targets the observed raw 14-channel
arrays, and defaults to zero-based array pairs 0:0, 0:1 and 1:0. Each pair is
mapped to sorted source subject and trial IDs exactly as in the converter.

For each sufficiently long stimulus or baseline segment, it scans crop starts
at 128-sample intervals plus start/center/end, then refines at sample resolution
near the three strongest coarse crops. It compares raw signals, linear
detrending, notch filtering, 4–30 Hz filtering, the canonical three-band sum,
and common-average referencing. It also checks crop-before-filter variants
at candidate positions. Matching uses temporal correlation and one-to-one
channel assignment; both declared and inferred orders are evaluated.

The report includes fitted global and per-channel scales/offsets, channel
correlations and errors, channel mappings, crop offsets and source sample
indices. It retains top candidates and the best result under unscaled,
global-affine and per-channel-affine error. A good fitted correlation is not
proof of provenance. Source trial permutations, alternate source files,
unlisted filters and transformations remain possible. The default crop scan
is coarse with local refinement, rather than exhaustive.

For custom pairs or exhaustive crop starts, invoke the Python script inside
the established cluster environment with --pairs 0:0,0:1 and --stride 1.

Send source-mismatch-trace-diagnostics.tar.gz from the path printed in the log.
