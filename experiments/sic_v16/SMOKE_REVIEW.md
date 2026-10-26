# Real-data smoke review: Longleaf jobs 6632324, 6632366, 6632367, 6632370

Source: `sicv16-diagnostics-20261025-220014.tar.gz`. All four tracked runs
completed two epochs for target subjects 0 and 1. All four data/bridge checks
passed in every job. The archive also contains earlier failed array logs;
those are historical, not failures of the new tracked runs.

Verified from JSON artifacts: exact configuration fingerprints, disjoint
training/validation/test subject sets, four completed subject/epoch records
per variant, finite training metrics, and equal-weight per-band reconstruction
MSE. With reconstruction enabled, total loss equals classification loss plus
0.1 times reconstruction loss within numerical reporting precision.

| Variant | Controller time | Peak TF GPU allocation | Epoch 2 mean | Estimated 50 epochs x 23 subjects |
| --- | ---: | ---: | ---: | ---: |
| CNN | 4.15 min | 3.58 GiB | 36.2 s | 11.6 h |
| CNN + reconstruction | 4.04 min | 3.58 GiB | 36.3 s | 11.7 h |
| CNN + Transformer | 3.77 min | 3.58 GiB | 35.7 s | 11.5 h |
| CNN + Transformer + reconstruction | 5.04 min | 3.58 GiB | 48.4 s | 15.6 h |

Estimates use the observed first-epoch and second-epoch times and assume later
epochs resemble epoch 2. They exclude additional fold process startup and
data preparation; other subjects and cluster contention may differ. They
are estimates, not completion guarantees. Measurements include L40 and L40S
nodes; these smoke timings are not a controlled hardware speed comparison.
The reported GPU figure is TensorFlow allocator peak, not total GPU usage.
The supplied full jobs request 24 hours with an 85,000-second soft budget.

## Learning results

Every saved epoch of every variant predicted class 1 for all 18 trials of
each held-out subject at the fixed 0.5 threshold. Thus validation-selected and
oracle balanced accuracy are both 0.5 for all variants; ordinary mean accuracy
is 0.5556 because the two subjects have different class proportions. This is
a functioning pipeline with uninformative thresholded predictions after two
epochs, not evidence of useful generalization or a winning architecture.

Training classification losses decrease. Reconstruction also learns: for CNN
alone its mean MSE falls from 1.2175 to 0.9453 (subject 0) and 1.1699 to 0.9699
(subject 1). With Transformers those values are 1.2310 to 0.9264 and 1.2409 to
0.9683. That does not establish a classification benefit from reconstruction.

No runtime/model correction is indicated by this archive. Added reporting
explicitly flags single-class predictions and retains probability ranges.
Architecture, class weights, thresholds and selection rules are unchanged.
Real-data checkpoint files were excluded from the archive; their existence
was checked by the on-cluster summarizer, while reload equivalence was tested
on synthetic models in the earlier GPU preflight.

## Next step

The smoke gate has passed. Use the existing tracker to plan all four baseline
runs (50 epochs, all 23 subjects), then submit the four generated scripts:

```bash
python experiments/sic_v16/tracker.py --variant all status
python experiments/sic_v16/tracker.py --variant all plan --target valence
```

Do not rerun the smoke jobs. These longer matched baselines are needed to see
whether the all-class-1 behavior persists. Do not change decision thresholds
using the held-out labels. If prediction collapse persists through full
training, investigate source-validation learning curves and representation
quality before interpreting architecture ablations or expanding the sweep.

All epochs' checkpoints are retained: the baseline comparison writes 4,600
models. Check available storage and the smoke `.keras` sizes before submission.
After completion, the usual tracker status/plot/sync workflow applies; no new
hand-written Slurm launcher is required. Upload a fresh diagnostics archive
when requesting review, retaining the model files on Longleaf.
