# Lower-learning-rate diagnostics on the same full fold

```bash
REBUILD_DIR=/work/users/p/m/pmucharl/sic-v16/crop-rebuild-comparison/job-8059671 bash experiments/sic_v16/submit_learning_rate_diagnostics.sh
```

Two runs: learning rate 0.0001 and 0.00001. Everything else matches the earlier
plain-CNN reference: seed 42, target subject 0 excluded, twenty training subjects,
two validation subjects, dropout 0.1, ReLU, no class weights, batch size one,
zero weight decay and the verified full-stimulus-filtered dataset.

Both run 500 epochs without early stopping, so slower training has a longer
observation window. Override via SWEEP_EPOCHS at submission. Compare the first
200 epochs with the earlier reference, and later epochs separately; different
learning rates do not imply equivalent optimization progress at equal epochs.
Each job requests 24 hours. No automatic resume is implemented.

All training/validation recordings are evaluated each epoch for prediction
collapse, BA, cross-entropy, classifier positive preactivations and probability
variation. Eight fixed training probes (two classes from each of four source
subjects) additionally measure representation activity before training and
every epoch at each band encoder output, concatenated encoder sequence,
pooled encoder, classifier preactivation, embedding and logits. Reported
between-trial variation uses both channel means and RMS, so a change in
waveform shape need not be represented fully by these summary statistics.

For every trainable layer, record actual weight differences across the first
20 optimizer steps, steps 50/100/200/500 and every 1000 steps thereafter.
Also record each epoch's net weight change. Relative change is update norm /
starting weight norm; zero starting norm reports null rather than a misleading
large ratio. Epoch net change is not the sum of all optimizer step magnitudes.
The probes are inference-only and never update weights. Test subject remains
untouched. Validation chooses best.keras; last.keras preserves final state.

The dependent summary emits learning-rate-diagnostics.tar.gz containing
provenance, configurations, per-epoch metrics, activity and sampled updates.
Checkpoints and large feature arrays remain on work storage. Submission prints
job IDs and output directory. If a task fails, preserve its logs and collect
partial diagnostics using the existing collect mode.
