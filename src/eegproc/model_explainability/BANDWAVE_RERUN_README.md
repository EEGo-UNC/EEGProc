# Bandwave continuity figure rerun

Regenerates "Comparison of Reconstructed Bandwave Continuity" (`waveform_zoom.png`, `connectivity_compare.png`) via `wpli_compare.py`.

Requires `pywt` installed (the `eegproc` package import needs it). The figures intentionally differ from the originally submitted ones: all four signals share one per-feature z-scale, and the stale hardcoded "real DREAMER ref" line is gone.

## Single counterfactual.npz

```bash
python -m eegproc.model_explainability.wpli_compare \
    --cf-npz runs/counterfactuals/<run>/counterfactual.npz \
    --dreamer-eeg PATH/TO/dreamer_eeg.npy \
    --dreamer-labels PATH/TO/dreamer_labels.npy \
    --task arousal \
    --output-dir runs/bandwave_rerun/<run>
```

Outputs `waveform_zoom.png` and `connectivity_compare.png` in `--output-dir`.

## Batch (all trials under a run root)

```bash
python -m eegproc.model_explainability.wpli_compare \
    --batch runs/counterfactuals/<run_root> \
    --dreamer-eeg PATH/TO/dreamer_eeg.npy \
    --dreamer-labels PATH/TO/dreamer_labels.npy \
    --task arousal \
    --output-dir runs/bandwave_rerun/<run_root>
```

Finds every `**/counterfactual.npz` under the root (override with `--batch-glob`) and writes `<trial-folder>_waveform_zoom.png` and `<trial-folder>_connectivity_compare.png` per trial. A failing trial is reported and skipped; the exit code is non-zero if any failed.
