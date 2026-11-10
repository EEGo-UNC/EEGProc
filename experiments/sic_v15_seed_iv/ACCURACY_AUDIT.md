# SEED-IV accuracy audit — 2026-11-09

Read-only audit of the isolated Longleaf SICv15 experiment. No model settings,
running jobs, preprocessed data, or SICv16 files were changed by this audit.

## Verified data contracts

Independently retrieved the Kaggle dataset ReadMe in memory. Class IDs are
0 neutral, 1 sad, 2 fear, 3 happy. All 37,575 stored window labels agree with
its three session-specific 24-trial label vectors. Checked headers of all 45
raw MATLAB files: all 1,080 trials have 62 channels, and stored window counts
equal floor(raw samples / 800). Trial counts are 270 per class.

Independently recomputed CAR, fourth-order zero-phase SOS filtering, constant
detrending, channel-major/band-minor layout, and per-window global RMS from raw
trial 1 for subject/session (1,1), (1,2), (1,3), (15,3), without calling the
adapter preprocessor. Maximum difference from the stored first windows was 0.
This verifies implementation consistency, not the optimality of CAR or RMS.

Official documentation: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
The published feature recipe uses 200 Hz, four-second windows, and five bands
1–4, 4–8, 8–14, 14–31, 31–50 Hz, then PSD/DE features. Our model consumes
band-filtered waveforms, not PSD/DE. That is a representation difference.

All five completed models have four outputs. Saved probabilities sum to one;
saved predictions equal their argmax. All 45 saved calibration splits have
disjoint calibration/evaluation movie trial IDs covering 72 trials. Subject
identity spans all three sessions. Source graph fitting excludes validation
and target subjects in the cross-validation implementation.

## Concrete training findings

1. Source class weights are silently discarded. The runner sets
   `source_use_class_weight=True`, but model config omits `use_class_weight`.
   The builder defaults that flag to false; `SICBaseMixin.fit` then replaces
   `class_weight` with None. Both flags must be true to enable weighting.
   Windows/class/subject: [678, 683, 615, 529]; majority accuracy is 27.27%.
   This is modest imbalance and cannot alone establish the cause of weak scores.

2. All five completed folds stopped after epoch 4 and restored epoch 1 using
   total validation loss. Total loss includes VC and reconstruction terms,
   not just the requested emotion-recognition metric. For subject 1 source
   balanced accuracy rose 27.02% -> 37.19%, while source-validation balanced
   accuracy was 28.40%, 27.42%, 29.02%, 28.98%. For subject 4 it rose from
   26.62% to 29.47% on validation, but epoch 1 was restored. The existing
   `best_epoch_metric='balanced_accuracy'` option selects
   `val_window_balanced_accuracy` using source-only validation subjects.
   A different monitor is an experiment, not a guaranteed improvement.

3. The run combines focal loss, VC auxiliary terms, subject adversity, and
   reconstruction using inherited DREAMER settings (VC beta .6, scale 64,
   reconstruction weight .6, subject-loss weight .2). These are not tuned for
   four-class SEED-IV. Rising source accuracy with flat validation accuracy
   indicates poor generalization. Simply adding epochs is not well supported.

4. This baseline uses ERM window classification. DREAMER launchers used
   full-trial MLDG. Those are different training protocols even with v15's
   encoder/decoder architecture retained. Current full-trial grouping rejects
   unequal trial window counts; restoring that recipe requires deliberate
   variable-length handling, not changing a dataset argument.

5. Calibration underperforms paired zero-shot balanced accuracy by about
   1.61, 2.16, and 0.57 percentage points at 4, 8, and 12 shots, respectively,
   averaged across the first five completed subjects. It unfreezes the recurrent
   classifier plus VC head for five epochs with VC regularization active.

## Class-level results (five completed subjects, zero-shot)

Pooled recall: neutral 35.13%, sad 41.79%, fear 21.07%, happy 20.38%.
Subject 1 predicted happy in only 21/2,505 windows (happy recall 2.08%). Other
subjects predict all classes, so this is not universal one-class collapse.
Mean subject balanced accuracy is 29.59%; four-class chance is 25%.

## Recommended controlled follow-ups

1. Overfit a small class-balanced source-only subset to test optimization, then
   enable both class-weight flags and select checkpoints by source-validation
   balanced accuracy. Increase patience modestly; do not select on target logs.
2. Run one-factor ablations: reconstruction .6 -> .1/0; subject adversity off;
   VC beta/lambda zero while retaining the VC head; focal gamma 1 -> 0. Check
   gradient magnitudes before claiming one loss dominates. Then combine only
   changes supported by source-only validation.
3. Derive log-bandpower/DE directly from the downloaded raw EEG for a cheap
   source-validation baseline. Separately ablate per-window RMS versus a scaler
   fitted on training subjects, and CAR on/off. Preserve session/movie boundaries.
4. Calibrate head-only with lower learning rate, fewer epochs, and VC auxiliary
   regularizers disabled, selecting the protocol within source subjects acting
   as pseudo-targets. Compare on paired non-calibration movie trials.
5. Evaluate later MLDG and temporal context experiments with explicit grouping
   or masking. Do not silently truncate full trials or treat sessions as subjects.

Existing target results have informed this audit; they must not be presented
as untouched validation evidence for follow-up choices. Use nested source
validation or a predeclared held-out participant set for honest comparisons.
