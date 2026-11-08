# CoMTE external comparator

The paired comparator and complete run protocol are documented in
[`comte/README.md`](../src/eegproc/model_explainability/comte/README.md).

Run `python -m eegproc.model_explainability.comte.runner --help` with the
project's deep-learning dependencies installed. Supply the same frozen LOSO
manifest, complete held-out trials, preprocessing, target thresholds, and CFO
settings as the study being compared. Source subjects alone provide reference
examples, independent typicality calibration, and physiological bounds.

The primary comparison scores delivered **signals** through the complete frozen
encoder–classifier for both methods, including re-encoding CFO's decoded output.
It includes failure cases, independent spectral typicality, physiology/joint
outcomes, displacement, electrode sparsity, runtime and query counts, and
held-out-subject summaries. Existing latent-only ablation reports are unchanged.

The Slurm launcher is
[`run_comte_comparison.sh`](../src/eegproc/model_explainability/slurm/run_comte_comparison.sh).
It does not submit jobs automatically.
