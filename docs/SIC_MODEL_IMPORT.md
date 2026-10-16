# SIC-model branch import

SIC, SICModelv11, and SICModelv15 were imported from `origin/SIC-model`,
commit `393858533020e3b3853decec29fcf08a4f607682`, onto `valence-enhancement`.
The model directories include their source-training entry points, argument
parsers, documentation, Slurm launchers, and calibration configurations.
The v11/v15 calibration-only drivers, shared dataset loader, SIC result reader,
and SIC comparison report are also included.

## Use the restored entry points

Install the optional dependencies in the existing environment:

```bash
python -m pip install -e '.[deep-learning]'
python -m eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_train --help
python -m eegproc.deep_learning.joint_architectures.SICModelv15.sic_calibration_only_train --help
python -m eegproc.plotting.sic_report --help
```

The same source-training entry point exists under `SICModel` and `SICModelv11`.
Training still requires the preconverted EEG and label arrays. Calibration
configuration files contain historical checkpoint locations; set those paths
to the models from the intended experiment before running calibration.

## Compatibility changes

- Model imports use `domain_generalization.meta_learning`, retaining v2's
  reusable encoder, recurrent-classifier, and variational-classifier fixes.
- Source training uses the modular `cross_validation` package. The restored
  calibration-only drivers import worker helpers from their actual module,
  rather than relying on private names in the deprecated `cross_val` wrapper.
- The shared calibration runtime now supports subject-specific training seeds,
  deterministic execution, disjoint GPU groups, and one fresh process per
  source fold, as required by the SIC-model branch's full-run launchers.
- Slurm environment setup installs the `deep-learning` extra instead of the
  removed root `requirements.txt`. Existing environments remain reusable.
- The upstream v15 valence launcher contained a leftover merge marker and
  selected 12-shot calibration while only running 3-shot calibration. The
  marker is removed, the selection level is corrected to 3, and the comments
  and logging describe the actual nine-configuration grid and 12 source epochs.
- Three v11 launcher comments were reworded to avoid a macOS Bash heredoc
  parsing issue. Calibration-only launchers now provide their missing help
  function and honor subject-selection arguments; their default remains users
  0 and 1.

The six-hour oracle sweep in `src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/full_run_v15_valence_sweep.sh` is preserved:
16 configurations, one per array task, and one shared epoch selected by maximum
mean held-out-subject balanced accuracy. The imported full-run/calibration
scripts retain their separate training budgets and reporting protocol.

## Validation

CPU checks cover all three variants' source training, target calibration,
checkpoint saving/loading, and v15's parallel fold processing across more
subjects than worker slots. The existing oracle sweep, SIC reporting tests,
two-device numerical and deterministic checks, library cross-validation
regressions, and module-layering checks also pass. All imported Slurm scripts
pass shell syntax checks. Full-size CUDA training and Slurm submission have
not been run locally.
