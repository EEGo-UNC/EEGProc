# DREAMER joint-v2 SLURM scripts

Each encoder folder contains a full valence run, a full arousal run, and an
arousal smoke test limited to four LOSO folds.

Run from the EEGProc repository root:

```bash
sbatch SLURM_scripts/CNN1D/smoke_test_joint_v2_dreamer_arousal_cnn1d.sh
sbatch SLURM_scripts/CNN2D/smoke_test_joint_v2_dreamer_arousal_cnn2d.sh
sbatch SLURM_scripts/GCN/smoke_test_joint_v2_dreamer_arousal_gcn.sh
```

After all three smoke tests pass, submit the six full runs in the same way.
CNN2D and GCN explicitly reshape each raw DREAMER timestep as 14 channels x 1 band.
