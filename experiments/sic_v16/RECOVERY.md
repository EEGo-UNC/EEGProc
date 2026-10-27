# Quota recovery and continuation of the October 26 baselines

The controller retained one full `.keras` model (including optimizer variables)
for every epoch under the home directory. That output policy filled home quota.
New planning and legacy launchers now direct run storage to shared `/work`.
All epoch models are preserved so common-epoch oracle selection stays intact.
No old model is pruned or replaced by this recovery.

## Recorded progress

Source archive: `sicv16-diagnostics-20261026-142212.tar.gz` (4,048 files).
The archive includes epoch JSONs but deliberately excludes model files.

| Variant | Fully finished subjects | Partial subject | Last recorded epoch | Recorded epochs |
| --- | --- | --- | ---: | ---: |
| CNN | 0–19 | 20 | 30/50 | 1,030 |
| CNN + reconstruction | 0–22 | none | complete | 1,150 |
| CNN + Transformer | 0–13 | 14 | 47/50 | 747 |
| CNN + Transformer + reconstruction | 0–13 | 14 | 43/50 | 743 |

The three failed workers hit quota while writing the next temporary model.
Their controllers then also hit quota when writing the summary, which obscured
the worker failure and left stale `preparing_data` execution status. Run recovery
audits actual per-epoch records/checkpoints instead of trusting that stale status.
There are 2,520 completed epoch records across the three failed variants, leaving
930 of their 3,450 planned epochs. The completed variant is never retrained.

The completed CNN + reconstruction result has mean validation-selected accuracy
0.61594, balanced accuracy 0.5, and ROC AUC 0.48644. All 23 selected checkpoints
predict one class. Preserve the matched baselines for diagnosis, but do not
expand the hyperparameter sweep on the assumption that accuracy means useful
discrimination. Other variants are incomplete and cannot be ranked yet.

## 1. Free enough home quota to pull the recovery code

Ensure the old training jobs are no longer active (`squeue -u "$USER"`). Move
the **completed** variant's run store to shared work and leave a link at the old
location. This moves all files, including model checkpoints, rather than deleting
training results. Do not run this while jobs are writing that variant's outputs.

```bash
src="$HOME/EEGProc/experiments/sic_v16/cnn_reconstruction/runs"
dst="/work/users/p/m/pmucharl/sic-v16/cnn_reconstruction/runs"
if [ -L "$src" ]; then
    echo "Already linked: $(readlink "$src")"
elif [ -d "$src" ] && [ ! -e "$dst" ] && [ ! -L "$dst" ]; then
    mkdir -p "$(dirname "$dst")" &&
    mv -- "$src" "$dst" && ln -s "$dst" "$src"
else
    echo "STOP: source/destination differs from the expected layout; do not overwrite it."
fi
```

GNU `mv` copies across filesystems before removing successfully copied source
files. If it reports any error, stop and report it rather than repeating a
different move or deleting either tree. The paths are deliberately limited to
this one run store; datasets, environment and source code remain in place.

Then update code and environment:

```bash
cd "$HOME/EEGProc"
git switch SICv16-Pranav
git pull --ff-only origin SICv16-Pranav
module load python/3.12.4
source venv312/bin/activate
```

## 2. Audit and migrate the remaining stores

First preview (no migration or job submission):

```bash
python experiments/sic_v16/recover.py --work-root /work/users/p/m/pmucharl/sic-v16
```

Apply the migration and checkpoint audit:

```bash
python experiments/sic_v16/recover.py \
  --work-root /work/users/p/m/pmucharl/sic-v16 --apply
```

The tool checks the scheduler for active recorded jobs. Every remaining home
file is copied to work, SHA-256 verified, rechecked against the source, and only
then removed from home. Directory links keep the original tracker paths valid.
The operation can continue after interruption; conflicting destination files
stop migration with originals preserved. All logs, prepared arrays and checkpoints
now live on shared work, including the already completed variant.

For each failed run, `recovery-audit.json` reports the checkpoint inventory.
All recorded checkpoint files must exist and be structurally valid ZIPs; the
latest checkpoint of each subject also receives a full CRC check. Temporary
models are preserved and ignored. An unpaired final model or missing/corrupt
recorded checkpoint stops automatic resumption for inspection; it is not silently
discarded. The diagnostic archive alone cannot prove these model checks pass.

## 3. Submit the three generated resume scripts

The tool prints exact `sbatch .../resume.slurm` commands on work storage. Submit
those three commands, **not** new tracker plans and **not** the old submit scripts.
The completed CNN + reconstruction run is skipped.

Each job first checks checkpoint/optimizer serialization on small synthetic GPU
models, then resumes the actual run. Keras restoration must preserve optimizer
moments and iteration count; failure stops the job rather than resetting AdamW.
The existing source-validation subject split, model settings, data metadata,
epoch budget and threshold are checked/preserved. Finished subjects are skipped;
partial subjects continue from their last paired, verified model/epoch record.

The interrupted epoch's unsaved updates cannot be recovered. Resumption keeps
saved weights and optimizer state, but future random shuffle/dropout draws need
not match an uninterrupted run bit for bit. Resume boundaries and attempts are
logged. Old logs are appended, never truncated, and old task/execution metadata
is archived. In-progress duplicate submissions are blocked by a run lock.

Monitor through the same tracker paths:

```bash
python experiments/sic_v16/tracker.py --scope all23 --variant all status
```

While running, inspect `resume-JOBID.out/.err` and `artifacts/subject_XX.log` in
the run store. Afterward, create an archive with the work-aware helper:

```bash
python experiments/sic_v16/diagnostics.py
```

This explicitly enters each linked run store, avoiding directory-symlink
traversal issues, and excludes `.keras`/`.npy` files. Shared work
is temporary cluster storage; archive required final artifacts separately.

## Future runs

On Longleaf, the tracker ensures each variant's `runs` directory points into
shared work before planning. Existing nonempty home stores require explicit
recovery migration. Training refuses a home output path when Longleaf work
storage is present. Set `SIC_V16_WORK_ROOT` to override the shared work root.
The legacy array launchers also use work storage by default.

All epochs are retained to preserve oracle selection and auditing. Moving them
off home fixes this quota failure without changing the scientific protocol.
Storage-aware pruning can be added separately after final checkpoint choices
are established; it is deliberately not part of lossless recovery.
