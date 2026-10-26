"""Audit and prepare resumption of existing runs; dry run unless --apply.

No jobs are submitted by this command. Models/datasets are never deleted to
reduce retention; migration verifies copied bytes before removing home copies.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from storage import VARIANTS, default_work_root, ensure_work_stores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    work = args.work_root or default_work_root()
    stores = [ROOT / "experiments/sic_v16" / v / "runs" for v in VARIANTS]
    candidates = [p for store in stores for p in store.glob("*/manifest.json")]
    jobs = [str(json.loads(p.read_text()).get("slurm_job_id", "")) for p in candidates]
    jobs = [job for job in jobs if job.isdigit()]
    if args.apply and jobs:
        status = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"],
                                capture_output=True, text=True)
        if status.returncode:
            raise ValueError(f"Cannot confirm scheduler inactivity: {status.stderr}")
        active = [job for job in status.stdout.split() if job in jobs or job.split('_')[0] in jobs]
        if active:
            raise ValueError(f"Run jobs are still active; do not migrate while writing: {active}")
    print(f"Shared work root: {work}")
    if not args.apply:
        for path in candidates:
            manifest = json.loads(path.read_text())
            print(manifest["status"], path.parent)
        print("Dry run only. Add --apply to migrate, audit and generate resume scripts.")
        return
    ensure_work_stores(ROOT, work, migrate=True)
    # All metadata and scripts now live on shared storage, including completed runs.
    from eegproc.deep_learning.joint_architectures.SICModelv16.recovery import checkpoint_progress
    for variant in VARIANTS:
        for manifest_path in (ROOT / "experiments/sic_v16" / variant / "runs").glob("*/manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            if manifest["status"] == "completed":
                print(f"KEEP completed: {variant}/{manifest['run_id']}")
                continue
            run = manifest_path.parent.resolve()
            lock = run / ".run.lock"
            if lock.exists():
                # Scheduler inactivity was verified above. Preserve stale lock
                # metadata rather than deleting it.
                lock.rename(run / f"previous-lock-{time.time_ns()}.json")
            task = run / "artifacts/task.json"
            if not task.exists():
                print(f"NO CHECKPOINT TASK: {run}; inspect rather than blindly restart.")
                continue
            spec = json.loads(task.read_text())
            progress = [checkpoint_progress(run / "artifacts", subject, spec["source_epochs"],
                        verify_latest=True) for subject in spec["target_subjects"]]
            audit = run / "recovery-audit.json"
            audit.write_text(json.dumps(progress, indent=2) + "\n")
            if any(p["orphan_checkpoints"] for p in progress):
                print(f"Unpaired checkpoint requires inspection: {audit}")
                continue
            q = shlex.quote
            script = run / "resume.slurm"
            # Slurm directives require a single unambiguous path token.
            if any(c.isspace() for c in str(run)):
                raise ValueError("Run paths containing whitespace are unsupported by these Slurm directives.")
            script.write_text(f'''#!/bin/bash
#SBATCH --job-name=sicv16_resume_{variant}
#SBATCH --output={run}/resume-%j.out
#SBATCH --error={run}/resume-%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
export PROJECT_DIR={q(str(ROOT))}
source "$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16/SLURM_scripts/environment.sh"
exec python "$PROJECT_DIR/experiments/sic_v16/train.py" --resume \\
    --config {q(str(run / 'config.json'))} --output-dir {q(str(run))} \\
    --data-dir "${{DATA_DIR:-$PROJECT_DIR/datasets}}"
''', encoding="utf-8", newline="\n")
            print(f"{variant}: {sum(p['complete'] for p in progress)} finished subjects; "
                  f"{sum(p['completed_epochs'] for p in progress)} saved epochs")
            print(f"Submit: sbatch {q(str(script))}")


if __name__ == "__main__":
    main()
