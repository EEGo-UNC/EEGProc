"""Plan, import, and summarize config-driven DREAMER experiments.

This module intentionally uses only the Python standard library so planning and
syncing work on a laptop without TensorFlow installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN = HERE / "campaign.json"
DEFAULT_RUNS = HERE / "runs"
RESULT_FILE = "result.json"
MANIFEST_FILE = "manifest.json"
CONFIG_FILE = "config.json"
VALID_TARGETS = ("valence", "arousal")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _result_metric(result: dict[str, Any], metric: str) -> tuple[float, float]:
    if metric.startswith("trial_"):
        level, name = "trial", metric.removeprefix("trial_")
    elif metric.startswith("window_"):
        level, name = "window", metric.removeprefix("window_")
    else:
        level, name = "trial", metric
    best_index = int(result.get("best_config_index", 0))
    configs = result.get("config_results", [])
    if not configs or best_index >= len(configs):
        raise ValueError("result.json has no selected configuration metrics.")
    selected = configs[best_index]
    mean = float(selected[f"{level}_mean_scores"][name])
    std = float(selected[f"{level}_std_scores"][name])
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError(f"Non-finite {metric} in result.json.")
    return mean, std


def scan_runs(runs_dir: Path, campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric = campaign["objective"]["metric"]
    for manifest_path in sorted(runs_dir.glob(f"*/{MANIFEST_FILE}")):
        run_dir = manifest_path.parent
        try:
            manifest = load_json(manifest_path)
            config = load_json(run_dir / CONFIG_FILE)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rows.append({"run_id": run_dir.name, "status": "invalid", "error": str(exc)})
            continue
        row = {
            "run_id": manifest.get("run_id", run_dir.name),
            "target": config.get("target", "?"),
            "stage": manifest.get("stage", "?"),
            "status": manifest.get("status", "planned"),
            "fingerprint": fingerprint(config),
            "config": config,
            "run_dir": run_dir,
            "job_id": manifest.get("slurm_job_id", ""),
            "created_at": manifest.get("created_at", ""),
        }
        result_path = run_dir / RESULT_FILE
        if result_path.exists():
            try:
                result = load_json(result_path)
                row["mean"], row["std"] = _result_metric(result, metric)
                row["status"] = "completed"
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                row["status"] = "invalid"
                row["error"] = str(exc)
        rows.append(row)
    return rows


def score(row: dict[str, Any], campaign: dict[str, Any]) -> float:
    objective = campaign["objective"]
    direction = 1.0 if objective.get("maximize", True) else -1.0
    penalty = float(objective.get("uncertainty_penalty", 0.0))
    return direction * float(row["mean"]) - penalty * float(row["std"])


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def suggest(campaign: dict[str, Any], rows: list[dict[str, Any]], target: str) -> tuple[str, dict[str, Any], str]:
    defaults = dict(campaign["defaults"])
    defaults["target"] = target
    relevant = [row for row in rows if row.get("target") == target]
    occupied = {_config_key(row["config"]) for row in relevant if "config" in row}
    completed = [row for row in relevant if row.get("status") == "completed"]
    active = [row for row in relevant if row.get("status") in {"planned", "running"}]
    if active:
        raise RuntimeError(
            f"Run {active[0].get('run_id', '<unknown>')} is already planned or running "
            f"for {target}. Sync its output before planning another."
        )

    smoke = {**defaults, **campaign.get("smoke_overrides", {})}
    if _config_key(smoke) not in occupied:
        return "smoke", smoke, "Validate the complete data/model/LOSO path cheaply before a full run."

    smoke_done = any(
        row.get("status") == "completed" and row.get("stage") == "smoke"
        for row in relevant
    )
    if not smoke_done:
        raise RuntimeError(
            f"A {target} smoke run is already planned or running. Sync its output before planning another."
        )

    if _config_key(defaults) not in occupied:
        return "full", defaults, "Establish the full 23-subject baseline after the smoke test passed."

    full_completed = [row for row in completed if row.get("stage") != "smoke"]
    if not full_completed:
        raise RuntimeError(
            f"A full {target} run is already planned or running. Sync its output before planning another."
        )
    incumbent = max(full_completed, key=lambda row: score(row, campaign))
    base = dict(incumbent["config"])
    for dimension in campaign["search_space"]:
        for value in dimension["values"]:
            candidate = dict(base)
            candidate[dimension["name"]] = value
            if _config_key(candidate) in occupied:
                continue
            reason = (
                f"One-factor test around incumbent {incumbent['run_id']}: "
                f"set {dimension['name']}={value!r}."
            )
            return "full", candidate, reason
    raise RuntimeError(
        f"All configured one-factor candidates for {target} have been evaluated. "
        "Extend search_space in campaign.json."
    )


def slurm_script(run_id: str, campaign: dict[str, Any]) -> str:
    cluster = campaign["longleaf"]
    experiment_dir = campaign.get("experiment_dir", "experiments/dreamer_3dcnn")
    training_script = campaign.get(
        "training_script", "experiments/dreamer_3dcnn/train.py"
    )
    return f'''#!/bin/bash
#SBATCH --job-name={run_id[:40]}
#SBATCH --output={experiment_dir}/runs/{run_id}/slurm-%j.out
#SBATCH --error={experiment_dir}/runs/{run_id}/slurm-%j.err
#SBATCH --partition={cluster["partition"]}
#SBATCH --qos={cluster["qos"]}
#SBATCH --gres=gpu:{cluster["gpus"]}
#SBATCH --cpus-per-task={cluster["cpus"]}
#SBATCH --mem={cluster["memory"]}
#SBATCH --time={cluster["time"]}

set -euo pipefail
module purge
module load {cluster["python_module"]}
module load {cluster["cuda_module"]}
module load {cluster["cudnn_module"]}

PROJECT_DIR="${{PROJECT_DIR:-$HOME/EEGProc}}"
VENV_DIR="${{VENV_DIR:-$PROJECT_DIR/venv312}}"
DATA_DIR="${{DATA_DIR:-$PROJECT_DIR/datasets}}"
RUN_DIR="$PROJECT_DIR/{experiment_dir}/runs/{run_id}"

cd "$PROJECT_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Missing virtual environment: $VENV_DIR" >&2
    echo "Create it and install this checkout with the deep-learning extra." >&2
    exit 2
fi
if [[ ! -f "$DATA_DIR/dreamer_eeg.npy" || ! -f "$DATA_DIR/dreamer_labels.npy" ]]; then
    echo "Missing DREAMER arrays in $DATA_DIR" >&2
    exit 2
fi

source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

# TensorFlow/XLA does not automatically discover the libdevice shipped by
# Longleaf's CUDA module. Resolve it from the active module (or the Python CUDA
# packages) and pass its CUDA root explicitly before TensorFlow starts.
MODULE_CUDA_ROOT=""
if [[ -n "${{EBROOTCUDA:-}}" && -d "$EBROOTCUDA" ]]; then
    MODULE_CUDA_ROOT="$EBROOTCUDA"
elif [[ -n "${{CUDA_HOME:-}}" && -d "$CUDA_HOME" ]]; then
    MODULE_CUDA_ROOT="$CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    MODULE_CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi

find_libdevice() {{
    local roots=()
    [[ -n "$MODULE_CUDA_ROOT" && -d "$MODULE_CUDA_ROOT" ]] && roots+=("$MODULE_CUDA_ROOT")
    [[ -d "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" ]] && roots+=("$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia")
    [[ -d /usr/local/cuda ]] && roots+=("/usr/local/cuda")
    [[ -d /opt/cuda ]] && roots+=("/opt/cuda")
    [[ ${{#roots[@]}} -eq 0 ]] && return 0
    find "${{roots[@]}}" -type f -path "*/nvvm/libdevice/libdevice.10.bc" -print -quit 2>/dev/null || true
}}

LIBDEVICE_PATH="$(find_libdevice)"
if [[ -z "$LIBDEVICE_PATH" || ! -f "$LIBDEVICE_PATH" ]]; then
    echo "Unable to locate nvvm/libdevice/libdevice.10.bc." >&2
    echo "Loaded CUDA root: ${{MODULE_CUDA_ROOT:-<none>}}" >&2
    echo "Try: python -m pip install nvidia-cuda-nvcc-cu12" >&2
    exit 2
fi
CUDA_XLA_ROOT="${{LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}}"
export XLA_FLAGS="${{XLA_FLAGS:+$XLA_FLAGS }}--xla_gpu_cuda_data_dir=$CUDA_XLA_ROOT"
[[ -n "$MODULE_CUDA_ROOT" ]] && export CUDA_HOME="$MODULE_CUDA_ROOT" CUDA_PATH="$MODULE_CUDA_ROOT"

echo "CUDA XLA root: $CUDA_XLA_ROOT"
echo "libdevice: $LIBDEVICE_PATH"

python {training_script} \\
    --config "$RUN_DIR/config.json" \\
    --data-dir "$DATA_DIR" \\
    --output-dir "$RUN_DIR"
'''


def _write_slurm_script(path: Path, run_id: str, campaign: dict[str, Any]) -> None:
    """Write a launcher with Unix newlines on every supported Python version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(slurm_script(run_id, campaign))


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(HERE.parent.parent).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _repair_partial_plan(
    campaign: dict[str, Any], runs_dir: Path, target: str
) -> Path | None:
    """Finish a planned run whose launcher was not written successfully."""
    for manifest_path in sorted(runs_dir.glob(f"*/{MANIFEST_FILE}")):
        run_dir = manifest_path.parent
        script = run_dir / "submit.slurm"
        if script.exists() and script.stat().st_size > 0:
            continue
        manifest = load_json(manifest_path)
        config_path = run_dir / CONFIG_FILE
        if (
            manifest.get("campaign") != campaign["campaign"]
            or manifest.get("status") != "planned"
            or not config_path.exists()
        ):
            continue
        config = load_json(config_path)
        if config.get("target") != target:
            continue
        if manifest.get("config_fingerprint") != fingerprint(config):
            raise ValueError(f"Config fingerprint mismatch in partial run {run_dir}.")
        run_id = str(manifest.get("run_id", run_dir.name))
        _write_slurm_script(script, run_id, campaign)
        print(
            f"Repaired partial plan {run_id}\nReason: {manifest.get('reason', 'planned run')}\n"
            f"Submit on Longleaf:\n  cd <repo> && sbatch {_display_path(script)}"
        )
        return run_dir
    return None


def plan_run(campaign: dict[str, Any], runs_dir: Path, target: str) -> Path:
    repaired = _repair_partial_plan(campaign, runs_dir, target)
    if repaired is not None:
        return repaired
    stage, config, reason = suggest(campaign, scan_runs(runs_dir, campaign), target)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    signature = fingerprint(config)
    run_id = f"{stamp}-{target}-{stage}-{signature[:8]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / CONFIG_FILE, config)
    write_json(
        run_dir / MANIFEST_FILE,
        {
            "schema_version": 1,
            "campaign": campaign["campaign"],
            "run_id": run_id,
            "config_fingerprint": signature,
            "stage": stage,
            "status": "planned",
            "created_at": utc_now(),
            "reason": reason,
        },
    )
    script = run_dir / "submit.slurm"
    _write_slurm_script(script, run_id, campaign)
    relative_script = _display_path(script)
    print(
        f"Planned {run_id}\nReason: {reason}\nSubmit on Longleaf:\n"
        f"  cd <repo> && sbatch {relative_script}"
    )
    return run_dir


def _bundle_dirs(source: Path) -> list[Path]:
    if (source / CONFIG_FILE).exists() and (source / MANIFEST_FILE).exists():
        return [source]
    return sorted(path.parent for path in source.rglob(MANIFEST_FILE) if (path.parent / CONFIG_FILE).exists())


def sync_bundles(source: Path, runs_dir: Path, campaign: dict[str, Any]) -> int:
    bundles = _bundle_dirs(source)
    if not bundles:
        raise ValueError(f"No run bundles found under {source}.")
    imported = 0
    for bundle in bundles:
        manifest = load_json(bundle / MANIFEST_FILE)
        config = load_json(bundle / CONFIG_FILE)
        run_id = str(manifest.get("run_id", bundle.name))
        if manifest.get("campaign") != campaign["campaign"]:
            raise ValueError(f"{bundle} belongs to campaign {manifest.get('campaign')!r}.")
        if manifest.get("config_fingerprint") != fingerprint(config):
            raise ValueError(f"Config fingerprint mismatch in {bundle}.")
        destination = runs_dir / run_id
        if destination.exists() and (destination / CONFIG_FILE).exists():
            existing = load_json(destination / CONFIG_FILE)
            if fingerprint(existing) != fingerprint(config):
                raise ValueError(f"Refusing to overwrite {run_id} with a different config.")
        destination.mkdir(parents=True, exist_ok=True)
        for item in bundle.iterdir():
            if item.is_file():
                target = destination / item.name
                if item.resolve() != target.resolve():
                    shutil.copy2(item, target)
        imported += 1
    rebuild_reports(runs_dir, campaign)
    return imported


def rebuild_reports(runs_dir: Path, campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows = scan_runs(runs_dir, campaign)
    runs_dir.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "target", "stage", "status", "mean", "std", "job_id", "created_at"]
    with (runs_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report_title = campaign.get("report_title", "DREAMER 3D-CNN runs")
    lines = [f"# {report_title}", "", f"Objective: `{campaign['objective']['metric']}`", ""]
    for target in VALID_TARGETS:
        lines.extend([f"## {target.title()}", "", "| run | stage | status | mean | std |", "|---|---:|---:|---:|---:|"])
        target_rows = [row for row in rows if row.get("target") == target]
        target_rows.sort(key=lambda row: score(row, campaign) if "mean" in row else -math.inf, reverse=True)
        if not target_rows:
            lines.append("| — | — | no runs | — | — |")
        for row in target_rows:
            mean = f"{row['mean']:.4f}" if "mean" in row else "—"
            std = f"{row['std']:.4f}" if "std" in row else "—"
            lines.append(f"| {row['run_id']} | {row['stage']} | {row['status']} | {mean} | {std} |")
        lines.append("")
    (runs_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def print_status(rows: list[dict[str, Any]], campaign: dict[str, Any]) -> None:
    if not rows:
        print("No runs yet.")
        return
    print(f"{'RUN':50} {'TARGET':8} {'STAGE':6} {'STATUS':10} {'MEAN':>8} {'STD':>8}")
    for row in rows:
        mean = f"{row['mean']:.4f}" if "mean" in row else "-"
        std = f"{row['std']:.4f}" if "std" in row else "-"
        print(f"{row['run_id'][:50]:50} {row.get('target','?'):8} {row.get('stage','?'):6} {row.get('status','?'):10} {mean:>8} {std:>8}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    result.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Run store (defaults to a runs folder beside the selected campaign).",
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("suggest", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--target", choices=VALID_TARGETS, default=None)
    commands.add_parser("status")
    sync = commands.add_parser("sync")
    sync.add_argument("source", type=Path)
    commands.add_parser("rebuild")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        campaign = load_json(args.campaign)
        runs_dir = args.runs_dir or args.campaign.resolve().parent / "runs"
        target = getattr(args, "target", None) or campaign["defaults"]["target"]
        if args.command == "suggest":
            stage, config, reason = suggest(campaign, scan_runs(runs_dir, campaign), target)
            print(f"Next: {target} / {stage}\nReason: {reason}\n\n{json.dumps(config, indent=2)}")
        elif args.command == "plan":
            plan_run(campaign, runs_dir, target)
            rebuild_reports(runs_dir, campaign)
        elif args.command == "sync":
            count = sync_bundles(args.source.resolve(), runs_dir.resolve(), campaign)
            print(f"Synced {count} run bundle(s) into {runs_dir}.")
        elif args.command == "rebuild":
            rows = rebuild_reports(runs_dir, campaign)
            print(f"Rebuilt reports for {len(rows)} run(s).")
        else:
            rows = rebuild_reports(runs_dir, campaign)
            print_status(rows, campaign)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
