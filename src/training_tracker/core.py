"""Plan, sync, summarize, and visualize config-driven training experiments.

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
import re

from .launchers import slurm_script
from .results import result_metric as _result_metric
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parents[2] / "experiments" / "dreamer_3dcnn"
DEFAULT_CAMPAIGN = HERE / "campaign.json"
DEFAULT_RUNS = HERE / "runs"
RESULT_FILE = "result.json"
MANIFEST_FILE = "manifest.json"
CONFIG_FILE = "config.json"
VALID_TARGETS = ("valence", "arousal")


def validate_run_id(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value.endswith("."):
        raise ValueError(f"Unsafe run identifier {value!r}.")
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError(f"Unsafe run identifier {value!r}.")


def campaign_targets(campaign: dict[str, Any]) -> list[str]:
    return campaign.get("targets", [campaign["defaults"]["target"]])


def load_campaign(path: Path) -> dict[str, Any]:
    campaign = load_json(path)
    for key in ("campaign", "defaults", "objective", "search_space"):
        if key not in campaign:
            raise ValueError(f"Campaign is missing {key!r}.")
    if not isinstance(campaign["campaign"], str) or not campaign["campaign"].strip():
        raise ValueError("campaign must be a nonempty name.")
    if not isinstance(campaign["defaults"], dict) or not campaign["defaults"].get("target"):
        raise ValueError("defaults.target must name a target.")
    targets = campaign_targets(campaign)
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets) or len(set(targets)) != len(targets):
        raise ValueError("targets must be a nonempty list of unique names.")
    for target in targets:
        validate_run_id(target)
    if campaign["defaults"]["target"] not in targets:
        raise ValueError("defaults.target must occur in targets.")
    if not isinstance(campaign["objective"], dict) or not isinstance(campaign["objective"].get("metric"), str) or not campaign["objective"]["metric"]:
        raise ValueError("objective.metric must name a metric.")
    if not isinstance(campaign["objective"].get("maximize", True), bool):
        raise ValueError("objective.maximize must be true or false.")
    penalty = float(campaign["objective"].get("uncertainty_penalty", 0))
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("uncertainty_penalty must be finite and nonnegative.")
    if not isinstance(campaign["search_space"], list):
        raise ValueError("search_space must be a list.")
    for dimension in campaign["search_space"]:
        if not isinstance(dimension, dict) or dimension.get("name") not in campaign["defaults"] or not isinstance(dimension.get("values"), list):
            raise ValueError("Each search dimension needs a defaults key and a values list.")
        if dimension["name"] == "target":
            raise ValueError("Use targets rather than searching over target.")
    overrides = campaign.get("smoke_overrides", {})
    if not isinstance(overrides, dict) or "target" in overrides:
        raise ValueError("smoke_overrides must be an object without target.")
    if overrides and {**campaign["defaults"], **overrides} == campaign["defaults"]:
        raise ValueError("Smoke overrides must differ from defaults; omit them to skip the smoke stage.")
    launcher = campaign.get("launcher", {"type": "longleaf"})
    if not isinstance(launcher, dict) or launcher.get("type") not in {"longleaf", "template"}:
        raise ValueError("launcher.type must be longleaf or template.")
    if launcher["type"] == "template" and not isinstance(launcher.get("template"), str):
        raise ValueError("A template launcher requires a template path.")
    campaign["_campaign_dir"] = str(path.resolve().parent)
    return campaign


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



def scan_runs(runs_dir: Path, campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric = campaign["objective"]["metric"]
    for manifest_path in sorted(runs_dir.glob(f"*/{MANIFEST_FILE}")):
        run_dir = manifest_path.parent
        try:
            manifest = load_json(manifest_path)
            config = load_json(run_dir / CONFIG_FILE)
            if manifest.get("campaign") != campaign["campaign"]:
                continue
            if manifest.get("config_fingerprint") != fingerprint(config):
                raise ValueError("Config fingerprint mismatch.")
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
        if manifest.get("error"):
            row["error"] = manifest["error"]
        result_path = run_dir / RESULT_FILE
        if result_path.exists():
            try:
                result = load_json(result_path)
                row["mean"], row["std"] = _result_metric(result, metric)
                row["status"] = "completed"
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
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
    if target not in campaign_targets(campaign):
        raise ValueError(f"Unknown target {target!r}.")
    if active:
        raise RuntimeError(
            f"Run {active[0].get('run_id', '<unknown>')} is already planned or running "
            f"for {target}. Sync its output before planning another."
        )

    if campaign.get("smoke_overrides"):
        smoke = {**defaults, **campaign.get("smoke_overrides", {})}
        if _config_key(smoke) not in occupied:
            return "smoke", smoke, "Validate the complete data/model/evaluation path cheaply before a full run."

        smoke_done = any(
            row.get("status") == "completed" and row.get("stage") == "smoke"
            for row in relevant
        )
        if not smoke_done:
            raise RuntimeError(
                f"A {target} smoke run is already planned or running. Sync its output before planning another."
            )

    if _config_key(defaults) not in occupied:
        return "full", defaults, "Establish the full baseline."

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



def _write_slurm_script(path: Path, run_id: str, campaign: dict[str, Any]) -> None:
    """Write a launcher with Unix newlines on every supported Python version."""
    contents = slurm_script(run_id, campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(contents)


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
    validate_run_id(target)
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
        validate_run_id(run_id)
        destination = runs_dir / run_id
        if destination.is_symlink():
            raise ValueError(f"Refusing symlink destination: {destination}.")
        if destination.exists() and (destination / CONFIG_FILE).exists():
            existing = load_json(destination / CONFIG_FILE)
            if fingerprint(existing) != fingerprint(config):
                raise ValueError(f"Refusing to overwrite {run_id} with a different config.")
        destination.mkdir(parents=True, exist_ok=True)
        for item in bundle.iterdir():
            if item.is_symlink():
                raise ValueError(f"Refusing symlink in bundle: {item}.")
            if item.is_file():
                target = destination / item.name
                if target.is_symlink():
                    raise ValueError(f"Refusing symlink destination: {target}.")
                if item.resolve() != target.resolve():
                    shutil.copy2(item, target)
        imported += 1
    rebuild_reports(runs_dir, campaign)
    return imported


def rebuild_reports(runs_dir: Path, campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows = scan_runs(runs_dir, campaign)
    runs_dir.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "target", "stage", "status", "mean", "std", "job_id", "created_at", "error"]
    with (runs_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report_title = campaign.get("report_title", campaign["campaign"])
    lines = [f"# {report_title}", "", f"Objective: `{campaign['objective']['metric']}`", ""]
    for target in campaign_targets(campaign):
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
        if row.get("error"):
            print(f"  {row['run_id']}: {row['error']}")
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
        command.add_argument("--target", default=None)
    commands.add_parser("status")
    sync = commands.add_parser("sync")
    sync.add_argument("source", type=Path)
    commands.add_parser("rebuild")
    plots = commands.add_parser("plot", help="Generate figures and source tables from completed runs.")
    plots.add_argument("--output-dir", type=Path)
    plots.add_argument("--run-id", action="append", help="Repeat to compare selected runs; default: all completed runs.")
    plots.add_argument("--format", choices=("png", "svg", "pdf"), default="png")
    plots.add_argument("--chance", type=float, help="Optional reference line in the metric’s units; descriptive only.")
    plots.add_argument("--compare-campaign", type=Path, action="append", help="Also read runs beside another campaign (repeatable).")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        campaign = load_campaign(args.campaign)
        runs_dir = args.runs_dir or args.campaign.resolve().parent / "runs"
        target = getattr(args, "target", None) or campaign["defaults"]["target"]
        if target not in campaign_targets(campaign):
            raise ValueError(f"Unknown target {target!r}; choose from {campaign_targets(campaign)}.")
        if args.command == "suggest":
            stage, config, reason = suggest(campaign, scan_runs(runs_dir, campaign), target)
            print(f"Next: {target} / {stage}\nReason: {reason}\n\n{json.dumps(config, indent=2)}")
        elif args.command == "plan":
            plan_run(campaign, runs_dir, target)
            rebuild_reports(runs_dir, campaign)
        elif args.command == "sync":
            count = sync_bundles(args.source.resolve(), runs_dir.resolve(), campaign)
            print(f"Synced {count} run bundle(s) into {runs_dir}.")
        elif args.command == "plot":
            from .plotting import generate_plots
            output = args.output_dir or runs_dir / "plots"
            paths = generate_plots(runs_dir, campaign, output, run_ids=args.run_id,
                                   image_format=args.format, chance=args.chance,
                                   compare_campaigns=args.compare_campaign)
            print(f"Generated {len(paths)} figures in {output}.")
        elif args.command == "rebuild":
            rows = rebuild_reports(runs_dir, campaign)
            print(f"Rebuilt reports for {len(rows)} run(s).")
        else:
            rows = rebuild_reports(runs_dir, campaign)
            print_status(rows, campaign)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
