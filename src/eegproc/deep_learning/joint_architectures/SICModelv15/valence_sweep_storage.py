#!/usr/bin/env python3
"""Resolve and probe sweep storage without importing TensorFlow or NumPy."""
from __future__ import annotations

import argparse
import getpass
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def outside_home(value):
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(Path.home().resolve()):
        raise ValueError(f"Sweep storage must be outside home: {path}. Use work/project storage or scratch.")
    if "\n" in str(path) or "\r" in str(path):
        raise ValueError("Storage paths cannot contain line breaks.")
    return path


def storage_paths(storage_root="", output_root="", log_dir=""):
    if not output_root:
        if not storage_root:
            user = getpass.getuser()
            account = Path("/work/users") / user[:1] / user[1:2] / user
            if not account.is_dir():
                raise ValueError(f"Work storage is unavailable at {account}. Set SWEEP_STORAGE_ROOT "
                                 "to your existing work/project or scratch directory; home is not a fallback.")
            storage_root = account / "EEGProc"
        output_root = Path(storage_root) / "runs/sweeps/sic_v15_valence_focal_scale_recon"
    output = outside_home(output_root)
    return output, outside_home(log_dir or output / "logs")


def check_storage(output_root, log_dir, min_free_gib=100, min_free_inodes=100000):
    if not math.isfinite(min_free_gib) or min_free_gib < 0 or min_free_inodes < 0:
        raise ValueError("Storage minimums must be finite and nonnegative.")
    checked = set()
    for value in (output_root, log_dir):
        path = outside_home(value)
        if path in checked:
            continue
        checked.add(path)
        path.mkdir(parents=True, exist_ok=True)
        # Flush actual bytes: permission checks alone do not detect EDQUOT.
        with tempfile.NamedTemporaryFile(prefix=".sweep-storage-", dir=path) as probe:
            probe.write(b"\0" * 4096)
            probe.flush()
            os.fsync(probe.fileno())
        free = shutil.disk_usage(path).free
        if free < min_free_gib * 1024**3:
            raise ValueError(f"Insufficient filesystem space at {path}: {free / 1024**3:.1f} GiB "
                             f"free; minimum {min_free_gib:g} GiB.")
        stats = os.statvfs(path)
        # Some filesystems do not expose meaningful inode counts.
        if stats.f_files and stats.f_favail < min_free_inodes:
            raise ValueError(f"Insufficient filesystem inodes at {path}: {stats.f_favail} "
                             f"available; minimum {min_free_inodes}.")
        print(f"Storage ready: {path} ({free / 1024**3:.1f} GiB filesystem free)", flush=True)


def show_quota():
    print("Storage checks do not reserve capacity or measure remaining user/group quota. "
          "Confirm the full sweep fits your allocation.", flush=True)
    if not shutil.which("quota"):
        print("Quota report unavailable: use your site's quota command.", flush=True)
        return
    try:
        result = subprocess.run(["quota", "-s"], capture_output=True, text=True, timeout=10)
        report = (result.stdout + result.stderr).strip()
        print("Quota report (quota -s):\n" + (report or "No quota information returned."), flush=True)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Quota report unavailable: {error}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    paths = commands.add_parser("paths")
    paths.add_argument("--storage-root", default="")
    paths.add_argument("--output-root", default="")
    paths.add_argument("--log-dir", default="")
    check = commands.add_parser("check")
    check.add_argument("--output-root", required=True)
    check.add_argument("--log-dir", required=True)
    check.add_argument("--min-free-gib", type=float, default=100)
    check.add_argument("--min-free-inodes", type=int, default=100000)
    check.add_argument("--show-quota", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "paths":
            print(*storage_paths(args.storage_root, args.output_root, args.log_dir), sep="\n")
        else:
            check_storage(args.output_root, args.log_dir, args.min_free_gib, args.min_free_inodes)
            if args.show_quota:
                show_quota()
    except (OSError, ValueError) as error:
        print(f"ERROR: sweep storage: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
