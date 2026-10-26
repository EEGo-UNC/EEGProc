"""Move run stores to shared work storage, verifying each copy before removal.

The original repository paths become directory symlinks, so tracker status,
existing Slurm output paths and relative model pointers continue to work.
Never moves datasets, environment files, code or unrelated directories.
"""
from pathlib import Path
import hashlib
import os
import shutil

VARIANTS = ("cnn", "cnn_reconstruction", "cnn_transformer", "cnn_transformer_reconstruction")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def default_work_root():
    user = os.environ.get("USER", "")
    if len(user) < 2:
        raise ValueError("Set SIC_V16_WORK_ROOT to your shared work directory.")
    return Path(os.environ.get("SIC_V16_WORK_ROOT",
        f"/work/users/{user[0]}/{user[1]}/{user}/sic-v16"))


def migrate_store(source, destination):
    source, destination = Path(source), Path(destination)
    destination = destination.resolve()
    if source.is_symlink():
        if source.resolve() != destination:
            raise ValueError(f"Unexpected run-store symlink: {source} -> {source.resolve()}")
        return
    if source.exists():
        source_root = source.resolve()
        if destination == source_root or destination.is_relative_to(source_root):
            raise ValueError("Destination must be outside the source run store.")
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Nested symlink requires inspection before migration: {path}")
        destination.mkdir(parents=True, exist_ok=True)
        # A verified file-by-file migration can safely continue after interruption.
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            if target.is_symlink() or not target.resolve().is_relative_to(destination):
                raise ValueError(f"Unsafe migration target: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            signature = digest(path)
            if target.exists():
                if digest(target) != signature:
                    raise ValueError(f"Conflicting destination preserved: {target}")
            else:
                temporary = target.with_name(target.name + ".migration-copy")
                shutil.copy2(path, temporary)
                if digest(temporary) != signature:
                    raise ValueError(f"Copy verification failed; source preserved: {path}")
                temporary.replace(target)
            # Check source once more: don't remove a file modified by another job.
            if digest(path) != signature:
                raise ValueError(f"Source changed during migration; preserved: {path}")
            path.unlink()
        # Only remove empty directories within the already checked source root.
        directories = sorted((p for p in source.rglob("*") if p.is_dir()),
                             key=lambda p: len(p.parts), reverse=True)
        for path in directories:
            if not path.resolve().is_relative_to(source_root):
                raise ValueError(f"Directory escaped run store: {path}")
            path.rmdir()
        source.rmdir()
    destination.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(destination, target_is_directory=True)
    print(f"Run store: {source} -> {destination}", flush=True)


def ensure_work_stores(repo, work_root, *, migrate=False):
    repo, work_root = Path(repo).resolve(), Path(work_root).resolve()
    if work_root.is_relative_to(Path.home().resolve()):
        raise ValueError("Work root must be outside the quota-limited home directory.")
    if not work_root.parent.is_dir():
        raise ValueError(f"Shared work parent does not exist: {work_root.parent}")
    for variant in VARIANTS:
        source = repo / "experiments/sic_v16" / variant / "runs"
        destination = work_root / variant / "runs"
        if source.exists() and not source.is_symlink() and any(source.iterdir()) and not migrate:
            raise ValueError("Existing home run stores need recovery migration before planning more runs.")
        migrate_store(source, destination)
