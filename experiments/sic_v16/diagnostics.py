"""Package lightweight diagnostics, following the four known run-store links."""
from datetime import datetime
from pathlib import Path
import tarfile
from storage import VARIANTS, default_work_root

ROOT = Path(__file__).resolve().parents[2]
extensions = {".json", ".log", ".out", ".err", ".txt", ".csv", ".slurm"}


def main():
    files = set()
    for prefix in (Path(), Path("first4")):
        for variant in VARIANTS:
            store = ROOT / "experiments/sic_v16" / prefix / variant / "runs"
            if store.is_dir():
                files.update(p for p in store.rglob("*") if p.is_file()
                             and not p.is_symlink() and p.suffix in extensions)
    for pattern in ("sicv16_*.out", "sicv16_*.err"):
        files.update(ROOT.glob(pattern))
    if not files:
        raise SystemExit("No diagnostic files found.")
    destination = default_work_root() / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"sicv16-diagnostics-{datetime.now():%Y%m%d-%H%M%S}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(files):
            bundle.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    print(f"Download: {archive}\nFiles: {len(files)}; size: {archive.stat().st_size/1024**2:.2f} MB")


if __name__ == "__main__":
    main()
