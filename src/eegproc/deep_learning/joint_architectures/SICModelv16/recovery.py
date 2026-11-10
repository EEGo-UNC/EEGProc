"""Framework-free checkpoint audit. Never delete records or partially saved models."""
import json
from pathlib import Path
import re
import zipfile

EPOCH = re.compile(r"epoch_(\d+)\.json$")


def checkpoint_progress(output, subject, epochs, *, verify_latest=False):
    output = Path(output)
    folder = output / f"subject_{subject:02d}"
    rows = []
    for path in sorted(folder.glob("epoch_*.json")):
        match = EPOCH.fullmatch(path.name)
        if match is None:
            continue
        row = json.loads(path.read_text())
        number = int(match[1])
        if row["target_subject"] != subject or row["epoch"] != number:
            raise ValueError(f"Mismatched epoch metadata: {path}")
        checkpoint = (output / row["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(output.resolve()) or not checkpoint.is_file():
            raise ValueError(f"Missing or unsafe checkpoint: {checkpoint}")
        if not zipfile.is_zipfile(checkpoint):
            raise ValueError(f"Invalid checkpoint ZIP: {checkpoint}. Original files are preserved.")
        rows.append(row)
    numbers = [r["epoch"] for r in rows]
    if numbers != list(range(1, len(rows) + 1)) or len(rows) > epochs:
        raise ValueError(f"Non-contiguous epoch records in {folder}: {numbers}")
    latest = output / rows[-1]["checkpoint"] if rows else None
    if latest and verify_latest:
        with zipfile.ZipFile(latest) as archive:
            if "model.weights.h5" not in archive.namelist() or "config.json" not in archive.namelist():
                raise ValueError(f"Incomplete checkpoint contents: {latest}")
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Checkpoint CRC failure: {latest}: {bad}")
    recorded = {r["checkpoint"] for r in rows}
    orphans = [str(p.relative_to(output)) for p in folder.glob("epoch_*.keras")
               if not p.name.endswith(".tmp.keras") and str(p.relative_to(output)) not in recorded]
    return {"subject": subject, "completed_epochs": len(rows), "complete": len(rows) == epochs,
            "checkpoint": str(latest) if latest else None, "orphan_checkpoints": sorted(orphans),
            "partial_files": [str(p) for p in folder.glob("*.tmp*")]}
