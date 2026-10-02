"""Convert downloaded public datasets to CSV: ``python -m eegproc.data.to_csv``."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

from ._csv_common import open_text
from .csv_cowen import cowen27_frames
from .csv_eegemotions import eegemotions_frames
from .csv_matlab import amigos_frames, deap_frames, dreamer_csv_frames, dreamer_mat_frames

DATASETS = ("amigos", "dreamer", "eegemotions", "cowen27", "deap")


def _publish_without_overwrite(temporary: Path, output: Path) -> None:
    """Publish ``temporary`` as ``output`` without replacing a file created meanwhile.

    A hard link is atomic and fails if ``output`` exists. Filesystems without
    hard links (FAT/exFAT drives, some network shares) fall back to an
    exclusive-create copy, which also refuses to overwrite and removes its own
    partial output if the copy fails.
    """
    try:
        output.hardlink_to(temporary)
        return
    except FileExistsError:
        raise
    except OSError:
        pass
    with open(temporary, "rb") as source:
        target = open(output, "xb")
        try:
            with target:
                shutil.copyfileobj(source, target)
        except BaseException:
            output.unlink(missing_ok=True)
            raise


def convert_dataset(dataset: str, input_path: str | Path, output_path: str | Path, *,
                    chunksize: int = 100_000, overwrite: bool = False,
                    labels_path: str | Path | None = None, source_label_kind: str = "cowen",
                    amigos_baseline_samples: int = 640) -> int:
    """Write a dataset CSV or CSV.gz and return its row count.

    Input is a dataset directory, or a single AMIGOS/DREAMER MAT or DEAP DAT file.
    DREAMER directories can contain either the three legacy CSV exports or
    DREAMER.mat. CSV exports take precedence if dreamer_eeg.csv[.gz] exists.
    Text/CSV readers stream chunks; MATLAB and DEAP readers load one source file
    at a time. Cowen loads only its small rating matrices.

    A temporary output is published only after successful conversion. Existing
    files require ``overwrite=True``. DEAP DAT files use Python pickle and must
    come from a trusted source. Signals are not filtered or standardized; the
    native DREAMER MAT reader resamples ECG onto the EEG sampling grid.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if amigos_baseline_samples < 0:
        raise ValueError("amigos_baseline_samples must be nonnegative")
    root, output = Path(input_path).expanduser().resolve(), Path(output_path).expanduser().absolute()
    if not root.exists():
        raise FileNotFoundError(root)
    if not (output.name.endswith(".csv") or output.name.endswith(".csv.gz")):
        raise ValueError("Output filename must end in .csv or .csv.gz")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    if output.is_symlink():
        raise ValueError("Output must not be a symbolic link")
    protected = {root}
    if root.is_dir():
        protected.update((root / name).resolve() for name in [
            "participants_info.csv", "training/eeg_features_extracted.csv",
            *[f"dreamer_{kind}{ext}" for kind in ("eeg", "ecg", "labels")
              for ext in (".csv", ".csv.gz")],
        ])
    if output.resolve() in protected:
        raise ValueError("Output must not replace a dataset input file")
    if labels_path is not None and output.resolve() == Path(labels_path).expanduser().resolve():
        raise ValueError("Output must not replace the labels file")
    if dataset == "amigos":
        frames = amigos_frames(root, chunksize, amigos_baseline_samples)
    elif dataset == "dreamer":
        if root.is_dir() and any((root / f"dreamer_eeg{ext}").exists() for ext in (".csv", ".csv.gz")):
            frames = dreamer_csv_frames(root, chunksize)
        else:
            frames = dreamer_mat_frames(root / "DREAMER.mat" if root.is_dir() else root, chunksize)
    elif dataset == "eegemotions":
        if labels_path is not None and not Path(labels_path).expanduser().is_file():
            raise FileNotFoundError(labels_path)
        frames = eegemotions_frames(root, chunksize, source_label_kind=source_label_kind,
                                   labels_path=Path(labels_path).expanduser() if labels_path else None)
    elif dataset == "cowen27":
        frames = cowen27_frames(root)
    else:
        frames = deap_frames(root, chunksize)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    columns = None
    # Keep the temporary file on the destination filesystem for atomic publication.
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.",
                                     suffix=".gz" if output.suffix == ".gz" else ".csv",
                                     delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with open_text(temporary) as handle:
            for frame in frames:
                if frame.empty:
                    continue
                if columns is None:
                    columns = list(frame.columns)
                elif list(frame.columns) != columns:
                    raise ValueError("CSV column layout changed between input records")
                frame.to_csv(handle, index=False, header=rows == 0)
                rows += len(frame)
        if rows == 0:
            raise ValueError("The input contained no convertible records")
        if overwrite:
            temporary.replace(output)
        else:
            # A concurrent writer cannot be overwritten after the initial existence check.
            _publish_without_overwrite(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--input", required=True, type=Path, help="Downloaded/extracted dataset directory or MAT/DAT file")
    parser.add_argument("--output", required=True, type=Path, help="Destination .csv or .csv.gz")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows per text/CSV chunk (default: 100000)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--labels", type=Path, help="EEGEmotions feature/label CSV; defaults to training/eeg_features_extracted.csv")
    parser.add_argument("--source-label-kind", choices=("cowen", "ekman"), default="cowen",
                        help="Emotion system encoded in EEGEmotions filenames (default: cowen); IDs are never shifted")
    parser.add_argument("--amigos-baseline-samples", type=int, default=640,
                        help="Leading AMIGOS baseline samples (default 640); use 0 for already-trimmed inputs")
    args = parser.parse_args(argv)
    try:
        rows = convert_dataset(args.dataset, args.input, args.output, chunksize=args.chunksize,
                               overwrite=args.overwrite, labels_path=args.labels,
                               source_label_kind=args.source_label_kind,
                               amigos_baseline_samples=args.amigos_baseline_samples)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")
    print(f"Wrote {rows:,} rows to {args.output}")


if __name__ == "__main__":
    main()
