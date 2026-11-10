"""Check converted arrays against the source CSV and evaluate trial identity."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def audit_source(csv_path, eeg, ratings):
    import numpy as np
    import pandas as pd
    from eegproc.deep_learning.prepare_datasets import DREAMER_EEG_COLS, _extract_centre, _filter_dreamer_trial
    from permutation_audit import digest
    if not csv_path.is_file():
        return dict(status="source_unavailable", path=str(csv_path),
                    unresolved=["Provide the original dreamer_joined.csv to check source correspondence."])
    df = pd.read_csv(csv_path)
    required = {"subject_id", "trial_id", "segment", "sample_idx", "valence", "arousal", *DREAMER_EEG_COLS}
    if not required <= set(df.columns):
        raise ValueError(f"Missing source columns: {sorted(required-set(df.columns))}")
    for col in ("subject_id", "trial_id", "sample_idx"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(np.int64)
    for col in ("valence", "arousal"):
        df[col] = df[col].astype(str).str.strip().str.strip("[]").astype(np.float32)
    ids = sorted(df.subject_id.unique().tolist())
    if len(ids) != 23:
        raise ValueError("Expected 23 source subjects.")
    records = []
    trial_sets = []
    for si, sid in enumerate(ids):
        sdf = df[df.subject_id == sid]
        tids = sorted(sdf.trial_id.unique().tolist())
        trial_sets.append(tids)
        if len(tids) != 18:
            raise ValueError("Expected 18 source trials per subject.")
        for ti, tid in enumerate(tids):
            group = sdf[sdf.trial_id == tid]
            stim = group[group.segment.astype(str).str.strip().str.lower().isin(["stimulus", "stimuli"])].sort_values("sample_idx", kind="stable")
            if len(stim) < 7680 or not np.all(np.diff(stim.sample_idx.to_numpy()) == 1):
                raise ValueError(f"Short, duplicated or discontinuous stimulus: {sid}/{tid}")
            if any(stim[c].isna().any() or stim[c].nunique() != 1 for c in ("valence", "arousal")):
                raise ValueError(f"Nonconstant/missing source ratings: {sid}/{tid}")
            raw = stim[DREAMER_EEG_COLS].to_numpy(dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError("Nonfinite source EEG.")
            signal = raw if eeg.shape[2] == 14 else _filter_dreamer_trial(raw)
            expected = _extract_centre(signal, 7680, repeat_pad=False).T
            label = stim[["valence", "arousal"]].iloc[0].to_numpy(dtype=np.float32)
            label_match = bool(np.allclose(ratings[si,ti,:2], label, rtol=0, atol=1e-6))
            eeg_match = bool(np.allclose(eeg[si,ti], expected, rtol=1e-5, atol=1e-5))
            start = (len(stim)-7680)//2
            records.append(dict(array_subject=si, array_trial=ti, source_subject=sid, source_trial=tid,
                source_samples=len(stim), crop_start_sample_idx=int(stim.sample_idx.iloc[start]),
                crop_end_sample_idx=int(stim.sample_idx.iloc[start+7679]),
                source_ratings=label.tolist(), labels_match=label_match, eeg_matches=eeg_match,
                maximum_absolute_eeg_difference=float(np.max(np.abs(eeg[si,ti]-expected)))))
    return dict(status="matched" if all(r["labels_match"] and r["eeg_matches"] for r in records) else "mismatch",
        csv_sha256=digest(csv_path), records=records, common_trial_ids=all(t == trial_sets[0] for t in trial_sets),
        unresolved=["CSV-to-original-recording/MAT-file correspondence is not verified.",
                    "Common trial IDs do not independently prove common film identity.",
                    "Sampling frequency is assumed 128 Hz; confirm against acquisition metadata."])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--source-csv", type=Path)
    args = p.parse_args()
    import numpy as np
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    eeg = np.load(args.data_dir / "dreamer_eeg.npy", mmap_mode="r", allow_pickle=False)
    ratings = np.load(args.data_dir / "dreamer_labels.npy", allow_pickle=False)
    if eeg.shape[:2] != (23,18) or eeg.shape[2] not in (14,42) or eeg.shape[-1] != 7680 or ratings.shape[:2] != (23,18) or ratings.shape[-1] < 2:
        raise ValueError("Unexpected array shapes.")
    if not np.isfinite(ratings).all() or not ((ratings[...,0]>=1)&(ratings[...,0]<=5)).all():
        raise ValueError("Invalid ratings.")
    audit = audit_source(args.source_csv or args.data_dir / "dreamer_joined.csv", eeg, ratings)
    (root / "source_alignment.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    cache = root / "prepared"
    cache.mkdir()
    labels = (ratings[...,0] >= 3).astype(int).reshape(-1)
    np.save(cache / "labels.npy", labels)
    np.save(cache / "subjects.npy", np.repeat(np.arange(23),18))
    np.save(cache / "trials.npy", np.tile(np.arange(18),23))
    np.save(cache / "log_power.npy", np.zeros((414,1)))
    (cache / "preprocessing.json").write_text(json.dumps(dict(feature="trial index only; no EEG features", source_status=audit["status"])), encoding="utf-8")
    subprocess.run([sys.executable, str(Path(__file__).with_name("nested_logpower.py")),
        "--data-dir", str(args.data_dir), "--output-dir", str(root / "baseline"),
        "--feature-cache", str(cache), "--trial-id-only"], check=True)
    archive = root / "source-stimulus-diagnostics.tar.gz"
    with tarfile.open(archive,"w:gz") as bundle:
        for relative in ("source_alignment.json", "baseline/config.json", "baseline/result.json"):
            bundle.add(root / relative, arcname=relative)
    print(f"Source audit: {audit['status']}; Download: {archive}", flush=True)


if __name__ == "__main__":
    main()
