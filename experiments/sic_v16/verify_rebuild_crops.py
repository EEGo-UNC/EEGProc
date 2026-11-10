"""Verify legacy crops, rebuild separate centred datasets, compare nested baselines."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def crop_rules(stimulus_length, baseline_length, length=7680):
    return dict(stimulus_centre=(stimulus_length-length)//2,
                combined_centre_relative_to_stimulus=(baseline_length+stimulus_length-length)//2-baseline_length,
                stimulus_start=0, stimulus_end=stimulus_length-length)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",required=True,type=Path)
    p.add_argument("--source-csv",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    args=p.parse_args()
    import numpy as np
    import pandas as pd
    from scipy.signal import fftconvolve
    from eegproc.deep_learning.prepare_datasets import DREAMER_EEG_COLS, _filter_dreamer_trial
    from permutation_audit import digest
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        (root/name).write_text(json.dumps(value,indent=2),encoding="utf-8")
    eeg=np.load(args.data_dir/"dreamer_eeg.npy",mmap_mode="r",allow_pickle=False)
    ratings=np.load(args.data_dir/"dreamer_labels.npy",allow_pickle=False)
    if eeg.shape != (23,18,14,7680) or ratings.shape[:2] != (23,18) or ratings.shape[-1]<2:
        raise ValueError("Expected the previously audited raw 14-channel dataset.")
    df=pd.read_csv(args.source_csv)
    for col in ("subject_id","trial_id","sample_idx"):
        df[col]=pd.to_numeric(df[col],errors="raise").astype(int)
    for col in ("valence","arousal"):
        df[col]=df[col].astype(str).str.strip().str.strip("[]").astype(np.float32)
    subject_ids=sorted(df.subject_id.unique())
    if len(subject_ids)!=23: raise ValueError("Expected 23 source subjects")
    directories={name:root/name for name in ("centred_raw","centred_full_filter")}
    outputs={}
    for name,path in directories.items():
        path.mkdir()
        outputs[name]=np.lib.format.open_memmap(path/"dreamer_eeg.npy",mode="w+",dtype=np.float32,
            shape=(23,18,14 if name=="centred_raw" else 42,7680))
    report=dict(source_sha256=digest(args.source_csv), original_eeg_sha256=digest(args.data_dir/"dreamer_eeg.npy"),
        original_labels_sha256=digest(args.data_dir/"dreamer_labels.npy"), sampling_rate_assumed=128,
        channel_order=list(DREAMER_EEG_COLS), records=[],
        interpretation="centred_raw isolates crop changes with the existing filter-after-crop loader; centred_full_filter additionally tests full-stimulus filtering before cropping")
    for si,sid in enumerate(subject_ids):
        sdf=df[df.subject_id==sid]; tids=sorted(sdf.trial_id.unique())
        if len(tids)!=18: raise ValueError("Expected 18 trials per source subject")
        for ti,tid in enumerate(tids):
            group=sdf[sdf.trial_id==tid]
            segments=group.segment.astype(str).str.strip().str.lower()
            stim=group[segments.isin(["stimulus","stimuli"])].sort_values("sample_idx",kind="stable")
            base=group[segments.isin(["baseline","base"])].sort_values("sample_idx",kind="stable")
            if len(stim)<7680 or not np.all(np.diff(stim.sample_idx.to_numpy())==1):
                raise ValueError(f"Invalid stimulus sequence {sid}/{tid}")
            if any(stim[c].isna().any() or stim[c].nunique()!=1 for c in ("valence","arousal")):
                raise ValueError("Nonconstant source ratings")
            label=stim[["valence","arousal"]].iloc[0].to_numpy(dtype=np.float32)
            if not np.isfinite(label).all() or not np.allclose(label,ratings[si,ti,:2],rtol=0,atol=1e-6):
                raise ValueError(f"Source-label mismatch {sid}/{tid}; rebuild halted")
            raw=stim[DREAMER_EEG_COLS].to_numpy(dtype=np.float32)
            saved=np.asarray(eeg[si,ti].T)
            if not np.isfinite(raw).all() or not np.isfinite(saved).all(): raise ValueError("Nonfinite EEG")
            rules=crop_rules(len(raw),len(base))
            # Search raw stimulus via channel-0 correlation if known rules fail.
            starts={v for v in rules.values() if 0<=v<=len(raw)-7680}
            exact=[s for s in starts if np.array_equal(raw[s:s+7680],saved)]
            if not exact:
                a=saved[:,0].astype(float); a-=a.mean()
                corr=fftconvolve(raw[:,0].astype(float),a[::-1],mode="valid")
                for index in np.argsort(corr)[-10:]:
                    starts.add(int(index))
                exact=[s for s in starts if np.array_equal(raw[s:s+7680],saved)]
            combined_match=False
            includes_baseline=False
            if len(base):
                b=base[DREAMER_EEG_COLS].to_numpy(dtype=np.float32)
                combined=np.concatenate([b,raw])
                start=(len(combined)-7680)//2
                combined_match=bool(np.array_equal(combined[start:start+7680],saved))
                includes_baseline=bool(combined_match and start<len(base))
            centre=rules["stimulus_centre"]
            outputs["centred_raw"][si,ti]=raw[centre:centre+7680].T
            filtered=_filter_dreamer_trial(raw)
            outputs["centred_full_filter"][si,ti]=filtered[centre:centre+7680].T
            if not np.isfinite(filtered).all(): raise ValueError("Nonfinite filtered output")
            report["records"].append(dict(array_subject=si,array_trial=ti,source_subject=int(sid),source_trial=int(tid),
                stimulus_samples=len(raw),baseline_samples=len(base),rules=rules,
                exact_stimulus_match_offsets=sorted(exact),combined_centre_exact_match=combined_match,
                verified_baseline_overlap=includes_baseline,source_ratings=label.tolist(),
                centred_start_sample_idx=int(stim.sample_idx.iloc[centre]),
                centred_end_sample_idx=int(stim.sample_idx.iloc[centre+7679]),
                status="identified" if exact or combined_match else "unresolved"))
        write("crop_verification.json",report)
        print(f"Verified/rebuilt subject {si}",flush=True)
    for output in outputs.values(): output.flush()
    outputs.clear()
    for name,path in directories.items():
        np.save(path/"dreamer_labels.npy",ratings)
        # Read back the rebuilt arrays to verify shape and finite output.
        rebuilt=np.load(path/"dreamer_eeg.npy",mmap_mode="r")
        if rebuilt.shape[:2]!=(23,18) or not np.isfinite(rebuilt).all(): raise ValueError("Invalid rebuilt dataset")
        report[name]=dict(eeg_sha256=digest(path/"dreamer_eeg.npy"), labels_sha256=digest(path/"dreamer_labels.npy"))
    report["summary"]=dict(trials=len(report["records"]),
        identified=sum(r["status"]=="identified" for r in report["records"]),
        combined_centre_matches=sum(r["combined_centre_exact_match"] for r in report["records"]),
        verified_baseline_overlaps=sum(r["verified_baseline_overlap"] for r in report["records"]))
    write("crop_verification.json",report)
    results={}
    for name,data in [("original",args.data_dir),*directories.items()]:
        subprocess.run([sys.executable,str(Path(__file__).with_name("nested_logpower.py")),
            "--data-dir",str(data),"--output-dir",str(root/f"baseline-{name}")],check=True)
        results[name]=json.loads((root/f"baseline-{name}/result.json").read_text())
    write("comparison.json",{name:{k:v for k,v in result.items() if k!="folds"} for name,result in results.items()})
    archive=root/"crop-rebuild-comparison-diagnostics.tar.gz"
    with tarfile.open(archive,"w:gz") as bundle:
        for relative in ["crop_verification.json","comparison.json"]+[f"baseline-{name}/{file}" for name in results for file in ("config.json","result.json","data/preprocessing.json")]:
            bundle.add(root/relative,arcname=relative)
    print(f"Download: {archive}",flush=True)


if __name__=="__main__": main()
