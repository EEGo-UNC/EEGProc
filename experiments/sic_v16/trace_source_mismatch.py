"""Rank source-crop/filter/channel/scale explanations for raw 14-channel arrays."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def compare(saved, candidate):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    a = saved.astype(float)
    b = candidate.astype(float)
    ac, bc = a-a.mean(0), b-b.mean(0)
    corr = ac.T @ bc / np.maximum(np.linalg.norm(ac, axis=0)[:,None]*np.linalg.norm(bc, axis=0)[None,:], 1e-20)
    _, order = linear_sum_assignment(-np.abs(corr))
    rows = []
    for name, mapping in (("declared_order", np.arange(a.shape[1])), ("inferred_order", order)):
        x = b[:,mapping]
        xc = x-x.mean(0)
        scale = np.sum(xc*ac,axis=0)/np.maximum(np.sum(xc*xc,axis=0),1e-20)
        offset = a.mean(0)-scale*x.mean(0)
        residual = a-(x*scale+offset)
        global_scale = float(np.sum(xc*ac)/max(float(np.sum(xc*xc)),1e-20))
        global_offset = float(np.mean(a-global_scale*x))
        denominator = max(float(np.linalg.norm(ac)),1e-20)
        rows.append(dict(channel_order=name, source_channel_indices=mapping.tolist(),
            channel_correlations=corr[np.arange(a.shape[1]),mapping].tolist(),
            mean_absolute_correlation=float(np.mean(np.abs(corr[np.arange(a.shape[1]),mapping]))),
            unscaled_relative_error=float(np.linalg.norm(a-x)/denominator),
            global_affine_relative_error=float(np.linalg.norm(a-(global_scale*x+global_offset))/denominator),
            global_scale=global_scale, global_offset=global_offset,
            per_channel_affine_relative_error=float(np.linalg.norm(residual)/denominator),
            per_channel_scale=scale.tolist(), per_channel_offset=offset.tolist(),
            numerically_matches_unscaled=bool(np.allclose(a,x,rtol=1e-5,atol=1e-5))))
    return rows


def crop_candidates(saved, signal, stride):
    """Coarse whole-record scan plus sample-resolution refinement near top crops."""
    import numpy as np
    from scipy.signal import fftconvolve
    from scipy.optimize import linear_sum_assignment
    length, channels = saved.shape
    count = len(signal)-length+1
    if count < 1:
        return []
    a = saved.astype(float)-saved.mean(0)
    b = signal.astype(float)
    sums = np.vstack([np.zeros((1,channels)), np.cumsum(b,axis=0)])
    squares = np.vstack([np.zeros((1,channels)), np.cumsum(b*b,axis=0)])
    norms = np.sqrt(np.maximum(squares[length:]-squares[:-length]-(sums[length:]-sums[:-length])**2/length,1e-20))
    correlations = np.empty((count,channels,channels))
    for i in range(channels):
        for j in range(channels):
            correlations[:,i,j] = fftconvolve(b[:,j],a[::-1,i],mode="valid") / np.maximum(np.linalg.norm(a[:,i])*norms[:,j],1e-20)
    def score(start):
        c = np.abs(correlations[start])
        i,j = linear_sum_assignment(-c)
        return float(c[i,j].mean())
    coarse = sorted(set(range(0,count,stride)) | {0,count//2,count-1})
    top = sorted(coarse,key=score,reverse=True)[:3]
    refined = set(coarse)
    for start in top:
        refined.update(range(max(0,start-stride),min(count,start+stride+1)))
    return sorted(refined,key=score,reverse=True)[:5]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",required=True,type=Path)
    p.add_argument("--source-csv",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--pairs",default="0:0,0:1,1:0",help="Zero-based array subject:trial pairs")
    p.add_argument("--stride",type=int,default=128)
    args=p.parse_args()
    if args.stride < 1: p.error("Stride must be positive")
    import numpy as np
    import pandas as pd
    from scipy.signal import detrend, butter, sosfiltfilt, iirnotch, filtfilt
    from eegproc.deep_learning.prepare_datasets import DREAMER_EEG_COLS, _filter_dreamer_trial
    from permutation_audit import digest
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    eeg=np.load(args.data_dir/"dreamer_eeg.npy",mmap_mode="r")
    if eeg.ndim!=4 or eeg.shape[2]!=14:
        raise ValueError("This trace targets the observed raw 14-channel arrays.")
    pairs=[tuple(map(int,item.split(":"))) for item in args.pairs.split(",")]
    if any(len(pair)!=2 or not 0<=pair[0]<eeg.shape[0] or not 0<=pair[1]<eeg.shape[1] for pair in pairs):
        raise ValueError("Invalid array subject/trial pair")
    df=pd.read_csv(args.source_csv)
    for col in ("subject_id","trial_id","sample_idx"):
        df[col]=pd.to_numeric(df[col],errors="raise").astype(int)
    subjects=sorted(df.subject_id.unique())
    report=dict(source_sha256=digest(args.source_csv), eeg_sha256=digest(args.data_dir/"dreamer_eeg.npy"),
        channel_names=list(DREAMER_EEG_COLS), sampling_rate_assumed=128, stride=args.stride,
        limitations=["Ranks tested transformations; fitted scales/channel permutations are diagnostic explanations, not verified provenance.",
                     "Searches corresponding sorted CSV trial only; does not exclude a different source recording or trial permutation.",
                     "Coarse scan plus local refinement can miss other crop locations; use stride=1 for exhaustive crop ranking."], trials=[])
    for si,ti in pairs:
        sdf=df[df.subject_id==subjects[si]]; tid=sorted(sdf.trial_id.unique())[ti]
        group=sdf[sdf.trial_id==tid]
        saved=np.asarray(eeg[si,ti].T,dtype=float)
        if not np.isfinite(saved).all(): raise ValueError("Nonfinite saved EEG")
        ranked=[]
        for segment in ("stimulus","baseline"):
            names=["stimulus","stimuli"] if segment=="stimulus" else ["baseline","base"]
            g=group[group.segment.astype(str).str.strip().str.lower().isin(names)].sort_values("sample_idx",kind="stable")
            if len(g)<len(saved): continue
            if not np.all(np.diff(g.sample_idx.to_numpy())==1): raise ValueError("Discontinuous source segment")
            raw=g[DREAMER_EEG_COLS].to_numpy(dtype=np.float32)
            if not np.isfinite(raw).all(): raise ValueError("Nonfinite source EEG")
            d=detrend(raw,axis=0)
            notch=filtfilt(*iirnotch(50,30,fs=128),d,axis=0)
            modes={"raw":raw,"linear_detrend":d,"detrend_notch50":notch,
                   "detrend_notch_bandpass4_30":sosfiltfilt(butter(4,[4,30],fs=128,btype="bandpass",output="sos"),notch,axis=0),
                   "canonical_theta_alpha_beta_sum":_filter_dreamer_trial(raw).reshape(len(raw),14,3).sum(-1),
                   "common_average_reference":raw-raw.mean(axis=1,keepdims=True)}
            starts=set()
            for mode,signal in modes.items():
                candidates=crop_candidates(saved,signal,args.stride)
                candidates=sorted(set(candidates)|{0,(len(signal)-len(saved))//2,len(signal)-len(saved)})
                starts.update(candidates)
                for start in candidates:
                    for row in compare(saved,signal[start:start+len(saved)]):
                        ranked.append(dict(segment=segment,filter=mode,crop_start_offset=start,
                            crop_start_sample_idx=int(g.sample_idx.iloc[start]),**row))
            # Filtering a crop differs from filtering the complete recording.
            for start in sorted(starts):
                cropped=_filter_dreamer_trial(raw[start:start+len(saved)]).reshape(len(saved),14,3).sum(-1)
                for row in compare(saved,cropped):
                    ranked.append(dict(segment=segment,filter="crop_then_canonical_band_sum",crop_start_offset=start,
                        crop_start_sample_idx=int(g.sample_idx.iloc[start]),**row))
        ranked.sort(key=lambda r:r["per_channel_affine_relative_error"])
        report["trials"].append(dict(array_subject=si,array_trial=ti,source_subject=int(subjects[si]),source_trial=int(tid),
            candidates_tested=len(ranked),best_candidates=ranked[:30],
            best_by_error={key:min(ranked,key=lambda r:r[key]) for key in
                ("unscaled_relative_error","global_affine_relative_error","per_channel_affine_relative_error")} if ranked else {}))
        (root/"trace.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"Finished {si}:{ti}; best candidate: {ranked[0] if ranked else 'no sufficiently long source segments'}",flush=True)
    archive=root/"source-mismatch-trace-diagnostics.tar.gz"
    with tarfile.open(archive,"w:gz") as t: t.add(root/"trace.json",arcname="trace.json")
    print(f"Download: {archive}",flush=True)


if __name__=="__main__": main()
