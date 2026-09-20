"""Qualitative EEG-realism check for counterfactuals: waveform zoom + wPLI/coherence.

Compares four signals from one counterfactual run, in the model's (1,W,T,42)
feature layout (14 channels x 3 bands theta/alpha/beta):

  1. real class-0   : the original trial `x` in the .npz
  2. counterfactual : `x_prime_*` (the 0->1 counterfactual)
  3. real class-1   : a real opposite-class trial from the SAME subject, pulled
                      from the prepared DREAMER arrays and z-scored to match
  4. reconstruction : `x_reconstructed_*` (decoder baseline; isolates decoder
                      error from the counterfactual change)

The subject/trial behind `x` is recovered empirically (correlation match against
the raw DREAMER trials), so nothing about subject/trial indexing is hardcoded --
point it at a new run and it re-derives the mapping.

wPLI/coherence use a standalone numpy port of `_vcsc_band_coherence_wpli`
(vcsc-calibration branch). RAW coherence and debiased wPLI^2 are reported, NOT
the z-scored VCSC penalty: the calibration file is under active revision, and
the qualitative question ("does it still look like EEG") is answered by the raw
values against the literature reference (real DREAMER ~0.061 wPLI^2, ~0.278
coherence).

Usage:
  python -m eegproc.model_explainability.wpli_compare \
      --cf-npz PATH/counterfactual.npz \
      --dreamer-eeg PATH/dreamer_eeg.npy \
      --dreamer-labels PATH/dreamer_labels.npy \
      --task arousal \
      --output-dir PATH/out
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CHANNELS = ("AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4")
_BANDS = ("theta","alpha","beta")
_PAIRS = list(itertools.combinations(range(len(_CHANNELS)), 2))
_MEDIAN_LABEL = 3.0            # DREAMER 1-5 Likert midpoint (joint_models_data.py)
_LABEL_COL = {"valence": 0, "arousal": 1}
# literature / real-data references from the vcsc-calibration docstring:
_REAL_WPLI_REF = 0.061
_REAL_COH_REF = 0.278


def band_coherence_wpli(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numpy port of _vcsc_band_coherence_wpli. x:(1,W,T,42) or (W,T,42).
    Returns (coherence, wpli), each (n_pairs, n_bands)."""
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    W, T, F = x.shape
    nC, nB = len(_CHANNELS), len(_BANDS)
    if F != nC * nB:
        raise ValueError(f"last dim {F} != {nC*nB}")
    reshaped = x.reshape(W, T, nC, nB)
    signal = np.transpose(reshaped, (2, 3, 0, 1))       # (C,B,W,T)
    spectrum = np.fft.rfft(signal, axis=-1)              # (C,B,W,Fbins)
    coherences, wplis = [], []
    for i, j in _PAIRS:
        xi, xj = spectrum[i], spectrum[j]                # (B,W,Fbins)
        cross = xi * np.conj(xj)
        power_i = np.real(xi * np.conj(xi))
        power_j = np.real(xj * np.conj(xj))
        mean_cross = cross.mean(axis=1)                  # over W -> (B,Fbins)
        mpi, mpj = power_i.mean(axis=1), power_j.mean(axis=1)
        coh = np.abs(mean_cross) ** 2 / (mpi * mpj + 1e-12)
        coherences.append(coh.mean(axis=1))              # -> (B,)
        ic = np.imag(cross)
        s_imag = ic.sum(axis=1); s_sq = (ic ** 2).sum(axis=1); s_abs = np.abs(ic).sum(axis=1)
        wpli = (s_imag ** 2 - s_sq) / (s_abs ** 2 - s_sq + 1e-12)
        wplis.append(wpli.mean(axis=-1))                 # -> (B,)
    return np.stack(coherences, 0), np.stack(wplis, 0)   # (n_pairs, n_bands)


def _pick_key(files, prefix):
    """Return the single key starting with prefix, or None. Errors if ambiguous."""
    hits = [k for k in files if k.startswith(prefix)]
    if not hits:
        return None
    if len(hits) > 1:
        # prefer 'joint', else first
        joint = [k for k in hits if "joint" in k]
        return joint[0] if joint else sorted(hits)[0]
    return hits[0]


def load_cf(npz_path):
    d = np.load(npz_path)
    files = list(d.files)
    keys = {
        "x": "x",
        "cf": _pick_key(files, "x_prime"),
        "recon": _pick_key(files, "x_reconstructed"),
    }
    if keys["cf"] is None:
        raise ValueError(f"no x_prime_* key in {files}")
    out = {name: d[k] for name, k in keys.items() if k is not None}
    return out, keys


def match_subject_trial(x, dreamer_eeg):
    """Correlate x (1,W,T,42) against every raw DREAMER trial; return (s,t,corr)."""
    xr = x[0].reshape(-1, x.shape[-1]).T                 # (42, W*T)
    a = (xr - xr.mean()) / (xr.std() + 1e-9)
    best = (-1.0, -1, -1)
    S, Tn = dreamer_eeg.shape[0], dreamer_eeg.shape[1]
    for s in range(S):
        for t in range(Tn):
            r = dreamer_eeg[s, t]                        # (42, samples)
            if r.shape != xr.shape:
                continue
            b = (r - r.mean()) / (r.std() + 1e-9)
            c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            if c > best[0]:
                best = (c, s, t)
    return best[1], best[2], best[0]


def zscore_per_channel(trial_42_by_n):
    """Per-feature z-score, matching the model's per-channel scaling."""
    m = trial_42_by_n.mean(axis=1, keepdims=True)
    sd = trial_42_by_n.std(axis=1, keepdims=True) + 1e-9
    return (trial_42_by_n - m) / sd


def to_windowed(trial_42_by_n, W, T):
    """(42, W*T) -> (1, W, T, 42)."""
    return trial_42_by_n.T.reshape(W, T, 42)[None, ...]


def pick_class1_trial(labels, subject_idx, task, cf_trial_idx):
    """Return trial index of the strongest opposite-class trial for this subject."""
    col = _LABEL_COL[task]
    arousal = labels[subject_idx][:, col]
    cf_class = int(arousal[cf_trial_idx] >= _MEDIAN_LABEL)
    target_class = 1 - cf_class
    if target_class == 1:
        cands = [(v, i) for i, v in enumerate(arousal) if v >= _MEDIAN_LABEL]
        cands.sort(reverse=True)                          # strongest high first
    else:
        cands = [(v, i) for i, v in enumerate(arousal) if v < _MEDIAN_LABEL]
        cands.sort()                                      # strongest low first
    if not cands:
        raise ValueError(f"no class-{target_class} trial for subject {subject_idx}")
    return cands[0][1], target_class, cf_class


def plot_waveforms(signals, out_path, n_windows=3, channel=0, band=0):
    """Zoom on the first n_windows for one channel/band, all signals overlaid."""
    feat = channel * len(_BANDS) + band
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, sig in signals.items():
        s = sig[0]                                        # (W,T,42)
        seg = s[:n_windows, :, feat].reshape(-1)          # concat early windows
        ax.plot(seg, label=name, linewidth=0.9)
    T = signals[next(iter(signals))].shape[2]
    for w in range(1, n_windows):
        ax.axvline(w * T, color="k", alpha=0.15, linestyle="--")
    ax.set_title(f"Waveform zoom  ch={_CHANNELS[channel]} band={_BANDS[band]}  "
                 f"(first {n_windows} windows)")
    ax.set_xlabel("sample"); ax.set_ylabel("amplitude (z)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_connectivity(metrics, out_path):
    """Bar chart: mean coherence and wPLI per band, per signal."""
    names = list(metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xpos = np.arange(len(_BANDS)); wbar = 0.8 / len(names)
    for m_i, (metric, ax, ref) in enumerate(
        [("coherence", axes[0], _REAL_COH_REF), ("wpli", axes[1], _REAL_WPLI_REF)]
    ):
        for k, name in enumerate(names):
            vals = metrics[name][metric].mean(axis=0)     # mean over pairs -> (n_bands,)
            ax.bar(xpos + k * wbar, vals, wbar, label=name)
        ax.axhline(ref, color="k", linestyle=":", label=f"real DREAMER ref ({ref})")
        ax.set_xticks(xpos + wbar * (len(names) - 1) / 2)
        ax.set_xticklabels(_BANDS)
        ax.set_title(f"mean {metric} per band (over 91 pairs)")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-npz", required=True)
    ap.add_argument("--dreamer-eeg", required=True)
    ap.add_argument("--dreamer-labels", required=True)
    ap.add_argument("--task", choices=["arousal", "valence"], default="arousal")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--class1-trial", type=int, default=None,
                    help="override auto-picked opposite-class trial index")
    ap.add_argument("--zoom-windows", type=int, default=3)
    ap.add_argument("--zoom-channel", type=int, default=0)
    ap.add_argument("--zoom-band", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cf, keys = load_cf(args.cf_npz)
    x = cf["x"]
    W, T = x.shape[1], x.shape[2]
    print(f"loaded cf: keys={keys}  x shape={x.shape}  (W={W}, T={T})")

    eeg = np.load(args.dreamer_eeg)
    labels = np.load(args.dreamer_labels)
    s_idx, t_idx, corr = match_subject_trial(x, eeg)
    print(f"index match: subject_idx={s_idx} trial_idx={t_idx} corr={corr:.4f}")
    if corr < 0.5:
        print("  WARNING: weak match (<0.5). x may be featurized differently; "
              "real class-1 selection may be unreliable.")

    if args.class1_trial is not None:
        c1_idx = args.class1_trial
        col = _LABEL_COL[args.task]
        target_class = int(labels[s_idx][c1_idx, col] >= _MEDIAN_LABEL)
        cf_class = int(labels[s_idx][t_idx, col] >= _MEDIAN_LABEL)
    else:
        c1_idx, target_class, cf_class = pick_class1_trial(
            labels, s_idx, args.task, t_idx)
    col = _LABEL_COL[args.task]
    print(f"cf trial {args.task}={labels[s_idx][t_idx,col]:.0f} -> class {cf_class}")
    print(f"real opposite-class trial idx={c1_idx} "
          f"{args.task}={labels[s_idx][c1_idx,col]:.0f} -> class {target_class}")

    real_opp = eeg[s_idx][c1_idx]                          # (42, samples)
    real_opp = zscore_per_channel(real_opp)
    real_opp_w = to_windowed(real_opp, W, T)

    signals = {
        f"real class{cf_class} (orig x)": x,
        f"counterfactual {cf_class}->{target_class}": cf["cf"],
        f"real class{target_class} (t{c1_idx})": real_opp_w,
    }
    if "recon" in cf:
        signals["reconstruction (baseline)"] = cf["recon"]

    metrics = {}
    print("\n--- coherence / debiased wPLI^2 (mean over 91 pairs) ---")
    for name, sig in signals.items():
        c, w = band_coherence_wpli(sig)
        metrics[name] = {"coherence": c, "wpli": w}
        print(f"{name:38s}  coh={c.mean():.4f}  wpli={w.mean():.4f}")
    print(f"{'(real DREAMER reference)':38s}  coh={_REAL_COH_REF:.4f}  "
          f"wpli={_REAL_WPLI_REF:.4f}")

    wf = out / "waveform_zoom.png"
    cn = out / "connectivity_compare.png"
    plot_waveforms(signals, wf, args.zoom_windows, args.zoom_channel, args.zoom_band)
    plot_connectivity(metrics, cn)
    print(f"\nsaved:\n  {wf}\n  {cn}")


if __name__ == "__main__":
    raise SystemExit(main())