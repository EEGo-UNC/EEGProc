"""Small array-only summaries for training diagnostics."""


def weight_changes(names, before, after):
    import numpy as np
    groups = {}
    for name, old, new in zip(names, before, after):
        layer = name.rsplit("/", 1)[0]
        row = groups.setdefault(layer, [0.0, 0.0])
        old = np.asarray(old, dtype=np.float64)
        delta = np.asarray(new, dtype=np.float64) - old
        row[0] += float(np.sum(old * old))
        row[1] += float(np.sum(delta * delta))
    return {name: dict(weight_norm=float(np.sqrt(old)), update_norm=float(np.sqrt(delta)),
                      relative_update=float(np.sqrt(delta / old)) if old > 0 else None)
            for name, (old, delta) in groups.items()}


def representation_summary(arrays):
    """Between-trial variation in channel means and RMS, plus tensor activity."""
    import numpy as np
    means, rms, zeros, norms = [], [], [], []
    for value in arrays:
        x = np.asarray(value, dtype=np.float64).reshape(-1, value.shape[-1])
        means.append(x.mean(0))
        rms.append(np.sqrt(np.mean(x*x, axis=0)))
        zeros.append(float(np.mean(np.abs(x) <= 1e-8)))
        norms.append(float(np.sqrt(np.mean(x*x))))
    return dict(between_trial_mean_std=float(np.std(means, axis=0).mean()),
                between_trial_RMS_std=float(np.std(rms, axis=0).mean()),
                near_zero_fraction=float(np.mean(zeros)), tensor_RMS=float(np.mean(norms)))
