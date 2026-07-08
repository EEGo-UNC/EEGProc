"""Publication-oriented figures from cross-validation result tables.

Each function takes one or more tidy DataFrames (see
:mod:`eegproc.plotting.results_io`) and draws a single panel. Functions accept an
optional ``ax``:

* ``ax=None`` -> a new standalone figure is created and returned (handy in
  notebooks / quick scripts);
* ``ax=<Axes>`` -> the panel is drawn into the given axes, so
  :mod:`eegproc.plotting.report` can compose several panels into one
  multi-panel figure.

Only matplotlib + scikit-learn are used (no extra dependencies). Confusion
matrices, per-subject accuracy bars, hyperparameter sweeps, calibration and
uncertainty are the building blocks reused across reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------

def _resolve_ax(ax, projection: str | None = None):
    """Return ``(fig, ax)``, creating a new figure when ``ax`` is None."""
    if ax is not None:
        return ax.figure, ax

    if projection == "3d":
        fig = plt.figure(figsize=(7, 6))
        return fig, fig.add_subplot(111, projection="3d")

    fig, ax = plt.subplots(figsize=(7, 5))
    return fig, ax


def _filter_model(df: pd.DataFrame, model: str | None) -> pd.DataFrame:
    """Restrict a frame to one model, defaulting to the only model present."""
    if df.empty or "model" not in df.columns:
        return df

    models = list(df["model"].unique())

    if model is not None:
        return df[df["model"] == model]

    if len(models) == 1:
        return df

    raise ValueError(
        f"Multiple models present {models}; pass model=... to select one."
    )


def _class_names(labels: np.ndarray, class_names: list[str] | None) -> list[str]:
    """Resolve display names for the observed integer classes."""
    classes = np.unique(labels)

    if class_names is None:
        return [str(c) for c in classes]

    if len(class_names) != len(classes):
        raise ValueError(
            f"class_names has {len(class_names)} entries but data has "
            f"{len(classes)} classes {classes.tolist()}."
        )

    return list(class_names)


# ---------------------------------------------------------------------
# Headline panels
# ---------------------------------------------------------------------

def plot_confusion_matrix(
    predictions: pd.DataFrame,
    model: str | None = None,
    normalize: bool = True,
    class_names: list[str] | None = None,
    ax=None,
):
    """Confusion matrix over all pooled test predictions.

    Pools ``y_true``/``y_pred`` across every outer fold (each test window appears
    once), which is the standard way LOSO emotion papers report a single
    confusion matrix.
    """
    predictions = _filter_model(predictions, model)

    if predictions.empty:
        raise ValueError("No predictions to plot a confusion matrix from.")

    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    names = _class_names(labels, class_names)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
    )

    # Fixed 0..vmax color scale (vmax=1 when normalized) so the panel reads the
    # same across runs and a uniform matrix is still clearly colored -- unlike
    # ConfusionMatrixDisplay, which auto-scales to the data and can wash out.
    vmax = 1.0 if normalize else float(matrix.max() or 1)

    fig, ax = _resolve_ax(ax)
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value):d}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > 0.5 * vmax else "black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    return fig


def plot_metric_summary(
    fold_metrics: pd.DataFrame,
    model: str | None = None,
    metrics: tuple[str, ...] = ("accuracy", "f1", "precision", "recall"),
    ax=None,
):
    """Mean +/- std bars per metric, with individual outer-fold points overlaid.

    Shows both the headline number (bar height = mean across folds, error bar =
    std) and the fold-to-fold spread (scatter), which matters for LOSO where
    variance across subjects is large.
    """
    fold_metrics = _filter_model(fold_metrics, model)

    if fold_metrics.empty:
        raise ValueError("No fold metrics to summarize.")

    metrics = tuple(m for m in metrics if m in fold_metrics.columns)
    means = [fold_metrics[m].mean() for m in metrics]
    stds = [fold_metrics[m].std(ddof=0) for m in metrics]

    fig, ax = _resolve_ax(ax)
    x = np.arange(len(metrics))

    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.85)

    for i, metric in enumerate(metrics):
        values = fold_metrics[metric].to_numpy()
        jitter = (np.random.RandomState(0).rand(len(values)) - 0.5) * 0.25
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            color="#1f1f1f",
            s=18,
            alpha=0.6,
            zorder=3,
        )
        ax.text(i, means[i] + 0.02, f"{means[i]:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean across outer folds)")
    ax.set_title("Outer-fold performance")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    return fig


# ---------------------------------------------------------------------
# Subject / task breakdown
# ---------------------------------------------------------------------

def plot_per_subject_accuracy(
    user_metrics: pd.DataFrame,
    model: str | None = None,
    metric: str = "accuracy",
    subject_labels: dict | None = None,
    ax=None,
):
    """Sorted per-subject score bars with the cohort mean drawn as a line."""
    user_metrics = _filter_model(user_metrics, model)

    if user_metrics.empty:
        raise ValueError("No per-user metrics to plot.")

    per_subject = (
        user_metrics.groupby("subject_id")[metric].mean().sort_values(ascending=False)
    )

    if subject_labels:
        index = [subject_labels.get(str(s), s) for s in per_subject.index]
    else:
        index = per_subject.index.tolist()

    fig, ax = _resolve_ax(ax)
    x = np.arange(len(per_subject))
    ax.bar(x, per_subject.to_numpy(), color="#55A868", alpha=0.85)

    mean_value = float(per_subject.mean())
    ax.axhline(
        mean_value,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"mean = {mean_value:.2f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(index, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Subject")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Per-subject {metric}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    return fig


def plot_subject_task_heatmap(
    predictions: pd.DataFrame,
    model: str | None = None,
    ax=None,
):
    """Accuracy heatmap with subjects on one axis and tasks on the other.

    Requires a ``task_id`` column in ``predictions`` (logged by ``nested_lnso_cv``
    when ``task_id_array`` is supplied). Raises if tasks were not logged so the
    report can skip this panel gracefully.
    """
    predictions = _filter_model(predictions, model)

    if predictions.empty:
        raise ValueError("No predictions to plot.")

    if "task_id" not in predictions.columns or predictions["task_id"].isna().all():
        raise ValueError(
            "predictions has no usable 'task_id' values; re-run cross-validation with "
            "task_id_array to enable the per-subject/per-task heatmap."
        )
    correct = (predictions["y_true"] == predictions["y_pred"]).astype(float)
    table = predictions.assign(correct=correct)
    # Prefer the real subject id (subject_label) on the axis when available so it
    # matches the per-subject bar chart; fall back to the integer code.
    subject_axis = "subject_label" if "subject_label" in table.columns else "subject_id"
    matrix = table.pivot_table(
        index=subject_axis,
        columns="task_id",
        values="correct",
        aggfunc="mean",
    )

    fig, ax = _resolve_ax(ax)
    image = ax.imshow(matrix.to_numpy(), cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_xlabel("Task")
    ax.set_ylabel("Subject")
    ax.set_title("Per-subject / per-task accuracy")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    return fig


# ---------------------------------------------------------------------
# Hyperparameter analysis (programmer-selectable axes)
# ---------------------------------------------------------------------

def plot_hyperparameter_sweep(
    inner_cv: pd.DataFrame,
    params: str | list[str],
    metric: str = "accuracy",
    model: str | None = None,
    surface: bool = False,
    ax=None,
):
    """Plot one or two hyperparameters against an inner-CV validation metric.

    The score is averaged over outer folds (and any hyperparameters not in
    ``params``) for each grid point. Using inner-CV validation scores keeps the
    held-out test set untouched by hyperparameter analysis.

    * 1 hyperparameter -> line plot.
    * 2 hyperparameters -> heatmap, or a 3D surface when ``surface=True``.
    """
    inner_cv = _filter_model(inner_cv, model)

    if inner_cv.empty:
        raise ValueError("No inner-CV results to plot a hyperparameter sweep from.")

    params = [params] if isinstance(params, str) else list(params)

    if len(params) not in (1, 2):
        raise ValueError("params must name 1 or 2 hyperparameters.")

    missing = [p for p in [*params, metric] if p not in inner_cv.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not found. Available: {list(inner_cv.columns)}"
        )

    grouped = inner_cv.groupby(params, as_index=False)[metric].mean()

    if len(params) == 1:
        return _sweep_line(grouped, params[0], metric, ax)

    if surface:
        return _sweep_surface(grouped, params, metric, ax)

    return _sweep_heatmap(grouped, params, metric, ax)


def _sweep_line(grouped: pd.DataFrame, param: str, metric: str, ax):
    grouped = grouped.sort_values(param)
    fig, ax = _resolve_ax(ax)
    ax.plot(grouped[param], grouped[metric], marker="o", color="#4C72B0")
    ax.set_xlabel(param)
    ax.set_ylabel(f"mean inner-CV {metric}")
    ax.set_title(f"{metric} vs {param}")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    return fig


def _sweep_heatmap(grouped: pd.DataFrame, params: list[str], metric: str, ax):
    pivot = grouped.pivot(index=params[1], columns=params[0], values=metric)
    fig, ax = _resolve_ax(ax)
    image = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto", origin="lower")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{c:g}" if isinstance(c, (int, float)) else c for c in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{r:g}" if isinstance(r, (int, float)) else r for r in pivot.index])
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    ax.set_title(f"mean inner-CV {metric}")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="w", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=metric)
    return fig


def _sweep_surface(grouped: pd.DataFrame, params: list[str], metric: str, ax):
    pivot = grouped.pivot(index=params[1], columns=params[0], values=metric)
    fig, ax = _resolve_ax(ax, projection="3d")

    try:
        x_vals = np.asarray(pd.to_numeric(pivot.columns, errors="raise"), dtype=float)
        y_vals = np.asarray(pd.to_numeric(pivot.index, errors="raise"), dtype=float)
    except Exception as exc:
        raise ValueError(
            "surface=True requires numeric hyperparameter values on both axes."
        ) from exc

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    ax.plot_surface(x_grid, y_grid, pivot.to_numpy(), cmap="viridis", edgecolor="none")
    ax.set_ylabel(params[1])
    ax.set_zlabel(f"mean inner-CV {metric}")
    ax.set_title(f"{metric} surface")
    return fig


# ---------------------------------------------------------------------
# Reliability / uncertainty (variational models)
# ---------------------------------------------------------------------

def plot_reliability(
    predictions: pd.DataFrame,
    model: str | None = None,
    n_bins: int = 10,
    ax=None,
):
    """Confidence-reliability diagram: mean predicted confidence vs accuracy.

    Bins test windows by the predicted-class probability ``p_pred`` and plots
    the empirical accuracy in each bin against the mean confidence. A perfectly
    calibrated model lies on the diagonal.
    """
    predictions = _filter_model(predictions, model)

    if predictions.empty or "p_pred" not in predictions.columns:
        raise ValueError("predictions lack 'p_pred'; cannot build a reliability plot.")

    confidence = predictions["p_pred"].to_numpy()
    correct = (predictions["y_true"] == predictions["y_pred"]).to_numpy().astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)

    mean_conf, mean_acc = [], []
    for b in range(n_bins):
        mask = bin_index == b
        if mask.any():
            mean_conf.append(confidence[mask].mean())
            mean_acc.append(correct[mask].mean())

    fig, ax = _resolve_ax(ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="ideal")
    ax.plot(mean_conf, mean_acc, marker="o", color="#8172B3", label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    return fig


def plot_uncertainty(
    intervals: pd.DataFrame,
    model: str | None = None,
    ax=None,
):
    """Histogram of predictive credible-interval widths per test window."""
    intervals = _filter_model(intervals, model)

    needed = {"p_pred_ci_low", "p_pred_ci_high"}
    if intervals.empty or not needed.issubset(intervals.columns):
        raise ValueError(
            "intervals lack CI columns; run with log_variational_intervals=True."
        )

    width = (intervals["p_pred_ci_high"] - intervals["p_pred_ci_low"]).to_numpy()

    fig, ax = _resolve_ax(ax)
    ax.hist(width, bins=20, color="#937860", alpha=0.85)
    ci_level = float(intervals["ci_level"].iloc[0]) if "ci_level" in intervals else None
    title = "Predictive interval width"
    if ci_level is not None:
        title += f" ({ci_level:.0%} CI)"
    ax.set_xlabel("CI width (predicted class)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    return fig
