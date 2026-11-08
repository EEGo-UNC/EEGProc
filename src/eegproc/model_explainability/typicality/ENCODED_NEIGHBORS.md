# Independent encoded-space neighbour evaluation

The evaluator measures the Euclidean distance to the **fifth-nearest real
target-class trial embedding**. It uses the classifier's frozen full-trial
`vc_trial_embedding` coordinates, just like the Gaussian analysis. The score
is the fifth distance, not the average of five distances. Smaller distances
mean that five real target-class examples are nearby.

No clustering, new encoder, Gaussian fit, or classifier retraining is involved.
This is an independent score in the existing representation, not an
independently learned representation or a physiological assessment.

## Source-only reference and calibration

Within each outer LOSO fold and each target class:

1. Select real source trials by **true target label only**. Do not filter on
   classifier correctness or the Gaussian typicality criterion.
2. Assign 25% of source participants with target-class trials to cutoff
   calibration, rounded up, using seed 42. Keep the other participants in the
   reference bank. The outer held-out participant is excluded from both sets.
3. Score calibration trials against that fixed bank. Because participants are
   disjoint, neither a trial itself nor another trial from the same participant
   can supply its reference neighbours.
4. Set the cutoff to the empirical **10th percentile** of calibration distances,
   using NumPy's `higher` convention. Accept when distance <= cutoff. This
   selects a tight, locally dense core, not a broad 90%-coverage region.

The finite-sample source calibration fraction can exceed 10%. Each outer fold
has its own bank and cutoff. Absolute distances from different classifiers
are not pooled; reports summarize distance/cutoff and calibration percentiles.
The same procedure applies to class 0 for the reverse direction.

## Before and after optimization

The bank and cutoff are fixed before counterfactual generation. The original
embedding and every arm's optimized embedding are scored against exactly the
same bank. The scorer is invoked only after `optimize()` returns. It never
affects losses, gradients, early stopping, or iterate selection.

Decoded reconstruction and counterfactual EEG are also passed through the
frozen encoder and scored separately. Their results are explicitly named
`decoded_reconstruction` and `decoded_counterfactual`; they must not be
confused with the optimized classification embedding.

Full terminal embedding vectors are retained in source/observation summaries
and per-attempt `classification_embeddings`. Window-level latents, gradients
and signals are not needed for the neighbour report. These small exports also
enable future density evaluators without rerunning the optimizer.

## Run and rebuild

The study runner enables evaluation by default with these settings:

```text
--knn-evaluation --knn-k 5 --knn-quantile 0.10 --knn-calibration-fraction 0.25
```

`--no-knn-evaluation` disables online neighbour scoring while keeping terminal
embedding exports. Gaussian optimization and its quantile are separate from
this evaluation-only metric. Altering k or the neighbour cutoff must be based
on source-only development, not improvements on the outer held-out trials.

Rebuild the independent report without TensorFlow or a model:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.knn_report \
  --roots runs/MY_STUDY/0_to_1 runs/MY_STUDY/1_to_0 \
  --out-dir runs/MY_STUDY/knn_report \
  --k 5 --quantile 0.10 --calibration-fraction 0.25 --seed 42
```

The output includes per-trial scores, before/after rows for each objective,
population summaries, missing/failed/pending counts, and reproducible banks
and calibration vectors. Both directions are reported separately. Missing
vectors cannot be replaced by scalar Gaussian discrepancy scores; older
scalar-only archives require embedding re-export or counterfactual regeneration.

## Interpretation

Compare full-objective counterfactuals with the baseline arms and real target
trials. Distance reductions and inclusion under this unoptimized score assess
support from actual real target-class neighbours. They do not by themselves
demonstrate subject-identity removal, match of the full class distribution,
or physiological realism. A 10th-percentile neighbour region and a
10th-percentile Gaussian-center region generally contain different points.

Reference: Sun et al., *Out-of-Distribution Detection with Deep Nearest
Neighbors*, ICML 2022: https://proceedings.mlr.press/v162/sun22d.html.
The present implementation uses unnormalized frozen trial embeddings with
source-calibrated Euclidean distances; it is an adaptation, not a claim to
reproduce every preprocessing or hyperparameter choice in that paper.
