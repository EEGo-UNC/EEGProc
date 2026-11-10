# Table 2 displacement audit: remaining work

Status checked on 2026-11-09, after integrating the completed counterfactual
features into `CFX-update`. This is a work list, not a completed new analysis.

## Already available

The saved original DREAMER suites contain 596 completed objective endpoints:
228 arousal (57 trials x 4 objectives) and 368 valence (92 trials x 4
objectives). Their per-attempt `result.json` files already contain:

- `d_z`: RMSE between original and optimized full-trial classification
  embeddings, as defined in each saved study manifest.
- `decoder_latent_rmse`: displacement in the latent coordinates optimized
  by the decoder objective. This is distinct from `d_z`.
- `decoded_trials.<report_output>.original_reconstruction_mse`.
- `decoded_trials.<report_output>.decoded_change_mse`.
- `decoded_trials.<report_output>.counterfactual_to_original_mse`.

The existing CSV report exposes `d_z`, `e_rec`, and `delta_dec` with sample
counts, medians, and quartiles. The earlier calculation audit verified the
Table 2 rounded summaries and the decoded MSE-to-RMSE conversions. Thus the
basic historical distances are available without regenerating signals.

The merged code additionally supports an optional target-free physiological
repair baseline. It exports `delta_dec_repair`, `delta_dec_explanation`, their
envelope analogues, and repair feasibility/prediction preservation. These
measurements require new runs: none of the 596 original endpoints has a
`repair_baseline` result. New completed arms are also re-encoded for separate
decoded prediction and typicality checks in both target directions.

## Remaining analysis using existing results

1. **Create one verified, paired displacement audit export.** Keep task,
   direction, subject, trial, objective, selected attempt, normalization and
   distance definitions attached. Include both latent distances and all
   three signal RMSEs:

   - `e_rec = sqrt(original_reconstruction_mse) = RMSE(R(z), x)`;
   - `delta_dec = sqrt(decoded_change_mse) = RMSE(R(z_cf), R(z))`;
   - `e_cf = sqrt(counterfactual_to_original_mse) = RMSE(R(z_cf), x)`.

   The reporting module still needs to expose `decoder_latent_rmse` and
   `e_cf`; both scalars are already available in the original result files.
   Verify hashes and unique trial/arm identities before combining outputs.

2. **Complete the distribution comparison.** Compare latent displacement
   against decoded displacement across paired trials and objectives, by
   task and held-out subject. Show distributions or paired plots in addition
   to medians/IQRs, including tails and small nonzero values rounded to 0.00.
   Describe the different latent coordinate systems and units; their raw
   magnitudes are not interchangeable. Retain all finite unsuccessful
   endpoints in new studies and report how many endpoints are unavailable.

3. **Quantify reconstruction versus counterfactual change.** Analyze
   `e_rec`, `delta_dec`, and `e_cf` together, showing the size of the edit
   relative to the pre-existing reconstruction error and whether `e_cf`
   increases or decreases. Check the RMSE triangle bounds on the same
   arrays/normalization. These distances are not additive and are not a
   causal decomposition. Reconcile the case-study MSE/RMSE labels separately
   before reusing its numerical example.

## Remaining experiments and reporting

4. **Run the matched repair-only control and interpret it.** Freeze the
   source-selected settings, checkpoint, initialization, optimization
   budget, normalization, and physiology definition. Compare each arm's
   change from the ordinary reconstruction with its change from the
   repaired reconstruction. Report repair failures and prediction changes;
   a completed repair need not be feasible or preserve the prediction.
   A first feasible endpoint is not a proof of a globally minimal repair.
   Existing historical scalars cannot supply these unmeasured outputs.

5. **Produce the final audit table/figures and manuscript corrections.**
   State metric definitions, precision, cohort sizes, eligible/finite
   denominators, exclusions, and the separate latent and decoded validity
   outcomes. Keep historical results separate from new 10th-percentile,
   bidirectional or differently regularized studies. Distinguish evidence
   that distances were calculated correctly from evidence that decoded
   changes explain the classifier's decision.

See [the integrated run protocol](CFX_COVERAGE.md). The optional repair
control and the audit exporter/comparisons are separate steps; merging the
code does not complete the experimental audit.
