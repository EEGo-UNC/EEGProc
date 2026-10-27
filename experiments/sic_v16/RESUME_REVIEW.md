# Resumed baseline review, October 27

Archive: `sicv16-diagnostics-20261027-153447.tar.gz` (5,060 files).

All three resumed jobs completed. Their GPU optimizer-restoration tests passed,
and actual run sidecars and logs record:

| Variant | Checkpoint restored | Optimizer iterations restored | First resumed epoch | Finished subjects skipped |
| --- | --- | ---: | ---: | ---: |
| CNN | subject 20, epoch 30 | 10,800 | 31 | 20 |
| CNN + Transformer | subject 14, epoch 47 | 16,920 | 48 | 14 |
| CNN + Transformer + reconstruction | subject 14, epoch 43 | 15,480 | 44 | 14 |

These step counts equal saved epochs times the original 360 training steps per
epoch. Resume paths point into `/work/users/p/m/pmucharl/sic-v16/`. The earlier
3,670 full-run epoch records (including the already completed reconstruction
variant) are byte-for-byte unchanged compared with the pre-resume archive.
Each variant now has all 1,150 epoch records (23 subjects x 50 epochs); execution
reports have no failures or interruption. No further recovery run is required.
The diagnostics do not contain model bytes, so this review relies on the
on-cluster checkpoint checks, optimizer tests and recorded resume evidence.

## Completed original all-23 results

Validation-selected held-out metrics, averaged over all 23 subjects:

| Variant | Accuracy | Balanced accuracy | ROC AUC | Oracle balanced accuracy | Selected folds with one predicted class |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.5966 | 0.5000 | 0.4969 | 0.5000 | 23/23 |
| CNN + reconstruction | 0.6159 | 0.5000 | 0.4864 | 0.5000 | 23/23 |
| CNN + Transformer | 0.5773 | 0.4959 | 0.5168 | 0.5000 | 21/23 |
| CNN + Transformer + reconstruction | 0.5483 | 0.5000 | 0.4674 | 0.5000 | 23/23 |

Recovery and storage are working. Learning quality remains poor: the models
mostly predict one class, and accuracy is influenced by class proportions.
These results do not establish a useful architecture or a reconstruction
benefit. Training/representation diagnosis should be distinguished from
infrastructure failures; thresholds or architecture are not silently changed
in this update.

## New requested scope

New experiments use only original subject IDs 0, 1, 2 and 3 for the whole data
pool, including training/validation. Other subjects are excluded before filtering
and windowing. With subject-separated LOSO and one validation subject per fold,
each model trains on two subjects (36 trials), validates on one (18 trials), and
tests on one (18 trials). This is a smaller debugging protocol, not a directly
comparable replacement score for the original 23-subject protocol.

The tracker defaults to the new `first4` scope and keeps separate stores. Old
all-23 results are accessible with `--scope all23`. The next required cluster
run is a short tracked smoke of the new selection path. It also tests exclusion
of poisoned out-of-pool subjects, original subject-ID preservation, and the
two-train/one-validation/one-test split. Model training has not run locally.
