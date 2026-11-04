# Verify trial-to-film metadata

```bash
SOURCE_CSV=/absolute/path/to/dreamer_joined.csv sbatch experiments/sic_v16/verify_film_mapping.slurm
```

Defaults to datasets/dreamer_joined.csv. The job streams metadata columns,
checks all 414 subject/trial records for consistent film titles and identifiers,
and reports cross-subject conflicts. Array positions map to sorted CSV IDs.
This does not independently verify arrays or original MATLAB recording order.

If an authoritative reference map is available:

```bash
SOURCE_CSV=/absolute/path/to/dreamer_joined.csv FILM_MAP=/absolute/path/to/dreamer_film_mapping.csv sbatch experiments/sic_v16/verify_film_mapping.slurm
```

The reference CSV needs trial_id,film_title,source_reference. Trial IDs must
match source CSV IDs exactly; no zero/one offset is guessed. Every title needs
an explicit reference, such as an official source table or export manifest.
The default map location is dreamer_film_mapping.csv next to the source CSV.
References are recorded and hashed; their contents are not independently
retrieved or authenticated. Source-title spelling differences are flagged.

Status is source_titles_consistent, reference_supplied_only, inconsistent,
or film_identity_unresolved. Existing exports often contain only trial indices;
in that case the verifier cannot name films. It emits a blank reference template
rather than inventing a mapping. Shared lengths and rating patterns do not
establish film identity. A supplied global reference cannot establish a
subject-specific randomized presentation order without additional metadata.

Send film-mapping-diagnostics.tar.gz from the printed path, including the
report and mapping template. Do not treat consistency alone as independent
confirmation of source authenticity.
