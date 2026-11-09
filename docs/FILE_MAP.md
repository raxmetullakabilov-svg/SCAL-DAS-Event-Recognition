# Interpretation and file map

**Final B1 partition**: `benchmark/b1/B1R_SAMPLE_ASSIGNMENTS.csv` (column `b1r_role`). The prefix `B1R_` is an immutable internal scientific identifier; it maps to **B1** in the submitted paper. A duplicate exact runtime copy is in `frozen/B1R_SAMPLE_ASSIGNMENTS.csv`.

**Historical Phase-2 partition**: `provenance/phase2_source_metadata/SAMPLE_REGISTRY.csv` (column `b1_role`) is maintained only for sample derivation and archival traceability. Its role labels must NOT be used for manuscript Table 3.

**B3 training**: `benchmark/b3/B3_SPECTRAL_CAMPAIGN_ASSIGNMENTS.csv` carries the role `spectral_campaign_role`; use `train` for source-only training, `audit_only_source_val` only for overlap diagnostics, and `test` only for post-epoch-80 target inference.

**Primary B1 results**: `results/b1/<model>/seed_<seed>/test_event_predictions__primary.csv`. Other saved checkpoint CSVs are strictly sensitivity evidence; do not select the headline checkpoint using test results.

**Primary B3 results**: `results/b3/<model>/seed_<seed>/target_event_predictions.csv` after fixed 80 epochs.

**Run provenance**: `COMPLETED.json` and `TRAINING_COMPLETE.json` for every run; `history.csv` for all epochs. Original per-run model `.pt` checkpoints remain only in the source archive/authorized Drive and are not present in this lightweight release.

**Runtime frozen files**: `frozen/` provides the exact unmodified selected immutable inputs needed by original training packages; it is separate from the public explanatory `docs/`.
