# Phase-2 metadata provenance (not the final B1 partition)

These files are untouched outputs of original Phase-2 signal extraction. Their `SAMPLE_REGISTRY.csv` has a historical `b1_role` from before the raw crop-support overlap audit; these roles are NOT the final paper evaluation split. The final B1 assignments are in `benchmark/b1/B1R_SAMPLE_ASSIGNMENTS.csv` (`b1r_role`). Joining by `sample_id` is safe, and both tables contain the same 2522 sample IDs. Original Phase-2 metadata is included only for raw recording mapping, shard geometry, provenance and extraction regeneration.

Original Phase-2 fingerprint: `c05ef6f99c1f9d235995f3db1f2067b60b988cf4b03a3d905c3c76d5fd826eef`. Metadata file SHA256 hashes match the original Phase-2 file hash manifest.
