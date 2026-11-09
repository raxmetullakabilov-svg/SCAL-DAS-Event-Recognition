# Provenance and source integrity

The artifacts here were curated from the authors' `JXU.ZIP` and original B1/B3 audit exports. No scientific outcome or training log was edited during this packaging. Newly authored wrappers/README are **not** claimed to have the same SHA256 as the original frozen protocol.

| Artifact | Original scientific fingerprint (SHA256) |
| --- | --- |
| Phase-1 event benchmark | `d5248a7d5730e7ad6857c5475f248e27cd315d9c5e26235eaba57421936cfe44` |
| Phase-2 signal-extraction package | `c05ef6f99c1f9d235995f3db1f2067b60b988cf4b03a3d905c3c76d5fd826eef` |
| Corrected B1 event/patch assignment (historical B1R_v1.0 ID) | `9fab73d477624ff3878ee3ec3cb2e86f5a4aa9bb17ce00e47fedf260d350fce7` |
| B1 training protocol | `1ea5a907f4e0f7d5bc87d43bd82f649089d5350329afb0133d3de86f1ca33dab` |
| B3 spectral-transfer protocol | `2db80f1e58721fc1bab060df59d2232597607b600e98d367573d120f19a71172` |

`frozen/` contains **byte-identical** B1/B3 runtime input files from the original audit ZIPs, including `LOCATION_LABELS_20.csv`, so that the frozen training source can refer to its actual file names. The `code/b1r_campaign` and `code/b3_spectral_campaign` training sources are preserved as originally exported. The original research B1 split builder is preserved at `provenance/legacy_code/build_b1r_v1_0_reference.py`; its reparameterized convenience adaptation is in `scripts/` and must not be mistaken for the frozen original.

Some archival scientific comments may contain historical internal QA/retraining or review terminology. These are **preserved, not retroactively rewritten**, to maintain authenticity; they should not be interpreted as evidence of previous submission to a journal. The manuscript-facing benchmark is **B1**, while the implementation retains its frozen `B1R_` IDs. The original Phase-2 registry's `b1_role` is an **earlier** split and not the published B1 evaluation membership.

The repository-wide `CHECKSUMS_SHA256.txt` is a **new release manifest**, separate from the scientific protocol fingerprints above. Regenerating that release manifest does not change original experimental hashes. The included SHA verifier verifies every listed repository file except the manifest itself, and refuses unexpected unlisted files.

**Notebook packaging exception (public release only):** The B1 notebook’s in-cell base64 ZIP payload was recompressed to discard seven CPython `__pycache__/*.pyc` files. All 13 remaining embedded files—including all seven original training `.py` modules and six frozen scientific inputs—were verified byte-identical to the original notebook payload; the notebook’s operational Python statements and original scientific source files remain unchanged. Thus the *notebook file SHA256* differs from the archival source, although its scientific payload and configuration are preserved.

## Independent B1 split regeneration (release audit)

The path-parameterized B1 split builder was executed during release QA using the included original Phase-2 metadata files (without HDF5 payload reads), Python 3.13, pandas 2.2.3, SciPy 1.17.0 and NumPy 2.3.5. It reproduced the same 70 groups, 100 event-level raw-support edges, 15 changed events, and the original B1 fingerprint `9fab73d477624ff3878ee3ec3cb2e86f5a4aa9bb17ce00e47fedf260d350fce7`. All 12 regenerated named B1 builder output files were byte-for-byte identical to their frozen counterparts in `benchmark/b1/`. This is **split regeneration**, not GPU training or original HDF5 extraction reproduction.
