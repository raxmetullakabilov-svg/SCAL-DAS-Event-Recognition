B1-R v1.0 — Corrected crop-support-overlap-controlled B1 benchmark

Scientific split SHA-256:
9fab73d477624ff3878ee3ec3cb2e86f5a4aa9bb17ce00e47fedf260d350fce7

Source Phase-1 SHA-256:
d5248a7d5730e7ad6857c5475f248e27cd315d9c5e26235eaba57421936cfe44

Source Phase-2 SHA-256:
c05ef6f99c1f9d235995f3db1f2067b60b988cf4b03a3d905c3c76d5fd826eef

Purpose
-------
B1-R corrects the reviewer-identified raw-input support leakage in the original B1 split without re-extracting signal payloads. It reuses all 2522 frozen Phase-2 patches and changes only benchmark membership metadata.

Core files for retraining
-------------------------
B1R_TRAIN_VAL_TEST.csv
    Event-level B1-R role mapping and group IDs.
B1R_SAMPLE_ASSIGNMENTS.csv
    Sample-level role mapping. Join on sample_id or event_id to the frozen Phase-2 SAMPLE_REGISTRY/shards.
B1R_GROUP_REGISTRY.csv
    Atomic B1-R group composition and assigned role.
B1R_INTEGRITY_GATES.csv
    Mandatory integrity gate results.
B1R_MANIFEST.json / B1R_SHA256.txt
    Frozen split identity and construction contract.

Important benchmark semantics
-----------------------------
- 108 training events, 11 validation events, 18 test events.
- 1920 training patches, 179 validation patches, 423 test patches.
- All 9 classes occur in training and test.
- Validation has 7 classes. construction and running have only two independent B1-R atomic support components each, so three-way independent class coverage is impossible.
- Cross-partition actual raw time x 128-channel support overlap = 0.
- B1-R is NOT recording-held-out. It is a within-deployment, event-instance-level, crop-support-overlap-controlled benchmark.
- Because membership changed, all four B1 models must be retrained. Frozen Phase-2 shards can be reused; raw ~44 GB signal re-extraction is not required.

Construction principle
----------------------
Atomic groups are the transitive union of:
(1) original Phase-1 leakage groups; and
(2) event-level edges created when any pair of extracted patches from the same recording overlaps in both raw temporal and spatial support.

After hard integrity and class-coverage constraints, the deterministic assignment minimizes the number of events moved relative to frozen B1, then class event-count deviation, then class patch-count deviation.

The minimum correction changes 15 of 137 event roles.
