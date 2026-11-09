# Reproduction guide — precise scope and prerequisites

## 1. Verify the stored evidence (no original signal data needed)

```bash
python scripts/verify_release.py
```

This is the recommended starting point for numerical verification of the results recorded in the manuscript. It checks metadata integrity, overlap counts, primary B1 validation NLL checkpoint epochs, and all 18 run prediction CSVs without requiring NumPy, PyTorch or a GPU.

## 2. Access the underlying signals

The original experimental process requires (a) the published DAS recordings, (b) the original annotation-derived event registry and/or Phase-1 extraction inputs, (c) the **exact** derived Phase-2 HDF5 shards and the six original Phase-2 metadata files, or verified regeneration of those shards. The 16 HDF5 shards are **not** provided in this repository. The official dataset DOI is https://doi.org/10.1038/s41597-025-05088-4 .

The expected Phase-2 disk layout is:

```text
Phase2_Compact_v0.1/
  PHASE2_SHA256.txt
  metadata/
    SAMPLE_REGISTRY.csv
    EVENT_TO_SAMPLE_MAP.csv
    EXCLUSIONS.csv
    EXTRACTION_STATS.csv
    RAW_RECORDING_MAP.csv
    RECORDING_INVENTORY.csv
  shards/
    <16 original .h5 files under the exact relative names specified in SAMPLE_REGISTRY.csv>
```

If authorized to use the authors' original Drive, the `Phase2_Compact_v0.1` folder contains the frozen shards and metadata. **Do not publish a private Drive folder link as a public data source.** The public `provenance/phase2_source_metadata/` is a reference copy of the six Phase-2 metadata files, but lacks the shards. `PHASE2_MANIFEST.json` lists original binary-file hashes; the 16 shard bytes were not freshly re-hashed in this release audit.

Crucial: `SAMPLE_REGISTRY.csv` has **historical B1 roles** and must not be used to evaluate the manuscript's B1 results. The final assignment file is `benchmark/b1/B1R_SAMPLE_ASSIGNMENTS.csv`, or the same immutable input in `frozen/`.

## 3. Software environment (experimental, not rerun-validated)

The archived B1 environment reports Python 3.13.16, PyTorch `2.11.0+cu130`, CUDA 13.0, NVIDIA L4, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 and h5py 3.14.0. See `requirements-reproduce.txt`. Use an appropriate CUDA-enabled PyTorch build from the vendor's documented installation route. Reproducing exactly may require the original Colab GPU image and external sources.

```bash
pip install -r requirements-reproduce.txt
# Install CUDA-compatible PyTorch separately.
python scripts/run_campaign.py --campaign b1 --mode validate --phase2-root /PATH/TO/Phase2_Compact_v0.1
python scripts/run_campaign.py --campaign b3 --mode validate --phase2-root /PATH/TO/Phase2_Compact_v0.1
```

`validate` performs the data/split gates through the original model source modules without training. These entrypoints are path adapters around the **unchanged scientific code**; they have not been executed with the omitted HDF5 shards in this release environment. They require third-party Python packages and the original HDF5 folder. No actual GPU retraining occurred during curation.

## 4. Optional GPU runs (separate output directories)

Do not run against the original experiment output directories. Choose newly created, empty output locations with adequate available disk and GPU memory.

```bash
python scripts/run_campaign.py --campaign b1 --mode smoke --phase2-root /PATH/TO/Phase2_Compact_v0.1 --output /NEW/OUTPUT/B1_SMOKE
python scripts/run_campaign.py --campaign b1 --mode full --phase2-root /PATH/TO/Phase2_Compact_v0.1 --output /NEW/OUTPUT/B1_FULL
python scripts/run_campaign.py --campaign b3 --mode smoke --phase2-root /PATH/TO/Phase2_Compact_v0.1 --output /NEW/OUTPUT/B3_SMOKE
python scripts/run_campaign.py --campaign b3 --mode full --phase2-root /PATH/TO/Phase2_Compact_v0.1 --output /NEW/OUTPUT/B3_FULL
```

**These commands have not been end-to-end tested here.** `--mode full` triggers all 12 B1 (4 models × 3 seeds; 60 epochs) or all 6 B3 (2 models × 3 seeds; 80 epochs) runs and requires GPU. Original notebooks instead assume a Google Drive workspace rooted at `/content/drive/MyDrive/Project_Paper01_SCAL_DAS`; avoid treating that absolute path as a universally portable location. Before execution, inspect compute/disk capacity and preserve the original frozen experiment files.

## 5. Rebuild the corrected B1 split, if needed

The archive-preserved original B1 MILP script had hardcoded research sandbox paths. An explicit-path adaptation, **not a new scientific model**, is provided at `scripts/build_b1_split_portable.py`. It retains the original group construction and optimizer logic, but asks for the Phase-2 root and an empty output directory. During release QA, this path-parameterized builder reproduced all 12 original frozen B1 builder outputs byte-for-byte (including the original B1 SHA256) using available Phase-2 metadata. Other solver versions or environments must still be audited against frozen hashes rather than assumed equivalent.

```bash
python scripts/build_b1_split_portable.py --phase2-root /PATH/TO/Phase2_Compact_v0.1 --output /NEW/EMPTY/B1_SPLIT
```

## 6. Verification boundaries

- Confirmed: stored split/event metadata, actual-support overlap counts, frozen SHA256 of included files, all 18 saved prediction records and main aggregate metrics, recorded 60/80 epoch lengths, B1 primary checkpoint selection from stored validation history.
- Not confirmed here: downloading the original dataset, rerunning Phase-1/2 extraction, original HDF5 shard byte hashes, PyTorch checkpoint loading/inference, fully matching environment on a fresh GPU, reproducing large embedding-based probes from this small repository alone.

**Important:** Never use B1 primary results to infer external-recording generalization or B3 target outcomes to infer independent-site generalization. Do not treat overlapping historical B3 source validation as a valid checkpoint selector.
