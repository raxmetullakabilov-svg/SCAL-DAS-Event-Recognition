# Frequency-domain and spatial-confounding-aware DAS event recognition

**Research artifacts for the manuscript** *Frequency-Domain and Spatial-Confounding-Aware Deep Learning for Event Recognition in Distributed Acoustic Sensing Using Telecommunication Fiber*.

This repository contains benchmark metadata, the original training and preprocessing source, Colab notebooks, frozen protocol fingerprints, and **stored** B1/B3 experimental outputs. It is a curated **core reproducibility release**, not a self-contained copy of the raw dataset or a claim that training was rerun during curation.

## Scope and results

| Benchmark | Model | Event-level macro-F1 (three-seed mean) |
| --- | --- | ---: |
| B1 | Raw TS-ResNet | 0.2575 |
| B1 | Raw SCAL-DAS | 0.0916 |
| B1 | Spectrum-RDFT | 0.5198 |
| B1 | SCAL-RDFT | 0.5917 |
| B3 (F4 → F1) | Spectrum-RDFT | 0.7690 |
| B3 (F4 → F1) | SCAL-RDFT | 0.7460 |

**B1:** 137 annotation-derived events and 2,522 patches; 108/11/18 train/validation/test events and 1,920/179/423 patches; 70 atomic event groups, no raw crop-support cross-partition overlap. Four models, seeds **2026, 2027, 2028**, 60 training epochs. The headline checkpoint is selected using the **minimum nine-class validation event-level NLL**, not test performance. B1 is a within-deployment split; some recordings are shared between partitions. The results do **not** establish recording-independent generalization.

**B3:** Target-blind two-class (**car** and **construction**) F4 → F1 spatial-transfer stress test. The 15-event source-train subset alone trains each spectral model for a **fixed 80 epochs**; 4 source-validation events were excluded from training/checkpoint selection after 30 source-train/source-validation raw-support overlap pairs were found. Fourteen independent target events are evaluated after epoch 80. Source-to-target raw-support overlap is zero; B3 is **not** independent-site generalization. A larger macro-F1 for SCAL-RDFT on B1 does not imply an advantage on B3.

## Quick independent check (no GPU or dependencies)

Run from the repository root:

```bash
python scripts/verify_release.py
python scripts/verify_release.py --json
```

GitHub Actions also runs this same stdlib-only audit on pushes and pull requests (`.github/workflows/verify-stored-results.yml`).

The portable B1 split builder was also run independently during package QA; its 12 generated output files were byte-identical to the original frozen B1 split outputs (see `docs/PROVENANCE.md`).

This recomputes B1/B3 event-level accuracy, balanced accuracy, macro-F1, NLL and Brier scores from the **stored prediction CSVs**, checks 60/80-epoch histories, verifies minimum-NLL B1 checkpoint selection, checks final sample/event partitions and actual-support overlaps, and validates SHA256 for every included file. It does **not** retrain or load model checkpoints.

## Repository layout

| Directory | Contents |
| --- | --- |
| `benchmark/b1/` | Final B1 event and patch assignments, provenance and overlap audit (historical file prefix `B1R_`). |
| `benchmark/b3/` | Fixed B3 assignments and source/target overlap report. |
| `frozen/` | Original B1 and B3 training runtime inputs, restored byte-for-byte from campaign audit archives. |
| `provenance/phase2_source_metadata/` | Phase-2 sample/recording mappings and original source metadata hashes; **do not** use its historical `b1_role` as the final B1 split. |
| `code/` | Original Phase-1/2 and B1/B3 model/training source. |
| `notebooks/` | Four research notebooks, with execution output omitted. The B1 embedded ZIP was repacked to strip CPython bytecode only; source `.py` files remain byte-identical. Includes B1 `...FIXED.ipynb`. |
| `results/b1/`, `results/b3/` | Original per-seed event predictions, histories, completion records and compact summaries. |
| `analysis/` | Saved auxiliary statistical analysis tables. |
| `scripts/` | Offline result verifier, explicit-path training wrapper, and noncanonical path-port of B1 builder. |
| `provenance/legacy_code/` | Immutable original B1 builder for historical code comparison. |
| `docs/` | Reproduction steps, provenance, limitations and rights notes. |

## Source data and research code

Original public DAS dataset: **Tomasov et al.**, *Scientific Data* (2025), https://doi.org/10.1038/s41597-025-05088-4 . Please retrieve and cite the dataset from its official distribution page, observing its redistribution conditions.

The experiment used **16 Phase-2 HDF5 shards** (original archive approximately **12.3 GB**), which are **not** distributed here. Binary PyTorch model checkpoints and large patch-level embedding tables are also omitted. The original training and notebooks assume the original directory topology. For exact dataset staging and GPU training, read [docs/REPRODUCTION.md](docs/REPRODUCTION.md); for exact source fingerprints, read [docs/PROVENANCE.md](docs/PROVENANCE.md); for release constraints, read [docs/RIGHTS_AND_LIMITATIONS.md](docs/RIGHTS_AND_LIMITATIONS.md).

**Publication and licensing:** This release does not itself grant a software license for authors' code; the rights holders must approve release terms. The source dataset, associated paper and any third-party packages may have different licenses. The GitHub URL must be publicly accessible before a manuscript can truthfully state that code/results are publicly available there.
