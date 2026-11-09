# B1-R Training Protocol v1.0 — Frozen Before Retraining

**Status:** FROZEN  
**Benchmark:** B1R_v1.0  
**B1-R SHA-256:** `9fab73d477624ff3878ee3ec3cb2e86f5a4aa9bb17ce00e47fedf260d350fce7`  
**Source Phase-1 SHA-256:** `d5248a7d5730e7ad6857c5475f248e27cd315d9c5e26235eaba57421936cfe44`  
**Source Phase-2 SHA-256:** `c05ef6f99c1f9d235995f3db1f2067b60b988cf4b03a3d905c3c76d5fd826eef`

## 1. Purpose

This protocol governs reviewer-facing retraining after correction of cross-partition raw-support overlap. It was frozen before any B1-R model retraining or B1-R test evaluation. It reuses the frozen Phase-2 signal shards read-only and changes only B1 benchmark membership according to B1-R.

B1-R is a **within-deployment, event-instance-level, crop-support-overlap-controlled evaluation**. It is not recording-held-out or deployment-independent.

## 2. Frozen split

- Train: **108 events / 1920 patches**
- Validation: **11 events / 179 patches**
- Test: **18 events / 423 patches**
- All nine classes occur in train and test.
- Validation contains seven classes. `construction` and `running` are structurally absent.
- Cross-partition actual raw-support overlap: **0**.

## 3. Models

The four pre-existing model definitions are retrained without architecture search:

1. Raw TS-ResNet
2. Raw SCAL-DAS
3. Spectrum-RDFT
4. SCAL-RDFT

No new model variant may be introduced into the main corrected comparison.

## 4. Common training settings

- Seeds: **2026, 2027, 2028**
- Epochs: **60 fixed epochs**
- Early stopping: **disabled**
- Batch size: **8**
- Optimizer: **AdamW**
- Learning rate: **3 × 10^-4**
- Weight decay: **1 × 10^-4**
- Gradient clipping: **5.0**
- Mixed precision: enabled when CUDA is available
- Embedding dimension: **192**
- Sampler: class-event-balanced

The class-event-balanced sampler assigns expected equal mass to each class, then each train event within class, then each patch within event.

## 5. Checkpoint selection

The previous validation macro-F1 rule is retired for B1-R because two classes are absent from validation. The primary checkpoint is the epoch with minimum **event-level 9-way validation NLL**. All models train the full 60 epochs; validation never controls training duration. Ties within `1e-8` select the earlier epoch.

Predeclared sensitivity checkpoints are (i) best observed-seven-class validation macro-F1 and (ii) epoch 60. They cannot replace the primary checkpoint after test inspection.

## 6. Representation-specific settings

### Raw models

Raw TS-ResNet and raw SCAL-DAS use the frozen sample-wise robust-normalized 8192 × 128 input. SCAL-DAS retains the frozen class-conditioned location adversary with GRL λ_max=0.15, γ=10, SmoothL1 location loss weight=1.0.

### Spectral models

Spectrum-RDFT and SCAL-RDFT use the frozen raw patch and the existing spectral pipeline: Hann (`periodic=False`) → one-sided RFFT with redundancy 3 (`n_fft=24576`) → magnitude clamp `1e-6` → `log10` magnitude → per-channel spectral mean subtraction → first 2048 bins. SCAL-RDFT retains the same frozen adversarial settings.

## 7. Initialization-equivalence gates

For every seed:

- Raw TS-ResNet and raw SCAL-DAS must have identical initial backbone and event-head weights before the location head is added.
- Spectrum-RDFT and SCAL-RDFT must have identical initial backbone and event-head weights; the location head is the only additional randomly initialized module.

Failure of either gate invalidates that seed run.

## 8. Test isolation

B1-R test rows may not be used for:

- training;
- epoch count;
- checkpoint selection;
- hyperparameter selection;
- architecture changes;
- sampler changes;
- RDFT/GRL tuning.

The test set is loaded for evaluation only after the 60-epoch run is complete and the primary NLL checkpoint has been frozen.

## 9. Primary evaluation

Patch probabilities are averaged within each annotation-derived event instance. Primary metrics are event-level accuracy, balanced accuracy, macro-F1, NLL, Brier score, and ECE-5. Patch-level metrics are secondary only. Class-wise recall is mandatory because B1-R test support remains small.

## 10. Uncertainty

For each seed and model, report 2000-repetition class-stratified event bootstrap 95% intervals for macro-F1 and balanced accuracy. Singleton test classes remain fixed within their class stratum and this limitation must be disclosed. Across-seed SD represents optimization variability only.

## 11. Predeclared checkpoint sensitivity

For reviewer Comment 3, the same training trajectory preserves three checkpoint definitions: primary minimum-NLL, secondary maximum observed-seven-class macro-F1, and epoch 60. Any test-side sensitivity analysis must report the full grid for all 4 models × 3 seeds and must not use test results to select the headline checkpoint.

## 12. Development freeze

After this protocol freeze, the following are prohibited for corrected main-paper results: split changes, architecture changes, optimizer/lr changes, seed changes, checkpoint-rule changes, RDFT settings changes, GRL schedule changes, location-loss changes, or any test-informed retuning. Only protocol-preserving bug fixes, reporting, visualization, and explicitly predeclared sensitivity analyses are allowed.

## 13. Required run artifacts

Each model/seed run must save:

- `CONFIG.json` with B1-R SHA and protocol SHA;
- initialization-equivalence gate record;
- epoch-wise train/validation history;
- primary best-NLL checkpoint;
- secondary best-macro-F1 checkpoint;
- last epoch checkpoint;
- primary train/val/test patch predictions;
- primary event predictions;
- metrics JSON;
- bootstrap CI CSV;
- completion manifest with hashes.

No run is accepted without all mandatory artifacts and the exact frozen B1-R/protocol hashes.
