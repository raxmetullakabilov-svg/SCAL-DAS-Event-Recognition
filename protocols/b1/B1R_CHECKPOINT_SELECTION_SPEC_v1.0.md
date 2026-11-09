# B1-R Checkpoint-Selection Specification v1.0

**Status:** FROZEN before retraining  
**B1-R SHA-256:** `9fab73d477624ff3878ee3ec3cb2e86f5a4aa9bb17ce00e47fedf260d350fce7`

## Structural problem

The B1-R validation partition contains 11 event instances from seven classes. `construction` and `running` are absent because each class has only two independent B1-R atomic support components; retaining independent train and held-out test support leaves no third component for validation. Therefore, the previous nine-class validation macro-F1 checkpoint rule is not scientifically valid for B1-R.

## Primary checkpoint rule

All models train for exactly **60 epochs**. There is **no early stopping**. After each epoch, patch probabilities are aggregated to event probabilities by arithmetic mean. The primary checkpoint is the epoch with the **minimum event-level multiclass negative log-likelihood (NLL)** over the 11 validation events using the full **9-way softmax**.

For event i with true class y_i and aggregated probability p_i(y_i),

`NLL_val = -(1/N_val) * sum_i log p_i(y_i)`, with `N_val = 11`.

If two epochs differ by no more than `1e-8`, the **earlier epoch** is selected.

### Why NLL rather than validation macro-F1?

1. Nine-class macro-F1 would conflate two structurally absent validation classes with model failure.
2. Seven-class macro-F1 is based on only 11 events and contains several singleton classes, so it is highly discontinuous and unstable as a control signal.
3. NLL is smooth, event-level, defined for every validation event, and uses the full nine-class probability vector. Probability mass assigned to absent classes is still penalized indirectly because it reduces the probability of the observed true class.
4. NLL does **not** solve the absence of direct validation evidence for construction/running. That limitation remains explicit; those classes are evaluated only on the final held-out B1-R test set.

## Predeclared sensitivity checkpoints

Two additional checkpoints are saved during the same fixed 60-epoch trajectory:

- **Secondary:** maximum validation event macro-F1 computed only over the seven represented validation classes; ties → earliest epoch.
- **Last:** epoch 60.

These checkpoints are **not** eligible to replace the primary checkpoint after seeing test results. If checkpoint sensitivity is reported, all three definitions must be evaluated and reported for all four models and all three seeds.

## Consequence

The training horizon is independent of the incomplete validation class coverage. Validation controls only which checkpoint from the fixed 60-epoch trajectory becomes the primary reported model. The B1-R test set remains untouched until training finishes and the primary checkpoint is already frozen.
