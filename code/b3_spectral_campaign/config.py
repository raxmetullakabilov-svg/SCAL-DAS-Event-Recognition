from dataclasses import dataclass
EXPECTED_PHASE2_SHA = 'c05ef6f99c1f9d235995f3db1f2067b60b988cf4b03a3d905c3c76d5fd826eef'
EXPECTED_PROTOCOL_SHA = '2db80f1e58721fc1bab060df59d2232597607b600e98d367573d120f19a71172'

@dataclass(frozen=True)
class Config:
    version: str = "B3_Spectral_Validation_v1.0"
    n_classes: int = 2
    embedding_dim: int = 192
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 80
    grad_clip_norm: float = 5.0
    amp: bool = True
    adv_lambda_max: float = 0.15
    adv_gamma: float = 10.0
    location_loss_weight: float = 1.0
    bootstrap_reps: int = 2000
    seeds: tuple = (2026,2027,2028)
    models: tuple = ("spectrum_rdft","scal_rdft")
    smoke_epochs: int = 2
    smoke_max_train_batches: int = 6
