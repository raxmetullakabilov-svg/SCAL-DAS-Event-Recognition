from dataclasses import dataclass

EXPECTED_PHASE1_SHA = 'd5248a7d5730e7ad6857c5475f248e27cd315d9c5e26235eaba57421936cfe44'
EXPECTED_PHASE2_SHA = 'c05ef6f99c1f9d235995f3db1f2067b60b988cf4b03a3d905c3c76d5fd826eef'
EXPECTED_B1R_SHA = '9fab73d477624ff3878ee3ec3cb2e86f5a4aa9bb17ce00e47fedf260d350fce7'
EXPECTED_PROTOCOL_SHA = '1ea5a907f4e0f7d5bc87d43bd82f649089d5350329afb0133d3de86f1ca33dab'

@dataclass(frozen=True)
class CampaignConfig:
    version: str = 'B1R_Retraining_Campaign_v1.0'
    model_time: int = 8192
    model_channels: int = 128
    embedding_dim: int = 192
    n_classes: int = 9
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 60
    grad_clip_norm: float = 5.0
    amp: bool = True
    adv_lambda_max: float = 0.15
    adv_gamma: float = 10.0
    location_loss_weight: float = 1.0
    bootstrap_reps: int = 2000
    tie_tolerance: float = 1e-8
    seeds: tuple = (2026, 2027, 2028)
    models: tuple = ('raw_ts_resnet','raw_scal_das','spectrum_rdft','scal_rdft')
    smoke_epochs: int = 2
    smoke_max_train_batches: int = 8
