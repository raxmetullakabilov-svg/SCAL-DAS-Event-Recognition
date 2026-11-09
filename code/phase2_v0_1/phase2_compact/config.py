from dataclasses import dataclass, asdict
import hashlib, json

@dataclass(frozen=True)
class Phase2Config:
    code_version: str = 'Paper01_PHASE2_MASTER_v0.1'
    benchmark_version: str = 'Phase2_v0.1'
    expected_phase1_sha: str = 'd5248a7d5730e7ad6857c5475f248e27cd315d9c5e26235eaba57421936cfe44'
    expected_phase1_events: int = 137
    b3_pair: str = 'F4_to_F1'
    b3_active_classes: tuple[str, ...] = ('car', 'construction')
    model_time: int = 8192
    model_channels: int = 128
    normalization_eps: float = 1e-6
    hdf5_compression: str = 'gzip'
    hdf5_compression_opts: int = 4


def config_fingerprint(cfg: Phase2Config, extra: dict | None = None) -> str:
    payload={'config':asdict(cfg),'extra':extra or {}}
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()
