from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_version: str = "Phase1_v0.1"
    connectedness: int = 8
    window_size: int = 8192
    window_shift: int = 2048
    temporal_guard_windows: int = 5
    reference_split: Tuple[float, float, float] = (0.80, 0.10, 0.10)
    n_spatial_folds: int = 4
    min_transfer_events_per_class: int = 3
    min_transfer_classes: int = 2
    min_transfer_source_groups_per_class: int = 2
    random_seed: int = 42
    code_version: str = "Paper01_PHASE1_MASTER_v0.1.4"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reference_split"] = list(self.reference_split)
        return data
