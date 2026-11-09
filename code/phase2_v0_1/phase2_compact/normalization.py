from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class NormalizationStats:
    median: float
    mad: float
    eps: float
    version: str = 'median_mad_v1'


def robust_normalize(x: np.ndarray, eps: float) -> tuple[np.ndarray, NormalizationStats]:
    arr=np.asarray(x)
    med=float(np.median(arr))
    mad=float(np.median(np.abs(arr-med)))
    y=(arr.astype(np.float32, copy=True)-med)/(mad+float(eps))
    return y, NormalizationStats(med,mad,float(eps))
