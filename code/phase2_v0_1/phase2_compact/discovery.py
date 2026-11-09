from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import h5py
import numpy as np
import pandas as pd

EXPECTED_DATASET_PATH = '/Acquisition/Raw[0]/RawData'

@dataclass(frozen=True)
class RawFileStructure:
    path: Path
    dataset_path: str
    shape: tuple[int, int]
    dtype: str
    attributes: dict[str, Any]


def _py(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='replace')
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def inspect_hdf5_file(path: Path) -> RawFileStructure:
    path = Path(path)
    candidates: list[str] = []
    with h5py.File(path, 'r') as f:
        if EXPECTED_DATASET_PATH in f:
            d = f[EXPECTED_DATASET_PATH]
            if isinstance(d, h5py.Dataset) and d.ndim == 2 and np.issubdtype(d.dtype, np.number):
                attrs = {str(k): _py(v) for k, v in d.attrs.items()}
                return RawFileStructure(path, EXPECTED_DATASET_PATH, tuple(map(int, d.shape)), str(d.dtype), attrs)
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
                candidates.append('/' + name.strip('/'))
        f.visititems(visitor)
        candidates = sorted(set(candidates))
        if not candidates:
            raise ValueError(f'No eligible 2-D numeric signal dataset in {path}')
        if len(candidates) > 1:
            raise ValueError(f'ambiguous 2-D signal datasets in {path}: {candidates}')
        ds_path = candidates[0]
        d = f[ds_path]
        attrs = {str(k): _py(v) for k, v in d.attrs.items()}
        return RawFileStructure(path, ds_path, tuple(map(int, d.shape)), str(d.dtype), attrs)


def discover_raw_files(root: Path) -> list[Path]:
    root = Path(root)
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.h5', '.hdf5'}]
    return sorted(files, key=lambda p: str(p.relative_to(root)).lower())


def write_source_structure(records: list[RawFileStructure], out_csv: Path) -> None:
    rows=[]
    for r in records:
        rows.append({
            'source_raw_file': str(r.path),
            'dataset_path': r.dataset_path,
            'raw_shape': json.dumps(list(r.shape)),
            'raw_dtype': r.dtype,
            'attributes_json': json.dumps(r.attributes, sort_keys=True, separators=(',', ':')),
        })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
