from pathlib import Path
import hashlib
import pandas as pd
from .discovery import RawFileStructure


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _stem(path: Path) -> str:
    name=path.name
    low=name.lower()
    for suffix in ('.hdf5','.h5'):
        if low.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def build_recording_map(contract, structures: list[RawFileStructure], manual_map: dict[str,str] | None = None) -> pd.DataFrame:
    manual_map = manual_map or {}
    by_path={str(Path(s.path).resolve()): s for s in structures}
    rows=[]
    for rid in sorted(set(contract.events['recording_id'].astype(str))):
        method='exact_stem'
        status='missing'
        chosen=None
        if rid in manual_map:
            target=str(Path(manual_map[rid]).resolve())
            chosen=by_path.get(target)
            status='manual_confirmed' if chosen is not None else 'missing'
            method='manual_map'
        else:
            candidates=[s for s in structures if _stem(Path(s.path)) == rid]
            if len(candidates)==1:
                chosen=candidates[0]; status='exact'
            elif len(candidates)>1:
                status='ambiguous'
        rows.append({
            'recording_id': rid,
            'source_raw_file': str(chosen.path) if chosen else '',
            'raw_file_sha256': _sha256(chosen.path) if chosen else '',
            'dataset_path': chosen.dataset_path if chosen else '',
            'raw_shape': 'x'.join(map(str, chosen.shape)) if chosen else '',
            'raw_dtype': chosen.dtype if chosen else '',
            'mapping_method': method,
            'mapping_status': status,
        })
    return pd.DataFrame(rows)


def validate_recording_map(df: pd.DataFrame) -> None:
    bad=df[df['mapping_status'].isin(['ambiguous','missing'])]
    if not bad.empty:
        vals=', '.join(f"{r.recording_id}:{r.mapping_status}" for r in bad.itertuples(index=False))
        raise ValueError(f'raw recording mapping contains ambiguous or missing entries: {vals}')
    if df['recording_id'].duplicated().any():
        raise ValueError('duplicate recording_id in raw recording map')
