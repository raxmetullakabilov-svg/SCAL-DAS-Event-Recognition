from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import pandas as pd
from .config import Phase2Config

@dataclass(frozen=True)
class Phase1Contract:
    root: Path
    events: pd.DataFrame
    b1: pd.DataFrame
    b3: pd.DataFrame
    manifest: dict
    phase1_sha: str

_REQUIRED = (
    'registry/EVENT_REGISTRY.csv',
    'splits/TRAIN_VAL_TEST.csv',
    'splits/SPATIAL_TRANSFER_SPLITS.csv',
    'BENCHMARK_MANIFEST.json',
    'BENCHMARK_FILE_HASHES.json',
    'BENCHMARK_SHA256.txt',
)
_HASHED_CONSUMED = (
    'registry/EVENT_REGISTRY.csv',
    'splits/TRAIN_VAL_TEST.csv',
    'splits/SPATIAL_TRANSFER_SPLITS.csv',
)
_HASHED_OPTIONAL = ('registry/LOCATION_LABELS_20.csv','registry/SOURCE_STRUCTURE.csv')

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def verify_phase1_sha(root: Path, expected_sha: str) -> str:
    root = Path(root)
    p = root / 'BENCHMARK_SHA256.txt'
    if not p.is_file(): raise FileNotFoundError(p)
    actual = p.read_text(encoding='utf-8').strip()
    if actual != expected_sha:
        raise ValueError(f'Phase-1 SHA mismatch: expected {expected_sha}, got {actual}')
    manifest_path=root/'BENCHMARK_MANIFEST.json'; hashes_path=root/'BENCHMARK_FILE_HASHES.json'
    if not manifest_path.is_file(): raise FileNotFoundError(manifest_path)
    if not hashes_path.is_file(): raise FileNotFoundError(hashes_path)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest_sha=manifest.get('benchmark_sha256')
    if manifest_sha != actual:
        raise ValueError(f'Phase-1 manifest SHA mismatch: {manifest_sha} != {actual}')
    hashes_doc=json.loads(hashes_path.read_text(encoding='utf-8'))
    # Phase-1 v0.1.4 exports BENCHMARK_FILE_HASHES.json as {\"file_hashes\": {...}}.
    file_hashes=hashes_doc.get('file_hashes') if isinstance(hashes_doc,dict) and 'file_hashes' in hashes_doc else hashes_doc
    if not isinstance(file_hashes,dict):
        raise ValueError('Phase-1 BENCHMARK_FILE_HASHES.json has invalid schema')
    if file_hashes != manifest.get('file_hashes'):
        raise ValueError('Phase-1 file hash map mismatch between manifest and BENCHMARK_FILE_HASHES.json')
    for rel in _HASHED_CONSUMED + tuple(x for x in _HASHED_OPTIONAL if (root/x).is_file()):
        if rel not in file_hashes:
            raise ValueError(f'Phase-1 persisted file hash missing for {rel}')
        got=_sha256(root/rel); want=str(file_hashes[rel])
        if got != want:
            raise ValueError(f'Phase-1 persisted file hash mismatch for {rel}: {want} != {got}')
    return actual

def load_phase1_contract(root: Path, cfg: Phase2Config) -> Phase1Contract:
    root = Path(root)
    for rel in _REQUIRED:
        p = root / rel
        if not p.is_file(): raise FileNotFoundError(p)
    phase1_sha = verify_phase1_sha(root, cfg.expected_phase1_sha)
    events = pd.read_csv(root/'registry'/'EVENT_REGISTRY.csv')
    b1 = pd.read_csv(root/'splits'/'TRAIN_VAL_TEST.csv')
    b3 = pd.read_csv(root/'splits'/'SPATIAL_TRANSFER_SPLITS.csv')
    manifest = json.loads((root/'BENCHMARK_MANIFEST.json').read_text(encoding='utf-8'))
    if len(events) != cfg.expected_phase1_events:
        raise ValueError(f'Phase-1 event count mismatch: expected {cfg.expected_phase1_events}, got {len(events)}')
    if set(events['event_id']) != set(b1['event_id']): raise ValueError('Phase-1 B1 event coverage mismatch')
    active = b3[b3['transfer_role'].isin(['source_train','source_val','target_test'])]
    if set(active['transfer_pair_id'].dropna().astype(str)) != {cfg.b3_pair}: raise ValueError('Phase-1 B3 active pair mismatch')
    if set(active['class_name']) != set(cfg.b3_active_classes): raise ValueError('Phase-1 B3 active classes mismatch')
    return Phase1Contract(root, events, b1, b3, manifest, phase1_sha)
