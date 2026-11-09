from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib, json
from .config import Phase2Config
from .shards import sha256_file


def hash_frozen_files(root: Path, include_paths: list[str]) -> dict[str,str]:
    root=Path(root)
    out={}
    for rel in sorted(map(str,include_paths)):
        p=root/rel
        if not p.is_file(): raise FileNotFoundError(p)
        out[rel]=sha256_file(p)
    return out


def _cfg_dict(cfg):
    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


def phase2_fingerprint(cfg: Phase2Config | dict, file_hashes: dict[str,str]) -> str:
    payload={'config':_cfg_dict(cfg),'file_hashes':{k:file_hashes[k] for k in sorted(file_hashes)}}
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def write_freeze_manifest(root: Path, cfg: Phase2Config, include_paths: list[str], *, phase1_sha: str) -> str:
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    file_hashes=hash_frozen_files(root,include_paths)
    sha=phase2_fingerprint(cfg,file_hashes)
    (root/'PHASE2_FILE_HASHES.json').write_text(json.dumps(file_hashes,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    manifest={
        'benchmark_version':cfg.benchmark_version,
        'code_version':cfg.code_version,
        'upstream_phase1_sha256':phase1_sha,
        'fingerprint_basis':'canonical config + SHA-256 of exact persisted scientific artifact bytes',
        'config':_cfg_dict(cfg),
        'file_hashes':file_hashes,
        'phase2_sha256':sha,
    }
    (root/'PHASE2_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    (root/'PHASE2_SHA256.txt').write_text(sha+'\n',encoding='utf-8')
    return sha


def verify_frozen_phase2(root: Path) -> str:
    root=Path(root)
    manifest=json.loads((root/'PHASE2_MANIFEST.json').read_text(encoding='utf-8'))
    hashes=json.loads((root/'PHASE2_FILE_HASHES.json').read_text(encoding='utf-8'))
    if hashes != manifest.get('file_hashes'):
        raise ValueError('file hash map mismatch between manifest and PHASE2_FILE_HASHES.json')
    for rel,want in hashes.items():
        p=root/rel
        got=sha256_file(p) if p.is_file() else ''
        if got != want:
            raise ValueError(f'hash mismatch for {rel}: {want} != {got}')
    sha=phase2_fingerprint(manifest['config'],hashes)
    stored=(root/'PHASE2_SHA256.txt').read_text(encoding='utf-8').strip()
    if sha != stored or sha != manifest.get('phase2_sha256'):
        raise ValueError('portable Phase-2 SHA mismatch')
    return sha
