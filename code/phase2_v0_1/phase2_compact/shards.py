from dataclasses import dataclass
from pathlib import Path
import hashlib, json, os
import h5py
import numpy as np
from .config import Phase2Config, config_fingerprint

@dataclass(frozen=True)
class ExtractedSample:
    sample_id: str
    event_id: str
    raw_patch: np.ndarray
    model_patch: np.ndarray
    metadata: dict

@dataclass(frozen=True)
class ShardRecord:
    path: str
    sha256: str
    sample_count: int
    source_raw_sha256: str
    phase1_sha: str
    phase2_config_sha256: str


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_shard_atomic(path: Path, samples: list[ExtractedSample], cfg: Phase2Config, config_sha256: str | None = None) -> ShardRecord:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        raise ValueError('cannot write empty shard')
    expected=(cfg.model_time,cfg.model_channels)
    for s in samples:
        if tuple(s.raw_patch.shape)!=expected or tuple(s.model_patch.shape)!=expected:
            raise ValueError(f'sample shape mismatch for {s.sample_id}')
    source_shas={str(s.metadata.get('source_raw_sha256','')) for s in samples}
    phase1_shas={str(s.metadata.get('phase1_sha','')) for s in samples}
    if len(source_shas)!=1 or len(phase1_shas)!=1:
        raise ValueError('a shard must have one source SHA and one Phase-1 SHA')
    config_sha256=config_sha256 or config_fingerprint(cfg)
    partial=Path(str(path)+'.partial')
    partial.unlink(missing_ok=True)
    dt=h5py.string_dtype('utf-8')
    with h5py.File(partial,'w') as f:
        f.create_dataset('raw', data=np.stack([s.raw_patch for s in samples]), compression=cfg.hdf5_compression, compression_opts=cfg.hdf5_compression_opts, chunks=(1,*expected), track_times=False)
        f.create_dataset('model', data=np.stack([s.model_patch for s in samples]).astype(np.float32), compression=cfg.hdf5_compression, compression_opts=cfg.hdf5_compression_opts, chunks=(1,*expected), track_times=False)
        f.create_dataset('sample_id', data=np.array([s.sample_id for s in samples],dtype=object), dtype=dt, track_times=False)
        f.create_dataset('event_id', data=np.array([s.event_id for s in samples],dtype=object), dtype=dt, track_times=False)
        f.create_dataset('metadata_json', data=np.array([json.dumps(s.metadata,sort_keys=True,separators=(',',':')) for s in samples],dtype=object), dtype=dt, track_times=False)
        f.attrs['source_raw_sha256']=next(iter(source_shas))
        f.attrs['phase1_sha']=next(iter(phase1_shas))
        f.attrs['code_version']=cfg.code_version
        f.attrs['phase2_config_sha256']=config_sha256
        f.flush()
    os.replace(partial,path)
    rec=ShardRecord(str(path), sha256_file(path), len(samples), next(iter(source_shas)), next(iter(phase1_shas)), config_sha256)
    validate_shard(path,rec)
    return rec


def validate_shard(path: Path, expected: ShardRecord) -> None:
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    if sha256_file(path)!=expected.sha256: raise ValueError('shard SHA mismatch')
    with h5py.File(path,'r') as f:
        n=len(f['sample_id'])
        if n!=expected.sample_count: raise ValueError('shard sample count mismatch')
        if f['raw'].shape[0]!=n or f['model'].shape[0]!=n: raise ValueError('shard dataset count mismatch')
        if str(f.attrs.get('source_raw_sha256',''))!=expected.source_raw_sha256: raise ValueError('shard source SHA mismatch')
        if str(f.attrs.get('phase1_sha',''))!=expected.phase1_sha: raise ValueError('shard Phase-1 SHA mismatch')
        if str(f.attrs.get('phase2_config_sha256',''))!=expected.phase2_config_sha256: raise ValueError('shard Phase-2 config SHA mismatch')


def read_shard_sample_metadata(path: Path) -> list[dict]:
    out=[]
    with h5py.File(path,'r') as f:
        sids=[x.decode() if isinstance(x,bytes) else str(x) for x in f['sample_id'][:]]
        eids=[x.decode() if isinstance(x,bytes) else str(x) for x in f['event_id'][:]]
        metas=[x.decode() if isinstance(x,bytes) else str(x) for x in f['metadata_json'][:]]
        for sid,eid,m in zip(sids,eids,metas):
            d=json.loads(m); d['sample_id']=sid; d['event_id']=eid; out.append(d)
    return out
