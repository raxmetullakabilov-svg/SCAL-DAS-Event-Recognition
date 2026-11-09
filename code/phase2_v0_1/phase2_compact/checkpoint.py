from dataclasses import dataclass
from pathlib import Path
import json, os, tempfile
from .shards import ShardRecord, sha256_file

@dataclass(frozen=True)
class CheckpointState:
    phase1_sha: str
    shards: dict


def _path(root: Path) -> Path:
    return Path(root)/'checkpoints'/'SHARD_STATUS.json'


def load_checkpoint(root: Path) -> CheckpointState:
    p=_path(root)
    if not p.is_file(): return CheckpointState('',{})
    d=json.loads(p.read_text(encoding='utf-8'))
    return CheckpointState(str(d.get('phase1_sha','')), dict(d.get('shards',{})))


def _atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=Path(str(path)+'.partial')
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def mark_shard_complete(root: Path, record: ShardRecord, phase1_sha: str) -> None:
    state=load_checkpoint(root)
    shards={} if state.phase1_sha not in ('',phase1_sha) else dict(state.shards)
    key=str(Path(record.path).resolve())
    shards[key]={
        'sha256':record.sha256,
        'sample_count':record.sample_count,
        'source_raw_sha256':record.source_raw_sha256,
        'phase1_sha':record.phase1_sha,
        'phase2_config_sha256':record.phase2_config_sha256,
    }
    _atomic_json(_path(root),{'phase1_sha':phase1_sha,'shards':shards})


def should_skip_shard(root: Path, path: Path, source_sha: str, phase1_sha: str, config_sha256: str | None = None) -> bool:
    path=Path(path)
    if Path(str(path)+'.partial').exists() or not path.is_file(): return False
    state=load_checkpoint(root)
    if state.phase1_sha != phase1_sha: return False
    rec=state.shards.get(str(path.resolve()))
    if not rec: return False
    if rec.get('source_raw_sha256') != source_sha or rec.get('phase1_sha') != phase1_sha: return False
    if config_sha256 is not None and rec.get('phase2_config_sha256') != config_sha256: return False
    try:
        return sha256_file(path)==rec.get('sha256')
    except OSError:
        return False
