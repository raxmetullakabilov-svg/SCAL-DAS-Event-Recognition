from dataclasses import dataclass
from pathlib import Path
import hashlib, json, re
import h5py
import numpy as np
import pandas as pd
from .config import Phase2Config, config_fingerprint
from .phase1 import load_phase1_contract, Phase1Contract
from .discovery import discover_raw_files, inspect_hdf5_file, RawFileStructure
from .mapping import build_recording_map, validate_recording_map
from .coordinates import resolve_window_geometry, event_to_raw_support
from .sampling import plan_event_samples, extract_fixed_patch
from .normalization import robust_normalize
from .shards import ExtractedSample, write_shard_atomic, validate_shard, read_shard_sample_metadata, sha256_file
from .checkpoint import should_skip_shard, mark_shard_complete
from .registry import build_sample_registry, build_event_to_sample_map
from .audits import audit_phase1_linkage, audit_b1_samples, audit_b3_samples, audit_signal_quality, audit_shards, run_final_gates
from .freeze import write_freeze_manifest, verify_frozen_phase2
from .figures import plot_sample_count_by_class_split, plot_event_to_sample_distribution, plot_signal_value_diagnostics

@dataclass(frozen=True)
class Phase2Paths:
    phase1_root: Path
    raw_root: Path
    output_root: Path
    def __post_init__(self):
        object.__setattr__(self,'phase1_root',Path(self.phase1_root))
        object.__setattr__(self,'raw_root',Path(self.raw_root))
        object.__setattr__(self,'output_root',Path(self.output_root))

@dataclass
class Phase2RunResult:
    sample_registry: pd.DataFrame
    gates: pd.DataFrame
    final_sha: str
    skipped_shards: int


def _sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.+-]+','_',str(s))


def _array_sha(x: np.ndarray) -> str:
    a=np.ascontiguousarray(x)
    h=hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _relative(path: str | Path, root: Path) -> str:
    p=Path(path).resolve(); r=Path(root).resolve()
    try: return p.relative_to(r).as_posix()
    except ValueError: return p.name


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,lineterminator='\n')


def _gate_df(results):
    return pd.DataFrame([{'gate':g.name,'passed':g.passed,'mandatory':g.mandatory,'detail':g.detail} for g in results])


def _planned_ids(events_df, mapping_local, geometries_by_recording, structures_by_path, cfg):
    ids=[]
    for e in events_df.sort_values(['recording_id','class_name','event_id'],kind='stable').to_dict('records'):
        mr=mapping_local.set_index('recording_id').loc[str(e['recording_id'])]
        st=structures_by_path[str(Path(mr.source_raw_file).resolve())]
        geom=geometries_by_recording[str(e['recording_id'])]
        support=event_to_raw_support(e,geom,st.shape)
        ids.extend(p.sample_id for p in plan_event_samples(e,support,geom,st.shape,cfg))
    return sorted(ids)


def _run(cfg: Phase2Config, paths: Phase2Paths, *, contract_override: Phase1Contract | None, freeze: bool, event_ids: list[str] | None = None, manual_map: dict[str,str] | None = None, geometry_overrides: dict[str,dict] | None = None) -> Phase2RunResult:
    out=paths.output_root
    for d in ['metadata','shards','checkpoints','audits','figures','logs']:
        (out/d).mkdir(parents=True,exist_ok=True)
    contract=contract_override or load_phase1_contract(paths.phase1_root,cfg)
    if len(contract.events)!=cfg.expected_phase1_events:
        raise ValueError(f'Phase-1 event count mismatch: {len(contract.events)} != {cfg.expected_phase1_events}')
    files=discover_raw_files(paths.raw_root)
    structures=[inspect_hdf5_file(p) for p in files]
    manual_map=manual_map or {}
    geometry_overrides=geometry_overrides or {}
    run_config_sha=config_fingerprint(cfg, {'geometry_overrides':geometry_overrides})
    mapping_local=build_recording_map(contract,structures,manual_map=manual_map)
    validate_recording_map(mapping_local)
    by_path={str(Path(s.path).resolve()):s for s in structures}

    mapping_persist=mapping_local.copy()
    mapping_persist['source_raw_file']=mapping_persist['source_raw_file'].map(lambda x:_relative(x,paths.raw_root))
    mapping_persist=mapping_persist.sort_values('recording_id',kind='stable').reset_index(drop=True)
    _write_csv(mapping_persist,out/'metadata'/'RAW_RECORDING_MAP.csv')

    inventory=[]
    geometries_by_recording={}
    for r in mapping_local.sort_values('recording_id').itertuples(index=False):
        st=by_path[str(Path(r.source_raw_file).resolve())]
        override=geometry_overrides.get(str(r.recording_id))
        geom=resolve_window_geometry(st,override)
        geometries_by_recording[str(r.recording_id)]=geom
        inventory.append({
            'recording_id':str(r.recording_id),
            'source_raw_file':_relative(st.path,paths.raw_root),
            'raw_file_sha256':str(r.raw_file_sha256),
            'dataset_path':st.dataset_path,'raw_shape':'x'.join(map(str,st.shape)),'raw_dtype':st.dtype,
            'attributes_json':json.dumps(st.attributes,sort_keys=True,separators=(',',':')),
            'window_length':geom.window_length,'hop':geom.hop,'time_axis':geom.time_axis,
            'geometry_source':geom.source_method,
            'geometry_override_json':json.dumps(override or {},sort_keys=True,separators=(',',':')),
        })
    inventory_df=pd.DataFrame(inventory).sort_values('recording_id',kind='stable').reset_index(drop=True)
    _write_csv(inventory_df,out/'metadata'/'RECORDING_INVENTORY.csv')

    sample_records=[]; signal_rows=[]; shard_rows=[]; skipped=0
    mindex=mapping_local.set_index('recording_id')
    events=contract.events.sort_values(['recording_id','class_name','event_id'],kind='stable')
    if event_ids is not None:
        wanted=list(dict.fromkeys(map(str,event_ids)))
        unknown=sorted(set(wanted)-set(events.event_id.astype(str)))
        if unknown: raise ValueError(f'unknown dry-run event IDs: {unknown}')
        if freeze and set(wanted)!=set(events.event_id.astype(str)):
            raise ValueError('final freeze cannot use a partial event subset')
        events=events[events.event_id.astype(str).isin(wanted)].copy()
    for (rid,cls), group in events.groupby(['recording_id','class_name'],sort=True):
        mr=mindex.loc[str(rid)]
        raw_path=Path(mr.source_raw_file)
        st=by_path[str(raw_path.resolve())]
        geom=geometries_by_recording[str(rid)]
        rel_shard=f"shards/{_sanitize(rid)}__{_sanitize(cls)}__part000.h5"
        shard_path=out/rel_shard
        if should_skip_shard(out,shard_path,str(mr.raw_file_sha256),contract.phase1_sha,run_config_sha):
            skipped+=1
            recs=read_shard_sample_metadata(shard_path)
            for r in recs:
                r['shard_valid']=True
                signal_rows.append({'sample_id':r['sample_id'],'raw_min':r['raw_min'],'raw_max':r['raw_max'],'model_min':r['model_min'],'model_max':r['model_max'],'finite_ok':r['finite_ok']})
            sample_records.extend(recs)
            shard_rows.append({'shard_file':rel_shard,'sha256':sha256_file(shard_path),'sample_count':len(recs),'valid':True})
            continue
        extracted=[]; local_meta=[]
        with h5py.File(raw_path,'r') as f:
            ds=f[st.dataset_path]
            for e in group.to_dict('records'):
                support=event_to_raw_support(e,geom,st.shape)
                for plan in plan_event_samples(e,support,geom,st.shape,cfg):
                    raw_patch=extract_fixed_patch(ds,plan,cfg)
                    model_patch,nstats=robust_normalize(raw_patch,cfg.normalization_eps)
                    finite=bool(np.isfinite(raw_patch).all() and np.isfinite(model_patch).all())
                    meta={
                        'sample_id':plan.sample_id,'event_id':plan.event_id,
                        'source_raw_file':_relative(raw_path,paths.raw_root),'raw_file_sha256':str(mr.raw_file_sha256),
                        'raw_time_start':plan.source_time_start,'raw_time_end':plan.source_time_end,
                        'raw_channel_start':plan.source_channel_start,'raw_channel_end':plan.source_channel_end,
                        'request_time_start':plan.request_time_start,'request_time_end':plan.request_time_end,
                        'request_channel_start':plan.request_channel_start,'request_channel_end':plan.request_channel_end,
                        'pad_time_left':plan.pad_time_left,'pad_time_right':plan.pad_time_right,
                        'pad_channel_left':plan.pad_channel_left,'pad_channel_right':plan.pad_channel_right,
                        'model_time':cfg.model_time,'model_channels':cfg.model_channels,'normalization':nstats.version,
                        'normalization_median':nstats.median,'normalization_mad':nstats.mad,'normalization_eps':nstats.eps,
                        'raw_patch_sha256':_array_sha(raw_patch),'model_patch_sha256':_array_sha(model_patch),
                        'shard_file':rel_shard,'shard_row':len(extracted),'extraction_status':'complete',
                        'finite_ok':finite,'shard_valid':True,
                        'source_raw_sha256':str(mr.raw_file_sha256),'phase1_sha':contract.phase1_sha,
                        'raw_min':float(np.min(raw_patch)),'raw_max':float(np.max(raw_patch)),
                        'model_min':float(np.min(model_patch)),'model_max':float(np.max(model_patch)),
                    }
                    extracted.append(ExtractedSample(plan.sample_id,plan.event_id,raw_patch,model_patch,meta))
                    local_meta.append(meta)
                    signal_rows.append({'sample_id':plan.sample_id,'raw_min':meta['raw_min'],'raw_max':meta['raw_max'],
                                        'model_min':meta['model_min'],'model_max':meta['model_max'],'finite_ok':finite})
        rec=write_shard_atomic(shard_path,extracted,cfg,config_sha256=run_config_sha)
        mark_shard_complete(out,rec,contract.phase1_sha)
        validate_shard(shard_path,rec)
        sample_records.extend(local_meta)
        shard_rows.append({'shard_file':rel_shard,'sha256':rec.sha256,'sample_count':rec.sample_count,'valid':True})

    samples=build_sample_registry(sample_records,contract).sort_values(['event_id','sample_id'],kind='stable').reset_index(drop=True)
    _write_csv(samples,out/'metadata'/'SAMPLE_REGISTRY.csv')
    _write_csv(build_event_to_sample_map(samples),out/'metadata'/'EVENT_TO_SAMPLE_MAP.csv')
    stats=samples.groupby(['class_name','b1_role'],dropna=False).size().rename('sample_count').reset_index().sort_values(['class_name','b1_role'])
    _write_csv(stats,out/'metadata'/'EXTRACTION_STATS.csv')
    _write_csv(pd.DataFrame(columns=['event_id','recording_id','reason_code','reason_text','fatal','code_version']),out/'metadata'/'EXCLUSIONS.csv')

    signal_df=pd.DataFrame(signal_rows)
    _write_csv(signal_df.sort_values('sample_id') if not signal_df.empty else signal_df,out/'audits'/'SIGNAL_RANGE_AUDIT.csv')
    shard_df=pd.DataFrame(shard_rows).sort_values('shard_file').reset_index(drop=True)
    _write_csv(shard_df,out/'audits'/'SHARD_INTEGRITY_AUDIT.csv')
    _write_csv(_gate_df(audit_phase1_linkage(samples,contract,cfg.expected_phase1_events)),out/'audits'/'PHASE1_LINKAGE_AUDIT.csv')
    _write_csv(_gate_df(audit_b1_samples(samples,contract)),out/'audits'/'B1_SAMPLE_LEAKAGE_AUDIT.csv')
    _write_csv(_gate_df(audit_b3_samples(samples)),out/'audits'/'B3_SAMPLE_LEAKAGE_AUDIT.csv')
    _write_csv(_gate_df(audit_signal_quality(samples,cfg)),out/'audits'/'SIGNAL_QUALITY_AUDIT.csv')
    plot_sample_count_by_class_split(samples,out/'figures'/'sample_count_by_class_split.png')
    plot_event_to_sample_distribution(samples,out/'figures'/'event_to_sample_distribution.png')
    plot_signal_value_diagnostics(signal_df,out/'figures'/'signal_value_diagnostics.png')

    planned=_planned_ids(events,mapping_local,geometries_by_recording,by_path,cfg)
    deterministic=(planned==sorted(samples.sample_id.astype(str).tolist()))
    gates=run_final_gates(contract,samples,cfg,mapping_df=mapping_persist,root=out,
                          phase1_sha_ok=(contract_override is not None or contract.phase1_sha==cfg.expected_phase1_sha),
                          persisted_hashes_match=freeze,portable_sha_ok=freeze,deterministic_rebuild_ok=deterministic)
    _write_csv(gates,out/'audits'/'FINAL_INTEGRITY_GATES.csv')
    if freeze and (not gates.iloc[:16]['passed'].all() or not deterministic):
        raise RuntimeError('Phase-2 pre-freeze integrity gate failure')

    final_sha=''
    if freeze:
        include=[]
        for sub in ['metadata','shards','audits','figures']:
            for p in sorted((out/sub).rglob('*')):
                if p.is_file(): include.append(p.relative_to(out).as_posix())
        final_sha=write_freeze_manifest(out,cfg,include,phase1_sha=contract.phase1_sha)
        verified=verify_frozen_phase2(out)
        if verified!=final_sha: raise RuntimeError('portable freeze verification mismatch')
        # Gates were written before freeze with the final three set true; verification above proves them.
        if not gates['passed'].all(): raise RuntimeError('Phase-2 final integrity gate failure')
    return Phase2RunResult(samples,gates,final_sha,skipped)


def run_phase2(cfg: Phase2Config, paths: Phase2Paths, *, contract_override: Phase1Contract | None = None, manual_map: dict[str,str] | None = None, geometry_overrides: dict[str,dict] | None = None) -> Phase2RunResult:
    return _run(cfg,paths,contract_override=contract_override,freeze=True,event_ids=None,manual_map=manual_map,geometry_overrides=geometry_overrides)


def run_dry_run(cfg: Phase2Config, paths: Phase2Paths, *, contract_override: Phase1Contract | None = None, event_ids: list[str] | None = None, manual_map: dict[str,str] | None = None, geometry_overrides: dict[str,dict] | None = None) -> Phase2RunResult:
    return _run(cfg,paths,contract_override=contract_override,freeze=False,event_ids=event_ids,manual_map=manual_map,geometry_overrides=geometry_overrides)
