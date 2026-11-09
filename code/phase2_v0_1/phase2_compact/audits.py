from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .config import Phase2Config

@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str = ''
    mandatory: bool = True


def _g(name, passed, detail=''):
    return GateResult(name,bool(passed),str(detail),True)


def audit_phase1_linkage(samples: pd.DataFrame, contract, expected_events: int = 137) -> list[GateResult]:
    event_set=set(contract.events.event_id.astype(str)); sample_events=set(samples.event_id.astype(str))
    expected_b1=contract.b1.set_index('event_id')['split'].astype(str).to_dict()
    actual_b1=samples.groupby('event_id')['b1_role'].agg(lambda x:set(map(str,x))).to_dict()
    b1_ok=all(actual_b1.get(e)=={r} for e,r in expected_b1.items())
    b3_expected=contract.b3.drop_duplicates('event_id').set_index('event_id')['transfer_role'].fillna('').astype(str).to_dict()
    b3_ok=True
    for e in event_set:
        want=b3_expected.get(e,'')
        got=set(samples.loc[samples.event_id.astype(str)==e,'b3_role'].fillna('').astype(str))
        if got != {want}: b3_ok=False; break
    return [
        _g('phase1_event_coverage_137_of_137', sample_events==event_set and len(event_set)==expected_events, f'{len(sample_events)}/{len(event_set)} events'),
        _g('sample_ids_unique', not samples['sample_id'].duplicated().any(), f'{samples.sample_id.nunique()}/{len(samples)} unique'),
        _g('event_to_sample_linkage_complete', sample_events==event_set and sample_events.issubset(event_set), 'every frozen event has >=1 sample and no unknown parent'),
        _g('b1_roles_inherited_exactly', b1_ok, 'sample B1 roles equal frozen parent roles'),
        _g('b3_roles_inherited_exactly', b3_ok, 'sample B3 roles equal frozen parent roles'),
    ]


def audit_b1_samples(samples: pd.DataFrame, contract=None) -> list[GateResult]:
    event_cross=(samples.groupby('event_id')['b1_role'].nunique(dropna=False)>1).any()
    group_cross=(samples.groupby('b1_leakage_group_id')['b1_role'].nunique(dropna=False)>1).any()
    return [_g('b1_sample_leakage_zero', not (event_cross or group_cross), f'event_cross={bool(event_cross)}, group_cross={bool(group_cross)}')]


def audit_b3_samples(samples: pd.DataFrame) -> list[GateResult]:
    active=samples[samples['b3_role'].isin(['source_train','source_val','target_test'])].copy()
    source=active[active.b3_role.isin(['source_train','source_val'])]
    source_group_cross=(source.groupby('b1_leakage_group_id')['b3_role'].nunique(dropna=False)>1).any() if not source.empty else False
    overlap=False
    if not active.empty:
        role_sets=active.groupby('event_id')['b3_role'].agg(set)
        overlap=any(('target_test' in rs and bool(rs & {'source_train','source_val'})) for rs in role_sets)
    return [
        _g('b3_source_train_val_leakage_zero', not source_group_cross, f'group_cross={bool(source_group_cross)}'),
        _g('b3_source_target_overlap_zero', not overlap, f'event_overlap={bool(overlap)}'),
    ]


def audit_signal_quality(samples: pd.DataFrame, cfg: Phase2Config) -> list[GateResult]:
    idx_ok=(
        (pd.to_numeric(samples['raw_time_end'],errors='coerce') > pd.to_numeric(samples['raw_time_start'],errors='coerce')) &
        (pd.to_numeric(samples['raw_channel_end'],errors='coerce') > pd.to_numeric(samples['raw_channel_start'],errors='coerce'))
    ).all()
    shape_ok=(pd.to_numeric(samples['model_time'],errors='coerce').eq(cfg.model_time) & pd.to_numeric(samples['model_channels'],errors='coerce').eq(cfg.model_channels)).all()
    finite_ok=('finite_ok' in samples.columns and samples['finite_ok'].fillna(False).astype(bool).all())
    return [
        _g('sample_raw_indices_valid',idx_ok),
        _g('sample_shapes_valid',shape_ok),
        _g('nan_inf_invalid_samples_zero',finite_ok),
    ]


def audit_shards(samples: pd.DataFrame, root: Path | None) -> list[GateResult]:
    valid=('shard_valid' in samples.columns and samples['shard_valid'].fillna(False).astype(bool).all())
    partials=[] if root is None else list(Path(root).rglob('*.partial'))
    return [_g('shard_integrity_complete',valid), _g('temporary_partial_files_zero',len(partials)==0,f'{len(partials)} partial files')]


def run_final_gates(contract, samples: pd.DataFrame, cfg: Phase2Config, *, mapping_df: pd.DataFrame | None, root: Path | None,
                    phase1_sha_ok: bool, persisted_hashes_match: bool, portable_sha_ok: bool, deterministic_rebuild_ok: bool) -> pd.DataFrame:
    map_complete=False; hashes=False
    if mapping_df is not None and not mapping_df.empty:
        required=set(contract.events.recording_id.astype(str))
        got=set(mapping_df.recording_id.astype(str))
        map_complete=(required==got and mapping_df.mapping_status.isin(['exact','manual_confirmed']).all())
        hashes=(mapping_df.raw_file_sha256.fillna('').astype(str).str.fullmatch(r'[0-9a-fA-F]{64}').all())
    gates=[
        _g('phase1_sha_exact',phase1_sha_ok),
        _g('raw_recording_mapping_complete',map_complete),
        _g('raw_source_hashes_recorded',hashes),
    ]
    link=audit_phase1_linkage(samples,contract,cfg.expected_phase1_events)
    # order according to the 19-gate contract
    by={x.name:x for x in link}
    gates += [by['phase1_event_coverage_137_of_137'],by['sample_ids_unique'],by['event_to_sample_linkage_complete'],by['b1_roles_inherited_exactly']]
    gates += audit_b1_samples(samples,contract)
    gates += [by['b3_roles_inherited_exactly']]
    gates += audit_b3_samples(samples)
    gates += audit_signal_quality(samples,cfg)
    gates += audit_shards(samples,root)
    gates += [
        _g('persisted_file_hashes_match',persisted_hashes_match),
        _g('portable_phase2_sha_recomputable',portable_sha_ok),
        _g('deterministic_rebuild_match',deterministic_rebuild_ok),
    ]
    if len(gates)!=19:
        raise RuntimeError(f'internal gate count error: {len(gates)}')
    return pd.DataFrame([{'gate':g.name,'passed':g.passed,'mandatory':g.mandatory,'detail':g.detail} for g in gates])
