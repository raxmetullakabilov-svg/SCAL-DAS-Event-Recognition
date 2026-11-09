import pandas as pd

_REQUIRED_OUTPUT = [
    'sample_id','event_id','class_id','class_name','recording_id','source_raw_file','raw_file_sha256',
    'b1_role','b1_leakage_group_id','b3_pair','b3_role',
    'phase1_window_start','phase1_window_end','phase1_channel_start','phase1_channel_end',
    'raw_time_start','raw_time_end','raw_channel_start','raw_channel_end',
    'pad_time_left','pad_time_right','pad_channel_left','pad_channel_right',
    'model_time','model_channels','normalization','raw_patch_sha256','model_patch_sha256',
    'shard_file','shard_row','extraction_status'
]

def build_sample_registry(sample_records, contract) -> pd.DataFrame:
    df=pd.DataFrame(list(sample_records))
    if df.empty:
        return pd.DataFrame(columns=_REQUIRED_OUTPUT)
    for c in ('sample_id','event_id'):
        if c not in df.columns: raise ValueError(f'missing required sample field {c}')
    if df['sample_id'].duplicated().any(): raise ValueError('duplicate sample_id')
    event_ids=set(contract.events['event_id'].astype(str))
    unknown=sorted(set(df['event_id'].astype(str))-event_ids)
    if unknown: raise ValueError(f'unknown event IDs: {unknown[:5]}')
    ev=contract.events.set_index('event_id')
    b1=contract.b1.set_index('event_id')
    b3=contract.b3.drop_duplicates('event_id').set_index('event_id') if not contract.b3.empty else pd.DataFrame()
    rows=[]
    for r in df.to_dict('records'):
        eid=str(r['event_id']); e=ev.loc[eid]; s=b1.loc[eid]
        t=b3.loc[eid] if isinstance(b3,pd.DataFrame) and not b3.empty and eid in b3.index else None
        base={
            'sample_id':str(r['sample_id']),'event_id':eid,
            'class_id':int(e['class_id']),'class_name':str(e['class_name']),'recording_id':str(e['recording_id']),
            'b1_role':str(s['split']),'b1_leakage_group_id':str(s['leakage_group_id']),
            'b3_pair':str(t['transfer_pair_id']) if t is not None and pd.notna(t['transfer_pair_id']) else '',
            'b3_role':str(t['transfer_role']) if t is not None and pd.notna(t['transfer_role']) else '',
            'phase1_window_start':int(e['window_start']),'phase1_window_end':int(e['window_end']),
            'phase1_channel_start':int(e['channel_start']),'phase1_channel_end':int(e['channel_end']),
        }
        base.update(r)
        rows.append(base)
    out=pd.DataFrame(rows)
    for c in _REQUIRED_OUTPUT:
        if c not in out.columns: out[c]=''
    extras=sorted(c for c in out.columns if c not in _REQUIRED_OUTPUT)
    return out[_REQUIRED_OUTPUT + extras]


def build_event_to_sample_map(samples: pd.DataFrame) -> pd.DataFrame:
    return samples[['event_id','sample_id']].sort_values(['event_id','sample_id'],kind='stable').reset_index(drop=True)
