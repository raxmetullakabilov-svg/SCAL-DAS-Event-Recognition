from pathlib import Path
from collections import OrderedDict
import hashlib, json, shutil
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .config import EXPECTED_PHASE1_SHA, EXPECTED_PHASE2_SHA, EXPECTED_B1R_SHA
from .models import is_spectral

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def auto_detect_phase2(project_root):
    root=Path(project_root)
    preferred=root/'Phase2_Compact_v0.1'
    candidates=[]
    if (preferred/'PHASE2_SHA256.txt').is_file(): candidates.append(preferred)
    for p in root.rglob('PHASE2_SHA256.txt'):
        if p.parent not in candidates: candidates.append(p.parent)
    good=[]
    for c in candidates:
        try: sha=(c/'PHASE2_SHA256.txt').read_text().strip()
        except Exception: continue
        if sha==EXPECTED_PHASE2_SHA and (c/'metadata'/'SAMPLE_REGISTRY.csv').is_file() and (c/'shards').is_dir(): good.append(c)
    if not good: raise FileNotFoundError('Full frozen Phase-2 root with expected SHA, metadata, and shards not found in Google Drive.')
    return sorted(good,key=lambda p:len(str(p)))[0]

def load_b1r_registry(phase2_root,frozen_dir):
    phase2_root=Path(phase2_root); frozen_dir=Path(frozen_dir)
    if (phase2_root/'PHASE2_SHA256.txt').read_text().strip()!=EXPECTED_PHASE2_SHA: raise ValueError('Phase-2 SHA mismatch')
    if (frozen_dir/'B1R_SHA256.txt').read_text().strip()!=EXPECTED_B1R_SHA: raise ValueError('B1-R SHA mismatch')
    p2=pd.read_csv(phase2_root/'metadata'/'SAMPLE_REGISTRY.csv')
    a=pd.read_csv(frozen_dir/'B1R_SAMPLE_ASSIGNMENTS.csv')
    loc=pd.read_csv(frozen_dir/'LOCATION_LABELS_20.csv')
    if len(p2)!=2522 or p2.event_id.nunique()!=137: raise ValueError('Phase-2 registry cardinality mismatch')
    if set(p2.phase1_sha.astype(str))!={EXPECTED_PHASE1_SHA}: raise ValueError('Phase-1 inheritance mismatch')
    chk=p2[['sample_id','event_id','class_id','class_name','recording_id','shard_file','shard_row','raw_patch_sha256','model_patch_sha256']].merge(
        a[['sample_id','event_id','class_id','class_name','recording_id','shard_file','shard_row','raw_patch_sha256','model_patch_sha256']],
        on='sample_id',suffixes=('_p2','_b1r'),validate='one_to_one')
    for c in ['event_id','class_id','class_name','recording_id','shard_file','shard_row','raw_patch_sha256','model_patch_sha256']:
        if not (chk[c+'_p2'].astype(str)==chk[c+'_b1r'].astype(str)).all(): raise ValueError(f'B1-R/Phase-2 mismatch: {c}')
    df=a.merge(loc,on=['event_id','class_name'],how='left',validate='many_to_one')
    if df.u_centroid.isna().any(): raise ValueError('Missing location labels')
    df['task_class_id']=df.class_id.astype(int)
    if set(df.task_class_id.unique())!=set(range(9)): raise ValueError('Expected class IDs 0..8')
    exp_p={'train':1920,'val':179,'test':423}; exp_e={'train':108,'val':11,'test':18}
    if df.b1r_role.value_counts().to_dict()!=exp_p: raise ValueError(f'B1-R patch counts mismatch: {df.b1r_role.value_counts().to_dict()}')
    ev=df.drop_duplicates('event_id')
    if ev.b1r_role.value_counts().to_dict()!=exp_e: raise ValueError(f'B1-R event counts mismatch: {ev.b1r_role.value_counts().to_dict()}')
    val_classes=set(ev.loc[ev.b1r_role=='val','class_name'])
    if val_classes!={'car','fence','longboard','manipulation','openclose','regular','walk'}: raise ValueError(f'Unexpected validation classes {val_classes}')
    return df

def assert_zero_cross_partition_raw_overlap(df):
    roles={'train','val','test'}; n=0
    for rec,g in df.groupby('recording_id'):
        z=g[['sample_id','b1r_role','raw_time_start','raw_time_end','raw_channel_start','raw_channel_end']].sort_values('raw_time_start').reset_index(drop=True)
        arr=z.to_dict('records')
        for i,a in enumerate(arr):
            for b in arr[i+1:]:
                if b['raw_time_start']>=a['raw_time_end']: break
                if a['b1r_role']==b['b1r_role'] or a['b1r_role'] not in roles or b['b1r_role'] not in roles: continue
                spatial=(a['raw_channel_start']<b['raw_channel_end'] and b['raw_channel_start']<a['raw_channel_end'])
                temporal=(a['raw_time_start']<b['raw_time_end'] and b['raw_time_start']<a['raw_time_end'])
                if temporal and spatial: n+=1
    if n: raise RuntimeError(f'B1-R raw-support overlap gate FAIL: {n} cross-partition patch pairs')
    return 0

def frames(df):
    return {s:df[df.b1r_role==s].copy().reset_index(drop=True) for s in ('train','val','test')}

def class_event_balanced_sampler(train_df,seed,num_samples=None):
    event_counts=train_df.groupby('class_name')['event_id'].nunique().to_dict(); patch_counts=train_df.groupby('event_id').size().to_dict()
    w=[1.0/(event_counts[r.class_name]*patch_counts[r.event_id]) for r in train_df.itertuples(index=False)]
    gen=torch.Generator(); gen.manual_seed(int(seed))
    return WeightedRandomSampler(torch.tensor(w,dtype=torch.double),num_samples=num_samples or len(train_df),replacement=True,generator=gen)

class ShardDataset(Dataset):
    def __init__(self,frame,phase2_root,model_name,cache_files=3):
        self.frame=frame.reset_index(drop=True).copy(); self.root=Path(phase2_root); self.model_name=model_name; self.cache_files=int(cache_files); self._handles=OrderedDict()
        self.dataset_name='raw' if is_spectral(model_name) else 'model'
    def __len__(self): return len(self.frame)
    def _handle(self,rel):
        rel=str(rel)
        if rel in self._handles:
            f=self._handles.pop(rel); self._handles[rel]=f; return f
        f=h5py.File(self.root/rel,'r'); self._handles[rel]=f
        while len(self._handles)>self.cache_files:
            _,old=self._handles.popitem(last=False); old.close()
        return f
    def __getitem__(self,i):
        r=self.frame.iloc[int(i)]; f=self._handle(r.shard_file); row=int(r.shard_row)
        x=np.asarray(f[self.dataset_name][row],dtype=np.float32)
        if x.shape!=(8192,128): raise ValueError(f'bad patch shape {x.shape}')
        return {'x':torch.from_numpy(x).unsqueeze(0),'y':torch.tensor(int(r.task_class_id),dtype=torch.long),
                'u':torch.tensor(float(r.u_centroid),dtype=torch.float32),'sample_id':str(r.sample_id),
                'event_id':str(r.event_id),'class_name':str(r.class_name)}
    def close(self):
        for f in self._handles.values():
            try:f.close()
            except Exception:pass
        self._handles.clear()
    def __del__(self): self.close()

def collate(batch):
    return {'x':torch.stack([b['x'] for b in batch]),'y':torch.stack([b['y'] for b in batch]),'u':torch.stack([b['u'] for b in batch]),
            'sample_id':[b['sample_id'] for b in batch],'event_id':[b['event_id'] for b in batch],'class_name':[b['class_name'] for b in batch]}

def make_loaders(fr,phase2_root,model_name,cfg,seed,include_test=False):
    train_ds=ShardDataset(fr['train'],phase2_root,model_name); val_ds=ShardDataset(fr['val'],phase2_root,model_name)
    samp=class_event_balanced_sampler(fr['train'],seed)
    out={'train':DataLoader(train_ds,batch_size=cfg.batch_size,sampler=samp,num_workers=cfg.num_workers,pin_memory=True,collate_fn=collate),
         'train_eval':DataLoader(train_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=True,collate_fn=collate),
         'val':DataLoader(val_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=True,collate_fn=collate)}
    if include_test:
        test_ds=ShardDataset(fr['test'],phase2_root,model_name)
        out['test']=DataLoader(test_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=True,collate_fn=collate)
    return out

def stage_shards_to_local(phase2_root,registry,local_root,verify_sha=False):
    phase2_root=Path(phase2_root); local_root=Path(local_root); (local_root/'shards').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((phase2_root/'PHASE2_MANIFEST.json').read_text()) if (phase2_root/'PHASE2_MANIFEST.json').is_file() else {'file_hashes':{}}
    expected=manifest.get('file_hashes',{}); rels=sorted(set(registry.shard_file.astype(str)))
    for i,rel in enumerate(rels,1):
        src=phase2_root/rel; dst=local_root/rel; dst.parent.mkdir(parents=True,exist_ok=True)
        if not src.is_file(): raise FileNotFoundError(src)
        good=dst.is_file() and dst.stat().st_size==src.stat().st_size
        if good and verify_sha and rel in expected: good=(sha256_file(dst)==expected[rel])
        if not good:
            tmp=Path(str(dst)+'.partial'); tmp.unlink(missing_ok=True); print(f'[{i:02d}/{len(rels):02d}] staging {rel} ({src.stat().st_size/1024**2:.1f} MB)')
            shutil.copy2(src,tmp); tmp.replace(dst)
            if verify_sha and rel in expected and sha256_file(dst)!=expected[rel]: raise ValueError(f'local staged SHA mismatch: {rel}')
        else: print(f'[{i:02d}/{len(rels):02d}] reuse local {rel}')
    # Metadata marker for local root.
    (local_root/'PHASE2_SHA256.txt').write_text(EXPECTED_PHASE2_SHA+'\n')
    return local_root
