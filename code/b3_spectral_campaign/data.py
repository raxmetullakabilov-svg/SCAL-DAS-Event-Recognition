from pathlib import Path
from collections import OrderedDict
import json, shutil, hashlib
import pandas as pd, numpy as np, h5py, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .config import EXPECTED_PHASE2_SHA

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def auto_detect_phase2(project_root):
    root=Path(project_root); preferred=root/'Phase2_Compact_v0.1'; candidates=[]
    if (preferred/'PHASE2_SHA256.txt').is_file(): candidates.append(preferred)
    for p in root.rglob('PHASE2_SHA256.txt'):
        if p.parent not in candidates: candidates.append(p.parent)
    good=[]
    for c in candidates:
        try: sha=(c/'PHASE2_SHA256.txt').read_text().strip()
        except Exception: continue
        if sha==EXPECTED_PHASE2_SHA and (c/'metadata'/'SAMPLE_REGISTRY.csv').is_file() and (c/'shards').is_dir(): good.append(c)
    if not good: raise FileNotFoundError('Full frozen Phase-2 root not found.')
    return sorted(good,key=lambda p:len(str(p)))[0]

def load_registry(phase2_root,frozen_dir):
    phase2_root=Path(phase2_root); frozen_dir=Path(frozen_dir)
    if (phase2_root/'PHASE2_SHA256.txt').read_text().strip()!=EXPECTED_PHASE2_SHA: raise ValueError('Phase-2 SHA mismatch')
    p2=pd.read_csv(phase2_root/'metadata'/'SAMPLE_REGISTRY.csv')
    a=pd.read_csv(frozen_dir/'B3_SPECTRAL_CAMPAIGN_ASSIGNMENTS.csv')
    chk=p2[['sample_id','event_id','class_name','recording_id','shard_file','shard_row']].merge(
        a[['sample_id','event_id','class_name','recording_id','shard_file','shard_row']],on='sample_id',suffixes=('_p2','_f'),validate='one_to_one')
    for c in ['event_id','class_name','recording_id','shard_file','shard_row']:
        if not (chk[c+'_p2'].astype(str)==chk[c+'_f'].astype(str)).all(): raise ValueError('assignment mismatch '+c)
    a['task_class_id']=a.class_name.map({'car':0,'construction':1}).astype(int)
    # Exact role gates.
    tr=a[a.spectral_campaign_role=='train']; te=a[a.spectral_campaign_role=='test']; va=a[a.spectral_campaign_role=='audit_only_source_val']
    assert len(tr)==296 and tr.event_id.nunique()==15
    assert len(va)==19 and va.event_id.nunique()==4
    assert len(te)==300 and te.event_id.nunique()==14
    return a

def assert_target_isolation(df):
    # Actual support overlap must be zero between train and target test.
    z=df[df.spectral_campaign_role.isin(['train','test'])]
    n=0
    for rec,g in z.groupby('recording_id'):
        g=g.sort_values('raw_time_start').reset_index(drop=True); rows=g.to_dict('records')
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                if b['raw_time_start']>=a['raw_time_end']: break
                if a['spectral_campaign_role']==b['spectral_campaign_role']: continue
                if a['raw_channel_start']<b['raw_channel_end'] and b['raw_channel_start']<a['raw_channel_end']:
                    n+=1
    if n: raise RuntimeError(f'train-target actual-support overlap: {n}')
    return 0

def frames(df):
    return {'train':df[df.spectral_campaign_role=='train'].copy().reset_index(drop=True),
            'audit_val':df[df.spectral_campaign_role=='audit_only_source_val'].copy().reset_index(drop=True),
            'test':df[df.spectral_campaign_role=='test'].copy().reset_index(drop=True)}

def sampler(train_df,seed):
    ec=train_df.groupby('class_name').event_id.nunique().to_dict(); pc=train_df.groupby('event_id').size().to_dict()
    w=[1.0/(ec[r.class_name]*pc[r.event_id]) for r in train_df.itertuples(index=False)]
    g=torch.Generator(); g.manual_seed(int(seed))
    return WeightedRandomSampler(torch.tensor(w,dtype=torch.double),num_samples=len(train_df),replacement=True,generator=g)

class DS(Dataset):
    def __init__(self,frame,root):
        self.frame=frame.reset_index(drop=True); self.root=Path(root); self._h=OrderedDict()
    def __len__(self): return len(self.frame)
    def _handle(self,rel):
        rel=str(rel)
        if rel in self._h:
            f=self._h.pop(rel); self._h[rel]=f; return f
        f=h5py.File(self.root/rel,'r'); self._h[rel]=f
        while len(self._h)>3:
            _,old=self._h.popitem(last=False); old.close()
        return f
    def __getitem__(self,i):
        r=self.frame.iloc[int(i)]; f=self._handle(r.shard_file); x=np.asarray(f['raw'][int(r.shard_row)],dtype=np.float32)
        if x.shape!=(8192,128): raise ValueError(x.shape)
        return {'x':torch.from_numpy(x).unsqueeze(0),'y':torch.tensor(int(r.task_class_id)),
                'u':torch.tensor(float(r.u_centroid),dtype=torch.float32),'sample_id':str(r.sample_id),
                'event_id':str(r.event_id),'class_name':str(r.class_name)}
    def __del__(self):
        for f in self._h.values():
            try:f.close()
            except:pass

def collate(b):
    return {'x':torch.stack([q['x'] for q in b]),'y':torch.stack([q['y'] for q in b]),'u':torch.stack([q['u'] for q in b]),
            'sample_id':[q['sample_id'] for q in b],'event_id':[q['event_id'] for q in b],'class_name':[q['class_name'] for q in b]}

def train_loader(fr,root,cfg,seed):
    ds=DS(fr['train'],root)
    return DataLoader(ds,batch_size=cfg.batch_size,sampler=sampler(fr['train'],seed),num_workers=0,pin_memory=True,collate_fn=collate)

def eval_loader(frame,root,cfg):
    return DataLoader(DS(frame,root),batch_size=cfg.batch_size,shuffle=False,num_workers=0,pin_memory=True,collate_fn=collate)

def stage_shards(phase2,registry,local_root):
    phase2=Path(phase2); local_root=Path(local_root); rels=sorted(set(registry.shard_file.astype(str)))
    for i,rel in enumerate(rels,1):
        src=phase2/rel; dst=local_root/rel; dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.is_file() or dst.stat().st_size!=src.stat().st_size:
            tmp=Path(str(dst)+'.partial'); tmp.unlink(missing_ok=True); print(f'[{i}/{len(rels)}] stage {rel}'); shutil.copy2(src,tmp); tmp.replace(dst)
    (local_root/'PHASE2_SHA256.txt').write_text(EXPECTED_PHASE2_SHA+'\n')
    return local_root
