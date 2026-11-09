from pathlib import Path
import pandas as pd, numpy as np, hashlib, json
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

ROOT=Path('/mnt/data/b1r_work/Phase2_FINAL_AUDIT')
OUT=Path('/mnt/data/b1r_minchange'); OUT.mkdir(exist_ok=True)
SRC=ROOT/'metadata'/'SAMPLE_REGISTRY.csv'
df=pd.read_csv(SRC)
roles=['train','val','test']; ridx={r:i for i,r in enumerate(roles)}

# Event metadata
em=df.groupby('event_id').agg(
    class_id=('class_id','first'), class_name=('class_name','first'), recording_id=('recording_id','first'),
    old_role=('b1_role','first'), old_group=('b1_leakage_group_id','first'), n_patches=('sample_id','size'),
).reset_index()
assert em.event_id.nunique()==137

# Union-find: preserve old leakage-group atomicity and add actual support-overlap edges.
parent={e:e for e in em.event_id}; rank={e:0 for e in em.event_id}
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra==rb:return
    if rank[ra]<rank[rb]: ra,rb=rb,ra
    parent[rb]=ra
    if rank[ra]==rank[rb]: rank[ra]+=1

for _,sub in em.groupby('old_group',sort=True):
    ev=sorted(sub.event_id.tolist())
    for e in ev[1:]: union(ev[0],e)

support_edges=set()
# raw_time_end / raw_channel_end are treated as exclusive request bounds.
for rec, sub in df.groupby('recording_id',sort=True):
    a=sub[['sample_id','event_id','raw_time_start','raw_time_end','raw_channel_start','raw_channel_end']].sort_values(['raw_time_start','raw_channel_start','sample_id']).to_records(index=False)
    n=len(a)
    for i in range(n):
        j=i+1
        while j<n and a[j].raw_time_start < a[i].raw_time_end:
            if (a[i].event_id != a[j].event_id and
                a[j].raw_time_end > a[i].raw_time_start and
                a[j].raw_channel_start < a[i].raw_channel_end and
                a[j].raw_channel_end > a[i].raw_channel_start):
                x,y=sorted((str(a[i].event_id),str(a[j].event_id)))
                support_edges.add((x,y))
            j+=1
for x,y in support_edges: union(x,y)

members=defaultdict(list)
for e in sorted(em.event_id): members[find(e)].append(e)
map_rows=[]
for evs in members.values():
    evs=sorted(evs)
    gid='B1R_'+hashlib.sha256('|'.join(evs).encode()).hexdigest()[:16]
    for e in evs: map_rows.append((e,gid))
gmap=pd.DataFrame(map_rows,columns=['event_id','b1r_group_id'])
em=em.merge(gmap,on='event_id',validate='one_to_one')

# Group-level stats and old role counts.
groups=[]
for gid,s in em.groupby('b1r_group_id',sort=True):
    assert s.class_name.nunique()==1 and s.class_id.nunique()==1 and s.recording_id.nunique()==1
    row={
        'b1r_group_id':gid,'class_id':int(s.class_id.iloc[0]),'class_name':s.class_name.iloc[0],
        'recording_id':s.recording_id.iloc[0],'n_events':len(s),'n_patches':int(s.n_patches.sum()),
        'event_ids':'|'.join(sorted(s.event_id)),
    }
    for r in roles:
        ss=s[s.old_role==r]
        row[f'old_{r}_events']=len(ss); row[f'old_{r}_patches']=int(ss.n_patches.sum())
    groups.append(row)
g=pd.DataFrame(groups).sort_values(['class_id','b1r_group_id']).reset_index(drop=True)
G=len(g); R=3

# Original per-class targets are metadata-only and frozen before model training.
old_event_targets=em.groupby(['class_name','old_role']).size().unstack(fill_value=0).reindex(columns=roles,fill_value=0)
old_patch_targets=df.groupby(['class_name','b1_role']).size().unstack(fill_value=0).reindex(columns=roles,fill_value=0)
classes=sorted(em.class_name.unique(), key=lambda c:int(em.loc[em.class_name==c,'class_id'].iloc[0]))

# MILP x[g,r] binary plus absolute-deviation variables for class event counts and class patch counts.
# Lexicographic-like weights: class event-count preservation >> event-role changes >> patch-count preservation.
# This uses no model predictions/results.
nx=G*R
# auxiliary d_event c,r and d_patch c,r
ce_index={}; cp_index={}; k=nx
for c in classes:
    for r in roles: ce_index[(c,r)]=k; k+=1
for c in classes:
    for r in roles: cp_index[(c,r)]=k; k+=1
nvar=k
cvec=np.zeros(nvar)
EVENT_DEV_W=1_000.0; CHANGE_W=1_000_000.0; PATCH_DEV_W=1.0
for idx in ce_index.values(): cvec[idx]=EVENT_DEV_W
for idx in cp_index.values(): cvec[idx]=PATCH_DEV_W
# change coefficient: number of events in group whose old role differs
for i,row in g.iterrows():
    for r in roles:
        changed=int(row['n_events']-row[f'old_{r}_events'])
        cvec[i*R+ridx[r]]=CHANGE_W*changed
        # deterministic tiny tie-breaker; cannot affect higher-priority integer objectives
        cvec[i*R+ridx[r]]+=1e-6*((i+1)*10+(ridx[r]+1))

constraints=[]; lbs=[]; ubs=[]
# each group exactly one role
for i in range(G):
    row=np.zeros(nvar); row[i*R:(i+1)*R]=1
    constraints.append(row); lbs.append(1); ubs.append(1)
# coverage constraints
for c in classes:
    inds=g.index[g.class_name==c].tolist(); ng=len(inds)
    for r in ['train','test']:
        row=np.zeros(nvar)
        for i in inds: row[i*R+ridx[r]]=g.loc[i,'n_events']
        constraints.append(row); lbs.append(1); ubs.append(np.inf)
    # validation is required when >=3 independent B1-R groups exist; otherwise impossible together with train/test.
    row=np.zeros(nvar)
    for i in inds: row[i*R+ridx['val']]=g.loc[i,'n_events']
    constraints.append(row)
    if ng>=3: lbs.append(1); ubs.append(np.inf)
    else: lbs.append(0); ubs.append(0)
# absolute deviations class event counts: d >= +/- (count-target)
for c in classes:
    inds=g.index[g.class_name==c].tolist()
    for r in roles:
        target=int(old_event_targets.loc[c,r])
        # count - d <= target
        row=np.zeros(nvar)
        for i in inds: row[i*R+ridx[r]]=g.loc[i,'n_events']
        row[ce_index[(c,r)]]=-1
        constraints.append(row); lbs.append(-np.inf); ubs.append(target)
        # -count - d <= -target
        row2=np.zeros(nvar)
        for i in inds: row2[i*R+ridx[r]]=-g.loc[i,'n_events']
        row2[ce_index[(c,r)]]=-1
        constraints.append(row2); lbs.append(-np.inf); ubs.append(-target)
# absolute deviations class patch counts
for c in classes:
    inds=g.index[g.class_name==c].tolist()
    for r in roles:
        target=int(old_patch_targets.loc[c,r])
        row=np.zeros(nvar)
        for i in inds: row[i*R+ridx[r]]=g.loc[i,'n_patches']
        row[cp_index[(c,r)]]=-1
        constraints.append(row); lbs.append(-np.inf); ubs.append(target)
        row2=np.zeros(nvar)
        for i in inds: row2[i*R+ridx[r]]=-g.loc[i,'n_patches']
        row2[cp_index[(c,r)]]=-1
        constraints.append(row2); lbs.append(-np.inf); ubs.append(-target)
A=csr_matrix(np.vstack(constraints))
lc=LinearConstraint(A,np.array(lbs),np.array(ubs))
lower=np.zeros(nvar); upper=np.full(nvar,np.inf); upper[:nx]=1
integrality=np.zeros(nvar,dtype=int); integrality[:nx]=1
res=milp(cvec,integrality=integrality,bounds=Bounds(lower,upper),constraints=lc,options={'time_limit':60})
if not res.success: raise RuntimeError(res.message)
x=res.x[:nx].reshape(G,R)
assign=[]
for i in range(G):
    rr=int(np.argmax(x[i])); assert x[i,rr]>.5
    assign.append(roles[rr])
g['b1r_role']=assign

# Event and sample assignments
emap=em.merge(g[['b1r_group_id','b1r_role']],on='b1r_group_id',validate='many_to_one')
smap=df[['sample_id','event_id','class_id','class_name','recording_id','b1_role','b1_leakage_group_id','raw_time_start','raw_time_end','raw_channel_start','raw_channel_end','n_patches' if 'n_patches' in df else 'sample_id']].copy() if False else df[['sample_id','event_id','class_id','class_name','recording_id','b1_role','b1_leakage_group_id','raw_time_start','raw_time_end','raw_channel_start','raw_channel_end','raw_patch_sha256','model_patch_sha256','shard_file','shard_row']].copy()
smap=smap.merge(emap[['event_id','b1r_group_id','b1r_role']],on='event_id',validate='many_to_one')

# Verify actual support overlap across B1-R roles.
overlap_rows=[]
rolepairs={tuple(sorted(p)) for p in [('train','val'),('train','test'),('val','test')]}
for rec,sub in smap.groupby('recording_id',sort=True):
    a=sub.sort_values(['raw_time_start','raw_channel_start','sample_id']).to_records(index=False)
    n=len(a)
    for i in range(n):
        j=i+1
        while j<n and a[j].raw_time_start < a[i].raw_time_end:
            if a[i].b1r_role!=a[j].b1r_role and a[j].raw_time_end>a[i].raw_time_start and a[j].raw_channel_start<a[i].raw_channel_end and a[j].raw_channel_end>a[i].raw_channel_start:
                overlap_rows.append({
                    'recording_id':rec,'sample_id_a':a[i].sample_id,'event_id_a':a[i].event_id,'role_a':a[i].b1r_role,
                    'sample_id_b':a[j].sample_id,'event_id_b':a[j].event_id,'role_b':a[j].b1r_role,
                    'time_overlap':int(min(a[i].raw_time_end,a[j].raw_time_end)-max(a[i].raw_time_start,a[j].raw_time_start)),
                    'channel_overlap':int(min(a[i].raw_channel_end,a[j].raw_channel_end)-max(a[i].raw_channel_start,a[j].raw_channel_start)),
                })
            j+=1
over=pd.DataFrame(overlap_rows)
assert len(over)==0, f'B1-R overlap gate FAIL: {len(over)}'
# Verify old leakage groups don't cross
chk=emap.groupby('old_group').b1r_role.nunique()
assert chk.max()==1
# exact duplicate SHA cross roles
for col in ['raw_patch_sha256','model_patch_sha256']:
    z=smap.groupby(col).b1r_role.nunique()
    assert z.max()==1
# event counts and coverage
class_stats=emap.groupby(['class_name','b1r_role']).size().unstack(fill_value=0).reindex(index=classes,columns=roles,fill_value=0)
patch_stats=smap.groupby(['class_name','b1r_role']).size().unstack(fill_value=0).reindex(index=classes,columns=roles,fill_value=0)
assert (class_stats['train']>=1).all() and (class_stats['test']>=1).all()
for c in classes:
    if g[g.class_name==c].b1r_group_id.nunique()>=3: assert class_stats.loc[c,'val']>=1
    else: assert class_stats.loc[c,'val']==0
# total consistency
assert class_stats.values.sum()==137 and patch_stats.values.sum()==2522

# Changed roles
emap['role_changed']=emap.old_role!=emap.b1r_role
changed=emap[emap.role_changed].copy()

# recording partition audit
rec_rows=[]
for rec,s in emap.groupby('recording_id',sort=True):
    rec_rows.append({
        'recording_id':rec,'class_name':s.class_name.iloc[0],'n_events':len(s),
        **{f'{r}_events':int((s.b1r_role==r).sum()) for r in roles},
        'n_partitions':int(s.b1r_role.nunique()),'partitions':'|'.join(sorted(s.b1r_role.unique()))
    })
recdf=pd.DataFrame(rec_rows)

# Persist core outputs before manifest hash.
g.to_csv(OUT/'B1R_GROUP_REGISTRY.csv',index=False)
emap[['event_id','class_id','class_name','recording_id','old_group','old_role','n_patches','b1r_group_id','b1r_role','role_changed']].sort_values(['class_id','recording_id','event_id']).to_csv(OUT/'B1R_TRAIN_VAL_TEST.csv',index=False)
smap[['sample_id','event_id','class_id','class_name','recording_id','b1r_group_id','b1r_role','raw_time_start','raw_time_end','raw_channel_start','raw_channel_end','raw_patch_sha256','model_patch_sha256','shard_file','shard_row']].sort_values(['b1r_role','class_id','event_id','sample_id']).to_csv(OUT/'B1R_SAMPLE_ASSIGNMENTS.csv',index=False)
pd.DataFrame(sorted(support_edges),columns=['event_id_a','event_id_b']).to_csv(OUT/'B1R_ACTUAL_SUPPORT_EVENT_EDGES.csv',index=False)
class_out=pd.concat({'old':old_event_targets,'B1-R':class_stats},axis=1)
class_out.to_csv(OUT/'B1R_CLASS_EVENT_COUNTS.csv')
patch_out=pd.concat({'old':old_patch_targets,'B1-R':patch_stats},axis=1)
patch_out.to_csv(OUT/'B1R_CLASS_PATCH_COUNTS.csv')
changed.to_csv(OUT/'B1R_ROLE_CHANGES.csv',index=False)
recdf.to_csv(OUT/'B1R_RECORDING_PARTITION_AUDIT.csv',index=False)
# gate table
gates=[
 ('source_phase2_samples_2522',len(df)==2522),('source_events_137',em.event_id.nunique()==137),
 ('source_recordings_16',em.recording_id.nunique()==16),('old_leakage_groups_preserved',chk.max()==1),
 ('b1r_actual_raw_support_cross_partition_overlap_zero',len(over)==0),
 ('b1r_raw_patch_sha_cross_partition_duplicate_zero',smap.groupby('raw_patch_sha256').b1r_role.nunique().max()==1),
 ('b1r_model_patch_sha_cross_partition_duplicate_zero',smap.groupby('model_patch_sha256').b1r_role.nunique().max()==1),
 ('train_all_9_classes',(class_stats.train>0).sum()==9),('test_all_9_classes',(class_stats.test>0).sum()==9),
 ('validation_classes_maximally_supported',(class_stats.val>0).sum()==7),
 ('construction_validation_infeasible_groups_lt3',g[g.class_name=='construction'].b1r_group_id.nunique()==2 and class_stats.loc['construction','val']==0),
 ('running_validation_infeasible_groups_lt3',g[g.class_name=='running'].b1r_group_id.nunique()==2 and class_stats.loc['running','val']==0),
]
pd.DataFrame(gates,columns=['gate','passed']).to_csv(OUT/'B1R_INTEGRITY_GATES.csv',index=False)
assert all(v for _,v in gates)

# hashes and manifest
files=[p for p in OUT.iterdir() if p.is_file() and p.name not in {'B1R_MANIFEST.json','B1R_SHA256.txt','B1R_FILE_HASHES.json'}]
def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
file_hashes={p.name:sha(p) for p in sorted(files)}
(OUT/'B1R_FILE_HASHES.json').write_text(json.dumps(file_hashes,indent=2,sort_keys=True))
phase2_sha=(ROOT/'PHASE2_SHA256.txt').read_text().strip()
phase1_sha=str(df.phase1_sha.iloc[0])
manifest={
 'benchmark_name':'B1-R','version':'B1R_v1.0','builder':'actual-support-overlap union + deterministic constrained assignment',
 'source_phase1_sha256':phase1_sha,'source_phase2_sha256':phase2_sha,
 'n_events':137,'n_samples':2522,'n_recordings':16,'old_leakage_groups':int(em.old_group.nunique()),'b1r_atomic_groups':int(g.b1r_group_id.nunique()),
 'actual_support_event_edges':len(support_edges),
 'split_event_counts':{r:int((emap.b1r_role==r).sum()) for r in roles},
 'split_patch_counts':{r:int((smap.b1r_role==r).sum()) for r in roles},
 'class_event_counts':{c:{r:int(class_stats.loc[c,r]) for r in roles} for c in classes},
 'validation_missing_classes':['construction','running'],
 'validation_missing_reason':'Each class has only two independent B1-R atomic support components; train+test coverage leaves no third independent component for validation.',
 'optimization_contract':{
   'inputs':'metadata/provenance only; no model predictions or metrics used',
   'hard_constraints':['each atomic group assigned to one role','old leakage groups remain atomic','actual raw-support overlap edges remain atomic','all 9 classes in train and test','validation required for classes with >=3 independent groups; forced absent when only 2 groups'],
   'objective_priority':['minimize number of events changing role from original frozen B1','minimize per-class event-count deviation from original frozen B1','minimize per-class patch-count deviation','deterministic group/role tie-break'],
 },
 'interval_semantics':'raw_time_start <= t < raw_time_end; raw_channel_start <= c < raw_channel_end',
 'file_hashes':file_hashes,
}
raw=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode(); fp=hashlib.sha256(raw).hexdigest(); manifest['b1r_sha256']=fp
(OUT/'B1R_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
(OUT/'B1R_SHA256.txt').write_text(fp+'\n')

# summary
print('MILP objective',res.fun)
print('groups',G,'support edges',len(support_edges))
print('event counts\n',class_stats,'\nTOTAL',class_stats.sum().to_dict())
print('patch totals',patch_stats.sum().to_dict())
print('changed events',int(emap.role_changed.sum()))
print('changed by class',changed.class_name.value_counts().to_dict())
print('recording n_partitions',recdf.n_partitions.value_counts().sort_index().to_dict())
print('B1R SHA',fp)
