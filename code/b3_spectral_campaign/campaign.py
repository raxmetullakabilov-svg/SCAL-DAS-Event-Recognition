from pathlib import Path
import json, zipfile, hashlib, shutil
import pandas as pd
from .config import Config,EXPECTED_PROTOCOL_SHA
from .data import auto_detect_phase2,load_registry,assert_target_isolation,frames,stage_shards
from .train import init_gate,train_one

def bootstrap(project_root,package_root,stage_local=True):
    cfg=Config(); phase2=auto_detect_phase2(project_root)
    reg=load_registry(phase2,Path(package_root)/'frozen'); assert_target_isolation(reg); fr=frames(reg)
    data_root=phase2
    if stage_local: data_root=stage_shards(phase2,reg[reg.spectral_campaign_role.isin(['train','test'])],Path('/content/b3_spectral_phase2_staged'))
    return cfg,phase2,data_root,reg,fr

def smoke(fr,data_root,cfg,out_root):
    out=Path(out_root)/'SMOKE_seed2026'; out.mkdir(parents=True,exist_ok=True)
    g=init_gate(2026,cfg); rows=[]
    for name in cfg.models:
        d=out/name
        if d.exists(): shutil.rmtree(d)
        rows.append(train_one(name,2026,fr,data_root,cfg,d,smoke=True))
    obj={'status':'PASS','initialization_gate':g,'runs':rows,'test_accessed':False,'protocol_sha':EXPECTED_PROTOCOL_SHA}
    (out/'SMOKE_GATE.json').write_text(json.dumps(obj,indent=2,sort_keys=True)); return obj

def full(fr,data_root,cfg,out_root):
    out_root=Path(out_root); rows=[]
    for seed in cfg.seeds:
        init_gate(seed,cfg)
        for name in cfg.models:
            out=out_root/'runs'/name/f'seed_{seed}'; comp=out/'COMPLETED.json'
            if comp.is_file():
                o=json.loads(comp.read_text())
                if o.get('status')=='COMPLETE' and o.get('protocol_sha')==EXPECTED_PROTOCOL_SHA:
                    print('SKIP VERIFIED COMPLETE',name,seed); rows.append(o); continue
            rows.append(train_one(name,seed,fr,data_root,cfg,out,smoke=False))
    return rows

def aggregate(out_root,cfg):
    rows=[]; boots=[]; recalls=[]
    for name in cfg.models:
        for seed in cfg.seeds:
            d=Path(out_root)/'runs'/name/f'seed_{seed}'
            c=json.loads((d/'COMPLETED.json').read_text()); m=c['target_event_metrics']
            rows.append({'model':name,'seed':seed,**m})
            b=pd.read_csv(d/'target_event_bootstrap_ci.csv'); b.insert(0,'seed',seed); b.insert(0,'model',name); boots.append(b)
            r=pd.read_csv(d/'target_class_recall.csv'); r.insert(0,'seed',seed); r.insert(0,'model',name); recalls.append(r)
    per=pd.DataFrame(rows); per.to_csv(Path(out_root)/'B3_SPECTRAL_PER_SEED.csv',index=False)
    agg=[]
    for name,g in per.groupby('model'):
        r={'model':name,'n_seeds':len(g)}
        for c in ['accuracy','balanced_accuracy','macro_f1','nll','brier','ece5']:
            r[c+'_mean']=float(g[c].mean()); r[c+'_std']=float(g[c].std(ddof=1))
        agg.append(r)
    ag=pd.DataFrame(agg); ag.to_csv(Path(out_root)/'B3_SPECTRAL_AGGREGATE.csv',index=False)
    pd.concat(boots,ignore_index=True).to_csv(Path(out_root)/'B3_SPECTRAL_BOOTSTRAP_CI.csv',index=False)
    pd.concat(recalls,ignore_index=True).to_csv(Path(out_root)/'B3_SPECTRAL_CLASS_RECALL.csv',index=False)
    return per,ag

def make_zip(out_root,package_root):
    out_root=Path(out_root); package_root=Path(package_root); out=out_root/'B3_SPECTRAL_VALIDATION_AUDIT.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out_root.rglob('*')):
            if p.is_file() and p!=out and p.suffix.lower()!='.pt': z.write(p,arcname='results/'+p.relative_to(out_root).as_posix())
        for p in sorted((package_root/'frozen').rglob('*')):
            if p.is_file(): z.write(p,arcname='frozen/'+p.name)
        for p in sorted((package_root/'b3_spectral_campaign').glob('*.py')):
            z.write(p,arcname='source/'+p.name)
    h=hashlib.sha256(out.read_bytes()).hexdigest(); (out_root/'B3_SPECTRAL_VALIDATION_AUDIT_SHA256.txt').write_text(h+'\n'); return out,h
