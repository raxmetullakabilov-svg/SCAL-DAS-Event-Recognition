from pathlib import Path
import json, zipfile, hashlib
import numpy as np
import pandas as pd
from .config import CampaignConfig,EXPECTED_B1R_SHA,EXPECTED_PROTOCOL_SHA
from .data import auto_detect_phase2,load_b1r_registry,assert_zero_cross_partition_raw_overlap,frames,stage_shards_to_local
from .train import initialization_equivalence_gate,train_run

def bootstrap_campaign(project_root,package_root,stage_local=True):
    cfg=CampaignConfig(); project_root=Path(project_root); package_root=Path(package_root)
    phase2=auto_detect_phase2(project_root); registry=load_b1r_registry(phase2,package_root/'frozen'); assert_zero_cross_partition_raw_overlap(registry)
    fr=frames(registry)
    data_root=phase2
    if stage_local:
        data_root=stage_shards_to_local(phase2,registry,Path('/content/b1r_phase2_staged'),verify_sha=False)
    return cfg,phase2,data_root,registry,fr

def run_smoke(fr,data_root,cfg,output_root):
    output_root=Path(output_root); smoke_root=output_root/'SMOKE_seed2026'; smoke_root.mkdir(parents=True,exist_ok=True)
    init=initialization_equivalence_gate(2026,cfg)
    rows=[]
    for m in ('raw_ts_resnet','raw_scal_das'):
        d=smoke_root/m
        if d.exists():
            import shutil; shutil.rmtree(d)
        rows.append(train_run(m,2026,fr,data_root,cfg,d,smoke=True))
    gate={'status':'PASS','initialization_equivalence':init,'runs':rows,'test_accessed':False,'b1r_sha':EXPECTED_B1R_SHA,'protocol_sha':EXPECTED_PROTOCOL_SHA}
    (smoke_root/'SMOKE_GATE.json').write_text(json.dumps(gate,indent=2,sort_keys=True))
    return gate

def run_full(fr,data_root,cfg,output_root):
    output_root=Path(output_root); runs=[]
    for seed in cfg.seeds:
        initialization_equivalence_gate(seed,cfg)
        for model in cfg.models:
            out=output_root/'runs'/model/f'seed_{seed}'
            comp=out/'COMPLETED.json'
            if comp.is_file():
                obj=json.loads(comp.read_text())
                if obj.get('status')=='COMPLETE' and obj.get('b1r_sha')==EXPECTED_B1R_SHA and obj.get('protocol_sha')==EXPECTED_PROTOCOL_SHA:
                    print('SKIP VERIFIED COMPLETE',model,seed); runs.append(obj); continue
            runs.append(train_run(model,seed,fr,data_root,cfg,out,smoke=False))
    return runs

def aggregate_campaign(output_root,cfg):
    output_root=Path(output_root); rows=[]; sens=[]; boots=[]; recalls=[]
    for model in cfg.models:
        for seed in cfg.seeds:
            rd=output_root/'runs'/model/f'seed_{seed}'
            comp=json.loads((rd/'COMPLETED.json').read_text()); mets=json.loads((rd/'metrics_all_checkpoints.json').read_text())
            for ck,v in mets.items():
                e=v['event']; sens.append({'model':model,'seed':seed,'checkpoint':ck,'checkpoint_epoch':v['checkpoint_epoch'],**{f'event_{k}':e[k] for k in ('accuracy','balanced_accuracy','macro_f1','nll','brier','ece5')}})
            e=mets['primary_nll']['event']; rows.append({'model':model,'seed':seed,'primary_epoch':comp['primary_epoch'],'secondary_epoch':comp['secondary_epoch'],**{f'event_{k}':e[k] for k in ('accuracy','balanced_accuracy','macro_f1','nll','brier','ece5')}})
            b=pd.read_csv(rd/'test_event_bootstrap_ci__primary.csv'); b.insert(0,'seed',seed); b.insert(0,'model',model); boots.append(b)
            c=pd.read_csv(rd/'test_class_recall__primary.csv'); c.insert(0,'seed',seed); c.insert(0,'model',model); recalls.append(c)
    per=pd.DataFrame(rows); per.to_csv(output_root/'B1R_CAMPAIGN_PRIMARY_PER_SEED.csv',index=False)
    agg=[]
    for model,g in per.groupby('model'):
        r={'model':model,'n_seeds':len(g)}
        for c in ('event_accuracy','event_balanced_accuracy','event_macro_f1','event_nll','event_brier','event_ece5'):
            r[c+'_mean']=float(g[c].mean()); r[c+'_std']=float(g[c].std(ddof=1))
        agg.append(r)
    pd.DataFrame(agg).to_csv(output_root/'B1R_CAMPAIGN_PRIMARY_AGGREGATE.csv',index=False)
    pd.DataFrame(sens).to_csv(output_root/'B1R_CHECKPOINT_SENSITIVITY_ALL.csv',index=False)
    pd.concat(boots,ignore_index=True).to_csv(output_root/'B1R_BOOTSTRAP_CI_ALL.csv',index=False)
    pd.concat(recalls,ignore_index=True).to_csv(output_root/'B1R_CLASS_RECALL_ALL.csv',index=False)
    return per,pd.DataFrame(agg)

def make_audit_zip(output_root,package_root):
    output_root=Path(output_root); package_root=Path(package_root); out=output_root/'B1R_RETRAINING_CAMPAIGN_AUDIT.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output_root.rglob('*')):
            if p.is_file() and p!=out and p.suffix.lower()!='.pt': z.write(p,arcname='results/'+p.relative_to(output_root).as_posix())
        for p in sorted((package_root/'frozen').rglob('*')):
            if p.is_file(): z.write(p,arcname='frozen/'+p.name)
        for p in sorted((package_root/'b1r_campaign').rglob('*.py')):
            if p.is_file(): z.write(p,arcname='source/'+p.relative_to(package_root).as_posix())
    h=hashlib.sha256(out.read_bytes()).hexdigest(); (output_root/'B1R_RETRAINING_CAMPAIGN_AUDIT_SHA256.txt').write_text(h+'\n')
    return out,h
