from pathlib import Path
import json, os, random
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from .config import EXPECTED_PROTOCOL_SHA
from .models import build_model,is_scal,grl_schedule
from .data import train_loader,eval_loader
from .metrics import aggregate_events,metrics,bootstrap,recall_table

def seed_all(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark=False

def atomic_save(obj,path):
    path=Path(path); tmp=Path(str(path)+'.partial'); tmp.unlink(missing_ok=True); torch.save(obj,tmp); os.replace(tmp,path)

def copy_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def predict(model,loader,device,name,emb=False):
    model.eval(); rows=[]; scal=is_scal(name)
    with torch.no_grad():
        for b in loader:
            x=b['x'].to(device); y=b['y'].to(device)
            if scal: logits,_,z=model(x,y=None,grl_lambda=0.0,return_embedding=True)
            else: logits,z=model(x,return_embedding=True)
            p=torch.softmax(logits.float(),1).cpu().numpy(); zz=z.float().cpu().numpy()
            for i in range(len(y)):
                r={'sample_id':b['sample_id'][i],'event_id':b['event_id'][i],'class_name':b['class_name'][i],'y':int(y[i].cpu())}
                r.update({f'p_{j}':float(p[i,j]) for j in range(2)})
                if emb: r.update({f'z_{j}':float(zz[i,j]) for j in range(zz.shape[1])})
                rows.append(r)
    return pd.DataFrame(rows)

def init_gate(seed,cfg):
    seed_all(seed); a=build_model('spectrum_rdft',2,cfg.embedding_dim)
    seed_all(seed); b=build_model('scal_rdft',2,cfg.embedding_dim)
    for k,v in a.backbone.state_dict().items():
        assert torch.equal(v,b.backbone.state_dict()[k]), k
    for k,v in a.head.state_dict().items():
        assert torch.equal(v,b.event_head.state_dict()[k]), k
    return {'seed':int(seed),'spectral_backbone_head_equal':True}

def train_one(name,seed,fr,data_root,cfg,out_dir,smoke=False):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    full_epochs=cfg.epochs; target=cfg.smoke_epochs if smoke else full_epochs
    seed_all(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(name,2,cfg.embedding_dim).to(device); scal=is_scal(name)
    loader=train_loader(fr,data_root,cfg,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    amp=bool(cfg.amp and device.type=='cuda'); scaler=torch.amp.GradScaler('cuda',enabled=amp)
    resume=out_dir/'resume_last.pt'; hist=[]; start=0
    if resume.is_file() and not smoke:
        st=torch.load(resume,map_location='cpu',weights_only=False)
        if st.get('name')==name and int(st.get('seed',-1))==int(seed) and st.get('protocol_sha')==EXPECTED_PROTOCOL_SHA:
            model.load_state_dict(st['state_dict']); opt.load_state_dict(st['optimizer'])
            if amp and st.get('scaler') is not None: scaler.load_state_dict(st['scaler'])
            start=int(st['epoch']); hist=list(st['history'])
            if st.get('sampler_state') is not None: loader.sampler.generator.set_state(st['sampler_state'].cpu().to(torch.uint8))
            if st.get('torch_state') is not None: torch.set_rng_state(st['torch_state'].cpu().to(torch.uint8))
            if torch.cuda.is_available() and st.get('cuda_state') is not None: torch.cuda.set_rng_state_all([x.cpu().to(torch.uint8) for x in st['cuda_state']])
            if st.get('np_state') is not None: np.random.set_state(st['np_state'])
            if st.get('py_state') is not None: random.setstate(st['py_state'])
            print(f'RESUME {name}/seed{seed} from epoch {start}')
    nsteps=max(1,len(loader))
    for ep in range(start,target):
        model.train(); losses=[]; ces=[]; locs=[]; lams=[]
        for step,b in enumerate(loader):
            if smoke and step>=cfg.smoke_max_train_batches: break
            x=b['x'].to(device); y=b['y'].to(device); u=b['u'].to(device)
            progress=(ep*nsteps+step)/max(1,full_epochs*nsteps-1); lam=grl_schedule(progress,cfg.adv_lambda_max,cfg.adv_gamma) if scal else 0.0
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                if scal:
                    logits,up=model(x,y,lam); ce=F.cross_entropy(logits,y); loc=F.smooth_l1_loss(up,u); loss=ce+cfg.location_loss_weight*loc
                else:
                    logits=model(x); ce=F.cross_entropy(logits,y); loc=torch.zeros((),device=device); loss=ce
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip_norm); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu())); ces.append(float(ce.detach().cpu())); locs.append(float(loc.detach().cpu())); lams.append(float(lam))
        rec={'epoch':ep+1,'train_loss':float(np.mean(losses)),'train_ce':float(np.mean(ces)),'train_loc':float(np.mean(locs)),
             'grl_lambda_last':float(lams[-1]) if lams else 0.0}
        hist.append(rec); pd.DataFrame(hist).to_csv(out_dir/'history.csv',index=False)
        state={'state_dict':copy_state(model),'optimizer':opt.state_dict(),'scaler':scaler.state_dict() if amp else None,
               'epoch':ep+1,'name':name,'seed':seed,'history':hist,'protocol_sha':EXPECTED_PROTOCOL_SHA,
               'sampler_state':loader.sampler.generator.get_state(),'torch_state':torch.get_rng_state(),
               'cuda_state':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
               'np_state':np.random.get_state(),'py_state':random.getstate()}
        atomic_save(state,resume)
        print(f'[{name}/seed{seed}] epoch {ep+1:03d}/{target} loss={rec["train_loss"]:.4f}')
    if smoke:
        return {'status':'PASS','name':name,'seed':seed,'epochs':target,'test_accessed':False}
    if target!=full_epochs: raise RuntimeError('full trajectory incomplete')
    complete={'status':'TRAINING_COMPLETE','name':name,'seed':seed,'epochs_completed':full_epochs,'protocol_sha':EXPECTED_PROTOCOL_SHA}
    (out_dir/'TRAINING_COMPLETE.json').write_text(json.dumps(complete,indent=2,sort_keys=True))
    # Only now construct target-test loader.
    test=eval_loader(fr['test'],data_root,cfg)
    st=torch.load(resume,map_location='cpu',weights_only=False); model.load_state_dict(st['state_dict']); model.to(device)
    pp=predict(model,test,device,name,emb=True); pp.to_csv(out_dir/'target_patch_predictions_embeddings.csv',index=False)
    ev=aggregate_events(pp); ev.to_csv(out_dir/'target_event_predictions.csv',index=False)
    pcols=['p_0','p_1']; m=metrics(ev.y,ev[pcols],2)
    pd.DataFrame([bootstrap(ev,q,cfg.bootstrap_reps,seed) for q in ('macro_f1','balanced_accuracy')]).to_csv(out_dir/'target_event_bootstrap_ci.csv',index=False)
    recall_table(ev).to_csv(out_dir/'target_class_recall.csv',index=False)
    completed={**complete,'status':'COMPLETE','headline_checkpoint':'epoch_80_fixed','target_event_metrics':m}
    (out_dir/'COMPLETED.json').write_text(json.dumps(completed,indent=2,sort_keys=True))
    return completed
