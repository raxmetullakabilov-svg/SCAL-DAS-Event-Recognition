from pathlib import Path
import json, os, random, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from .models import build_model,is_scal,grl_schedule
from .data import make_loaders
from .metrics import aggregate_events,classification_metrics,validation_metrics,stratified_event_bootstrap,class_recall_table
from .config import EXPECTED_B1R_SHA, EXPECTED_PROTOCOL_SHA

def seed_everything(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark=False

def _atomic_torch_save(obj,path):
    path=Path(path); tmp=Path(str(path)+'.partial'); tmp.unlink(missing_ok=True); torch.save(obj,tmp); os.replace(tmp,path)

def _predict(model,loader,device,model_name,collect_embeddings=False,max_batches=None):
    model.eval(); rows=[]; scal=is_scal(model_name)
    with torch.no_grad():
        for bi,b in enumerate(loader):
            if max_batches is not None and bi>=max_batches: break
            x=b['x'].to(device,non_blocking=True); y=b['y'].to(device,non_blocking=True)
            if scal:
                logits,_,z=model(x,y=None,grl_lambda=0.0,return_embedding=True)
            else:
                logits,z=model(x,return_embedding=True)
            probs=torch.softmax(logits.float(),dim=1).cpu().numpy(); zz=z.float().cpu().numpy()
            yy=y.cpu().numpy()
            for i in range(len(yy)):
                r={'sample_id':b['sample_id'][i],'event_id':b['event_id'][i],'class_name':b['class_name'][i],'y':int(yy[i])}
                r.update({f'p_{j}':float(probs[i,j]) for j in range(probs.shape[1])})
                if collect_embeddings: r.update({f'z_{j}':float(zz[i,j]) for j in range(zz.shape[1])})
                rows.append(r)
    return pd.DataFrame(rows)

def _load_state(model,path,device):
    st=torch.load(path,map_location='cpu',weights_only=False); model.load_state_dict(st['state_dict']); model.to(device); return st

def _copy_state_dict_cpu(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def initialization_equivalence_gate(seed,cfg):
    seed_everything(seed); a=build_model('raw_ts_resnet',cfg.n_classes,cfg.embedding_dim)
    seed_everything(seed); b=build_model('raw_scal_das',cfg.n_classes,cfg.embedding_dim)
    for k,v in a.backbone.state_dict().items():
        if not torch.equal(v,b.backbone.state_dict()[k]): raise AssertionError(f'raw backbone init mismatch {k}')
    for k,v in a.head.state_dict().items():
        if not torch.equal(v,b.event_head.state_dict()[k]): raise AssertionError(f'raw head init mismatch {k}')
    seed_everything(seed); c=build_model('spectrum_rdft',cfg.n_classes,cfg.embedding_dim)
    seed_everything(seed); d=build_model('scal_rdft',cfg.n_classes,cfg.embedding_dim)
    for k,v in c.backbone.state_dict().items():
        if not torch.equal(v,d.backbone.state_dict()[k]): raise AssertionError(f'spectral backbone init mismatch {k}')
    for k,v in c.head.state_dict().items():
        if not torch.equal(v,d.event_head.state_dict()[k]): raise AssertionError(f'spectral head init mismatch {k}')
    return {'seed':int(seed),'raw_pair_equal':True,'spectral_pair_equal':True}

def train_run(model_name,seed,fr,phase2_root,cfg,out_dir,smoke=False):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    full_epochs=cfg.epochs; target_epochs=cfg.smoke_epochs if smoke else full_epochs
    seed_everything(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(model_name,cfg.n_classes,cfg.embedding_dim).to(device); scal=is_scal(model_name)
    loaders=make_loaders(fr,phase2_root,model_name,cfg,seed,include_test=False)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    amp_enabled=bool(cfg.amp and device.type=='cuda'); scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    primary=out_dir/'checkpoint_primary_nll.pt'; secondary=out_dir/'checkpoint_secondary_macroF1.pt'; resume=out_dir/'resume_last.pt'
    hist=[]; start_epoch=0; best_nll=float('inf'); best_f1=-float('inf'); best_nll_epoch=None; best_f1_epoch=None
    if resume.is_file() and not smoke:
        st=torch.load(resume,map_location='cpu',weights_only=False)
        if st.get('model_name')==model_name and int(st.get('seed',-1))==int(seed) and st.get('b1r_sha')==EXPECTED_B1R_SHA and st.get('protocol_sha')==EXPECTED_PROTOCOL_SHA:
            model.load_state_dict(st['state_dict']); opt.load_state_dict(st['optimizer'])
            if amp_enabled and st.get('scaler') is not None: scaler.load_state_dict(st['scaler'])
            start_epoch=int(st['epoch']); hist=list(st['history']); best_nll=float(st['best_nll']); best_f1=float(st['best_f1']); best_nll_epoch=st['best_nll_epoch']; best_f1_epoch=st['best_f1_epoch']
            if st.get('sampler_generator_state') is not None: loaders['train'].sampler.generator.set_state(st['sampler_generator_state'].detach().cpu().to(torch.uint8))
            if st.get('torch_rng_state') is not None: torch.set_rng_state(st['torch_rng_state'].detach().cpu().to(torch.uint8))
            if torch.cuda.is_available() and st.get('cuda_rng_state_all') is not None: torch.cuda.set_rng_state_all([x.detach().cpu().to(torch.uint8) for x in st['cuda_rng_state_all']])
            if st.get('numpy_rng_state') is not None: np.random.set_state(st['numpy_rng_state'])
            if st.get('python_rng_state') is not None: random.setstate(st['python_rng_state'])
            print(f'RESUME {model_name}/seed{seed} from epoch {start_epoch}')
    nsteps=max(1,len(loaders['train']))
    for epoch in range(start_epoch,target_epochs):
        model.train(); losses=[]; ces=[]; locs=[]; lams=[]
        for step,b in enumerate(loaders['train']):
            if smoke and step>=cfg.smoke_max_train_batches: break
            x=b['x'].to(device,non_blocking=True); y=b['y'].to(device,non_blocking=True); u=b['u'].to(device,non_blocking=True)
            progress=(epoch*nsteps+step)/max(1,full_epochs*nsteps-1); lam=grl_schedule(progress,cfg.adv_lambda_max,cfg.adv_gamma) if scal else 0.0
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_enabled):
                if scal:
                    logits,up=model(x,y,lam); ce=F.cross_entropy(logits,y); loc=F.smooth_l1_loss(up,u); loss=ce+cfg.location_loss_weight*loc
                else:
                    logits=model(x); ce=F.cross_entropy(logits,y); loc=torch.zeros((),device=device); loss=ce
            if not torch.isfinite(loss): raise RuntimeError('non-finite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip_norm); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu())); ces.append(float(ce.detach().cpu())); locs.append(float(loc.detach().cpu())); lams.append(float(lam))
        vp=_predict(model,loaders['val'],device,model_name,collect_embeddings=False)
        ve=aggregate_events(vp); vm=validation_metrics(ve,cfg.n_classes)
        rec={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'train_ce':float(np.mean(ces)),'train_loc':float(np.mean(locs)),
             'grl_lambda_last':float(lams[-1]) if lams else 0.0,**{k:v for k,v in vm.items() if k!='observed_class_ids'}}
        hist.append(rec); pd.DataFrame(hist).to_csv(out_dir/'history.csv',index=False)
        if vm['event_nll_9way'] < best_nll-cfg.tie_tolerance:
            best_nll=float(vm['event_nll_9way']); best_nll_epoch=epoch+1
            _atomic_torch_save({'state_dict':_copy_state_dict_cpu(model),'epoch':epoch+1,'seed':seed,'model_name':model_name,'val_event_nll_9way':best_nll,'b1r_sha':EXPECTED_B1R_SHA,'protocol_sha':EXPECTED_PROTOCOL_SHA},primary)
        if vm['event_macro_f1_observed7'] > best_f1+cfg.tie_tolerance:
            best_f1=float(vm['event_macro_f1_observed7']); best_f1_epoch=epoch+1
            _atomic_torch_save({'state_dict':_copy_state_dict_cpu(model),'epoch':epoch+1,'seed':seed,'model_name':model_name,'val_event_macro_f1_observed7':best_f1,'b1r_sha':EXPECTED_B1R_SHA,'protocol_sha':EXPECTED_PROTOCOL_SHA},secondary)
        state={'state_dict':_copy_state_dict_cpu(model),'optimizer':opt.state_dict(),'scaler':scaler.state_dict() if amp_enabled else None,'epoch':epoch+1,'seed':seed,'model_name':model_name,
               'history':hist,'best_nll':best_nll,'best_f1':best_f1,'best_nll_epoch':best_nll_epoch,'best_f1_epoch':best_f1_epoch,
               'sampler_generator_state':loaders['train'].sampler.generator.get_state(),'torch_rng_state':torch.get_rng_state(),
               'cuda_rng_state_all':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'numpy_rng_state':np.random.get_state(),'python_rng_state':random.getstate(),
               'b1r_sha':EXPECTED_B1R_SHA,'protocol_sha':EXPECTED_PROTOCOL_SHA}
        _atomic_torch_save(state,resume)
        print(f'[{model_name}/seed{seed}] epoch {epoch+1:03d}/{target_epochs} loss={rec["train_loss"]:.4f} valNLL9={vm["event_nll_9way"]:.5f} valF1obs7={vm["event_macro_f1_observed7"]:.4f}')
    if smoke:
        if not primary.is_file() or not secondary.is_file(): raise RuntimeError('smoke checkpoint gate failed')
        s1=_load_state(model,primary,device); s2=_load_state(model,secondary,device)
        return {'smoke':True,'model_name':model_name,'seed':seed,'epochs':target_epochs,'primary_epoch':int(s1['epoch']),'secondary_epoch':int(s2['epoch']),'finite':True}
    if target_epochs!=full_epochs: raise RuntimeError('Full run must reach frozen 60 epochs')
    # Training complete before any test loader is constructed.
    training_complete={'model_name':model_name,'seed':seed,'epochs_completed':full_epochs,'primary_epoch':int(best_nll_epoch),'primary_val_nll_9way':float(best_nll),
                       'secondary_epoch':int(best_f1_epoch),'secondary_val_macro_f1_observed7':float(best_f1),'b1r_sha':EXPECTED_B1R_SHA,'protocol_sha':EXPECTED_PROTOCOL_SHA}
    (out_dir/'TRAINING_COMPLETE.json').write_text(json.dumps(training_complete,indent=2,sort_keys=True))
    loaders_test=make_loaders(fr,phase2_root,model_name,cfg,seed,include_test=True)
    # Evaluate three predeclared checkpoints. Headline = primary only.
    checkpoints={'primary_nll':primary,'secondary_macroF1':secondary,'last_epoch60':resume}
    result_bundle={}
    for cname,cpath in checkpoints.items():
        st=_load_state(model,cpath,device)
        # resume_last stores state_dict too.
        tp=_predict(model,loaders_test['test'],device,model_name,collect_embeddings=False)
        te=aggregate_events(tp); pcols=[c for c in te if c.startswith('p_')]
        met={'patch':classification_metrics(tp.y,tp[[c for c in tp if c.startswith('p_')]],cfg.n_classes),
             'event':classification_metrics(te.y,te[pcols],cfg.n_classes),'checkpoint_epoch':int(st['epoch'])}
        result_bundle[cname]=met
        te.to_csv(out_dir/f'test_event_predictions__{cname}.csv',index=False)
    # Primary detailed artifacts + embeddings for later probe/reviewer analyses.
    _load_state(model,primary,device)
    detailed={}
    for split,key in [('train','train_eval'),('val','val'),('test','test')]:
        pp=_predict(model,loaders_test[key],device,model_name,collect_embeddings=True)
        pp=pp.merge(fr[split][['sample_id','u_centroid','location_bin_20','b1r_role']].drop_duplicates('sample_id'),on='sample_id',how='left',validate='one_to_one')
        pp.to_csv(out_dir/f'{split}_patch_predictions_embeddings__primary.csv',index=False)
        ee=aggregate_events(pp); ee.to_csv(out_dir/f'{split}_event_predictions__primary.csv',index=False)
        detailed[split]={'patch':classification_metrics(pp.y,pp[[c for c in pp if c.startswith('p_')]],cfg.n_classes),
                         'event':classification_metrics(ee.y,ee[[c for c in ee if c.startswith('p_')]],cfg.n_classes)}
    ev=pd.read_csv(out_dir/'test_event_predictions__primary.csv')
    cis=[stratified_event_bootstrap(ev,m,cfg.bootstrap_reps,seed,cfg.n_classes) for m in ('macro_f1','balanced_accuracy')]
    pd.DataFrame(cis).to_csv(out_dir/'test_event_bootstrap_ci__primary.csv',index=False)
    class_recall_table(ev,cfg.n_classes).to_csv(out_dir/'test_class_recall__primary.csv',index=False)
    (out_dir/'metrics_all_checkpoints.json').write_text(json.dumps(result_bundle,indent=2,sort_keys=True))
    (out_dir/'metrics_primary_detailed.json').write_text(json.dumps(detailed,indent=2,sort_keys=True))
    completed={**training_complete,'status':'COMPLETE','headline_checkpoint':'primary_nll','test_event_metrics_primary':result_bundle['primary_nll']['event']}
    (out_dir/'COMPLETED.json').write_text(json.dumps(completed,indent=2,sort_keys=True))
    return completed
