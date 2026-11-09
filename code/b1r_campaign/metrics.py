import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix

def _ece(y,probs,n_bins=5):
    probs=np.asarray(probs); y=np.asarray(y); conf=probs.max(1); pred=probs.argmax(1); correct=(pred==y).astype(float)
    if len(y)==0: return np.nan
    edges=np.quantile(conf,np.linspace(0,1,n_bins+1)); edges[0]-=1e-12; edges[-1]+=1e-12
    out=0.0
    for a,b in zip(edges[:-1],edges[1:]):
        m=(conf>a)&(conf<=b)
        if m.any(): out += m.mean()*abs(correct[m].mean()-conf[m].mean())
    return float(out)

def classification_metrics(y,probs,n_classes=9):
    y=np.asarray(y,dtype=int); probs=np.asarray(probs,float); probs=probs/np.clip(probs.sum(axis=1,keepdims=True),1e-12,None); pred=probs.argmax(1)
    labels=list(range(n_classes)); one=np.eye(n_classes)[y]
    return {
        'n':int(len(y)),
        'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
        'weighted_f1':float(f1_score(y,pred,labels=labels,average='weighted',zero_division=0)),
        'macro_precision':float(precision_score(y,pred,labels=labels,average='macro',zero_division=0)),
        'macro_recall':float(recall_score(y,pred,labels=labels,average='macro',zero_division=0)),
        'nll':float(log_loss(y,probs,labels=labels)),
        'brier':float(np.mean(np.sum((probs-one)**2,axis=1))),
        'ece5':_ece(y,probs,5),
    }

def validation_metrics(event_df,n_classes=9):
    pcols=[c for c in event_df.columns if c.startswith('p_')]
    y=event_df.y.to_numpy(int); probs=event_df[pcols].to_numpy(float); probs=probs/np.clip(probs.sum(axis=1,keepdims=True),1e-12,None); pred=probs.argmax(1)
    observed=sorted(np.unique(y).tolist())
    return {
        'n':int(len(y)),
        'event_nll_9way':float(log_loss(y,probs,labels=list(range(n_classes)))),
        'event_macro_f1_observed7':float(f1_score(y,pred,labels=observed,average='macro',zero_division=0)),
        'event_accuracy':float(accuracy_score(y,pred)),
        'event_balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'observed_class_ids':observed,
    }

def aggregate_events(pred_df):
    pcols=[c for c in pred_df.columns if c.startswith('p_')]
    rows=[]
    for eid,g in pred_df.groupby('event_id',sort=True):
        probs=g[pcols].to_numpy(float).mean(0); yvals=g.y.unique()
        if len(yvals)!=1: raise ValueError('event contains multiple labels')
        r={'event_id':eid,'y':int(yvals[0]),'class_name':str(g.class_name.iloc[0]),'n_patches':len(g)}
        r.update({c:float(v) for c,v in zip(pcols,probs)}); rows.append(r)
    return pd.DataFrame(rows)

def stratified_event_bootstrap(event_df,metric='macro_f1',reps=2000,seed=2026,n_classes=9):
    rng=np.random.default_rng(seed); pcols=[c for c in event_df.columns if c.startswith('p_')]; vals=[]
    groups={int(k):g.reset_index(drop=True) for k,g in event_df.groupby('y')}
    for _ in range(int(reps)):
        pieces=[]
        for _,g in groups.items():
            pieces.append(g.iloc[rng.integers(0,len(g),size=len(g))])
        b=pd.concat(pieces,ignore_index=True)
        vals.append(float(classification_metrics(b.y.to_numpy(),b[pcols].to_numpy(),n_classes)[metric]))
    estimate=float(classification_metrics(event_df.y,event_df[pcols],n_classes)[metric])
    return {'metric':metric,'estimate':estimate,'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),'reps':int(reps)}

def class_recall_table(event_df,n_classes=9):
    pcols=[c for c in event_df.columns if c.startswith('p_')]
    y=event_df.y.to_numpy(int); pred=event_df[pcols].to_numpy(float).argmax(1)
    cm=confusion_matrix(y,pred,labels=list(range(n_classes)))
    names={int(r.y):str(r.class_name) for r in event_df[['y','class_name']].drop_duplicates().itertuples(index=False)}
    rows=[]
    for k in range(n_classes):
        denom=cm[k].sum(); rec=float(cm[k,k]/denom) if denom else np.nan
        rows.append({'class_id':k,'class_name':names.get(k,''),'n_events':int(denom),'recall':rec})
    return pd.DataFrame(rows)
