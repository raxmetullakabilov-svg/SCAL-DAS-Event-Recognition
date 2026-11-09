import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix

def _ece(y,probs,n_bins=5):
    y=np.asarray(y); probs=np.asarray(probs,float); conf=probs.max(1); pred=probs.argmax(1); ok=(pred==y).astype(float)
    edges=np.quantile(conf,np.linspace(0,1,n_bins+1)); edges[0]-=1e-12; edges[-1]+=1e-12
    out=0.0
    for a,b in zip(edges[:-1],edges[1:]):
        m=(conf>a)&(conf<=b)
        if m.any(): out += m.mean()*abs(ok[m].mean()-conf[m].mean())
    return float(out)

def metrics(y,probs,n_classes=2):
    y=np.asarray(y,int); probs=np.asarray(probs,float); probs=probs/np.clip(probs.sum(1,keepdims=True),1e-12,None); pred=probs.argmax(1)
    one=np.eye(n_classes)[y]
    return {'n':int(len(y)),'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=list(range(n_classes)),average='macro',zero_division=0)),
            'nll':float(log_loss(y,probs,labels=list(range(n_classes)))),
            'brier':float(np.mean(np.sum((probs-one)**2,axis=1))),
            'ece5':_ece(y,probs,5)}

def aggregate_events(df):
    pcols=[c for c in df if c.startswith('p_')]; rows=[]
    for eid,g in df.groupby('event_id',sort=True):
        y=g.y.unique()
        if len(y)!=1: raise ValueError('multiple labels in event')
        p=g[pcols].to_numpy(float).mean(0)
        r={'event_id':eid,'y':int(y[0]),'class_name':str(g.class_name.iloc[0]),'n_patches':len(g)}
        r.update({c:float(v) for c,v in zip(pcols,p)}); rows.append(r)
    return pd.DataFrame(rows)

def bootstrap(ev,metric='macro_f1',reps=2000,seed=2026):
    rng=np.random.default_rng(seed); pcols=[c for c in ev if c.startswith('p_')]; vals=[]
    groups={int(k):g.reset_index(drop=True) for k,g in ev.groupby('y')}
    for _ in range(int(reps)):
        b=pd.concat([g.iloc[rng.integers(0,len(g),size=len(g))] for g in groups.values()],ignore_index=True)
        vals.append(metrics(b.y,b[pcols],2)[metric])
    est=metrics(ev.y,ev[pcols],2)[metric]
    return {'metric':metric,'estimate':est,'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),'reps':int(reps)}

def recall_table(ev):
    pcols=[c for c in ev if c.startswith('p_')]; y=ev.y.to_numpy(int); pred=ev[pcols].to_numpy(float).argmax(1)
    cm=confusion_matrix(y,pred,labels=[0,1]); names={0:'car',1:'construction'}; out=[]
    for k in [0,1]:
        n=int(cm[k].sum()); out.append({'class_id':k,'class_name':names[k],'n_events':n,'recall':float(cm[k,k]/n) if n else np.nan})
    return pd.DataFrame(out)
