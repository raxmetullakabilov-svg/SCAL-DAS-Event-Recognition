#!/usr/bin/env python3
"""Independent, CPU-only audit of included B1/B3 splits, predictions and checksums.
No third-party packages required. This does not rerun model inference or training.
"""
import csv, json, hashlib, statistics, math, argparse
from pathlib import Path
from collections import Counter, defaultdict

ROOT=Path(__file__).resolve().parents[1]
B1_MODELS=['raw_ts_resnet','raw_scal_das','spectrum_rdft','scal_rdft']
B3_MODELS=['spectrum_rdft','scal_rdft']
SEEDS=[2026,2027,2028]
EXPECTED_B1={'raw_ts_resnet':0.2574955908289241,'raw_scal_das':0.091600096828855,
             'spectrum_rdft':0.5198412698412699,'scal_rdft':0.591710758377425}
EXPECTED_B3={'spectrum_rdft':0.7689865689865689,'scal_rdft':0.7460317460317459}

def readcsv(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def loadjson(p): return json.loads(p.read_text(encoding='utf-8'))
def require(ok,msg):
    if not ok: raise AssertionError(msg)
def approximate(a,b,eps=2e-5): return abs(float(a)-float(b)) <= eps

def verify_sha():
    manifest=ROOT/'CHECKSUMS_SHA256.txt'
    require(manifest.is_file(),'Missing CHECKSUMS_SHA256.txt')
    expected={}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        parts=line.split('  ',1)
        require(len(parts)==2, 'Malformed hash line')
        digest,rel=parts
        require(rel not in expected,'Repeated hash entry: '+rel)
        require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Bad SHA digest: '+rel)
        expected[rel]=digest
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p!=manifest}
    require(actual==set(expected),'Manifest missing or extra files: '+str(sorted(actual^set(expected))[:10]))
    for rel,digest in expected.items():
        h=hashlib.sha256()
        with (ROOT/rel).open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        require(h.hexdigest()==digest,'Checksum mismatch: '+rel)
    return len(expected)

def verify_phase2():
    p=ROOT/'provenance'/'phase2_source_metadata'
    m=loadjson(p/'PHASE2_FILE_HASHES.json')
    matched=0
    for rel,sha in m.items():
        if not rel.startswith('metadata/'): continue
        target=p/Path(rel).name
        require(target.is_file(),'Missing Phase-2 metadata: '+rel)
        require(hashlib.sha256(target.read_bytes()).hexdigest()==sha,'Bad Phase-2 SHA: '+rel)
        matched+=1
    require(matched==6,'Expected 6 original Phase-2 metadata hashes')
    return matched

def verify_split():
    a=readcsv(ROOT/'benchmark'/'b1'/'B1R_SAMPLE_ASSIGNMENTS.csv')
    require(len(a)==2522,'B1 patch count !=2522')
    require(len({r['sample_id'] for r in a})==2522,'Duplicate B1 sample ID')
    require(Counter(r['b1r_role'] for r in a)=={'train':1920,'val':179,'test':423},'B1 patch split incorrect')
    ev={}
    for r in a:
        eid=r['event_id']; role=r['b1r_role']
        if eid in ev: require(ev[eid]==role,'Event spans B1 partitions: '+eid)
        ev[eid]=role
    require(len(ev)==137,'B1 event count !=137')
    require(Counter(ev.values())=={'train':108,'val':11,'test':18},'B1 event split incorrect')
    require(len({r['b1r_group_id'] for r in a})==70,'B1 group count !=70')
    groups={}
    for r in a:
        k=r['b1r_group_id'];role=r['b1r_role']
        if k in groups:require(groups[k]==role,'Group spans B1 partitions: '+k)
        groups[k]=role
    overlap_b1=overlap_count(a,'b1r_role')
    require(overlap_b1['cross'] == 0,'B1 actual-support overlap !=0')
    b=readcsv(ROOT/'benchmark'/'b3'/'B3_SPECTRAL_CAMPAIGN_ASSIGNMENTS.csv')
    require(Counter(r['spectral_campaign_role'] for r in b)=={'train':296,'audit_only_source_val':19,'test':300},'B3 patch counts incorrect')
    roles={}
    for r in b:
        k=r['event_id'];role=r['spectral_campaign_role']
        if k in roles: require(roles[k]==role,'B3 event spans partitions')
        roles[k]=role
    require(Counter(roles.values())=={'train':15,'audit_only_source_val':4,'test':14},'B3 event counts incorrect')
    by_pair=overlap_count(b,'spectral_campaign_role',b3=True)
    require(by_pair['source_val']==30,'B3 train/source-validation overlap should be 30')
    require(by_pair['target']==0,'B3 target cross-role overlap should be zero')
    return {'b1_events':dict(Counter(ev.values())),'b1_patches':dict(Counter(r['b1r_role'] for r in a)),
            'b1_atomic_groups':len(groups),'b1_cross_partition_overlap':overlap_b1['cross'],
            'b3_source_train_to_source_val_overlap':by_pair['source_val'],
            'b3_cross_target_overlap':by_pair['target']}

def overlap_count(rows,role_key,b3=False):
    by_rec=defaultdict(list)
    for r in rows:
        by_rec[r['recording_id']].append((int(r['raw_time_start']),int(r['raw_time_end']),
                                         int(r['raw_channel_start']),int(r['raw_channel_end']),r[role_key]))
    counts={'cross':0,'source_val':0,'target':0}
    for items in by_rec.values():
        items.sort(key=lambda r:r[0])
        for i,ri in enumerate(items):
            for rj in items[i+1:]:
                if rj[0]>=ri[1]:break
                if ri[4]==rj[4] or not(ri[2]<rj[3] and rj[2]<ri[3]):continue
                if not b3:counts['cross']+=1
                elif set((ri[4],rj[4]))=={'train','audit_only_source_val'}:counts['source_val']+=1
                elif 'test' in (ri[4],rj[4]):counts['target']+=1
    return counts

def calc_metrics(rows,classes):
    n=len(rows);ys=[];pred=[];pvec=[]
    for r in rows:
        y=int(r['y']); p=[float(r[f'p_{i}']) for i in range(classes)]
        require(0<=y<classes and all(-1e-8 <=z<=1+1e-8 for z in p),'Probabilities/classes out of range')
        require(abs(sum(p)-1)<1e-4,'Probability vector not normalized')
        ys.append(y); pred.append(max(range(classes),key=lambda i:p[i]));pvec.append(p)
    labels=sorted(set(ys)|set(pred))
    f1=[];rec=[]
    for k in labels:
        tp=sum(y==k and v==k for y,v in zip(ys,pred))
        fp=sum(y!=k and v==k for y,v in zip(ys,pred))
        fn=sum(y==k and v!=k for y,v in zip(ys,pred))
        f1.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.)
        if tp+fn: rec.append(tp/(tp+fn))
    return {'macro_f1':statistics.mean(f1), 'accuracy':sum(x==y for x,y in zip(ys,pred))/n,
            'balanced_accuracy':statistics.mean(rec),
            'nll':sum(-math.log(max(p[y],1e-15)) for p,y in zip(pvec,ys))/n,
            'brier':sum(sum((p[i]-(1 if i==y else 0))**2 for i in range(classes)) for p,y in zip(pvec,ys))/n}

def verify_runs():
    means={};num_runs=0
    for family,models,ref,nclass,n_events,fn in [
        ('b1',B1_MODELS,EXPECTED_B1,9,18,'test_event_predictions__primary.csv'),
        ('b3',B3_MODELS,EXPECTED_B3,2,14,'target_event_predictions.csv')]:
        means[family]={}
        for model in models:
            fs=[]
            for seed in SEEDS:
                d=ROOT/'results'/family/model/f'seed_{seed}'
                completed=loadjson(d/'COMPLETED.json')
                require(completed['status']=='COMPLETE','Run not complete: '+str(d))
                require(completed['epochs_completed']==(60 if family=='b1' else 80),'Wrong epochs: '+str(d))
                hist=readcsv(d/'history.csv')
                require(len(hist)==(60 if family=='b1' else 80),'History length wrong: '+str(d))
                require([int(r['epoch']) for r in hist]==list(range(1,len(hist)+1)),'Epoch sequence incorrect')
                if family=='b1':
                    require(completed.get('headline_checkpoint')=='primary_nll','B1 checkpoint selection wrong')
                    eps=1e-8
                    best=min(float(r['event_nll_9way']) for r in hist)
                    epoch=next(int(r['epoch']) for r in hist if float(r['event_nll_9way'])<=best+eps)
                    require(epoch==int(completed['primary_epoch']),'Primary B1 checkpoint not minimal validation NLL')
                else:
                    require(completed.get('headline_checkpoint')=='epoch_80_fixed','B3 checkpoint != fixed epoch 80')
                rows=readcsv(d/fn)
                require(len(rows)==n_events,'Wrong event prediction count: '+str(d))
                metrics=calc_metrics(rows,nclass)
                published=completed['test_event_metrics_primary'] if family=='b1' else completed['target_event_metrics']
                for key in ('macro_f1','accuracy','balanced_accuracy','nll','brier'):
                    require(approximate(metrics[key],published[key]),f'{d}: {key} {metrics[key]} != {published[key]}')
                fs.append(metrics['macro_f1']);num_runs+=1
            means[family][model]=statistics.mean(fs)
            require(approximate(means[family][model],ref[model],1e-10),f'{family}/{model} aggregate mismatch')
    return {'runs_verified':num_runs,'macro_f1_three_seed_means':means}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--json',action='store_true');args=ap.parse_args()
    result={'sha256_verified_files':verify_sha(),'phase2_verified_metadata_files':verify_phase2(),
            'split':verify_split(),'results':verify_runs(),
            'scope':'Stored numerical results and provenance only; no training, inference or HDF5 byte verification.'}
    print(json.dumps(result,indent=2,ensure_ascii=False) if args.json else
          f"PASS: {result['sha256_verified_files']} checksums, 6 Phase-2 metadata hashes, "
          f"B1/B3 overlap counts, 18 runs and event-level metrics.\n"
          'NOT TESTED: GPU retraining, checkpoint inference, HDF5 shard bytes, access rights.')

if __name__=='__main__':main()
