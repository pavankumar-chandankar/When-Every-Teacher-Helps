#!/usr/bin/env python3
"""MM-Fi per-sequence subject-disjoint F1 matrix, five modalities, 3 seeds.
Each baseline and directed pair uses dev-selected weights and test only once.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import numpy as np

MODS = ['wifi','mmwave','lidar','depth','rgbpose']
SEEDS = [11,22,33]
NCLASS = 27
DEV = {'S05','S15','S25','S35'}
TEST = {'S10','S20','S30','S40'}


def atomic_json(path,obj):
    path=Path(path)
    temp=path.with_suffix('.json.partial')
    with open(temp,'w') as f:
        json.dump(obj,f,indent=2)
    os.replace(temp,path)


def load_cohort(inp):
    samples={m:[] for m in MODS}
    labels,groups,scenes=[],[],[]
    for i in range(1,41):
        subject=f'S{i:02d}'
        p=Path(inp)/f'{subject}.npz'
        with np.load(p, allow_pickle=False) as d:
            for m in MODS:
                samples[m].append(d[m].astype(np.float32))
            labels.extend(d['y'].tolist())
            groups.extend([subject]*27)
            scenes.extend([str(d['scene'])]*27)
    x={m:np.concatenate(samples[m]) for m in MODS}
    y=np.asarray(labels,dtype=np.int64)
    groups=np.asarray(groups)
    splits={'train':np.flatnonzero(~np.isin(groups,list(DEV|TEST))),
            'dev':np.flatnonzero(np.isin(groups,list(DEV))),
            'test':np.flatnonzero(np.isin(groups,list(TEST)))}
    assert [len(splits[s]) for s in ('train','dev','test')]==[864,108,108]
    assert set(np.unique(y))==set(range(27))
    for s,ix in splits.items():
        assert np.bincount(y[ix],minlength=27).min() >= 4
    for m in MODS:
        assert x[m].shape[:2] == (1080,12) and np.isfinite(x[m]).all()
        train=x[m][splits['train']]
        mean=train.mean(axis=(0,1),keepdims=True)
        std=train.std(axis=(0,1),keepdims=True)
        std[std<1e-5]=1
        x[m] = np.clip((x[m]-mean)/std, -10, 10).astype(np.float32)
    print('TRAIN_GATE_PASSED train=864 dev=108 test=108 classes=27',flush=True)
    return x,y,splits


def make_model(in_dim):
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim,96),nn.GELU())
            self.temporal = nn.Sequential(nn.Conv1d(96,96,3,padding=1),nn.GELU(),
                                          nn.Conv1d(96,96,3,padding=1),nn.GELU())
            self.proj = nn.Sequential(nn.Linear(192,96), nn.GELU())
            self.head = nn.Linear(96,27)
        def forward(self,x):
            h=self.temporal(self.pre(x).transpose(1,2))
            h=self.proj(__import__('torch').cat((h.mean(-1),h.amax(-1)),dim=-1))
            return self.head(h),h
    return Net()


def macro_f1(y_true,y_pred):
    a=np.bincount(np.asarray(y_true)*NCLASS+np.asarray(y_pred),minlength=NCLASS*NCLASS).reshape(NCLASS,NCLASS)
    tp=a.diagonal()
    den=a.sum(0)+a.sum(1)
    f=np.divide(2*tp,den,out=np.zeros(NCLASS,dtype=float),where=den!=0)
    return float(f.mean()),a.astype(int).tolist()


def evaluate(net,x,y,ix,torch,keep_cm=False):
    net.eval()
    pr=[]
    with torch.no_grad():
        for start in range(0,len(ix),128):
            chunk=ix[start:start+128]
            logits,_=net(torch.from_numpy(x[chunk]))
            pr.extend(logits.argmax(dim=1).numpy().tolist())
    score,cm=macro_f1(y[ix],np.array(pr))
    return (score,cm) if keep_cm else score


def train_one(x,y,splits,student,seed,teacher=None,teacher_mod=None,epochs=30):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed+10000*MODS.index(student))
    net=make_model(x[student].shape[-1])
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    xs=torch.from_numpy(x[student][splits['train']])
    ys=torch.from_numpy(y[splits['train']])
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():p.requires_grad_(False)
        xt=torch.from_numpy(x[teacher_mod][splits['train']])
    rng=np.random.default_rng(seed+10000*MODS.index(student))
    best=-1.0
    best_state=None
    best_epoch=0
    for ep in range(1,epochs+1):
        net.train()
        perm=rng.permutation(len(xs))
        for ids in np.array_split(perm,max(1,(len(perm)+63)//64)):
            logits,embed=net(xs[ids])
            loss=F.cross_entropy(logits,ys[ids])
            if teacher is not None:
                with torch.no_grad():
                    _,t_embed=teacher(xt[ids])
                loss=loss + .2*(1-F.cosine_similarity(embed,t_embed,dim=-1).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        dev=evaluate(net,x[student],y,splits['dev'],torch)
        if dev>best+1e-10:
            best=dev
            best_state=copy.deepcopy(net.state_dict())
            best_epoch=ep
    net.load_state_dict(best_state)
    test,cm=evaluate(net,x[student],y,splits['test'],torch,True)
    return net,{'dev_f1':best,'test_f1':test,'epoch':best_epoch,'test_confusion':cm}


def run_seed(args):
    import torch
    torch.set_num_threads(min(int(os.environ.get('SLURM_CPUS_PER_TASK','4')),8))
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    x,y,splits=load_cohort(args.input)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    p=out/f'seed_{args.seed}.json'
    if p.exists():
        old=json.loads(p.read_text())
        if old.get('complete'):
            print('SKIP_COMPLETE_SEED',p,flush=True)
            return
        backup=p.with_name(p.name+'.interrupted')
        if backup.exists():
            raise RuntimeError('Existing incomplete-seed backup: '+str(backup))
        os.replace(p,backup)
        print('QUARANTINED_INCOMPLETE_SEED',backup,flush=True)
    data={'seed':args.seed,'protocol':'mmfi_sequence_12frames_subject_disjoint_v1',
          'baselines':{},'pairs':{},'complete':False}
    state={}
    for mod in MODS:
        net,r=train_one(x,y,splits,mod,args.seed,epochs=args.epochs)
        state[mod]=copy.deepcopy(net.state_dict())
        data['baselines'][mod]=r
        print('BASELINE',mod,'seed',args.seed,'TEST_F1',f"{r['test_f1']:.6f}",flush=True)
        atomic_json(p,data)
    for teacher in MODS:
        tmodel=make_model(x[teacher].shape[-1]);tmodel.load_state_dict(state[teacher]);tmodel.eval()
        for student in MODS:
            if student==teacher:continue
            _,r=train_one(x,y,splits,student,args.seed,tmodel,teacher,args.epochs)
            r['gain']=r['test_f1']-data['baselines'][student]['test_f1']
            data['pairs'][f'{teacher}->{student}']=r
            print('TRANSFER',teacher,'->',student,'seed',args.seed,
                  'F1',f"{r['test_f1']:.6f}",'DELTA',f"{r['gain']:+.6f}",flush=True)
            atomic_json(p,data)
    data['complete']=True
    atomic_json(p,data)
    print('SEED_COMPLETE',p,flush=True)


def files(out):
    out=Path(out)
    d=[]
    for seed in SEEDS:
        p=out/f'seed_{seed}.json'
        if not p.exists():continue
        obj=json.loads(p.read_text())
        d.append(obj)
    return d


def summarize(out,aggregate=False):
    d=files(out)
    print('SEEDS_FOUND',len(d),'BASELINES',sum(len(x['baselines']) for x in d),
          'PAIRS',sum(len(x['pairs']) for x in d))
    if aggregate and not (len(d)==3 and all(x['complete'] and len(x['baselines'])==5 and len(x['pairs'])==20 for x in d)):
        raise SystemExit('INCOMPLETE: refusing final aggregate')
    rows=[]
    for teacher in MODS:
        for student in MODS:
            diag=(teacher==student)
            values=[]
            for obj in d:
                if diag:
                    if student in obj['baselines']:
                        values.append(float(obj['baselines'][student]['test_f1']))
                else:
                    key=f'{teacher}->{student}'
                    if key in obj['pairs'] and student in obj['baselines']:
                        r=obj['pairs'][key]
                        # Recompute gains; never copy a potentially inconsistent saved delta.
                        values.append(float(r['test_f1'])-float(obj['baselines'][student]['test_f1']))
            mean=statistics.mean(values) if values else None
            sd=statistics.stdev(values) if len(values)>1 else (0. if values else None)
            rows.append([teacher,student,'BASELINE' if diag else 'GAIN',len(values),mean,sd])
    print('TEACHER/STUDENT, '+', '.join(MODS))
    for t in MODS:
        cells=[]
        for r in rows:
            if r[0]==t:
                cells.append('PENDING' if r[4] is None else f'{r[4]:+.4f} +/- {r[5]:.4f} (n={r[3]})')
        print(t,', '.join(cells))
    if aggregate:
        out=Path(out)
        p=out/'matrix_f1.csv'
        with open(p,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['teacher','student','status','n_seeds','mean','sample_sd'])
            w.writerows(rows)
        print('MATRIX_COMPLETE',p)


def synthetic_check():
    import torch
    torch.set_num_threads(2)
    assert len(MODS)==5 and len(SEEDS)==3
    a=np.array([0,1,2,2]);b=np.array([0,2,2,1])
    f,cm=macro_f1(a,b)
    assert abs(f-(1.5/27))<1e-9 and sum(sum(row) for row in cm)==4
    for dim in (14,22,34,256,570):
        m=make_model(dim)
        logits,emb=m(torch.ones(3,12,dim))
        assert tuple(logits.shape)==(3,27) and tuple(emb.shape)==(3,96)
    print('MATRIX_SYNTHETIC_SELF_TEST_PASSED')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['self-test','train','progress','aggregate'])
    ap.add_argument('--input',default='./data/perfail2027/mmfi/compact_v1')
    ap.add_argument('--out',default='./data/perfail2027/mmfi/matrix_v1')
    ap.add_argument('--seed',type=int,choices=SEEDS)
    ap.add_argument('--epochs',type=int,default=30)
    a=ap.parse_args()
    if a.action=='self-test':synthetic_check()
    elif a.action=='train':
        if a.seed is None:ap.error('--seed is required for train')
        run_seed(a)
    else:summarize(a.out,aggregate=a.action=='aggregate')
