#!/usr/bin/env python3
"""PerFail 2027 OPERAnet exploratory F1 matrix; Python/PyTorch only.

Reads *only* independently generated aligned_v7r1_10 NPZ files; does not read raw recordings or SensorTAL.
All modality baselines precede paired cosine-feature distillation.
Fixed 30 epochs, no checkpoint tuning. Distinct train/dev/test subjects.
Use the training/validation protocol as an exploratory cross-dataset replication.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('./data/perfail2027/operanet/aligned_v7r1_10')
EXPECTED_EXPS = (18, 29, 32, 33, 56, 57, 58, 59, 60, 61)
SUBJECTS = {18:'Three', 29:'Three', 30:'One', 31:'Six', 32:'Two', 33:'Five',
            56:'One', 57:'Seven', 58:'Two', 59:'Six', 60:'Three', 61:'Five'}
SPLITS = {'One':'train', 'Two':'train', 'Three':'train',
          'Five':'dev', 'Six':'test', 'Seven':'test'}
DIMS = {'wifi':270, 'kinect':114, 'pwr':600, 'uwb':42}
MODS = tuple(DIMS)
CLASSES = ('walk', 'sit', 'stand', 'liedown', 'standfromlie', 'bodyrotate')
SEEDS = (11, 22, 33)
EPOCHS = 30
BATCH = 64
LAMBDA = 1.0
WEIGHT_DECAY = 1e-4
LR = 1e-3


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_cohort(root):
    """Fail before training if cohort, labels, or splits are not valid."""
    arrays = {split:{m:[] for m in MODS} for split in ('train','dev','test')}
    labels = {split:[] for split in arrays}
    manifest = []
    for exp in EXPECTED_EXPS:
        path = root / f'exp{exp:03d}_aligned.npz'
        if not path.is_file():
            raise RuntimeError('MISSING_PREPROCESSING_OUTPUT '+str(path))
        with np.load(path, allow_pickle=False) as z:
            subject, room, split = str(z['subject']), int(z['room']), str(z['split'])
            if subject != SUBJECTS[exp] or split != SPLITS[subject] or room != (1 if exp < 55 else 2):
                raise RuntimeError(f'METADATA_OR_SPLIT_ERROR exp{exp:03d} {subject}/{room}/{split}')
            if int(z['experiment']) != exp:
                raise RuntimeError('EXPERIMENT_ID_MISMATCH '+str(path))
            y = np.asarray(z['y'], dtype=np.int64)
            if y.ndim != 1 or len(y) < 5 or np.any((y < 0) | (y >= len(CLASSES))):
                raise RuntimeError('BAD_LABELS '+str(path))
            times = np.asarray(z['start_time_s'], dtype=np.float64)
            if times.shape != y.shape or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise RuntimeError('INVALID_WINDOW_TIMES '+str(path))
            for mod, dim in DIMS.items():
                x = np.asarray(z['X_'+mod], dtype=np.float32)
                if x.shape != (len(y), 15, dim):
                    raise RuntimeError(f'BAD_SHAPE exp{exp:03d}/{mod}: {x.shape}')
                if (np.isfinite(x).mean(axis=(1,2)) < .80).any():
                    raise RuntimeError(f'INSUFFICIENT_FINITE_WINDOWS exp{exp:03d}/{mod}')
                arrays[split][mod].append(x)
            labels[split].append(y)
            manifest.append({'exp':exp, 'subject':subject, 'split':split,
                             'room':room, 'windows':len(y), 'y_counts':np.bincount(y,minlength=6).tolist(),
                             'sha256':sha256(path)})
    xs = {s:{m:np.concatenate(arrays[s][m]) for m in MODS} for s in arrays}
    ys = {s:np.concatenate(labels[s]) for s in labels}
    for split in ('train','dev','test'):
        counts = np.bincount(ys[split], minlength=6)
        print('SPLIT',split,'WINDOWS',len(ys[split]),'CLASSES',dict(zip(CLASSES,counts.tolist())),flush=True)
        if not counts.all():
            print('CLASS_SUPPORT_WARNING',split, 'zero-support=',[CLASSES[i] for i,c in enumerate(counts) if c==0],flush=True)
            if split=='train': raise RuntimeError(f'INCOMPLETE_SIX_CLASS_TRAINING_COVERAGE: {counts.tolist()}')
    subjects = {s:{x['subject'] for x in manifest if x['split']==s} for s in ('train','dev','test')}
    if any(subjects[s] & subjects[t] for s,t in [('train','dev'),('train','test'),('dev','test')]):
        raise RuntimeError('SUBJECT_LEAKAGE')
    print('COHORT_MATRIX_GATE_PASSED',flush=True)
    return xs,ys,manifest


def normalize_using_train(xs):
    """For each modality, fit moments on training only; nonfinite values imputed to training mean."""
    output = {split:{} for split in xs}
    stats = {}
    for mod in MODS:
        tr = xs['train'][mod]
        mask = np.isfinite(tr)
        cnt = mask.sum((0,1))
        sums = np.nansum(np.where(mask,tr,0), axis=(0,1), dtype=np.float64)
        means = np.divide(sums,cnt,out=np.zeros_like(sums),where=cnt>0)
        delta = np.where(mask,tr-means[None,None,:],0)
        sq = np.sum(np.square(delta, dtype=np.float64), axis=(0,1))
        sd = np.sqrt(np.divide(sq,cnt,out=np.zeros_like(sq),where=cnt>0))
        sd = np.where((sd>1e-5)&np.isfinite(sd),sd,1.)
        if np.count_nonzero(cnt==0):
            print('ALL_MISSING_CHANNELS',mod,int((cnt==0).sum()),flush=True)
        for split in xs:
            x = (xs[split][mod] - means[None,None,:]) / sd[None,None,:]
            x = np.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
            np.clip(x,-10,10,out=x)
            output[split][mod] = torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32))
        stats[mod]={'all_missing_channels':int(np.count_nonzero(cnt==0)),
                    'train_finite_fraction':float(mask.mean()),
                    'train_mean_abs':float(np.abs(means).mean())}
        del tr,mask,delta
    return output,stats


class DCLSTM(nn.Module):
    def __init__(self,dim):
        super().__init__()
        layers=[]
        d=dim
        for _ in range(4):
            layers += [nn.Conv1d(d,64,5,padding=2), nn.ReLU()]
            d=64
        self.conv=nn.Sequential(*layers)
        self.lstm=nn.LSTM(64,128,num_layers=2,dropout=.5,batch_first=True)
        self.head=nn.Linear(128,6)

    def forward(self,x):
        h=self.conv(x.transpose(1,2)).transpose(1,2)
        h,_=self.lstm(h)
        z=h[:,-1,:]
        return self.head(z),z


def f1_report(model,x,y,device):
    model.eval()
    predictions=[]
    with torch.no_grad():
        for a in range(0,len(y),BATCH):
            logits,_=model(x[a:a+BATCH].to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    pred=np.concatenate(predictions)
    true=y.cpu().numpy()
    conf=np.bincount(6*true+pred,minlength=36).reshape(6,6)
    tp=np.diag(conf).astype(float)
    denom=2*tp+(conf.sum(0)-tp)+(conf.sum(1)-tp)
    f1=np.divide(2*tp,denom,out=np.zeros(6),where=denom>0)
    return {'macro_f1':float(f1.mean()),'per_class_f1':f1.tolist(),
            'confusion':conf.tolist(),'support':conf.sum(1).tolist()}


def train_one(mod,seed,x,y,device,teacher_mod=None,teacher=None,epochs=EPOCHS):
    torch.manual_seed(seed)
    model=DCLSTM(DIMS[mod]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    yt=y['train']
    gen=torch.Generator().manual_seed(seed + 104729)
    tx=x['train'][mod]
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
    for ep in range(epochs):
        model.train()
        order=torch.randperm(len(yt),generator=gen)
        epoch_loss=0.
        for start in range(0,len(yt),BATCH):
            take=order[start:start+BATCH]
            inp=tx[take].to(device)
            target=yt[take].to(device)
            out,z=model(inp)
            loss=F.cross_entropy(out,target)
            if teacher is not None:
                with torch.no_grad():
                    _,zt=teacher(x['train'][teacher_mod][take].to(device))
                loss=loss+LAMBDA*(1-F.cosine_similarity(z,zt,dim=1).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss+=float(loss.detach())*len(take)
        if ep==0 or (ep+1)%10==0 or ep+1==epochs:
            print('EPOCH',mod,'teacher',teacher_mod,'seed',seed,'epoch',ep+1,'loss',round(epoch_loss/len(yt),5),flush=True)
    return model


def save_json_once(path, data):
    """Durably save an immutable completed baseline/pair artifact."""
    if path.exists() or path.with_name(path.name+'.part').exists():
        raise RuntimeError('EXISTING_RESULT_REFUSING_OVERWRITE '+str(path))
    part = path.with_name(path.name+'.part')
    with part.open('x') as fp:
        json.dump(data,fp,indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(part,path)


def run_seed(args):
    if args.seed not in SEEDS: raise RuntimeError(f'Only frozen seeds {SEEDS} permitted')
    if args.epochs != EPOCHS: raise RuntimeError(f'Only frozen epoch count {EPOCHS} permitted')
    root=Path(args.root)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    output=out/f'seed_{args.seed}.json'
    if output.exists(): raise RuntimeError('REFUSING_OVERWRITE '+str(output))
    torch.set_num_threads(min(int(os.environ.get('SLURM_CPUS_PER_TASK','4')),4))
    torch.use_deterministic_algorithms(True)
    device=torch.device('cpu') # CPU Slurm nodes; never assume GPU allocation.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    xs,ys,manifest=load_cohort(root)
    x,normalization=normalize_using_train(xs)
    del xs
    y={k:torch.from_numpy(v) for k,v in ys.items()}
    result={'seed':args.seed,'epochs':EPOCHS,'model':'4xConv1d64k5pad2+2xLSTM128',
            'modality_dims':DIMS,'classes':CLASSES,'method':'CE + lambda cosine; lambda=1',
            'manifest':manifest,'normalization':normalization,
            'baseline':{},'pairs':{},'baseline_only':list(args.baseline_only)}
    models={}
    for mod in MODS:
        print('\nBASELINE_START',mod,'SEED',args.seed,flush=True)
        model=train_one(mod,args.seed,x,y,device)
        models[mod]=model
        result['baseline'][mod]={'dev':f1_report(model,x['dev'][mod],y['dev'],device),
                                 'test':f1_report(model,x['test'][mod],y['test'],device)}
        print('BASELINE_RESULT',mod,'TEST_F1',result['baseline'][mod]['test']['macro_f1'],flush=True)
    # Preserve baseline F1 even if any subsequent distillation run fails.
    save_json_once(out/f'baseline_seed_{args.seed}.json',
                   {'seed':args.seed, 'baseline':result['baseline'],
                    'manifest':manifest, 'normalization':normalization})
    print('ALL_FOUR_BASELINES_SAVED',out/f'baseline_seed_{args.seed}.json',flush=True)
    # Every baseline has already been evaluated before any distillation run.
    for teacher in MODS:
        for student in MODS:
            if teacher==student: continue
            key=f'{teacher}->{student}'
            if teacher in args.baseline_only or student in args.baseline_only:
                result['pairs'][key]={'status':'N/A','reason':'explicit baseline-only modality'}
                print('PAIR_NOT_APPLICABLE',key,flush=True)
                continue
            print('\nPAIR_START',key,'SEED',args.seed,flush=True)
            model=train_one(student,args.seed,x,y,device,teacher_mod=teacher,teacher=models[teacher])
            val=f1_report(model,x['dev'][student],y['dev'],device)
            test=f1_report(model,x['test'][student],y['test'],device)
            delta=test['macro_f1']-result['baseline'][student]['test']['macro_f1']
            result['pairs'][key]={'status':'OK','dev':val,'test':test,'delta_macro_f1':delta}
            save_json_once(out/f'pair_{teacher}_to_{student}_seed_{args.seed}.json',
                           {'seed':args.seed,'pair':key, 'result':result['pairs'][key],
                            'manifest':manifest})
            print('PAIR_RESULT',key,'TEST_F1',test['macro_f1'],'DELTA',delta,flush=True)
            del model
    # Refuse overwrite and retain sidecars; written only after the full seed has finished.
    tmp=out/f'seed_{args.seed}.json.part'
    if tmp.exists(): raise RuntimeError('OLD_PARTIAL_RESULT '+str(tmp))
    with tmp.open('x') as fp:
        json.dump(result,fp,indent=2)
        fp.flush(); os.fsync(fp.fileno())
    os.replace(tmp,output)
    print('SEED_COMPLETE',output,flush=True)


def progress(args):
    """Show available baseline scores/pair gains, never impute unrun cells."""
    out=Path(args.out)
    baselines={}
    pairs={}
    for seed in SEEDS:
        path=out/f'baseline_seed_{seed}.json'
        if path.is_file(): baselines[seed]=json.loads(path.read_text())
        for teacher in MODS:
            for student in MODS:
                if student==teacher: continue
                path=out/f'pair_{teacher}_to_{student}_seed_{seed}.json'
                if path.is_file():
                    pairs[(teacher,student,seed)] = json.loads(path.read_text())
    print('RESULTS_AVAILABLE',len(baselines),'BASELINE_SEEDS',len(pairs),'PAIR_SEED_RUNS')
    print('TEACHER\\STUDENT,'+','.join(MODS))
    for teacher in MODS:
        cells=[]
        for student in MODS:
            if teacher==student:
                vals=[d['baseline'][student]['test']['macro_f1'] for d in baselines.values()]
            else:
                vals=[v['result']['delta_macro_f1'] for (t,s,seed),v in pairs.items() if t==teacher and s==student]
            if not vals:
                cells.append('PENDING')
            else:
                cell=(f'{np.mean(vals):+.3f}' if teacher!=student else f'{np.mean(vals):.3f}')
                cells.append(f'{cell} [n={len(vals)}/3]')
        print(teacher+','+','.join(cells))
    print('PROGRESS_REPORT_ONLY_NOT_A_COMPLETE_MATRIX')


def aggregate(args):
    out=Path(args.out)
    data=[]
    for seed in SEEDS:
        path=out/f'seed_{seed}.json'
        if not path.is_file(): raise RuntimeError('MISSING_SEED_RESULT '+str(path))
        data.append(json.loads(path.read_text()))
    mf=[[(m['exp'],m['sha256']) for m in d['manifest']] for d in data]
    if not all(v==mf[0] for v in mf): raise RuntimeError('INPUT_MANIFESTS_DIFFER_ACROSS_SEEDS')
    print('\n=== TEACHER ROW / STUDENT COLUMN: DIAGONAL=BASELINE F1, OFFDIAGONAL=F1 DELTA ===')
    print('teacher\\student,'+','.join(MODS))
    table=[]
    for teacher in MODS:
        row=[]
        for student in MODS:
            if teacher==student:
                vals=[d['baseline'][student]['test']['macro_f1'] for d in data]
                label=f'{np.mean(vals):.3f} +/- {np.std(vals,ddof=1):.3f}'
                status='BASELINE'
            else:
                key=f'{teacher}->{student}'
                pairs=[d['pairs'][key] for d in data]
                if all(p['status']=='N/A' for p in pairs):
                    vals=[]; label='N/A'; status='N/A'
                elif all(p['status']=='OK' for p in pairs):
                    vals=[p['delta_macro_f1'] for p in pairs]
                    label=f'{np.mean(vals):+.3f} +/- {np.std(vals,ddof=1):.3f}'
                    status='GAIN'
                else:
                    raise RuntimeError('INCONSISTENT_PAIR_STATUS '+key)
            row.append(label)
            table.append({'teacher':teacher,'student':student,'status':status,
                          'mean':float(np.mean(vals)) if vals else '',
                          'sd':float(np.std(vals,ddof=1)) if vals else '',
                          'per_seed':';'.join(str(v) for v in vals)})
        print(teacher+','+','.join(row))
    csvpath=out/'matrix_f1.csv'
    if csvpath.exists(): raise RuntimeError('REFUSING_OVERWRITE '+str(csvpath))
    with csvpath.open('x',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=['teacher','student','status','mean','sd','per_seed'])
        writer.writeheader(); writer.writerows(table)
    print('MATRIX_COMPLETE',csvpath,flush=True)


def selftest():
    """Synthetic known-answer and anti-leakage checks for F1 and architecture."""
    cm_true=np.array([0,0,1,1,2,2,3,3,4,4,5,5])
    class Exact(nn.Module):
        def forward(self,x):
            predictions=x[:,0,0].long()
            return F.one_hot(predictions,6).float()*10, x[:,0,:]
    x=torch.from_numpy(cm_true[:,None,None].astype(np.float32))
    report=f1_report(Exact(),x,torch.from_numpy(cm_true),torch.device('cpu'))
    if abs(report['macro_f1']-1)>1e-8: raise RuntimeError('F1_IDENTITY_TRAP')
    xbad=x.clone(); xbad[:,0,0]=0
    report=f1_report(Exact(),xbad,torch.from_numpy(cm_true),torch.device('cpu'))
    if report['macro_f1'] >= .8: raise RuntimeError('F1_ALL_ONE_CLASS_TRAP')
    torch.manual_seed(0)
    m=DCLSTM(42)
    logits,z=m(torch.randn(4,15,42))
    assert logits.shape==(4,6) and z.shape==(4,128)
    print('SYNTHETIC_F1_AND_ARCHITECTURE_CHECKS_PASSED')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['check','seed','aggregate','progress','selftest'])
    parser.add_argument('--root',default=str(ROOT))
    parser.add_argument('--out',default=str(ROOT.parent/'matrix_v1'))
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--epochs',type=int,default=EPOCHS)
    parser.add_argument('--baseline-only',nargs='*',choices=list(MODS),default=[])
    args=parser.parse_args()
    if args.mode=='selftest': selftest()
    elif args.mode=='check':
        _,_,manifest=load_cohort(Path(args.root))
        print('CHECK_COMPLETE',len(manifest),'experiments')
    elif args.mode=='seed': run_seed(args)
    elif args.mode=='aggregate': aggregate(args)
    elif args.mode=='progress': progress(args)

if __name__=='__main__': main()
