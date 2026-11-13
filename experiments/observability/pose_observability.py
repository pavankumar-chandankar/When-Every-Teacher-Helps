#!/usr/bin/env python3
"""Pose-motion observability on the existing OPERAnet and MM-Fi matrix caches.

This measures pose-motion *proxy* observability, not XRFv2 IMU observability.
No raw MAT files, MATLAB, retraining of distillation matrices, or sealed XRFv2 data.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

OP_EXPS = (18, 29, 32, 33, 56, 57, 58, 59, 60, 61)
OP_SUBJECTS = {18:'Three', 29:'Three', 32:'Two', 33:'Five', 56:'One',
               57:'Seven', 58:'Two', 59:'Six', 60:'Three', 61:'Five'}
OP_SPLITS = {'One':'train', 'Two':'train', 'Three':'train',
             'Five':'dev', 'Six':'test', 'Seven':'test'}
OP_DIMS = {'wifi':270, 'pwr':600, 'uwb':42, 'kinect':114}
MM_DIMS = {'wifi':570, 'mmwave':22, 'lidar':14, 'depth':256, 'rgbpose':34}
MM_DEV = {'S05','S15','S25','S35'}
MM_TEST = {'S10','S20','S30','S40'}
ALPHAS = (0.001, 0.01, 0.1, 1., 10., 100., 1000.)


def ensure(ok, message):
    if not ok:
        raise ValueError(message)


def hash_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for piece in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def parse_operanet(root):
    """Read exactly the matrix v7r1 cohort; never touch original MAT inputs."""
    x = {m: [] for m in OP_DIMS}
    targets, y, group, split, source = [], [], [], [], []
    for exp in OP_EXPS:
        file = root / f'exp{exp:03d}_aligned.npz'
        ensure(file.is_file(), f'missing aligned cohort: {file}')
        with np.load(file, allow_pickle=False) as d:
            subj = str(d['subject'])
            part = str(d['split'])
            ensure(subj == OP_SUBJECTS[exp] and part == OP_SPLITS[subj],
                   f'OPERAnet subject/split mismatch {file}')
            ensure(int(d['experiment']) == exp and int(d['room']) == (1 if exp < 55 else 2),
                   f'OPERAnet provenance mismatch {file}')
            labels = d['y'].astype(np.int64)
            n = len(labels)
            ensure(n >= 5 and labels.shape == (n,) and ((labels >= 0) & (labels < 6)).all(),
                   f'OPERAnet invalid labels {file}')
            t = np.asarray(d['start_time_s'])
            ensure(t.shape == (n,) and np.isfinite(t).all() and (np.diff(t) > 0).all(),
                   f'OPERAnet nonmonotonic times {file}')
            for m, dim in OP_DIMS.items():
                a = d['X_' + m].astype(np.float32)
                ensure(a.shape == (n,15,dim), f'OPERAnet {m}: bad shape {a.shape}')
                ensure((np.isfinite(a).mean(axis=(1,2)) >= 0.8).all(), f'OPERAnet {m}: poor coverage')
                x[m].append(a)
            k = x['kinect'][-1].reshape(n,15,2,19,3)
            # Squared frame-to-frame displacement / (0.1 s)^2; camera coordinates
            # stay separate, so no cross-camera frame-of-reference subtraction.
            dpos = np.diff(k, axis=1) / 0.1
            valid = np.isfinite(k[:,1:]).all(-1) & np.isfinite(k[:,:-1]).all(-1)
            counts = valid.sum(axis=(1,2,3))
            ensure((counts >= (14 * 2 * 19 * 0.8)).all(),
                   f'OPERAnet missing Kinect motion reference {file}')
            energy = np.divide(np.where(valid, np.sum(dpos*dpos,axis=-1), 0.).sum(axis=(1,2,3)),
                               counts, where=counts>0)
            ensure(np.isfinite(energy).all(), f'OPERAnet invalid pose energy {file}')
            targets.append(energy)
            y.append(labels)
            group.extend([subj]*n)
            split.extend([part]*n)
        source.append({'file':file.name,'sha256':hash_sha256(file)})
    return finish('operanet', {m:np.concatenate(v) for m,v in x.items()},
                  np.concatenate(y), np.asarray(group), np.asarray(split),
                  {'whole_body_kinect_speed2':np.concatenate(targets)}, source)


def mmfi_energy(pose, frames, indices):
    """2D pose-squared-speed proxy; wrist joint indices follow COCO-17 convention."""
    n, t, d = pose.shape
    ensure(d == 34 and t == len(frames) and np.all(np.diff(frames)>0), 'bad pose/frame schema')
    p = pose.reshape(n,t,17,2)
    frame_gap = np.diff(frames).astype(np.float64)[None,:,None,None]
    speed2 = np.sum((np.diff(p.astype(np.float64),axis=1)/frame_gap)**2, axis=-1)
    good = np.isfinite(p).all(-1) & (np.abs(p).sum(-1)>0.)
    edge_good = good[:,1:] & good[:,:-1]
    selected = edge_good[:,:,indices]
    coverage = selected.mean(axis=(1,2))
    ensure((coverage >= 0.5).all(), 'MM-Fi pose reference missing in >50% of edges')
    val = speed2[:,:,indices]
    counts = selected.sum(axis=(1,2))
    out = np.where(selected, val, 0.).sum(axis=(1,2))/counts
    ensure(np.isfinite(out).all(), 'non-finite MM-Fi pose motion')
    return out


def parse_mmfi(root):
    """Use matrix v2_48 subject cache, never read or extract the 83-GB archive."""
    x = {m: [] for m in MM_DIMS}
    factors = {'whole_body_rgbpose_speed2':[], 'wrist_rgbpose_speed2':[]}
    y, group, split, source = [], [], [], []
    frame_reference = None
    for idx in range(1,41):
        subj=f'S{idx:02d}'
        file = root/f'{subj}.npz'
        ensure(file.is_file(), f'missing MM-Fi compact cache {file}')
        with np.load(file, allow_pickle=False) as d:
            ensure(str(d['subject']) == subj, f'MM-Fi subject mismatch {file}')
            ensure(str(d['scene']) == f'E{(idx-1)//10+1:02d}', f'MM-Fi scene mismatch {file}')
            labels = d['y'].astype(np.int64)
            ensure(labels.shape==(27,) and np.array_equal(labels,np.arange(27)),
                   f'MM-Fi action labels mismatch {file}')
            frames = np.asarray(d['frames'], dtype=np.int64)
            expected_frames=np.asarray([int(v) for v in np.linspace(1,297,48)],dtype=np.int64)
            ensure(np.array_equal(frames,expected_frames),
                   f'MM-Fi frame selection mismatch {file}')
            if frame_reference is None: frame_reference=frames
            ensure(np.array_equal(frames,frame_reference), f'MM-Fi samples misaligned {file}')
            for m,dim in MM_DIMS.items():
                a = np.asarray(d[m],dtype=np.float32)
                ensure(a.shape == (27,48,dim) and np.isfinite(a).all(),
                       f'MM-Fi invalid {m} {file} {a.shape}')
                x[m].append(a)
            pose=x['rgbpose'][-1]
            factors['whole_body_rgbpose_speed2'].append(mmfi_energy(pose,frames,list(range(17))))
            factors['wrist_rgbpose_speed2'].append(mmfi_energy(pose,frames,[9,10]))
            part='test' if subj in MM_TEST else 'dev' if subj in MM_DEV else 'train'
            split.extend([part]*27); y.append(labels); group.extend([subj]*27)
        source.append({'file':file.name,'sha256':hash_sha256(file)})
    return finish('mmfi',{m:np.concatenate(v) for m,v in x.items()},
                  np.concatenate(y),np.asarray(group),np.asarray(split),
                  {k:np.concatenate(v) for k,v in factors.items()},source)


def finish(dataset,x,y,group,split,targets,source):
    n=len(y)
    ensure(all(a.shape[0]==n for a in x.values()), 'misaligned modality counts')
    ensure(set(split)=={'train','dev','test'}, 'missing split')
    ensure(set(group[split=='train']).isdisjoint(set(group[split=='dev'])) and
           set(group[split=='train']).isdisjoint(set(group[split=='test'])) and
           set(group[split=='dev']).isdisjoint(set(group[split=='test'])), 'SUBJECT LEAKAGE')
    counts={k:int((split==k).sum()) for k in ('train','dev','test')}
    if dataset=='mmfi':
        ensure(counts=={'train':864,'dev':108,'test':108}, f'bad MM-Fi split {counts}')
    ensure(all(counts[k]>0 for k in counts), f'empty split {counts}')
    classes=27 if dataset=='mmfi' else 6
    ensure(np.bincount(y[split=='train'],minlength=classes).min()>0,
           'training split lacks activity classes')
    ensure(all(v.shape==(n,) and np.isfinite(v).all() for v in targets.values()),
           'target length or finiteness error')
    ensure(all(float(v[split=='train'].var())>0 for v in targets.values()),
           'motion target has no training variance')
    return dict(dataset=dataset,x=x,y=y,group=group,split=split,targets=targets,
                sources=source,counts=counts,classes=classes)


def normalize_train_only(x, train):
    """Fit channel moments solely on training subjects; impute nonfinite to mean."""
    a=x[train]
    mask=np.isfinite(a)
    counts=mask.sum(axis=(0,1))
    mu=np.divide(np.where(mask,a,0).sum(axis=(0,1),dtype=np.float64),counts,
                 out=np.zeros(x.shape[-1],dtype=np.float64),where=counts>0)
    delta=np.where(mask,a-mu,0)
    std=np.sqrt(np.divide((delta.astype(np.float64)**2).sum(axis=(0,1)),counts,
                          out=np.ones_like(mu),where=counts>0))
    std[std<1e-5]=1.
    z=np.nan_to_num((x-mu)/std,nan=0,posinf=0,neginf=0)
    return np.ascontiguousarray(np.clip(z,-10.,10.),dtype=np.float32)


def ridge_fit(train_x,train_y,alpha):
    """Ridge with train-only feature centering/scaling and unpenalized intercept."""
    x=np.asarray(train_x,dtype=np.float64)
    y=np.asarray(train_y,dtype=np.float64)
    mean=x.mean(axis=0)
    sd=x.std(axis=0)
    sd[sd<1e-8]=1.
    z=(x-mean)/sd
    ymean=y.mean()
    mat=z.T@z
    mat.flat[::len(mat)+1] += alpha
    coef=np.linalg.solve(mat,z.T@(y-ymean))
    ensure(np.isfinite(coef).all(), 'NONFINITE_RIDGE_COEFFICIENTS')
    return mean,sd,ymean,coef


def ridge_predict(fit,x):
    mu,sd,ym,coef=fit
    return ((np.asarray(x,dtype=np.float64)-mu)/sd)@coef + ym


def ridge_select(train_x,train_y,dev_x,dev_y):
    for name, array in [('train embeddings',train_x),('train target',train_y),
                        ('dev embeddings',dev_x),('dev target',dev_y)]:
        ensure(np.isfinite(array).all(), f'NONFINITE_READOUT_INPUT: {name}')
    best=None
    for alpha in ALPHAS:
        fit=ridge_fit(train_x,train_y,alpha)
        err=mse(ridge_predict(fit,dev_x),dev_y)
        if best is None or err<best[0]: best=(float(err),alpha,fit)
    ensure(best is not None, 'NO_VALID_RIDGE_CANDIDATE')
    return best[1],best[2],best[0]


def label_oracle(train_y,train_target,other_y):
    """Reference knows true activity, but class means are learned on train only."""
    table={int(label):float(np.mean(train_target[train_y==label]))
           for label in np.unique(train_y)}
    ensure(set(np.unique(other_y)).issubset(table), 'unseen oracle class')
    return np.asarray([table[int(z)] for z in other_y],dtype=np.float64)


def mse(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    ensure(a.shape==b.shape and a.size>0, 'INVALID_MSE_SHAPE')
    ensure(np.isfinite(a).all() and np.isfinite(b).all(), 'NONFINITE_MSE_INPUT')
    result=float(np.mean((a-b)**2))
    ensure(np.isfinite(result), 'NONFINITE_MSE_RESULT')
    return result
