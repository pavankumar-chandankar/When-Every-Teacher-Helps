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
