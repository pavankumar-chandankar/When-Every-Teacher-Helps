#!/usr/bin/env python3
"""Resample the XRFv2 Plus side modalities onto the WiFi 50 Hz timeline.

Every modality runs on its own clock (AirPods 25 Hz, pose ~15 fps with per-frame
timestamps, RGB video features 1 per second). This writes one cache where each
sequence has, per modality, a [T_wifi, C] float32 array on the exact 50 Hz grid
the WiFi and IMU windows use, so a window [start, end) is the same moment in
every modality.

Resampling is nearest-frame (sample-and-hold): no blending of valid frames with
missing ones and no invented values between video frames.

Pose is made person-centric, as is standard for skeleton HAR: joints are centred
on the hip midpoint and divided by torso length. Raw camera coordinates would
encode where in the room the person stands, i.e. the room itself.
Joints with confidence < 0.2, missing depth (NaN) or no detection are zeroed and
their confidence channel set to 0, so the model can tell missing from present.

RGB features (1024-d) are reduced to 64 dims by PCA fitted on TRAINING subjects
only, so DeepConvLSTM keeps the same architecture for every modality.

Normalisation statistics (per-channel mean / std) are computed on training
subjects only and written next to the cache, in the same format as TAL's stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

HIPS, SHOULDERS = (11, 12), (5, 6)
FILES = {'airpods': 'airpods_25hz_853_video_aligned.h5',
         'pose2d': 'pose2d_coco17_yolov8m_853_video_aligned.h5',
         'pose3d': 'pose3d_coco17_depth_853_video_aligned.h5',
         'video': 'video_feature_853_video_aligned.h5'}
MIN_CONF = 0.2


def nearest(t_src, t_dst):
    if len(t_src) == 1:
        return np.zeros(len(t_dst), dtype=np.int64)
    i = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    return np.where(np.abs(t_dst - t_src[i - 1]) <= np.abs(t_src[i] - t_dst), i - 1, i)


def person_centric(xyz, valid):
    """xyz [T, 17, D], valid [T, 17] -> hip-centred, torso-scaled; zeros where invalid."""
    out = np.zeros_like(xyz, dtype=np.float32)
    for t in range(xyz.shape[0]):
        v = valid[t]
        if not v.any():
            continue
        hip = [j for j in HIPS if v[j]]
        centre = xyz[t, hip].mean(0) if hip else xyz[t, v].mean(0)
        sh = [j for j in SHOULDERS if v[j]]
        scale = float(np.linalg.norm(xyz[t, sh].mean(0) - xyz[t, hip].mean(0))) if (hip and sh) else 0.0
        if scale < 1e-6:
            pts = xyz[t, v]
            scale = float(np.linalg.norm(pts.max(0) - pts.min(0))) / 2.0
        out[t, v] = (xyz[t, v] - centre) / max(scale, 1e-6)
    return out


def load(mod, h, seq):
    """-> (values [N, C], timestamps [N], fraction of valid joints or None)."""
    g = h['samples'][seq]
    if mod == 'airpods':
        a = np.nan_to_num(np.asarray(g['airpods'], np.float32))
        return a, np.arange(a.shape[0], dtype=np.float64) / 25.0, None
    if mod == 'video':
        return (np.nan_to_num(np.asarray(g['video_feature'], np.float32)),
                np.asarray(g['timestamps_sec'], np.float64), None)
    if mod == 'pose2d':
        kp = np.asarray(g['keypoints'], np.float32)
        det = np.asarray(g['detection_count'])
        pts, conf = kp[..., :2], np.nan_to_num(kp[..., 2])
        valid = (conf >= MIN_CONF) & (det[:, None] > 0) & np.isfinite(pts).all(-1)
    elif mod == 'pose3d':
        p = np.asarray(g['pose3d_camera'], np.float32)
        pts, conf = p[..., :3], np.nan_to_num(p[..., 3])
        valid = (conf >= MIN_CONF) & np.isfinite(pts).all(-1)
    else:
        raise ValueError(mod)
    norm = person_centric(np.nan_to_num(pts), valid)
    c = np.where(valid, conf, 0.0).astype(np.float32)
    v = np.concatenate([norm, c[..., None]], -1).reshape(len(pts), -1)
    return v, np.asarray(g['timestamps_sec'], np.float64), float(valid.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--fold-dir', type=Path, required=True, help='TAL fold dir; train.json = training subjects')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--video-pca', type=int, default=64)
    ap.add_argument('--allow-missing', action='store_true', help='testing only')
    a = ap.parse_args()

    train_seqs = set(json.load(open(a.fold_dir / 'train.json', encoding='utf-8'))['database'])
    hs = {m: h5py.File(a.data_dir / f, 'r') for m, f in FILES.items()}
    wifi = h5py.File(a.data_dir / 'wifi_50hz_853_video_aligned.h5', 'r')
    seqs = sorted(wifi['samples'].keys())

    X = np.concatenate([np.nan_to_num(np.asarray(hs['video']['samples'][s]['video_feature'], np.float32))
                        for s in seqs if s in train_seqs and s in hs['video']['samples']])
    mu = X.mean(0)
    _, sv, vt = np.linalg.svd(X - mu, full_matrices=False)
    comps = vt[:a.video_pca].astype(np.float32)
    var_kept = float((sv[:a.video_pca] ** 2).sum() / (sv ** 2).sum())
    print('video PCA fitted on %d training frames: %d -> %d dims, variance kept %.3f'
          % (len(X), X.shape[1], a.video_pca, var_kept), flush=True)
    del X

    acc = {}
    rep = {m: {'coverage': [], 'valid': [], 'missing': 0} for m in FILES}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, 'w') as out:
        gs = out.create_group('samples')
        for n, seq in enumerate(seqs, 1):
            T = wifi['samples'][seq]['amp'].shape[0]
            t50 = np.arange(T, dtype=np.float64) / 50.0
            g = gs.create_group(seq)
            for m, h in hs.items():
                if seq not in h['samples']:
                    rep[m]['missing'] += 1
                    if not a.allow_missing:
                        raise SystemExit('%s is missing sequence %s' % (m, seq))
                    continue
                v, ts, vf = load(m, h, seq)
                if m == 'video':
                    v = (v - mu) @ comps.T
                order = np.argsort(ts, kind='mergesort'); v, ts = v[order], ts[order]
                arr = np.ascontiguousarray(v[nearest(ts, t50)], dtype=np.float32)
                step = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
                rep[m]['coverage'].append(float(((t50 >= ts[0] - step) & (t50 <= ts[-1] + step)).mean()))
                if vf is not None:
                    rep[m]['valid'].append(vf)
                g.create_dataset(m, data=arr, compression='lzf')
                if seq in train_seqs:
                    s = arr.astype(np.float64)
                    if m not in acc:
                        acc[m] = [np.zeros(s.shape[1]), np.zeros(s.shape[1]), 0]
                    acc[m][0] += s.sum(0); acc[m][1] += (s ** 2).sum(0); acc[m][2] += len(s)
            if n % 100 == 0 or n == len(seqs):
                print('  %d/%d' % (n, len(seqs)), flush=True)
        out.attrs['description'] = 'XRFv2 Plus side modalities resampled to the WiFi 50 Hz grid'

    stats = {'metadata': {'split': 'training subjects (fold train.json)', 'video_pca_dims': a.video_pca,
                          'video_pca_variance_kept': var_kept}}
    for m, (s1, s2, c) in acc.items():
        mean = s1 / c
        stats[m] = {'global_mean': mean.tolist(),
                    'global_std': np.sqrt(np.maximum(s2 / c - mean ** 2, 1e-12)).tolist(), 'count': int(c)}
    Path(str(a.out) + '.stats.json').write_text(json.dumps(stats))
    np.savez(str(a.out) + '.video_pca.npz', mean=mu, components=comps)

    print('\nmodality  channels  coverage min/mean   valid joints  missing')
    for m in FILES:
        r = rep[m]; cov = np.array(r['coverage'] or [np.nan])
        vf = ('%.3f' % np.mean(r['valid'])) if r['valid'] else '  -  '
        print('%-9s %8d   %.3f / %.3f        %s      %d'
              % (m, len(stats[m]['global_mean']) if m in stats else 0, cov.min(), cov.mean(), vf, r['missing']))
    print('written', a.out)


if __name__ == '__main__':
    main()
