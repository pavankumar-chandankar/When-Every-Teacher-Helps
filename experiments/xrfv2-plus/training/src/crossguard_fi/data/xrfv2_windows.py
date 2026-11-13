"""Sliding-window dataset for XRFv2 Plus HAR, any modality.

Reads a manifest from scripts/har_build_windows.py. Whole sequences are loaded
once into memory, normalised with training-subject statistics, and windows are
sliced on the fly.

WiFi and IMU are read from the original HDF5 files with TAL's statistics file,
exactly as before, so earlier WiFi / IMU runs remain valid. AirPods, 2D pose,
3D pose and RGB video features are read from the 50 Hz cache written by
scripts/har_build_modality_cache.py, with its own training-subject statistics.
Every modality therefore shares the same 50 Hz timeline and the same windows.

Returns ({modality: FloatTensor[C, T]}, class_index, row_index);
class_index is -1 for unknown windows.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FILES = {'wifi': ('wifi_50hz_853_video_aligned.h5', 'amp', 270),
         'imu': ('imu_50hz_853_video_aligned.h5', 'imu', 30)}
CACHE_MODALITIES = ('airpods', 'pose2d', 'pose3d', 'video')
ALL_MODALITIES = tuple(FILES) + CACHE_MODALITIES


class XRFv2WindowDataset(Dataset):
    def __init__(self, manifest, data_dir, stats_path, split, modalities=('wifi',),
                 roles=('known',), limit=None, limit_seed=0,
                 cache_path=None, cache_stats_path=None):
        self.modalities = tuple(modalities)
        with open(manifest, encoding='utf-8') as f:
            rows = [r for r in csv.DictReader(f)
                    if r['split'] == split and r['role'] in roles]
        if limit and len(rows) > limit:
            rows = random.Random(limit_seed).sample(rows, limit)
            rows.sort(key=lambda r: (r['sample_name'], int(r['start_frame'])))
        if not rows:
            raise ValueError('no windows for split=%s roles=%s' % (split, roles))
        self.rows = rows

        stats = json.load(open(stats_path, encoding='utf-8'))
        cstats = json.load(open(cache_stats_path, encoding='utf-8')) if cache_stats_path else None
        self.seqs = {m: {} for m in self.modalities}
        self._channels = {}
        needed = sorted({r['sample_name'] for r in rows})
        for m in self.modalities:
            if m in FILES:
                fname, key, nch = FILES[m]
                mean = np.asarray(stats[m]['global_mean'], dtype=np.float32)
                std = np.maximum(np.asarray(stats[m]['global_std'], dtype=np.float32), 1e-6)
                assert mean.size == nch, '%s stats have %d channels, expected %d' % (m, mean.size, nch)
                with h5py.File(Path(data_dir) / fname, 'r') as h:
                    for seq in needed:
                        v = np.asarray(h['samples'][seq][key], dtype=np.float32).reshape(-1, nch)
                        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        self.seqs[m][seq] = np.ascontiguousarray(((v - mean) / std).T)
                self._channels[m] = nch
            elif m in CACHE_MODALITIES:
                if cache_path is None or cstats is None or m not in cstats:
                    raise ValueError('modality %s needs --cache-path and --cache-stats' % m)
                mean = np.asarray(cstats[m]['global_mean'], dtype=np.float32)
                std = np.maximum(np.asarray(cstats[m]['global_std'], dtype=np.float32), 1e-6)
                with h5py.File(cache_path, 'r') as h:
                    for seq in needed:
                        v = np.asarray(h['samples'][seq][m], dtype=np.float32)
                        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        assert v.shape[1] == mean.size, '%s cache/stats channel mismatch' % m
                        self.seqs[m][seq] = np.ascontiguousarray(((v - mean) / std).T)
                self._channels[m] = int(mean.size)
            else:
                raise ValueError('unknown modality %r' % m)

        self.labels = np.array([int(r['class_index']) for r in rows], dtype=np.int64)
        self.envs = np.array([r['env'] for r in rows])
        self.subjects = np.array([str(r['subject']) for r in rows])
        self.roles = np.array([r['role'] for r in rows])
        self.names = np.array([r['sample_name'] for r in rows])

    def __len__(self):
        return len(self.rows)

    def channels(self, modality):
        return self._channels[modality]

    def __getitem__(self, i):
        r = self.rows[i]
        s, e = int(r['start_frame']), int(r['end_frame'])
        x = {m: torch.from_numpy(self.seqs[m][r['sample_name']][:, s:e].copy())
             for m in self.modalities}
        return x, int(self.labels[i]), i
