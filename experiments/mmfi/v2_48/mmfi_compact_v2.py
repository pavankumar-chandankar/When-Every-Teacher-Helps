#!/usr/bin/env python3
"""MM-Fi read-only ZIP -> compact per-sequence five-modal features.

Never extracts the 2.25M files. Actions come from path; ground_truth.npy is
3D pose annotation and is not used as action label or as input.
"""
import argparse
from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

MODS = ('wifi', 'mmwave', 'lidar', 'depth', 'rgbpose')
FREQ = 48
FRAMES = tuple(int(x) for x in np.linspace(1,297,FREQ))
DIMS = {'wifi': 570, 'mmwave': 22, 'lidar': 14, 'depth': 256, 'rgbpose': 34}


def finite(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        good = x[np.isfinite(x)]
        x = np.nan_to_num(x, nan=float(good.mean()), posinf=float(good.max()), neginf=float(good.min()))
    return x


def cloud(blob, width):
    if len(blob) == 0:
        return np.zeros(width * 4 + 2, dtype=np.float32)
    a = np.frombuffer(blob, dtype='<f8')
    if a.size % width:
        raise ValueError(f'Point cloud invalid length {a.size} for width {width}')
    a = a.reshape(-1, width)
    if len(a) == 0:
        return np.zeros(width * 4 + 2, dtype=np.float32)
    a = finite(a)
    # Per-channel moments and interquartile range; fixed-width descriptor.
    q25, q75 = np.quantile(a, (0.25, 0.75), axis=0)
    return np.concatenate([a.mean(0), a.std(0), q25, q75,
                           [np.log1p(len(a)), 1.0]]).astype(np.float32)


def features(mod, blob):
    if mod == 'mmwave':
        return cloud(blob, 5)
    if mod == 'lidar':
        return cloud(blob, 3)
    if mod == 'rgbpose':
        a = np.load(BytesIO(blob), allow_pickle=False)
        if a.shape != (17, 2):
            raise ValueError(f'RGB-pose shape {a.shape} != (17,2)')
        return finite(a).reshape(-1)
    if mod == 'wifi':
        d = loadmat(BytesIO(blob), variable_names=['CSIamp'])
        a = d.get('CSIamp')
        if a is None or a.shape != (3, 114, 10):
            raise ValueError(f'CSIamp shape {None if a is None else a.shape}')
        a = finite(a)
        # Six adjacent subcarriers per bin, preserve receiver and packet channels.
        return a.reshape(3, 19, 6, 10).mean(axis=2).reshape(-1)
    if mod == 'depth':
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError('Pillow required: python -m pip install pillow') from e
        with Image.open(BytesIO(blob)) as im:
            a = np.asarray(im)
        if a.shape != (480,640):
            raise ValueError(f'depth shape {a.shape} != (480,640)')
        # 16x16 spatial grid of mean depth in metres (raw 16-bit millimetres).
        return (a.astype(np.float32).reshape(16,30,16,40).mean(axis=(1,3)) * .001).reshape(-1)
    raise ValueError(mod)


def member(seq, mod, ix):
    ext = {'wifi': ('wifi-csi', 'mat'), 'mmwave': ('mmwave','bin'),
           'lidar': ('lidar','bin'), 'depth': ('depth','png'),
           'rgbpose': ('rgb', 'npy')}[mod]
    return f'MMFi_Dataset/{seq[0]}/{seq[1]}/{seq[2]}/{ext[0]}/frame{ix:03d}.{ext[1]}'


def prepare(archive, out):
    archive, out = Path(archive), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    print('MMFI_COMPACT_START', 'frame_indices', FRAMES, flush=True)
    with ZipFile(archive) as z:
        # Each z.open(...).read() checks its own CRC; never CRC-scan 83 GB.
        subjects = tuple(f'S{i:02d}' for i in range(1,41))
        for si, subject in enumerate(subjects, 1):
            path = out / f'{subject}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as old:
                    good = all(m in old and old[m].shape == (27,FREQ,DIMS[m]) for m in MODS)
                    good &= (old['y'].shape == (27,))
                    good &= np.array_equal(old['y'], np.arange(27))
                if not good:
                    raise RuntimeError(f'Existing checkpoint invalid: {path}')
                print('SKIP_VERIFIED', subject, flush=True)
                continue
            scene = f'E{(int(subject[1:])-1)//10+1:02d}'
            data = {m: np.empty((27,FREQ,DIMS[m]), dtype=np.float32) for m in MODS}
            missing_mmwave = 0
            for ai in range(1,28):
                seq = (scene, subject, f'A{ai:02d}')
                for t, frame in enumerate(FRAMES):
                    for mod in MODS:
                        name = member(seq,mod,frame)
                        with z.open(name) as f:
                            b = f.read()
                        if not b and mod == 'mmwave':
                            missing_mmwave += 1
                        elif not b:
                            raise ValueError(f'Empty sensor file {name}')
                        x = features(mod,b)
                        if x.shape != (DIMS[mod],) or not np.all(np.isfinite(x)):
                            raise ValueError(f'Invalid features {name}: {x.shape}')
                        data[mod][ai-1,t] = x
            data.update(y=np.arange(27,dtype=np.int64),
                        subject=np.asarray(subject), scene=np.asarray(scene),
                        frames=np.asarray(FRAMES,dtype=np.int16),
                        missing_mmwave=np.asarray(missing_mmwave))
            temp = out / (path.name+'.partial')
            with open(temp,'wb') as f:
                np.savez_compressed(f, **data)
            os.replace(temp,path)
            print('SUBJECT_COMPLETE',subject,scene,'sequences=27',
                  'empty_mmwave_sampled',missing_mmwave,'subjects_done',si,'/40',flush=True)
    # Full cohort gate is embedded, not a separate audit.
    for subject in subjects:
        with np.load(out/f'{subject}.npz',allow_pickle=False) as d:
            assert all(d[m].shape == (27,FREQ,DIMS[m]) for m in MODS)
            assert np.array_equal(d['y'], np.arange(27))
    report = {'status':'PREP_COMPLETE','archive_bytes':archive.stat().st_size,
              'subjects':40,'actions':27,'sequences':1080,
              'frames_sampled':list(FRAMES),'modalities':DIMS,
              'note':'Exploratory 48-frame replication; rgbpose=2D keypoints; labels from path; mmwave empty frames=zero descriptor'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    print('MMFI_COMPACT_GATE_PASSED',out,'sequences=1080',flush=True)


def self_test():
    a=np.arange(3*114*10,dtype=np.float64).reshape(3,114,10)
    from scipy.io import savemat
    b=BytesIO();savemat(b,{'CSIamp':a})
    assert features('wifi',b.getvalue()).shape==(570,)
    p=BytesIO();np.save(p,np.ones((17,2)))
    assert features('rgbpose',p.getvalue()).shape==(34,)
    assert features('mmwave', b'').shape==(22,)
    assert features('lidar',np.ones((2,3),dtype='<f8').tobytes()).shape==(14,)
    try:
        features('lidar', b'xyz')
        raise AssertionError('malformed cloud was accepted')
    except ValueError:
        pass
    from PIL import Image
    p=BytesIO();Image.fromarray(np.zeros((480,640),dtype=np.uint16)).save(p,format='PNG')
    assert features('depth',p.getvalue()).shape == (256,)
    assert FRAMES[0]==1 and FRAMES[-1]==297 and len(set(FRAMES))==FREQ
    print('SYNTHETIC_SENSOR_TRAPS_PASSED')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['self-test','prepare'])
    p.add_argument('--archive',default='./data/datasets/mmfi/MMFi_Dataset.zip')
    p.add_argument('--out',default='./data/perfail2027/mmfi/compact_v2_48')
    a=p.parse_args()
    if a.action=='self-test':self_test()
    else:prepare(a.archive,a.out)
