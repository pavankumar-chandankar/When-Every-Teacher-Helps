#!/usr/bin/env python3
"""Synthetic end-to-end test. Never loads research measurements."""
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pose_observability as obs

rng=np.random.default_rng(2026)
with tempfile.TemporaryDirectory() as d:
    root=Path(d)
    op=root/'operanet';op.mkdir()
    for exp in obs.OP_EXPS:
        n=12;subj=obs.OP_SUBJECTS[exp]
        labels=np.tile(np.arange(6,dtype=np.int16),2)
        k=np.ones((n,15,2,19,3),dtype=np.float32)
        velocity=(labels.astype(float)+1+0.05*exp)[:,None,None,None,None] * np.linspace(0,1,15)[None,:,None,None,None]
        k+=velocity.astype(np.float32)
        data={'X_kinect':k.reshape(n,15,114),
              'y':labels,'start_time_s':np.arange(n,dtype=float),
              'subject':np.asarray(subj),'room':np.int8(1 if exp<55 else 2),
              'experiment':np.int16(exp),'split':np.asarray(obs.OP_SPLITS[subj])}
        for mod,dim in obs.OP_DIMS.items():
            if mod!='kinect':data['X_'+mod]=(rng.random((n,15,dim))+.02).astype(np.float32)
        np.savez_compressed(op/f'exp{exp:03d}_aligned.npz',**data)
    cmd=['python',str(Path(__file__).parent/'pose_observability.py'),'run','--dataset','operanet',
         '--input',str(op),'--out',str(root/'opout'),'--seeds','11','--epochs','1',
         '--batch','16','--modalities','wifi','--device','cpu']
    subprocess.run(cmd,check=True)
    mm=root/'mmfi';mm.mkdir()
    frames=np.asarray(np.linspace(1,297,48),dtype=np.int16)
    for i in range(1,41):
        subj=f'S{i:02d}';scene=f'E{(i-1)//10+1:02d}'
        data={'y':np.arange(27,dtype=np.int64),'frames':frames,'subject':np.asarray(subj),
              'scene':np.asarray(scene)}
        for m,dim in obs.MM_DIMS.items():
            if m=='rgbpose':
                p=np.ones((27,48,17,2),dtype=np.float32)
                p[:,:,:,0]+=(np.arange(27)+1+0.10*i)[:,None,None] * np.arange(48)[None,:,None]/200.0
                p[:,:,:,1]+=rng.random((27,48,17)).astype(np.float32)*.02
                data[m]=p.reshape(27,48,34)
            elif m=='wifi':
                # Variable but motion-uninformative input. Constant tensors can
                # produce degenerate synthetic representations on some PyTorch versions.
                data[m]=(1. + .02*rng.standard_normal((27,48,dim))).astype(np.float32)
            else:
                data[m]=np.ones((27,48,dim),dtype=np.float32)
        np.savez_compressed(mm/f'{subj}.npz',**data)
    subprocess.run(['python',str(Path(__file__).parent/'pose_observability.py'),'run',
                    '--dataset','mmfi','--input',str(mm),'--out',str(root/'mmout'),
                    '--seeds','11','--epochs','1','--batch','128','--modalities','wifi',
                    '--device','cpu'],check=True)
    import csv
    for folder,n in (('opout',1),('mmout',2)):
        with open(root/folder/'observability.csv') as f:rows=list(csv.DictReader(f))
        assert len(rows)==n
        for row in rows:
            for metric in ('Err_oracle','Err_O','Err_E','R_O','R_E','H'):
                assert np.isfinite(float(row[metric])), (folder,metric,row)
    print('SYNTHETIC_END_TO_END_PASSED')
