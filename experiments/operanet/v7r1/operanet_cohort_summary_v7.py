#!/usr/bin/env python3
"""Read only per-experiment aligned-window outputs and audit split/label coverage."""
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path('./data/perfail2027/operanet/aligned_v7r1_10')
EXPS = (18,29,32,33,56,57,58,59,60,61)
CLASSES=('walk','sit','stand','liedown','standfromlie','bodyrotate')
expected={'train':{'One','Two','Three'},'dev':{'Five'},'test':{'Six','Seven'}}
counts={name:Counter() for name in expected}
subjects={name:set() for name in expected}
rooms={name:set() for name in expected}
valid=True
for exp in EXPS:
    path=ROOT/f'exp{exp:03d}_aligned.npz'
    if not path.exists():
        print('MISSING',path); valid=False; continue
    with np.load(path,allow_pickle=False) as a:
        lab=a['y']; split=str(a['split']); sub=str(a['subject']); room=int(a['room'])
        if not len(lab) or not np.all((lab>=0)&(lab<len(CLASSES))):
            raise RuntimeError('Invalid labels in '+str(path))
        if split not in expected or sub not in expected[split]:
            raise RuntimeError('Subject leakage or unknown split: '+str(path))
        for modality, dim in (('wifi',270),('kinect',114),('pwr',600),('uwb',42)):
            x=a['X_'+modality]
            if x.shape != (len(lab),15,dim): raise RuntimeError('Shape error '+str(path)+'/'+modality+': '+str(x.shape))
        counts[split].update(lab.tolist()); subjects[split].add(sub); rooms[split].add(room)
        print('FILE',exp,'SPLIT',split,'SUBJECT',sub,'ROOM',room,'WINDOWS',len(lab),'DISTRIBUTION',dict(Counter(lab.tolist())))
print('\n=== SPLIT AND CLASS COVERAGE ===')
for split in ('train','dev','test'):
    labels={CLASSES[i]:counts[split][i] for i in range(len(CLASSES))}
    print(split,'WINDOWS',sum(counts[split].values()),'SUBJECTS',sorted(subjects[split]),'ROOMS',sorted(rooms[split]),'LABELS',labels)
    if not subjects[split]: valid=False
    missing=[key for key,n in labels.items() if n==0]
    if missing:
        print('CLASS_SUPPORT_WARNING',split,'MISSING',missing,'FIXED_6_CLASS_F1_WILL_SCORE_THEM_AS_ZERO')
        if split=='train': valid=False
for a in ('train','dev','test'):
    for b in ('train','dev','test'):
        if a!=b and subjects[a]&subjects[b]: raise RuntimeError('SUBJECT LEAKAGE')
print('COHORT_TRAINING_GATE', 'PASSED' if valid else 'FAILED')
if not valid: raise SystemExit(2)
