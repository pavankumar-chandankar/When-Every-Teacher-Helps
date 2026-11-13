#!/usr/bin/env python3
"""Teacher x student matrix analysis for the DeepConvLSTM track.

Room score per modality (from each baseline's saved validation embeddings):
  mean cosine of different-activity/same-room centroids minus mean cosine of
  same-activity/different-room centroids. > 0 means features group by room.

Tests the pre-registered predictions:
  P1 WiFi as teacher gives ~zero gain;
  P2 gain rises with the invariance gap (student room - teacher room);
  P3 the gap predicts gain better than teacher accuracy or student headroom,
     checked by correlation and by a joint standardised regression.
"""
import json, sys, numpy as np, statistics as st
from pathlib import Path
R = Path(sys.argv[1]); M = ['imu', 'pose3d', 'pose2d', 'video', 'airpods', 'wifi']; SEEDS = (1, 2, 3)

def load(name):
    p = R / name / 'summary.json'
    return json.load(open(p)) if p.exists() else None

def student_dir(s, t, k):
    return ('student_seed%d' % k) if (t == 'imu' and s == 'wifi') else ('%s_from_%s_seed%d' % (s, t, k))

def room_score(npz):
    z = np.load(npz); k = z['role'] == 'known'
    e = z['emb'][k]; e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    y, r = z['class_index'][k], z['env'][k]
    shared = [c for c in np.unique(y) if len(np.unique(r[y == c])) >= 2]
    C = {(c, q): e[(y == c) & (r == q)].mean(0) for c in shared for q in np.unique(r) if ((y == c) & (r == q)).sum() >= 5}
    keys = list(C); sa, sr = [], []
    for i, (c1, q1) in enumerate(keys):
        for c2, q2 in keys[i + 1:]:
            u, v = C[(c1, q1)], C[(c2, q2)]; cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
            if c1 == c2 and q1 != q2: sa.append(cos)
            elif c1 != c2 and q1 == q2: sr.append(cos)
    return float(np.mean(sr) - np.mean(sa)) if sa and sr else float('nan')

alone = {m: st.mean([load('%s_seed%d' % (m, k))['val_known_macro_f1'] for k in SEEDS if load('%s_seed%d' % (m, k))]) for m in M}
room = {m: float(np.nanmean([room_score(R / ('%s_seed%d' % (m, k)) / 'val_embeddings.npz')
                             for k in SEEDS if (R / ('%s_seed%d' % (m, k)) / 'val_embeddings.npz').exists()])) for m in M}

G, pairs = {}, []
for t in M:
    for s in M:
        if s == t: continue
        g = [load(student_dir(s, t, k))['val_known_macro_f1'] - load('%s_seed%d' % (s, k))['val_known_macro_f1']
             for k in SEEDS if load(student_dir(s, t, k)) and load('%s_seed%d' % (s, k))]
        if g:
            G[(t, s)] = (st.mean(g), len(g), sum(x > 0 for x in g))
            pairs.append((t, s, st.mean(g)))

print('Modalities: alone macro-F1 and room score (>0 = room-dominated)')
for m in M: print('  %-8s F1 %.3f   room %+.3f' % (m, alone[m], room[m]))

print('\nGain matrix: rows = TEACHER, columns = STUDENT  (mean over seeds; [n+] = seeds with positive gain)')
print('%-9s' % 't \\ s' + ''.join('%-15s' % s for s in M))
for t in M:
    cells = []
    for s in M:
        if s == t: cells.append('%-15s' % '    -')
        elif (t, s) in G:
            m_, n, p = G[(t, s)]; cells.append('%-15s' % ('%+.3f [%d/%d]' % (m_, p, n)))
        else: cells.append('%-15s' % '   pending')
    row = [G[(t, s)][0] for s in M if (t, s) in G]
    print('%-9s' % t + ''.join(cells) + (('  row mean %+.3f' % np.mean(row)) if row else ''))

if len(pairs) >= 6:
    gain = np.array([p[2] for p in pairs])
    gap = np.array([room[s] - room[t] for t, s, _ in pairs])
    tacc = np.array([alone[t] for t, s, _ in pairs])
    head = np.array([1 - alone[s] for t, s, _ in pairs])
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    sp = lambda a, b: np.corrcoef(rk(a), rk(b))[0, 1]
    print('\nAcross %d pairs, Spearman correlation with gain:' % len(pairs))
    print('  invariance gap (student room - teacher room)  %+.2f' % sp(gain, gap))
    print('  teacher accuracy                               %+.2f' % sp(gain, tacc))
    print('  student headroom (1 - student F1)              %+.2f' % sp(gain, head))
    print('  [collinearity check] gap vs headroom           %+.2f' % sp(gap, head))
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    X = np.c_[np.ones(len(gain)), z(gap), z(tacc), z(head)]
    beta, *_ = np.linalg.lstsq(X, gain, rcond=None)
    r2 = 1 - (gain - X @ beta).var() / gain.var()
    print('\nJoint regression (standardised; which explanation survives when all compete):')
    print('  gain ~ %+.4f*gap  %+.4f*teacher_acc  %+.4f*headroom      R^2 = %.2f' % (beta[1], beta[2], beta[3], r2))
    nw = [(t, s, g) for t, s, g in pairs if 'wifi' not in (t, s)]
    if len(nw) >= 6:
        g2 = np.array([p[2] for p in nw]); gp2 = np.array([room[s] - room[t] for t, s, _ in nw])
        h2 = np.array([1 - alone[s] for t, s, _ in nw])
        print('\nRobustness, WiFi excluded (%d pairs): rho(gain, gap) %+.2f   rho(gain, headroom) %+.2f'
              % (len(nw), sp(g2, gp2), sp(g2, h2)))
    if all(('wifi', s) in G for s in M if s != 'wifi'):
        w = [G[('wifi', s)][0] for s in M if s != 'wifi']
        print('\nP1 check: WiFi-as-teacher gains', ' '.join('%+.3f' % x for x in w), '  mean %+.3f' % np.mean(w))
