#!/usr/bin/env python3
"""Train and evaluate DeepConvLSTM on XRFv2 Plus sliding windows, Fold A.

Any modality can be a baseline, a teacher or a student:
  --modality M                              M-only baseline (or teacher)
  --modality S --teacher-ckpt T.pt          S student distilled from teacher T
                                            (T's modality is read from its checkpoint)

Student loss = cross-entropy + align_weight * mean(1 - cos(student_emb, teacher_emb)).
The frozen teacher sees its own modality; the student sees only its own, so the
student is single-modality at inference by construction and its checkpoint holds
only the student.

Modalities: wifi, imu (original HDF5 files, TAL statistics) and airpods, pose2d,
pose3d, video (50 Hz cache, needs --cache-path and --cache-stats).

Model selection: best validation macro-F1 on KNOWN windows only (subjects 12-13).
The test split is never read. After training, the selected checkpoint is
evaluated for known-class accuracy / macro-F1 and for open-set rejection (MSP)
with validation-unknown windows as positives; validation embeddings are saved.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossguard_fi.data.xrfv2_windows import XRFv2WindowDataset, ALL_MODALITIES   # noqa: E402
from crossguard_fi.models.deepconvlstm import DeepConvLSTM                        # noqa: E402

NUM_KNOWN = 24


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def macro_f1(y, p):
    f1s = []
    for c in np.unique(y):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def ranking(pos, neg):
    if pos.size == 0 or neg.size == 0:
        return {'auroc': None, 'aupr': None, 'fpr_at_95_tpr': None, 'n_pos': int(pos.size), 'n_neg': int(neg.size)}
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    y = y[np.argsort(-s, kind='mergesort')]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / pos.size; fpr = fp / neg.size
    trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    prec = tp / np.maximum(tp + fp, 1e-12)
    i = np.searchsorted(tpr, 0.95, side='left')
    return {'auroc': float(trap(np.r_[0, tpr], np.r_[0, fpr])),
            'aupr': float(np.sum(np.diff(np.r_[0, tpr]) * prec)),
            'fpr_at_95_tpr': float(fpr[min(i, fpr.size - 1)]),
            'base_rate': float(pos.size / (pos.size + neg.size)),
            'n_pos': int(pos.size), 'n_neg': int(neg.size)}


@torch.no_grad()
def predict(model, loader, modality, device):
    model.eval()
    embs, logits = [], []
    for x, _, _ in loader:
        e, l = model(x[modality].to(device))
        embs.append(e.float().cpu()); logits.append(l.float().cpu())
    return torch.cat(embs).numpy(), torch.cat(logits).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--stats-path', type=Path, required=True)
    ap.add_argument('--cache-path', type=Path, default=None)
    ap.add_argument('--cache-stats', type=Path, default=None)
    ap.add_argument('--modality', choices=ALL_MODALITIES, required=True)
    ap.add_argument('--teacher-ckpt', type=Path, default=None)
    ap.add_argument('--align-weight', type=float, default=1.0)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--limit-train', type=int, default=None, help='smoke tests only')
    ap.add_argument('--limit-val', type=int, default=None, help='smoke tests only')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    seed_all(a.seed)                                    # before any model is built

    ck_t, t_mod = None, None
    if a.teacher_ckpt is not None:
        ck_t = torch.load(a.teacher_ckpt, map_location='cpu')
        t_mod = ck_t['args']['modality']
        if t_mod == a.modality:
            raise SystemExit('teacher and student must use different modalities')
    distill = ck_t is not None
    train_mods = (a.modality, t_mod) if distill else (a.modality,)
    kw = dict(cache_path=a.cache_path, cache_stats_path=a.cache_stats)

    t0 = time.time()
    train_ds = XRFv2WindowDataset(a.manifest, a.data_dir, a.stats_path, 'train', modalities=train_mods,
                                  roles=('known',), limit=a.limit_train, **kw)
    val_ds = XRFv2WindowDataset(a.manifest, a.data_dir, a.stats_path, 'val', modalities=(a.modality,),
                                roles=('known', 'unknown'), limit=a.limit_val, **kw)
    print('data loaded in %.0fs: train %d windows, val %d (known %d, unknown %d)'
          % (time.time() - t0, len(train_ds), len(val_ds),
             int((val_ds.roles == 'known').sum()), int((val_ds.roles == 'unknown').sum())), flush=True)

    g = torch.Generator(); g.manual_seed(a.seed)
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                          num_workers=a.num_workers, generator=g)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=a.num_workers)

    model = DeepConvLSTM(train_ds.channels(a.modality), NUM_KNOWN).to(dev)
    print('model: DeepConvLSTM  modality=%s  channels=%d  params=%d'
          % (a.modality, train_ds.channels(a.modality), sum(p.numel() for p in model.parameters())), flush=True)

    teacher = None
    if distill:
        teacher = DeepConvLSTM(**ck_t['model_config'])
        teacher.load_state_dict(ck_t['state_dict']); teacher.to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print('teacher: %s modality=%s (epoch %d, val macro-F1 %.4f)  align_weight=%g'
              % (a.teacher_ckpt, t_mod, ck_t['epoch'], ck_t['val_macro_f1'], a.align_weight), flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    known = val_ds.roles == 'known'
    yk = val_ds.labels[known]
    best, history = -1.0, []
    for ep in range(a.epochs):
        model.train()
        n, ce_sum, al_sum, t1 = 0, 0.0, 0.0, time.time()
        for x, y, _ in train_dl:
            y = y.to(dev)
            emb, logits = model(x[a.modality].to(dev))      # student / baseline: own modality only
            ce = F.cross_entropy(logits, y)
            loss = ce
            if teacher is not None:
                with torch.no_grad():
                    t_emb, _ = teacher(x[t_mod].to(dev))   # privileged input, loss only
                al = (1.0 - F.cosine_similarity(emb, t_emb, dim=1)).mean()
                loss = ce + a.align_weight * al
                al_sum += al.item() * y.size(0)
            opt.zero_grad(); loss.backward(); opt.step()
            ce_sum += ce.item() * y.size(0); n += y.size(0)

        _, lv = predict(model, val_dl, a.modality, dev)
        pk = lv[known].argmax(1)
        f1, acc = macro_f1(yk, pk), float((pk == yk).mean())
        row = {'epoch': ep, 'train_ce': ce_sum / n, 'train_align': (al_sum / n) if teacher is not None else None,
               'val_known_acc': acc, 'val_known_macro_f1': f1, 'seconds': time.time() - t1}
        history.append(row)
        tag = ''
        if f1 > best:
            best = f1; tag = '  *'
            torch.save({'state_dict': model.state_dict(), 'model_config': model.config,
                        'args': {k: str(v) for k, v in vars(a).items()}, 'epoch': ep,
                        'val_macro_f1': f1, 'val_acc': acc}, a.out_dir / 'best.pt')
        print('epoch %3d  ce %.4f%s  val acc %.4f  macro-F1 %.4f  (%.0fs)%s'
              % (ep, row['train_ce'], ('  align %.4f' % row['train_align']) if teacher is not None else '',
                 acc, f1, row['seconds'], tag), flush=True)

    with open(a.out_dir / 'history.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0].keys())); wr.writeheader(); wr.writerows(history)

    ck = torch.load(a.out_dir / 'best.pt', map_location='cpu')
    model.load_state_dict(ck['state_dict'])
    ev, lv = predict(model, val_dl, a.modality, dev)
    prob = torch.softmax(torch.from_numpy(lv), 1).numpy()
    msp_unknown = 1.0 - prob.max(1)
    pk = lv[known].argmax(1)
    summary = {
        'arm': 'student' if distill else a.modality,
        'modality': a.modality, 'teacher_modality': t_mod, 'modality_at_inference': a.modality,
        'seed': a.seed, 'best_epoch': ck['epoch'],
        'val_known_acc': float((pk == yk).mean()), 'val_known_macro_f1': macro_f1(yk, pk),
        'open_set_msp': ranking(msp_unknown[~known], msp_unknown[known]),
        'n_train': len(train_ds), 'n_val_known': int(known.sum()), 'n_val_unknown': int((~known).sum()),
        'teacher_ckpt': str(a.teacher_ckpt) if distill else None,
        'align_weight': a.align_weight if distill else None,
    }
    (a.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    np.savez_compressed(a.out_dir / 'val_embeddings.npz', emb=ev, logits=lv,
                        class_index=val_ds.labels, role=val_ds.roles, env=val_ds.envs,
                        subject=val_ds.subjects, sample_name=val_ds.names)
    o = summary['open_set_msp']
    print('\nBEST epoch %d | val known acc %.4f  macro-F1 %.4f | open-set MSP AUROC %s  AUPR %s (base %s)'
          % (summary['best_epoch'], summary['val_known_acc'], summary['val_known_macro_f1'],
             'n/a' if o['auroc'] is None else '%.4f' % o['auroc'],
             'n/a' if o['aupr'] is None else '%.4f' % o['aupr'],
             'n/a' if o.get('base_rate') is None else '%.4f' % o['base_rate']))
    print('written', a.out_dir)


if __name__ == '__main__':
    main()
