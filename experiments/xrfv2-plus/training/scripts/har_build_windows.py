#!/usr/bin/env python3
"""Build sliding-window HAR manifests for XRFv2 Plus, Fold A, from the TAL protocol.

Every protocol decision is inherited from the TAL track so the two tracks are
directly comparable:
  * subjects and splits come from the TAL fold annotation files
    (train = 0-11, validation = 12-13, test = 14-15);
  * known classes and their order come from TAL's class_mapping.json;
  * held-out activities are 'ignore' in training and 'unknown' in validation /
    test, exactly as in TAL. They are never relabelled as background.

Labelling rule. A window takes the label of the annotation that covers most of
it, provided that annotation covers at least --min-overlap of the window.
Otherwise it straddles two activities and is dropped as a transition: XRFv2
annotations tile every recording, so there is no genuine background class.

Contamination rule. A window is dropped if more than --max-ignore of it lies
inside ignored (held-out) intervals, so held-out activities do not leak into
training windows, and validation windows are not muddied by test-unknowns.

Test windows are written but carry split='test'; the trainer never reads them.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import h5py

SPLIT_FILES = {'train': 'train.json', 'val': 'validation.json', 'test': 'test.json'}
ROLE_FIELDS = (('known', 'annotations'),
               ('unknown', 'unknown_annotations'),
               ('ignore', 'ignore_annotations'))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold-dir', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--window-seconds', type=float, default=5.0)
    ap.add_argument('--stride-seconds', type=float, default=2.5)
    ap.add_argument('--sample-rate', type=float, default=50.0)
    ap.add_argument('--min-overlap', type=float, default=0.5)
    ap.add_argument('--max-ignore', type=float, default=0.2)
    ap.add_argument('--allow-missing', action='store_true',
                    help='skip sequences absent from the HDF5 (testing only)')
    a = ap.parse_args()

    sr = a.sample_rate
    win = int(round(a.window_seconds * sr))
    stride = int(round(a.stride_seconds * sr))
    assert win > 0 and stride > 0

    mapping = json.load(open(a.fold_dir / 'class_mapping.json', encoding='utf-8'))
    zh_to_index = {r['label_zh']: int(r['new_class_id']) - 1
                   for r in mapping['known_foreground_classes']}
    n_known = len(zh_to_index)
    assert sorted(zh_to_index.values()) == list(range(n_known)), 'non-contiguous classes'

    rows, drops, missing = [], Counter(), []
    with h5py.File(a.data_dir / 'wifi_50hz_853_video_aligned.h5', 'r') as hw, \
         h5py.File(a.data_dir / 'imu_50hz_853_video_aligned.h5', 'r') as hi:
        for split, fname in SPLIT_FILES.items():
            db = json.load(open(a.fold_dir / fname, encoding='utf-8'))['database']
            for seq in sorted(db):
                rec = db[seq]
                if seq not in hw['samples'] or seq not in hi['samples']:
                    missing.append(seq)
                    continue
                T = min(hw['samples'][seq]['amp'].shape[0], hi['samples'][seq]['imu'].shape[0])
                intervals = []
                for role, field in ROLE_FIELDS:
                    for ann in rec.get(field, []):
                        s = int(round(float(ann['segment'][0]) * sr))
                        e = int(round(float(ann['segment'][1]) * sr))
                        s, e = max(0, s), min(T, e)
                        if e > s:
                            intervals.append((s, e, role, str(ann['label'])))
                env = rec.get('scene_name') or rec.get('scene')
                subject = rec.get('volunteer_id')
                for w, start in enumerate(range(0, T - win + 1, stride)):
                    end = start + win
                    best, best_ov, ign = None, 0, 0
                    for s, e, role, label in intervals:
                        ov = max(0, min(end, e) - max(start, s))
                        if role == 'ignore':
                            ign += ov
                        if ov > best_ov:
                            best, best_ov = (role, label), ov
                    ratio, ign_frac = best_ov / win, ign / win
                    if best is None or ratio < a.min_overlap:
                        drops[(split, 'transition')] += 1
                        continue
                    if ign_frac > a.max_ignore:
                        drops[(split, 'ignore_contaminated')] += 1
                        continue
                    role, label = best
                    if role == 'ignore' or (split == 'train' and role != 'known'):
                        drops[(split, 'role_' + role)] += 1
                        continue
                    if role == 'known' and label not in zh_to_index:
                        raise SystemExit('known label %r missing from class_mapping' % label)
                    rows.append({
                        'sample_name': seq, 'subject': subject, 'env': env, 'split': split,
                        'window_index': w, 'start_frame': start, 'end_frame': end,
                        'role': role, 'label_zh': label,
                        'class_index': zh_to_index[label] if role == 'known' else -1,
                        'overlap_ratio': '%.4f' % ratio, 'ignore_fraction': '%.4f' % ign_frac,
                    })

    if missing and not a.allow_missing:
        raise SystemExit('%d sequences missing from HDF5, e.g. %s' % (len(missing), missing[:3]))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    kept = Counter((r['split'], r['role']) for r in rows)
    train_classes = Counter(r['class_index'] for r in rows if r['split'] == 'train')
    report = {
        'window_frames': win, 'stride_frames': stride, 'min_overlap': a.min_overlap,
        'max_ignore': a.max_ignore, 'known_classes': n_known,
        'kept': {'%s/%s' % k: v for k, v in sorted(kept.items())},
        'dropped': {'%s/%s' % k: v for k, v in sorted(drops.items())},
        'train_windows_per_class': {str(k): v for k, v in sorted(train_classes.items())},
        'missing_sequences': len(missing),
    }
    Path(str(a.out) + '.report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('window %d frames, stride %d frames' % (win, stride))
    for k, v in sorted(kept.items()):
        print('  kept    %-5s %-8s %6d' % (k[0], k[1], v))
    for k, v in sorted(drops.items()):
        print('  dropped %-5s %-20s %6d' % (k[0], k[1], v))
    print('  train classes present: %d / %d   (min %d, max %d windows per class)'
          % (len(train_classes), n_known, min(train_classes.values()), max(train_classes.values())))
    if missing:
        print('  missing sequences skipped:', len(missing))
    print('written', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
