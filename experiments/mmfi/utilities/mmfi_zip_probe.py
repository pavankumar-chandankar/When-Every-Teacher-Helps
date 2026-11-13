#!/usr/bin/env python3
"""Read-only MM-Fi ZIP inventory and representative format probe; no bulk extraction.

Run this on an HPC compute node (ZIP has >2 million central-directory entries).
"""
import argparse
import io
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import whosmat

MODS = ('wifi-csi', 'mmwave', 'lidar', 'depth', 'rgb', 'infra1', 'infra2')
EXTS = {'wifi-csi': '.mat', 'mmwave': '.bin', 'lidar': '.bin', 'depth': '.png',
        'rgb': '.npy', 'infra1': '.npy', 'infra2': '.npy'}
PAT = re.compile(r'^MMFi_Dataset/(E\d{2})/(S\d{2})/(A\d{2})/(wifi-csi|mmwave|lidar|depth|rgb|infra1|infra2)/frame(\d{3})\.(mat|bin|png|npy)$')
GT = re.compile(r'^MMFi_Dataset/(E\d{2})/(S\d{2})/(A\d{2})/ground_truth\.npy$')


def describe(z, name, mod):
    info = z.getinfo(name)
    payload = z.read(info)
    result = {'compressed_bytes': info.compress_size, 'bytes': info.file_size}
    if mod in ('rgb', 'infra1', 'infra2', 'ground_truth'):
        a = np.load(io.BytesIO(payload), allow_pickle=False)
        result.update(shape=list(a.shape), dtype=str(a.dtype), finite=float(np.isfinite(a).mean())
                      if np.issubdtype(a.dtype, np.number) and a.size else None)
    elif mod == 'wifi-csi':
        result['mat_variables'] = [(v, list(shape), typ) for v, shape, typ in whosmat(io.BytesIO(payload))]
    elif mod in ('lidar', 'mmwave'):
        width = 3 if mod == 'lidar' else 5
        if len(payload) % (8 * width):
            result['ERROR'] = f'Binary length {len(payload)} is not divisible by {8*width}'
        else:
            a = np.frombuffer(payload, dtype=np.float64).reshape(-1, width)
            result.update(shape=list(a.shape), dtype='float64', finite=float(np.isfinite(a).mean()) if a.size else None)
    elif mod == 'depth':
        if payload[:8] != b'\x89PNG\r\n\x1a\n' or payload[12:16] != b'IHDR':
            result['ERROR'] = 'Invalid PNG header'
        else:
            w, h, depth, color = struct.unpack('>IIBB', payload[16:26])
            result.update(shape=[h, w], png_bit_depth=depth, png_color_type=color)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    archive = Path(a.archive)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    print('MMFI_ZIP_PROBE_START', archive, 'BYTES', archive.stat().st_size, flush=True)
    counts = Counter()
    frames = defaultdict(lambda: [0, 10**9, 0])
    sequences = set()
    gt = set()
    other_files = Counter()
    with ZipFile(archive) as z:
        print('ZIP_DIRECTORY_READ', len(z.infolist()), 'entries', flush=True)
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            m = PAT.fullmatch(name)
            if m:
                env, subj, action, mod, frame, ext = m.groups()
                if '.' + ext != EXTS[mod]:
                    raise ValueError('Unexpected extension: ' + name)
                key = (env, subj, action)
                sequences.add(key)
                counts[mod] += 1
                spec = frames[key, mod]
                index = int(frame)
                spec[0] += 1
                spec[1] = min(spec[1], index)
                spec[2] = max(spec[2], index)
            else:
                m = GT.fullmatch(name)
                if m:
                    gt.add(m.groups())
                else:
                    other_files[name.rsplit('/', 1)[-1]] += 1
        print('SEQUENCES', len(sequences), 'GROUND_TRUTH_FILES', len(gt), flush=True)
        print('ACTIVITIES', sorted({k[2] for k in sequences}), flush=True)
        print('SUBJECTS', len({k[1] for k in sequences}), 'ENVIRONMENTS', sorted({k[0] for k in sequences}), flush=True)
        print('FRAME_FILES_BY_MODALITY', dict(sorted(counts.items())), flush=True)
        problems = []
        for key in sorted(sequences):
            if key not in gt:
                problems.append(f'{key}: missing ground_truth.npy')
            for mod in MODS:
                n, smallest, largest = frames[key, mod]
                if (n, smallest, largest) != (297, 1, 297):
                    problems.append(f'{key}/{mod}: frames={n} min={smallest} max={largest}')
        print('FRAME_SEQUENCE_ERRORS', len(problems), problems[:12], flush=True)
        print('OTHER_FILES', dict(other_files.most_common(8)), flush=True)
        chosen = [('E01', 'S01', 'A01'), ('E02', 'S11', 'A01'),
                  ('E03', 'S21', 'A01'), ('E04', 'S31', 'A01')]
        samples = {}
        for env, subj, act in chosen:
            seq = f'MMFi_Dataset/{env}/{subj}/{act}'
            if (env, subj, act) not in sequences:
                print('MISSING_SAMPLE_SEQUENCE', seq, flush=True)
                continue
            row = {}
            for mod in MODS:
                name = f'{seq}/{mod}/frame001{EXTS[mod]}'
                try:
                    row[mod] = describe(z, name, mod)
                except Exception as ex:
                    row[mod] = {'ERROR': type(ex).__name__ + ': ' + str(ex)}
            try:
                row['ground_truth'] = describe(z, f'{seq}/ground_truth.npy', 'ground_truth')
            except Exception as ex:
                row['ground_truth'] = {'ERROR': type(ex).__name__ + ': ' + str(ex)}
            samples[seq] = row
            print('SAMPLE', seq, json.dumps(row), flush=True)

    report = {'archive': str(archive), 'sequence_count': len(sequences),
              'ground_truth_count': len(gt), 'frame_file_counts': dict(counts),
              'issues_count': len(problems), 'issues_first': problems[:50], 'samples': samples}
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print('PROBE_REPORT', dest, flush=True)
    if len(sequences) == 1080 and len(gt) == 1080 and not problems and not any(
            'ERROR' in vals for row in samples.values() for vals in row.values()):
        print('MMFI_ARCHIVE_STRUCTURE_AND_SAMPLE_FORMATS_PASSED', flush=True)
    else:
        print('MMFI_ARCHIVE_REQUIRES_REVIEW_BEFORE_TRAINING', flush=True)


if __name__ == '__main__':
    main()
