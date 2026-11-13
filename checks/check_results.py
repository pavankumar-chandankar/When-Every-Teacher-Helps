"""Check classification summaries against the manuscript."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with (ROOT / path).open(newline='') as f:
        return list(csv.DictReader(f))

def check_matrix(name, expected_pairs):
    rows = read(name)
    baseline = [r for r in rows if r['kind'] == 'baseline']
    gain = [r for r in rows if r['kind'] == 'transfer_gain']
    n = len(baseline)
    assert len(gain) == expected_pairs and len(rows) == n * n
    assert len({r['student'] for r in baseline}) == n
    assert all(int(r['seeds']) == 3 for r in rows)
    return baseline, gain


def check():
    _, xrf = check_matrix('results/xrfv2-plus/matrix_published_rounded.csv', 30)
    assert all(float(r['value']) > 0 for r in xrf)
    assert min(float(r['value']) for r in xrf) == 0.7
    assert max(float(r['value']) for r in xrf) == 9.0

    op_base, op = check_matrix('results/operanet/matrix_published_rounded.csv', 12)
    assert all(float(r['value']) > 0 for r in op)
    assert {r['teacher']: float(r['value']) for r in op if r['student'] == 'wifi'} == {'kinect': 6.2, 'pwr': 3.5, 'uwb': 4.2}

    for frame, positives, avg, wifi, rgb in [
        (12, 8, -1.13, 0.0504, 0.9232),
        (48, 14, 0.63, 0.0973, 0.9440),
    ]:
        name = f'results/mmfi/v1_12frame_matrix.csv' if frame == 12 else 'results/mmfi/v2_48frame_matrix.csv'
        base, gains = check_matrix(name, 20)
        score = {r['student']:float(r['mean_macro_f1_or_gain']) for r in base}
        values = [float(r['mean_macro_f1_or_gain']) for r in gains]
        assert sum(v > 0 for v in values) == positives
        assert round(100 * sum(values) / len(values), 2) == avg
        assert score['wifi'] == wifi and score['rgbpose'] == rgb
        if frame == 48:
            assert all(float(r['mean_macro_f1_or_gain']) < 0 for r in gains if r['student'] == 'wifi')
    check_xrf_run_log()
    check_observability()
    print('Classification matrices, XRF run log and observability values validated.')


def check_xrf_run_log():
    import re
    text = (ROOT / 'experiments/xrfv2-plus/figures/matrix_raw.txt').read_text()
    names = ('imu', 'airpods', 'pose2d', 'pose3d', 'video', 'wifi')
    pattern = r'((?:' + '|'.join(names) + r'|student)(?:_from_(?:' + '|'.join(names) + r'))?)_seed([123]) "val_known_macro_f1": ([0-9.]+)'
    values = {(name, int(seed)): float(score) for name, seed, score in re.findall(pattern, text)}
    for seed in (1, 2, 3):
        values['wifi_from_imu', seed] = values['student', seed]
    gains = {}
    positive_runs = 0
    for student in names:
        for teacher in names:
            if student == teacher:
                continue
            per_seed = [values[f'{student}_from_{teacher}', seed] - values[student, seed] for seed in (1, 2, 3)]
            positive_runs += sum(v > 0 for v in per_seed)
            gains[student, teacher] = sum(per_seed) / 3
    assert len(gains) == 30 and positive_runs == 82 and all(v > 0 for v in gains.values())

    aliases = {'2d_pose': 'pose2d', '3d_pose': 'pose3d', 'body_imu': 'imu'}
    for row in read('results/xrfv2-plus/matrix_published_rounded.csv'):
        student = aliases.get(row['student'], row['student'])
        teacher = aliases.get(row['teacher'], row['teacher'])
        if row['kind'] == 'baseline':
            actual = sum(values[student, seed] for seed in (1, 2, 3)) / 3
            assert abs(actual - float(row['value'])) < .0051
        else:
            assert abs(100 * gains[student, teacher] - float(row['value'])) < .051


def check_observability():
    rows = read('results/xrfv2-plus/observability.csv')
    assert len(rows) == 156
    def values(modality, factor):
        return [float(row['R_O']) for row in rows
                if row['modality'] == modality and row['factor'] == factor and row['scale'] == 'per_second']
    assert all(.50 <= x <= .52 for x in values('AirPods (head control)', 'F3h'))
    assert all(.44 <= x <= .49 for x in values('Pose2D', 'F1'))
    assert all(.90 <= x <= .94 for x in values('WiFi', 'F1'))

if __name__ == '__main__':
    check()
