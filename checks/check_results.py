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
    print('Classification matrices match reported pair counts, signs and sampling summaries.')

if __name__ == '__main__':
    check()
