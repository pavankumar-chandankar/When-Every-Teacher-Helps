#!/usr/bin/env bash
# One-shot, fail-closed v7.1 fork; never modify frozen v7 or existing raw data.
set -euo pipefail
BASE="$HOME/perfail_ops_tmp"
OLD="$BASE/perfail_v7"
NEW="$BASE/perfail_v7r1"
PY="$BASE/perfail_py311/bin/python"
ROOT="./data/perfail2027/operanet"
[[ -d "$OLD" && -x "$PY" ]] || { echo 'STOP: v7 source or Python 3.11 missing'; exit 2; }
[[ ! -e "$NEW" && ! -e "$ROOT/aligned_v7r1_10" && ! -e "$ROOT/matrix_v7r1_10" ]] || {
  echo 'STOP: v7r1 already exists; inspect existing job/results instead of resubmitting'; exit 2;
}
(cd "$OLD" && sha256sum -c SOURCE_SHA256.txt)
cp -a -- "$OLD" "$NEW"
"$PY" - "$NEW" <<'PYCODE'
import hashlib
import sys
from pathlib import Path
import importlib.util
import numpy as np
from scipy.io.matlab import MatlabOpaque

new = Path(sys.argv[1])
files = [p for p in new.iterdir() if p.is_file() and p.name != 'SOURCE_SHA256.txt']
for path in files:
    data = path.read_text()
    data = data.replace('/perfail_v7/', '/perfail_v7r1/')
    data = data.replace('perfail_ops_tmp/perfail_v7"', 'perfail_ops_tmp/perfail_v7r1"')
    data = data.replace('aligned_v7_10', 'aligned_v7r1_10')
    data = data.replace('matrix_v7_10', 'matrix_v7r1_10')
    data = data.replace('opera_v7_', 'opera_v7r1_')
    data = data.replace('V7_', 'V7R1_')
    data = data.replace('perfail-opera-v7', 'perfail-opera-v7r1')
    path.write_text(data)
# Match actual Slurm filenames to the rewritten script references.
for path in list(files):
    if path.name.startswith('opera_v7_'):
        path.rename(path.with_name(path.name.replace('opera_v7_', 'opera_v7r1_', 1)))
files = [p for p in new.iterdir() if p.is_file() and p.name != 'SOURCE_SHA256.txt']

p = new / 'operanet_build_windows_v7.py'
s = p.read_text()
needle = 'def ms_of_day(strings):\n'
assert s.count(needle) == 1, 'Unexpected v7 function layout'
insert = '''def verified_room_or_opaque(value):
    """Return verified MAT room, or None only for an authentic MATLAB MCOS string.

    The opaque value is NOT interpreted as a room. Independent UWB and WiFi
    metadata must both verify room, subject and recording identity later.
    """
    if isinstance(value, MatlabOpaque):
        a = np.asarray(value)
        if a.size == 1 and a.dtype.names is not None:
            if {'s0', 's1', 's2', 'arr'}.issubset(a.dtype.names):
                row = a.reshape(-1)[0]
                if row['s1'] == b'MCOS' and row['s2'] == b'string':
                    return None
    return exact_int_scalar(value, 'room_no')


'''
s = s.replace('from scipy.io import loadmat\n', 'from scipy.io import loadmat\nfrom scipy.io.matlab import MatlabOpaque\n', 1)
s = s.replace(needle, insert + needle, 1)
old = "room=exact_int_scalar(table[i,col['room_no']], 'room_no')"
assert s.count(old)==1, 'Expected room parsing call not found uniquely'
s = s.replace(old,"room=verified_room_or_opaque(table[i,col['room_no']])",1)
old = 'if mat_exp != exp or room != int(expected_room):'
assert s.count(old)==1, 'Expected metadata comparison not found uniquely'
s = s.replace(old,'if mat_exp != exp or (room is not None and room != int(expected_room)):',1)
old = 'opaque_person_rows = 0\n'
assert s.count(old)==1, 'Expected counter not found uniquely'
s = s.replace(old,old+'        opaque_room_rows = 0\n',1)
old = 'if person is None:\n                opaque_person_rows += 1\n'
assert s.count(old)==1, 'Expected person branch not found uniquely'
s = s.replace(old,old+'            if room is None:\n                opaque_room_rows += 1\n',1)
old = "'OPAQUE_PERSON_UNVERIFIED_ROWS',opaque_person_rows,'TIME_RANGE'"
assert s.count(old)==1, 'Expected audit output not found uniquely'
s = s.replace(old,"'OPAQUE_PERSON_UNVERIFIED_ROWS',opaque_person_rows,'OPAQUE_ROOM_UNVERIFIED_ROWS',opaque_room_rows,'TIME_RANGE'",1)
old = "'KINECT_PWR_MCOS_PERSON_VALUES_UNREADABLE_WHEN_OPAQUE'"
assert s.count(old)==1, 'Expected provenance output not found uniquely'
s = s.replace(old,"'KINECT_PWR_MCOS_PERSON_AND_ROOM_VALUES_UNREADABLE_WHEN_OPAQUE'",1)
p.write_text(s)

# Syntax plus real function-level traps, including rejecting wrong numeric room.
compile(s, str(p), 'exec')
spec=importlib.util.spec_from_file_location('operanet_v7r1', p)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
opaque_dtype=np.dtype([('s0','O'),('s1','O'),('s2','O'),('arr','O')])
opaque=MatlabOpaque(np.array([(b'',b'MCOS',b'string',np.array([[1],[2]],dtype=np.uint32))],dtype=opaque_dtype))
assert mod.verified_room_or_opaque(opaque) is None
for value,want in [('room_1',1),('Room 2',2),(1,1),(np.array([[2.]]),2)]:
    assert mod.verified_room_or_opaque(value)==want
for invalid in ('room_9a',3.5,np.nan,np.array([1,2]),'floor_1'):
    try: mod.verified_room_or_opaque(invalid)
    except ValueError: continue
    raise AssertionError('Invalid room accepted: '+repr(invalid))
assert mod.verified_person_or_opaque(opaque) is None
assert mod.exact_int_scalar('exp_018','exp_no')==18
assert len(mod.UWB_FEATURES)==42 and mod.WIN==15 and mod.STRIDE==5
print('V7R1_ROOM_AND_ID_TRAPS_PASSED')

# Rehash all forked files, making v7r1 immutable under launcher guards.
with (new/'SOURCE_SHA256.txt').open('w') as fp:
    for path in sorted(files):
        if path.name=='SOURCE_SHA256.txt': continue
        fp.write(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n')
(new/'V7R1_CHANGELOG.txt').write_text(
    'V7R1 targeted post-logs correction: Kinect/PWR room_no MATLAB MCOS string '
    'cannot be decoded by scipy. Its value is NOT inferred. Require independent '
    'WiFi + UWB room/subject checks, MAT exp_no check, and timestamp/label agreement. '
    'Previous v7 untouched; 1.5s windows, subject split and models unchanged.\n')
print('V7R1_FORK_READY',new)
PYCODE
(cd "$NEW" && sha256sum -c SOURCE_SHA256.txt >/dev/null)
if [[ "${PERFAIL_BUILD_ONLY:-0}" == '1' ]]; then
  echo 'BUILD_ONLY: jobs not submitted'
  exit 0
fi
bash "$NEW/launch_operanet_v7.sh"
