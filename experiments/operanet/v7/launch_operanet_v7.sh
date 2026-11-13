#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/perfail_ops_tmp/perfail_v7"
ROOT="./data/perfail2027/operanet"
LOGS="$HOME/perfail_ops_tmp/perfail_logs"
PY="$HOME/perfail_ops_tmp/perfail_py311/bin/python"
[[ -x "$PY" ]] || { echo 'STOP: missing PerFail Python 3.11'; exit 1; }
[[ -d "$BASE" ]] || { echo 'STOP: missing extracted v7 bundle'; exit 1; }
(cd "$BASE" && sha256sum -c SOURCE_SHA256.txt)
# Never launch another pipeline over a previous attempt.
if [[ -e "$ROOT/aligned_v7_10" || -e "$ROOT/matrix_v7_10" ]]; then
    echo 'STOP: v7 output already exists; no overwrite'; exit 1
fi
mkdir -p "$LOGS" "$ROOT/aligned_v7_10"
sha256sum "$BASE/operanet_build_windows_v7.py" "$BASE/operanet_cohort_summary_v7.py" > "$ROOT/aligned_v7_10/PROTOCOL_SHA256.txt"
cat > "$ROOT/aligned_v7_10/PROTOCOL.txt" <<'PROTOCOL'
PHASE=EXPLORATORY_SUBJECT_DISJOINT_OPERANET_10_EXPERIMENT_SUBSET
EXPS=18,29,32,33,56,57,58,59,60,61
EXCLUDED_EXP030_EXP031=unavailable_Kinect_required_headers_in_v4_audit
WINDOW_SECONDS=1.5
STRIDE_SECONDS=0.5
ALL_FOUR_MODALITIES_REQUIRE_EXACT_UNIFORM_LABELS
SUBJECT_SPLITS=One,Two,Three/train;Five/dev;Six,Seven/test
MISSING_DEV_TEST_CLASSES_REPORTED_AS_ZERO_SUPPORT
MAT_KINECT_PWR_PERSON_MCOS=unreadable; require experiment room and WiFi UWB identity corroboration
NO_MATLAB=true
PROTOCOL
# Verify parser on synthetic inputs in the intended environment BEFORE using Slurm.
"$PY" - <<'PYTEST'
import ast, numpy as np
from pathlib import Path
p=Path.home()/'perfail_ops_tmp/perfail_v7/operanet_build_windows_v7.py'
t=ast.parse(p.read_text())
f=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='decode_person_id']
assert len(f)==1
ns={'np':np}
exec(compile(ast.fix_missing_locations(ast.Module(body=f,type_ignores=[])),str(p),'exec'),ns)
fun=ns['decode_person_id']
for value,expected in [(3.0,'Three'),(np.array([[3.0]]),'Three'),('Three','Three'),(b'Three','Three'),('3','Three'),(np.array([['Five']]),'Five'),(7,'Seven')]:
    assert fun(value)==expected,(value,fun(value))
for value in (0,9,3.5,np.nan,'unlabeled',np.array([1,2])):
    try: fun(value)
    except ValueError: continue
    raise AssertionError('MALFORMED_ID_NOT_REJECTED '+repr(value))
from scipy.io.matlab import MatlabOpaque
opaque_dtype=np.dtype([('s0','O'),('s1','O'),('s2','O'),('arr','O')])
opaque=MatlabOpaque(np.array([(b'', b'MCOS', b'string',np.array([[1],[2]],dtype=np.uint32))],dtype=opaque_dtype))
mod=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name in ('verified_person_or_opaque','exact_int_scalar')]
assert len(mod)==2
exec(compile(ast.fix_missing_locations(ast.Module(body=mod,type_ignores=[])),str(p),'exec'),ns)
assert ns['verified_person_or_opaque'](opaque) is None
assert ns['verified_person_or_opaque'](np.array([3.]))=='Three'
import re
ns['re']=re
assert ns['exact_int_scalar'](np.array([[18.]]),'exp_no')==18
assert ns['exact_int_scalar']('exp_018','exp_no')==18
assert ns['exact_int_scalar']('EXP 018','exp_no')==18
assert ns['exact_int_scalar']('Room 1','room_no')==1
assert ns['exact_int_scalar']('room_2','room_no')==2
assert ns['exact_int_scalar']('1.0','room_no')==1
for field, bad in (('exp_no','room_019'),('room_no','floor_1'),('exp_no','18.5'),('room_no','one')):
    try: ns['exact_int_scalar'](bad,field)
    except ValueError: continue
    raise AssertionError('BAD_TAGGED_METADATA_ACCEPTED '+repr((field,bad)))
for bad in (18.5, np.nan, np.array([18.,19.])):
    try: ns['exact_int_scalar'](bad,'exp_no')
    except ValueError: continue
    raise AssertionError('BAD_EXP_ID_ACCEPTED '+repr(bad))
ns2={}
import pandas as pd
f=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name in ('parse_uwb_complex_magnitude','uwb_features_to_float')]
assert len(f)==2, 'MISSING_UWB_HELPERS'
ns2.update({'np':np,'pd':pd,'UWB_FEATURES':tuple(f'cir{i}' for i in range(1,36))+('fp_pow_dbm','rx_pow_dbm','max_growth_cir','std_noise','fp_amp1','fp_amp2','fp_amp3')})
exec(compile(ast.fix_missing_locations(ast.Module(body=f,type_ignores=[])),str(p),'exec'),ns2)
for value,want in [('-0.24793-0.34711i',np.hypot(.24793,.34711)),('3+4i',5.),('3-4i',5.),('0+0i',0.)]:
    assert np.isclose(ns2['parse_uwb_complex_magnitude'](value),want)
for value in ('broken', '1+infi',None,'1+3k'):
    assert np.isnan(ns2['parse_uwb_complex_magnitude'](value))
cols=ns2['UWB_FEATURES']
test=pd.DataFrame({c:['3+4i','bad'] if c.startswith('cir') else ['2.0','3.0'] for c in cols})
floats=ns2['uwb_features_to_float'](test)
assert floats.shape==(2,42) and np.isfinite(floats[0]).all() and np.isnan(floats[1,:35]).all()
assert np.isfinite(floats[1,35:]).all()
print('V7_METADATA_UWB_AND_OPAQUE_ID_TRAPS_PASSED')
PYTEST
JOB=$(sbatch --parsable --job-name=perfail-opera-v7smoke   --cpus-per-task=4 --mem=64G --time=02:00:00   --output="$LOGS/opera_v7_smoke_%j.log" "$BASE/opera_v7_smoke.sbatch")
DISP=$(sbatch --parsable --dependency="afterany:$JOB"   --job-name=perfail-opera-v7dispatch --cpus-per-task=1 --mem=1G --time=00:10:00   --output="$LOGS/opera_v7_dispatch_%j.log" "$BASE/opera_v7_dispatch.sbatch")
echo "V7_SMOKE_JOB=$JOB V7_DISPATCH_JOB=$DISP"
echo "SMOKE_LOG=$LOGS/opera_v7_smoke_${JOB}.log"
echo "DISPATCH_LOG=$LOGS/opera_v7_dispatch_${DISP}.log"
echo 'Dispatch runs after smoke ends, successful or failed; no DependencyNeverSatisfied.'
