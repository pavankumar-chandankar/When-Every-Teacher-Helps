import re, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy import stats
txt = open(sys.argv[1]).read()
M = '(?:imu|airpods|pose2d|pose3d|video|wifi)'
rows = re.findall(r'((?:' + M[3:-1] + r'|student)(?:_from_' + M + r')?)_seed([123]) "val_known_macro_f1": ([0-9.]+)', txt)
f1 = {(n, int(s)): float(v) for n, s, v in rows}
mods = ['imu','airpods','pose2d','pose3d','video','wifi']
for s in (1,2,3):
    f1[('wifi_from_imu', s)] = f1[('student', s)]
base = {m: np.mean([f1[(m,s)] for s in (1,2,3)]) for m in mods}
print('baselines', {m: round(v,4) for m,v in base.items()})
G = {}; seedpos = 0; seedn = 0; seed_d = {}
for st_ in mods:
    for te in mods:
        if st_ == te: continue
        k = f'{st_}_from_{te}'
        diffs = [f1[(k,s)] - f1[(st_,s)] for s in (1,2,3)]
        seed_d[(st_,te)] = diffs
        seedpos += sum(d > 0 for d in diffs); seedn += 3
        G[(st_,te)] = np.mean([f1[(k,s)] for s in (1,2,3)]) - base[st_]
print('pairs', len(G), 'all pair means positive:', all(v > 0 for v in G.values()), 'min', round(min(G.values()),4), 'max', round(max(G.values()),4))
print('seed-level positive', seedpos, '/', seedn)
neg = [(k, [round(x,4) for x in v]) for k, v in seed_d.items() if any(x <= 0 for x in v)]
print('pairs with a non-positive seed:', neg)
print('\nGain matrix (rows=student, cols=teacher):')
print('student'.ljust(9), ' '.join(t.rjust(8) for t in mods))
for s_ in mods:
    print(s_.ljust(9), ' '.join((f'{G[(s_,t)]:+.3f}' if s_!=t else '--').rjust(8) for t in mods))
tavg = {t: np.mean([G[(s_,t)] for s_ in mods if s_!=t]) for t in mods}
savg = {s_: np.mean([G[(s_,t)] for t in mods if s_!=t]) for s_ in mods}
print('\nteacher F1 ', {t: round(base[t],3) for t in mods})
print('teacher avg gain', {t: round(v,4) for t,v in tavg.items()})
print('student avg gain', {s_: round(v,4) for s_,v in savg.items()})
r = spearmanr([base[t] for t in mods], [tavg[t] for t in mods])
print('Spearman teacher F1 vs teacher-avg gain (n=6):', round(r.correlation,3), 'p', round(r.pvalue,3))
cells = [(s_,t) for s_ in mods for t in mods if s_!=t]
y = np.array([G[c] for c in cells]); n = len(y)
def design(use_s, use_t):
    cols = [np.ones(n)]
    if use_s: cols += [np.array([1.0 if c[0]==m else 0.0 for c in cells]) for m in mods[1:]]
    if use_t: cols += [np.array([1.0 if c[1]==m else 0.0 for c in cells]) for m in mods[1:]]
    return np.column_stack(cols)
def rss(X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; return float(e @ e)
sst = float(((y - y.mean())**2).sum())
rs, rt, rst = rss(design(True,False)), rss(design(False,True)), rss(design(True,True))
print('\nstudent-only R2', round(1-rs/sst,3), '| teacher-only R2', round(1-rt/sst,3), '| additive R2', round(1-rst/sst,3))
print('unique student', round((rt-rst)/sst,3), '| unique teacher', round((rs-rst)/sst,3), '| residual', round(rst/sst,3))
print('\nWiFi as teacher -> gain per student:', {s_: round(G[(s_,'wifi')],4) for s_ in mods if s_!='wifi'})
print('WiFi as student <- gain per teacher:', {t: round(G[('wifi',t)],4) for t in mods if t!='wifi'})
print('IMU as teacher -> gain per student:', {s_: round(G[(s_,'imu')],4) for s_ in mods if s_!='imu'})
for k in ['imu_from_wifi','pose2d_from_wifi']:
    print(k, 'mean F1', round(np.mean([f1[(k,s)] for s in (1,2,3)]),4))
wifi = {'s2025':0.16865895,'s2026':0.18794980,'orig':0.17701827}
stud = {'s2025':0.15565241,'s2026':0.18052014,'orig':0.18582646}
d = np.array([stud[k]-wifi[k] for k in wifi]); t = stats.ttest_rel([stud[k] for k in wifi],[wifi[k] for k in wifi])
print('\nTAL wifi mean', round(np.mean(list(wifi.values())),4), '| student mean', round(np.mean(list(stud.values())),4), '| diffs', np.round(d,4), '| mean diff', round(d.mean(),4), '| t', round(t.statistic,3), 'p', round(t.pvalue,3))
