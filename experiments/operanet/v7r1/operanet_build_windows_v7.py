#!/usr/bin/env python3
"""PerFail OPERAnet: independent, fail-closed four-modality aligned-window extractor.

Input: acquired cohort12 experiment files; output: one NPZ per experiment.
No MATLAB, training, normalization, subject mixing, or access to SensorTAL.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque
from matio import load_from_mat

DATA = Path('./data/perfail2027/operanet')
SUBJECTS = {18:'Three',29:'Three',30:'One',31:'Six',32:'Two',33:'Five',
            56:'One',57:'Seven',58:'Two',59:'Six',60:'Three',61:'Five'}
SPLIT = {'One':'train','Two':'train','Three':'train','Five':'dev','Six':'test','Seven':'test'}
CLASSES = ('walk','sit','stand','liedown','standfromlie','bodyrotate')
LABELS = ('noactivity',) + CLASSES
RATE = 10
WIN = 15
STRIDE = 5
MAX_TIME_GAP_TICKS = 2
CSI_NAMES = tuple(f'tx{tx}rx{rx}_sub{sc}' for tx in range(1,4) for rx in range(1,4) for sc in range(1,31))
UWB_FEATURES = tuple(f'cir{i}' for i in range(1,36)) + ('fp_pow_dbm','rx_pow_dbm','max_growth_cir','std_noise','fp_amp1','fp_amp2','fp_amp3')


def as_text(obj):
    if isinstance(obj, str): return obj.strip()
    x = np.asarray(obj)
    for _ in range(6):
        if x.dtype.kind in 'US': return ''.join(x.ravel().astype(str)).strip()
        if x.dtype == object and x.size == 1: x = np.asarray(x.item())
        else: break
    raise ValueError('Unrecognized label/timestamp encoding')


def decode_person_id(value):
    """Canonicalize the dataset's numeric or textual participant identifiers.

    Numeric 1/2/3/5/6/7 map to named participants; unknown, missing,
    malformed, fractional, nonfinite or multi-valued IDs fail closed.
    """
    word_to_word={x:x for x in ('One','Two','Three','Five','Six','Seven')}
    digit_to_word={1:'One',2:'Two',3:'Three',5:'Five',6:'Six',7:'Seven'}
    value_raw=value
    for _ in range(8):
        if isinstance(value_raw,(str,bytes,np.str_,np.bytes_)):
            token=value_raw.decode('utf8') if isinstance(value_raw,(bytes,np.bytes_)) else str(value_raw)
            token=token.strip()
            if token in word_to_word: return token
            if token.isascii() and token.isdecimal():
                number=int(token)
                if number in digit_to_word: return digit_to_word[number]
            raise ValueError('Invalid participant ID token '+repr(token))
        a=np.asarray(value_raw)
        if a.dtype==object and a.size==1:
            value_raw=a.item()
            continue
        if a.dtype.kind in 'US':
            value_raw=''.join(a.reshape(-1).astype(str)).strip()
            continue
        if a.size==1 and a.dtype.kind in 'iuf':
            f=float(a.reshape(-1)[0])
            if np.isfinite(f) and f.is_integer() and int(f) in digit_to_word:
                return digit_to_word[int(f)]
            raise ValueError('Invalid numeric participant ID '+repr(f))
        raise ValueError('Unsupported participant ID encoding dtype='+str(a.dtype)+' shape='+str(a.shape))
    raise ValueError('Excessively nested participant ID')


def verified_person_or_opaque(value):
    """Decode supported IDs; flag authentic scipy MATLAB MCOS string objects as unreadable.

    This is NOT a recovered identity. Identity for any opaque MAT row must instead
    be corroborated by experiment ID, room, timestamp/label agreement and independently
    decoded WiFi/UWB participant metadata. Never convert an opaque object to text.
    """
    a=np.asarray(value)
    if a.size == 1 and a.dtype.names is not None:
        names=set(a.dtype.names)
        if {'s0','s1','s2','arr'}.issubset(names):
            row=a.reshape(-1)[0]
            if row['s1'] == b'MCOS' and row['s2'] == b'string':
                return None
    return decode_person_id(value)


def exact_int_scalar(value, field):
    """Read numeric or exact tagged dataset metadata (e.g., exp_018, room_1).

    The prefix must match the field, and a mismatched experiment or room still
    fails closed in get_stream. Do not confuse exp_018 with numeric 0 or 18.5.
    """
    a=np.asarray(value)
    for _ in range(8):
        if a.dtype.kind == 'O' and a.size == 1:
            a=np.asarray(a.item())
        else:
            break
    if a.size != 1 and a.dtype.kind not in 'US':
        raise ValueError(f'Invalid {field} scalar size {a.size}')
    if a.dtype.kind in 'iuf':
        if a.size!=1: raise ValueError(f'Non-scalar numeric {field}')
        n=float(a.reshape(-1)[0])
        if not np.isfinite(n) or not n.is_integer():
            raise ValueError(f'Invalid {field} value {n}')
        return int(n)
    if a.dtype.kind in 'US':
        token=''.join(a.reshape(-1).astype(str)).strip()
    elif a.dtype.kind == 'S':
        token=b''.join(a.reshape(-1)).decode('utf8').strip()
    else:
        raise ValueError(f'Unsupported {field} encoding dtype={a.dtype} shape={a.shape}')
    prefix={'exp_no':r'(?:exp(?:eriment)?[_\s-]*)?',
            'room_no':r'(?:room[_\s-]*)?'}.get(field,r'')
    matched=re.fullmatch(prefix+r'([0-9]+)(?:\.0+)?',token,flags=re.IGNORECASE)
    if not matched:
        raise ValueError(f'Invalid {field} token {token!r}')
    return int(matched.group(1))


def verified_room_or_opaque(value):
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


def ms_of_day(strings):
    s = pd.Series(strings, dtype='string')
    t = pd.to_timedelta(s, errors='coerce').dt.total_seconds().to_numpy(dtype=np.float64)
    if not np.isfinite(t).all(): raise ValueError('Timestamp parse failed')
    if np.any(np.diff(t) < -0.2): raise ValueError('Nonmonotonic timestamps')
    return t


def bins_labels(times, label_strings):
    idx = np.rint(times * RATE).astype(np.int64)
    labels = np.asarray([str(x).strip().lower() for x in label_strings])
    bad = sorted(set(labels) - set(LABELS))
    if bad: raise ValueError('Unrecognized activity labels: '+repr(bad[:20]))
    return idx, labels


def reduce_to_bins(idx, labels, values):
    first = int(idx.min()); length = int(idx.max() - first + 1)
    if length > 1_000_000: raise ValueError('Unreasonable recording length')
    local = idx - first
    counts = np.bincount(local, minlength=length)
    out = np.full((length, values.shape[1]), np.nan, dtype=np.float32)
    for j in range(values.shape[1]):
        v = values[:,j]
        ok = np.isfinite(v)
        n = np.bincount(local[ok], minlength=length)
        s = np.bincount(local[ok], weights=v[ok].astype(np.float64), minlength=length)
        out[n>0,j] = (s[n>0]/n[n>0]).astype(np.float32)
    label_votes = np.stack([np.bincount(local[labels==k], minlength=length) for k in LABELS], axis=1)
    winning = np.argmax(label_votes, axis=1)
    unanimity = label_votes[np.arange(length),winning] == counts
    final = np.where((counts>0)&unanimity, winning, -1).astype(np.int16)
    return np.arange(first, first+length, dtype=np.int64), final, out


def get_kinect_pwr(exp, base):
    """Decode two independently sampled MAT tables; skip only audited empty terminal rows.

    Both streams remain separate until timestamp-bin alignment. No padding, label
    rewriting, or substitution of synthetic marker values is permitted.
    """
    k = loadmat(base/f'kinect_exp{exp:03d}.mat', variable_names=['Kinect'])['Kinect']
    p = loadmat(base/f'pwr_exp{exp:03d}.mat', variable_names=['PWR'])['PWR']

    def get_stream(table, kind):
        if table.ndim != 2 or table.shape[0] < 2:
            raise ValueError(f'{kind} invalid table dimensions {table.shape}')
        hdr = [as_text(v) for v in table[0]]
        required = (('timestamp','activity','person_id','room_no','Kinect1_Markers','Kinect2_Markers')
                    if kind == 'kinect' else
                    ('timestamp','activity','person_id','room_no','PWR_ch1','PWR_ch2','PWR_ch3'))
        if len(set(hdr)) != len(hdr) or any(col not in hdr for col in required) or 'exp_no' not in hdr:
            raise ValueError(f'{kind} missing/duplicate required headers {hdr}')
        col = {name:hdr.index(name) for name in required}
        print('MAT_HEADERS',exp,kind,hdr,flush=True)
        sample_person=table[1,col['person_id']]
        print('MAT_PERSON_SAMPLE',exp,kind,'TYPE',type(sample_person).__name__, 'VALUE',repr(sample_person)[:120],flush=True)
        feature_names = required[4:]
        expected_room = str(1 if exp < 55 else 2)
        width = 114 if kind == 'kinect' else 600
        t, labs, arr = [], [], []
        skipped_terminal = 0
        opaque_person_rows = 0
        opaque_room_rows = 0
        for i in range(1, len(table)):
            vals = [np.asarray(table[i,col[name]]) for name in feature_names]
            if kind == 'kinect':
                good = all(v.shape == (19,3) for v in vals)
            else:
                good = all(v.size == 200 for v in vals)
            if not good:
                shapes = [v.shape for v in vals]
                # Audit found occasional one empty, terminal Kinect record.
                if i == len(table)-1 and all(v.size == 0 for v in vals):
                    skipped_terminal += 1
                    continue
                raise ValueError(f'{kind} exp{exp:03d} row{i}: invalid sensor shapes {shapes}')
            ts=as_text(table[i,col['timestamp']])
            label=as_text(table[i,col['activity']])
            # MATLAB string objects appear as MatlabOpaque under scipy.io.loadmat.
            # Their contents must never be guessed from their opaque reference bytes.
            person=verified_person_or_opaque(table[i,col['person_id']])
            mat_exp=exact_int_scalar(table[i,hdr.index('exp_no')], 'exp_no')
            room=verified_room_or_opaque(table[i,col['room_no']])
            if mat_exp != exp or (room is not None and room != int(expected_room)):
                raise ValueError(f'{kind} experiment/room mismatch exp{exp:03d} row{i}: {mat_exp}/{room}')
            if person is None:
                opaque_person_rows += 1
            if room is None:
                opaque_room_rows += 1
            elif person != SUBJECTS[exp]:
                raise ValueError(f'{kind} participant mismatch exp{exp:03d} row{i}: {person}')
            data=np.concatenate([np.asarray(v,dtype=np.float32).reshape(-1) for v in vals])
            if len(data)!=width:
                raise ValueError(f'{kind} exp{exp:03d} row{i}: wrong width {len(data)}')
            t.append(ts); labs.append(label); arr.append(data)
        if not arr:
            raise ValueError(f'{kind} has no valid samples')
        secs = ms_of_day(t)
        idx, lab = bins_labels(secs,labs)
        x = np.stack(arr,axis=0)
        print('MAT_STREAM',exp,kind,'RAW_ROWS',len(table)-1,'VALID_ROWS',len(arr),
              'EMPTY_TERMINAL_DROPPED',skipped_terminal,'OPAQUE_PERSON_UNVERIFIED_ROWS',opaque_person_rows,'OPAQUE_ROOM_UNVERIFIED_ROWS',opaque_room_rows,'TIME_RANGE',secs[0],secs[-1],flush=True)
        return reduce_to_bins(idx,lab,x)

    kin = get_stream(k, 'kinect')
    pwr = get_stream(p, 'pwr')
    common=np.intersect1d(kin[0],pwr[0],assume_unique=True)
    if len(common)<100:
        raise ValueError(f'Insufficient Kinect/PWR temporal overlap exp{exp:03d}: {len(common)}')
    ik=np.searchsorted(kin[0],common); ip=np.searchsorted(pwr[0],common)
    ok=(kin[1][ik]>=0)&(pwr[1][ip]>=0)
    checked=int(ok.sum()); disagreements=int(np.sum(kin[1][ik][ok]!=pwr[1][ip][ok]))
    print('MAT_TIME_LABEL_AGREEMENT',exp,'OVERLAP_BINS',len(common),
          'VALID_PAIRS',checked,'DISAGREEMENTS',disagreements,flush=True)
    if checked<100 or disagreements>max(2,int(np.ceil(.005*checked))):
        raise ValueError(f'Kinect/PWR label disagreement exp{exp:03d}: {disagreements}/{checked}')
    return kin,pwr


def parse_uwb_complex_magnitude(value):
    """Parse raw UWB CIR complex strings such as '-0.24793-0.34711i'."""
    if pd.isna(value): return np.nan
    token=str(value).strip().replace(' ','').replace('I','i').replace('i','j')
    if not token: return np.nan
    try:
        z=complex(token)
    except (TypeError,ValueError):
        return np.nan
    if not (np.isfinite(z.real) and np.isfinite(z.imag)):
        return np.nan
    return float(abs(z))


def uwb_features_to_float(part):
    """Convert 35 complex CIR coefficients to magnitude and seven real scalars."""
    result=np.full((len(part),len(UWB_FEATURES)),np.nan,dtype=np.float32)
    for i,name in enumerate(UWB_FEATURES):
        if name.startswith('cir'):
            result[:,i]=np.fromiter(
                (parse_uwb_complex_magnitude(val) for val in part[name].to_numpy()),
                dtype=np.float32,count=len(part))
        else:
            result[:,i]=pd.to_numeric(part[name],errors='coerce').to_numpy(dtype=np.float32)
    return result


def get_uwb(exp,base):
    path=base/f'uwb1_exp{exp:03d}.csv'
    header=set(pd.read_csv(path,nrows=0).columns)
    need={'timestamp','activity','person_id','room_no'}|set(UWB_FEATURES)
    if not need.issubset(header): raise ValueError('UWB missing columns: '+str(sorted(need-header)))
    parts=[]
    for part in pd.read_csv(path,usecols=sorted(need),chunksize=20000,low_memory=False):
        if set(part['person_id'].astype(str).str.strip()) != {SUBJECTS[exp]}: raise ValueError('UWB person mismatch')
        if set(part['room_no'].astype(str).str.strip()) != {str(1 if exp<55 else 2)}: raise ValueError('UWB room mismatch')
        t=ms_of_day(part['timestamp'].astype(str).tolist())
        ii,lab=bins_labels(t,part['activity'].astype(str).tolist())
        arr=uwb_features_to_float(part)
        parts.append((ii,lab,arr))
    if not parts: raise ValueError('UWB file empty')
    ids=np.concatenate([x[0] for x in parts]); lab=np.concatenate([x[1] for x in parts]); data=np.concatenate([x[2] for x in parts])
    if (np.diff(ids)<-2).any(): raise ValueError('UWB timestamps not monotonic')
    return reduce_to_bins(ids,lab,data)


def get_wifi(exp,base):
    path=base/f'wificsi2_exp{exp:03d}.mat'
    print('WIFI_DECODE_START',path,flush=True)
    df=load_from_mat(str(path),variable_names=['wifi_csi_2'],raw_data=False)['wifi_csi_2']
    expected=('timestamp','activity','exp_no','person_id','room_no')
    if any(k not in df.columns for k in expected): raise ValueError('WiFi metadata missing')
    if list(df.columns[12:])!=list(CSI_NAMES): raise ValueError('WiFi CSI column names/order not verified')
    if set(df['person_id'].astype(str).str.strip())!={SUBJECTS[exp]}: raise ValueError('WiFi person mismatch')
    if set(df['room_no'].astype(str).str.strip())!={str(1 if exp<55 else 2)}: raise ValueError('WiFi room mismatch')
    idx,lab=bins_labels(ms_of_day(df['timestamp']),df['activity'].to_numpy())
    counts=np.bincount(idx-idx.min(),minlength=int(idx.max()-idx.min()+1)); length=len(counts); start=int(idx.min())
    votes=np.stack([np.bincount((idx-start)[lab==v],minlength=length) for v in LABELS],axis=1)
    best=np.argmax(votes,axis=1); final=np.where((counts>0)&(votes[np.arange(length),best]==counts),best,-1).astype(np.int16)
    # Verify the raw HDF5 column mapping against the trusted decoder before any extraction.
    with h5py.File(path,'r') as h:
        refs=h['#refs#/o'][()].reshape(-1)
        if len(refs)!=282: raise ValueError('Expected 282 WiFi table columns')
        ch=h[refs[12]]
        first=ch[0,:min(16,len(df))]
        raw=first['real']+1j*first['imag']
        decoded=np.asarray(df[CSI_NAMES[0]].iloc[:len(raw)],dtype=np.complex128)
        if not np.allclose(raw,decoded,rtol=1e-7,atol=1e-9,equal_nan=True):
            raise ValueError('HDF5 CSI mapping does not match mat-io table')
        del df
        out=np.full((length,270),np.nan,dtype=np.float32)
        local=idx-start
        for j in range(270):
            x=h[refs[12+j]][0,:]
            if len(x)!=len(idx) or x.dtype.names!=('real','imag'): raise ValueError('CSI shape/dtype mismatch')
            amplitude=np.hypot(x['real'],x['imag'])
            valid=np.isfinite(amplitude)
            n=np.bincount(local[valid],minlength=length)
            s=np.bincount(local[valid],weights=amplitude[valid],minlength=length)
            out[n>0,j]=(s[n>0]/n[n>0]).astype(np.float32)
            if j%45==0: print('WIFI_CHANNEL',j,flush=True)
    return np.arange(start,start+length,dtype=np.int64),final,out


def align_one(src,grid):
    ticks,labels,x=src
    inds=np.searchsorted(ticks,grid)
    right=np.clip(inds,0,len(ticks)-1); left=np.clip(inds-1,0,len(ticks)-1)
    choose=np.where(abs(ticks[right]-grid)<abs(ticks[left]-grid),right,left)
    valid=abs(ticks[choose]-grid)<=MAX_TIME_GAP_TICKS
    labels_out=labels[choose].copy(); labels_out[~valid]=-1
    xout=x[choose].copy(); xout[~valid]=np.nan
    return labels_out,xout


def process(exp,outdir):
    if exp in (30,31):
        raise ValueError('EXCLUDED_FROM_V5: exp030/031 lack required four-modality Kinect table schema')
    basedir=DATA/f'exp{exp:03d}'
    for modality in ('kinect','pwr','uwb1','wificsi2'):
        suffix='csv' if modality=='uwb1' else 'mat'
        if not (basedir/f'{modality}_exp{exp:03d}.{suffix}').is_file(): raise FileNotFoundError(modality)
    print('MAT_METADATA_AND_SHAPE_START',exp,flush=True)
    kin,pwr=get_kinect_pwr(exp,basedir)
    print('MAT_METADATA_AND_SHAPE_PASS',exp,flush=True)
    uwb=get_uwb(exp,basedir)
    print('UWB_DECODE_PASS',exp,flush=True)
    wifi=get_wifi(exp,basedir)
    print('WIFI_DECODE_PASS',exp,flush=True)
    print('IDENTITY_PROVENANCE',exp,'UWB_AND_WIFI_VERIFIED',SUBJECTS[exp],
          'KINECT_PWR_MCOS_PERSON_AND_ROOM_VALUES_UNREADABLE_WHEN_OPAQUE',flush=True)
    streams={'wifi':wifi,'kinect':kin,'pwr':pwr,'uwb':uwb}
    lo=max(int(s[0][0]) for s in streams.values()); hi=min(int(s[0][-1]) for s in streams.values())
    if hi-lo < WIN: raise ValueError('Not enough simultaneous recording')
    grid=np.arange(lo,hi+1,dtype=np.int64)
    aligned={k:align_one(v,grid) for k,v in streams.items()}
    accepted=[]; label_counts={k:0 for k in CLASSES}; dropped={'label_mismatch_or_background':0,'missing_or_invalid':0}
    for start in range(0,len(grid)-WIN+1,STRIDE):
        chosen=[]
        for key,(labs,x) in aligned.items():
            section=labs[start:start+WIN]
            if section[0]<=0 or not np.all(section==section[0]): break
            chosen.append(int(section[0]))
        if len(chosen)!=4 or len(set(chosen))!=1:
            dropped['label_mismatch_or_background']+=1; continue
        example={k: aligned[k][1][start:start+WIN] for k in streams}
        if any((np.isfinite(x).mean(axis=1)<0.9).sum()>2 for x in example.values()):
            dropped['missing_or_invalid']+=1; continue
        accepted.append((chosen[0]-1,start,example))
        label_counts[CLASSES[chosen[0]-1]]+=1
    print('EXP',exp,'SUBJECT',SUBJECTS[exp],'ROOM',1 if exp<55 else 2,'CANDIDATE_WINDOWS',(len(grid)-WIN)//STRIDE+1,
          'ACCEPTED',len(accepted),'LABEL_COUNTS',label_counts,'DROPPED',dropped,flush=True)
    if len(accepted)<5: raise ValueError('Insufficient verified synchronized windows')
    result={}
    for key in streams:
        stacked=np.stack([item[2][key] for item in accepted]).astype(np.float32)
        result['X_'+key]=stacked
        result['finite_'+key]=np.isfinite(stacked).mean().astype(np.float32)
    result['y']=np.asarray([q[0] for q in accepted],dtype=np.int16)
    result['start_time_s']=np.asarray([grid[q[1]]/RATE for q in accepted],dtype=np.float64)
    result['subject']=np.asarray(SUBJECTS[exp]); result['room']=np.int8(1 if exp<55 else 2)
    result['experiment']=np.int16(exp); result['split']=np.asarray(SPLIT[SUBJECTS[exp]])
    outdir.mkdir(parents=True,exist_ok=True)
    final=outdir/f'exp{exp:03d}_aligned.npz'
    if final.exists(): raise FileExistsError('Will not overwrite existing result: '+str(final))
    temp=final.with_suffix('.npz.part')
    if temp.exists(): raise FileExistsError('Partial output already exists: '+str(temp))
    try:
        with open(temp,'xb') as f:
            np.savez_compressed(f,**result)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp,final)
    except Exception:
        # Retain partial for diagnostics; never overwrite without user review.
        raise
    print('EXTRACTION_COMPLETE',final,'WINDOWS',len(accepted),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--exp',type=int,required=True,choices=sorted(SUBJECTS))
    parser.add_argument('--output',type=Path,default=DATA/'aligned_v7r1_10')
    a=parser.parse_args(); process(a.exp,a.output)
