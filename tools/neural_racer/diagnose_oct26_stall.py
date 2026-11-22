"""Read the archived October 26 trials only. Run with NumPy; no simulator or weights."""
import csv, gzip, hashlib, io, json, math, sys, zipfile
from collections import Counter
from pathlib import Path

import numpy as np

archive, line_csv, output = map(Path, sys.argv[1:])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '50f16cc707c0f0e7a3fc07096793432b30c4bf2f66df8ade79343c7d60381082'
xy = np.loadtxt(line_csv, delimiter=',', skiprows=1)
v = np.roll(xy, -1, axis=0) - xy
length = np.linalg.norm(v, axis=1)
unit = v / length[:, None]
station = np.r_[0, np.cumsum(length[:-1])]

def project(p):
    delta = p[:, None, :] - xy
    along = np.clip(np.sum(delta * unit, axis=-1), 0, length)
    residual = delta - along[..., None] * unit
    j = np.argmin(np.sum(residual**2, axis=-1), axis=1)
    i = np.arange(len(p))
    det = residual[i,j,1]*unit[j,0] - residual[i,j,0]*unit[j,1]
    lateral = np.sign(det) * np.linalg.norm(residual[i,j],axis=1)
    return station[j] + along[i,j], lateral, j

def rounded(value):
    return float(round(float(value), 6)) if np.isfinite(value) else None

def select(row):
    return {k:rounded(arr[row]) for k, arr in cols.items()}

def points(p):
    return p.round(4).tolist()

results = []
with zipfile.ZipFile(archive) as z:
    assert hashlib.sha256(z.read('reservation.json')).hexdigest() == json.loads(z.read('freeze.json'))['reservation_sha256']
    for name in sorted(z.namelist()):
        if not (name.startswith('trials/car1_course_2026_10_26-') and name.endswith('.csv.gz')):
            continue
        trial = json.loads(z.read(name[:-7]+'.json'))
        rows = list(csv.DictReader(io.StringIO(gzip.decompress(z.read(name)).decode())))
        cols = {k:np.array([float(r[k]) if r[k] else np.nan for r in rows]) for k in rows[0] if k != 'binding'}
        t, clear, sp, ns, steer, vel = (cols[k] for k in ('time_s','clear_m','command_speed','net_speed','command_steer','true_v'))
        p = np.column_stack((cols['x'],cols['y']))
        s, offset, j = project(p)
        # Do not mistake a late near-zero projection at the finish for the start.
        s = np.unwrap(s * 2*np.pi / sum(length)) * sum(length) / (2*np.pi)
        s -= round(s[0] / sum(length)) * sum(length)
        cols.update(station=s, left_offset=offset)
        h = clear <= .20
        q = int(np.where(h)[0][0])
        assert len(rows) == trial['ticks'] and np.allclose(np.diff(t),.025)
        assert trial['start']['valid'] and not trial['contact']
        assert math.isclose(t[q],trial['margin_entry_s']) and h.sum()==trial['margin_ticks']
        assert math.isclose(np.nanmin(clear),trial['min_clear_m'])
        assert all(r['clear_m'] for r in rows[2:])
        assert np.all(sp[h] == 0) if trial['policy'] != 'gap' else True
        d={'seed':trial['seed'],'driver':trial['policy'],'summary':trial,'entry':select(q),
           'end':select(-1),'snapshots':{},'all_inside_after':bool(h[q:].all()),
           'binding_after':dict(Counter(r['binding'] for r in rows[q:])),
           'coast_m':rounded(np.linalg.norm(p[-1]-p[q])),
           'tail_5s_bounding_diagonal_m':rounded(np.linalg.norm(np.ptp(p[-200:],axis=0))),
           'tail_5s_max_speed':rounded(abs(vel[-200:]).max())}
        if trial['policy']=='learned':
            assert np.all(ns[q:]>0) and np.all(sp[q:]==0) and h[q:].all()
            assert all(row['binding']=='braking' for row in rows[q:])
            assert d['tail_5s_bounding_diagonal_m']==0 and d['tail_5s_max_speed']==0
            ds=np.diff(s)
            dist=np.linalg.norm(np.diff(p,axis=0),axis=1)
            d['path_check']={'xy_distance_m':rounded(dist.sum()),'xy_distance_to_entry_m':rounded(dist[:q].sum()),
                             'xy_distance_after_entry_m':rounded(dist[q:].sum()),
                             'max_station_m':rounded(s.max()),'min_station_m':rounded(s.min()),
                             'backtrack_total_m':rounded(-np.minimum(ds,0).sum()),
                             'peak_to_later_trough_m':rounded(max(np.maximum.accumulate(s)-s)),
                             'binding_before_entry':dict(Counter(r['binding'] for r in rows[2:q])),
                             'before_entry_zero_cmd_ticks':int(np.count_nonzero(sp[2:q]==0)),
                             'before_entry_min_cmd_mps':rounded(min(sp[2:q]))}
            d.update(entry_minus_1s=select(q-40), terminal_net_range=[rounded(x) for x in (min(ns[q:]),max(ns[q:]))],
                     max_final_slew_delta=rounded(max(abs(steer[q-80:q+1]-cols['net_steer'][q-80:q+1]))),
                     slows_below_05_s=rounded(t[np.where((np.arange(len(t))>=q)&(abs(vel)<.05))[0][0]]),
                     trace_near_entry=[dict(dt=rounded(t[k]-t[q]),**select(k)) for k in range(q-80,q+81)])
        for pos in (26,27,28,28.5,29,29.25,29.4,29.5,30):
            if max(s)<pos: continue
            k = np.argmin(abs(s-pos))
            d['snapshots'][str(pos)] = select(k)
        corner=(s>=25) & (s<=31.2)
        if trial['policy']=='gap':
            d['corner_min_clear']=rounded(min(clear[corner]))
            d['first_to_last_reserve_station']=[rounded(min(s[h])),rounded(max(s[h]))]
        mask=corner & (t <= t[q]+1.5) if trial['policy']=='learned' else corner
        d['corner_xy']=points(p[mask][::2])
        results.append(d)
assert len(results)==12
output.write_text(json.dumps(dict(results=results,centerline=points(xy),line_length=sum(length)),allow_nan=False))
