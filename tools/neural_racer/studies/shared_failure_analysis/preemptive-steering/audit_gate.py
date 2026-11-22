from pathlib import Path
import csv,gzip,json,math,hashlib
D=Path(__file__).parent
sha=lambda x:hashlib.sha256((D/x).read_bytes()).hexdigest()
protocol=json.loads((D/'protocol.json').read_bytes())
assert sha('protocol.json')=='0c2426163bc11577a060ae1dd929be707e96181e9618c129998f61b7c35e659d'
assert sha('steering.npz')==protocol['frozen_head']
assert sha('base.py')==json.loads((D/'freeze.json').read_bytes())['inference_code']
assert sha('guard.py')=='3620557d441160dcd06fa5e6b073bf20b118275324f6b0c4e311c526f64b37c9'
assert sha('dev_run.py')=='37c1bb387d31560394be51973bd251957435a6cab1bf36a34f1573413616f848'
rows=[]
for seed in protocol['development']['seeds']:
    stem=f'car1_course_2026_10_26-{seed}'
    r=json.loads((D/'trials'/f'{stem}-arc-prior.json').read_bytes())
    a=json.loads((D/'archived'/f'{stem}-learned.json').read_bytes())
    assert r['start']==a['start'] and r['start']['valid']
    with gzip.open(D/'trials'/f'{stem}-arc-prior.csv.gz','rt') as f: ticks=list(csv.DictReader(f))
    assert len(ticks)==r['ticks']==3600
    assert all(math.isclose(float(t['time_s']),i*.025,abs_tol=1e-6) for i,t in enumerate(ticks))
    v=[t for t in ticks if t['clear_m']]
    breached=[t for t in v if float(t['clear_m'])<=.20]
    t0=float(breached[0]['time_s']); assert t0==r['margin_entry_s']
    assert len(breached)==r['margin_ticks']==sum(float(t['time_s'])>=t0 for t in v)
    assert all(t['binding']=='braking' and float(t['command_speed'])==0. for t in breached)
    assert r['outcome']=='stall' and r['finish_s'] is None and not r['contact']
    assert max(abs(float(t['true_v'])) for t in ticks[-200:])<.05
    assert all(len({t[k] for t in ticks[-200:]})==1 for k in ('x','y'))
    changes=[t for t in v if float(t['time_s'])<t0 and abs(float(t['prior_steer'])-float(t['net_steer']))>1e-6]
    assert len(changes)>0
    rows.append(dict(seed=seed,prior_progress_m=a['progress_m'],new_progress_m=r['progress_m'],prior_entry_s=a['margin_entry_s'],new_entry_s=t0,interventions_before_entry=len(changes),interventions_final_2s=sum(float(t['time_s'])>=t0-2 for t in changes),stop_ticks=len(breached),min_clear_m=r['min_clear_m'],requested_speed_at_entry=float(breached[0]['net_speed'])))
assert not (D/'maps').exists() and len(list((D/'trials').glob('*.json')))==3
with (D/'development-comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Gate failed: 0/3 finishes, 0 contacts, 3 stalls; holdout unopened.')
