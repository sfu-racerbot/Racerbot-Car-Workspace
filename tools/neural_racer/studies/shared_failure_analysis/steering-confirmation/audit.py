"""Recompute outcome counts, margin entries and stalls from saved 40 Hz observations."""
from collections import Counter
from pathlib import Path
from statistics import median
import csv,json,gzip,math,hashlib
from itertools import product
D=Path(__file__).resolve().parent
reservation=json.loads((D/'reservation.json').read_text())
frozen=json.loads((D/'freeze.json').read_text())
maps=json.loads((D/'map-digests.json').read_text())
assert {s['digest'] for s in maps.values()}.isdisjoint(frozen['track_digests'])
for file,key in [('steering.npz','candidate_sha256'),('first-reservation.json','reservation_sha256'),('base.py','inference_code')]:
    assert hashlib.sha256((D/file).read_bytes()).hexdigest()==frozen[key]
policies=dict(learned='Learned steering',original='Frozen baseline',corrected='Frozen correction',gap='Gap-follow')
tracks=[f'new_{s}' for s,k in reservation['procedural_maps']]
seeds=reservation['sensor_seeds']
rows=[]; starts={}; n={}
for t,s,p in product(tracks,seeds,policies):
    f=D/'trials'/f'{t}-{s}-{p}.json'
    r=json.loads(f.read_text()); assert (t,s,p)==(r['track'],r['seed'],r['policy'])
    start=starts.setdefault((t,s),r['start']); assert r['start']==start
    assert start['valid']==(not start['spawn_contact'] and start['spawn_map_m']>.2 and start['first_stationary_scan_m']>.2)
    if start['spawn_contact']:
        assert r['outcome']=='void_start'
    else:
        with gzip.open(f.with_suffix('.csv.gz'),'rt') as stream: tick=list(csv.DictReader(stream))
        assert len(tick)==r['ticks']
        assert all(abs(float(x['time_s'])-k/40)<1e-9 for k,x in enumerate(tick))
        margins=[x for x in tick if x['clear_m'] and float(x['clear_m'])<=.2]
        assert len(margins)==r['margin_ticks']
        assert (None if not margins else float(margins[0]['time_s']))==r['margin_entry_s']
        if p!='gap':
            assert all(float(x['command_speed'])==0. for x in margins)
        if r['finish_s'] is not None:
            assert r['lap_count']>=1
            assert abs(len(tick)/40-r['finish_s'])<1e-5
        if r['finish_s'] is None and not r['contact']:
            assert len(tick)==3600
            tail=tick[-200:]; xy=[[float(x[k]) for x in tail] for k in ['x','y']]
            st=(max(abs(float(x['true_v'])) for x in tail)<.05 and math.hypot(*(max(a)-min(a) for a in xy))<.05)
            assert st==r['stall']
        assert r['outcome']==('contact' if r['contact'] else 'finish' if r['finish_s'] is not None else 'stall' if r['stall'] else 'other_timeout')
    rows.append({**{k:v for k,v in r.items() if k!='start'},'valid':start['valid'],'stationary_scan_clear_m':start['first_stationary_scan_m'],'spawn_map_m':start['spawn_map_m']})
assert len(list((D/'trials').glob('*.json')))==len(rows)
fields=sorted(set().union(*(r.keys() for r in rows)))
with (D/'paired-trials.csv').open('w') as f:
    out=csv.DictWriter(f,fields);out.writeheader();out.writerows(rows)
def counts(rs):
    return dict(n=len(rs),finishes=sum(r.get('finish_s') is not None for r in rs),contacts=sum(bool(r.get('contact')) for r in rs),stalls=sum(bool(r.get('stall')) for r in rs),other_timeouts=sum(r['outcome']=='other_timeout' for r in rs),voids=sum(r['outcome']=='void_start' for r in rs),margin_entries=sum(r['margin_entry_s'] is not None for r in rs))
def by_policy(cond):return {p:counts([r for r in rows if r['policy']==p and cond(r)]) for p in policies}
stats={'valid':by_policy(lambda r:r['valid']),'invalid':by_policy(lambda r:not r['valid']),'by_track':{t:by_policy(lambda r:r['track']==t) for t in tracks},'starts':[dict(track=t,seed=s,**f) for (t,s),f in starts.items()]}
(D/'counts.json').write_text(json.dumps(stats,indent=2))
print(json.dumps({k:v for k,v in stats.items() if k!='starts'},indent=2))
