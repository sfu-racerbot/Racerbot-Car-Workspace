import hashlib,itertools,json,os,subprocess
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np,yaml
H=Path(__file__).resolve().parent
root=Path(os.environ['DATA_WORKSPACE'])
def read(f):return json.loads(f.read_text())
plan=read(H/'november-reservation.json')
old=read(Path(os.environ['PRIOR_INVENTORY']))['bags']
intake_file=Path(os.environ['NOVEMBER_INTAKE'])
assert hashlib.sha256(intake_file.read_bytes()).hexdigest()=='5f8631dbefd61f274e8df695405d8e1f928c6d43ba392e9f1780175413acb53f'
intake=read(intake_file);assert intake['main']==plan['data_commit']
source={(r['car'],r['run']):r for r in intake['runs']}
assert set(source)=={(r['car'],r['run']) for r in plan['runs']}
old_chunks={v for bag in old.values() for k,v in bag.items() if k.endswith('.mcap')}
new_chunks=defaultdict(list); intervals=[]; grouped=defaultdict(lambda:defaultdict(list));rows=[]
def combine(ms):
    n=sum(m['ticks'] for m in ms); turns=sum(m['turns'] for m in ms)
    def weighted(k):return sum(m[k]*m['ticks'] for m in ms)/n
    return dict(ticks=n,steer_mae_rad=weighted('steer_mae_rad'),speed_mae_mps=weighted('speed_mae_mps'),turns=turns,
                turn_sign_accuracy=sum(round((m['turn_sign_accuracy'] or 0)*m['turns']) for m in ms)/turns if turns else None,
                interventions=sum(m['interventions'] for m in ms),stops=sum(m['stops'] for m in ms))
for r in plan['runs']:
    car,run=r['car'],r['run']; name=f"{car}/{run}"
    x=read(H/'scores'/f'{car}-{run}.json');assert x['id']==name
    path=f'data/{car}/racerbot_runs/{run}'; p=root/path; info=yaml.safe_load((p/'bag/metadata.yaml').read_text())['rosbag2_bagfile_information']
    s=info['starting_time']['nanoseconds_since_epoch'];intervals.append((name,s,s+info['duration']['nanoseconds']))
    for line in subprocess.check_output(['git','ls-tree','-r','HEAD','--',path+'/bag'],cwd=root,text=True).splitlines():
        attrs,f=line.split('\t')
        if f.endswith('.mcap'):new_chunks[attrs.split()[2]].append(f)
    assert name not in old
    a=yaml.safe_load((p/'auto_record.yaml').read_text()) if (p/'auto_record.yaml').exists() else {}
    x['recorded_activity']=a.get('activity');x['recorded_failures']=a.get('recorder_failures')
    x['publishers_from_auto_record']=r['recorded_publishers']
    ref=source[car,run]
    assert all(v==ref['topics'].get(k,{}).get('count',0) for k,v in x['topic_counts'].items())
    pub=','.join(n.lstrip('/') for n in ref['algorithm'])
    x['source']={k:ref[k] for k in ['algorithm_evidence','decoded_max_forward_odom_mps','decoded_lb_held_s','prior_reference_paths','research_check','url']}
    x['label_group']=pub;rows.append(x)
    for mode,gs in x['groups'].items():
        assert sum(m['ticks'] for key,m in gs.items() if key!='all')==gs['all']['ticks']==x['counts']['accepted']
        for axis in [r['cohort'],car+'/'+run[:8],car+'/'+run[:8]+'/'+pub]:
            for key,m in gs.items():grouped[axis+'|'+key][mode].append(m)
groups={group:{mode:combine(ms) for mode,ms in modes.items()} for group,modes in grouped.items()}
macro={mode:float(np.mean([x['groups'][mode]['all']['steer_mae_rad'] for x in rows if x['cohort']=='reserved-Nov16'])) for mode in ['original','corrected','learned']}
dup_old={k:v for k,v in new_chunks.items() if k in old_chunks}
dup_new={k:v for k,v in new_chunks.items() if len(v)>1}
overlap=[(a,b) for (a,sa,ea),(b,sb,eb) in itertools.combinations(intervals,2) if a.split('/')[0]==b.split('/')[0] and min(ea,eb)>max(sa,sb)]
qa=read(H/'qa-original.json')
for x in rows:
    if x['cohort']=='reserved-Nov16':
        m=x['groups']['original']['all'];z=qa['groups']['run:'+x['id']]['base']
        for k in m:assert np.isclose(m[k],z[k],rtol=0,atol=1e-12),k
window=read(H/'window.json');assert (window['paired'],round(window['max_rad'],4),round(window['rms_rad'],4),window['within_point02_mps'])==(148,.0164,.0026,139)
out=dict(reservation_sha256=hashlib.sha256((H/'november-reservation.json').read_bytes()).hexdigest(),intake_sha256=hashlib.sha256(intake_file.read_bytes()).hexdigest(),runs_scored=len(rows),
         scans=sum(x['counts']['scans'] for x in rows),paired=sum(x['counts']['accepted'] for x in rows),
         primary_macro_steer_mae_rad=macro,learned_minus_original_macro_rad=macro['learned']-macro['original'],
         overlap=dict(prior_chunk_count=len(old_chunks),new_chunk_count=sum(map(len,new_chunks.values())),identical_prior_chunks=dup_old,identical_november_chunks=dup_new,same_car_overlapping_bag_times=overlap,scope='Exact file hashes and bag log-time intervals only; not proof of course or tuning independence.'),
         source_reproduction=window,qa='Original policy scores on each reserved run matched an independent invocation of PR94 extract/evaluate to absolute tolerance 1e-12.',
         groups=groups,runs=rows)
(H/'november-scorecard.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
for k in ['scans','paired','overlap','primary_macro_steer_mae_rad']:print(k,out[k])
for k,v in groups.items():
    if k.startswith('reserved') or k.count('/')==1 and k.endswith('|all'):print(k,[(n,z['ticks'],round(z['steer_mae_rad'],4),round(z['speed_mae_mps'],3),z['interventions'],z['stops']) for n,z in v.items()])
