"""Paired solo evaluation with passive metrics (no ROS, no plant/controller changes)."""
from base import *
from neural_racer.safety_wrapper import SafetyWrapper, WrapperConfig, scan_angles
from gap_follow import gap_logic
from dataclasses import replace
import gzip,csv,sys
S=json.loads((D/'reservation.json').read_text())
F=json.loads((D/'freeze.json').read_text())
assert sha(D/'first-reservation.json')==F['reservation_sha256']
assert sha(D/'reservation.json')=='706582746b89f27b607c6ac2e193b23eda1cca58fa201c6330ac97786ac99cab'
assert F['candidate_sha256']==S['candidate_sha256'] and F['inference_code']==S['inference_sha256']
assert sha(D/'steering.npz')==F['candidate_sha256'] and sha(D/'base.py')==F['inference_code']
assert R.joinpath('tools/neural_racer/neural_racer_train/referee.py').is_file()
assert os.environ['RACERBOT_CAR']=='rb2'

def write_maps():
    from neural_racer_train.heldout import write_gym_map
    dm={}
    for seed,kind in S['procedural_maps']:
        r=tracks.rasterize(tracks.generate(seed,kind=kind))
        assert r.digest not in F['track_digests']
        assert r.digest not in {item['digest'] for item in json.loads((D/'first-map-digests.json').read_text()).values()}
        assert r.digest not in {item['digest'] for item in dm.values()}
        write_gym_map(r,D/'maps',f'new_{seed}')
        dm[str(seed)]=dict(kind=kind,digest=r.digest,lap_length_m=r.lap_length)
    (D/'map-digests.json').write_text(json.dumps(dm,indent=2))

def geom(scan,wrapper):
    scan=np.asarray(scan,dtype=float)
    angles=scan_angles(-np.radians(135),np.radians(270)/(len(scan)-1),len(scan))
    valid=observation.beam_validity(scan,.05,25.)
    if 1-valid.mean()>wrapper.cfg.max_invalid_fraction:return None
    return float(gap_logic.minimum_footprint_clearance(scan,valid,wrapper.boundary(angles)))

def preflight(track,seed,wrapper):
    rv=referee._rv()
    plant=rv.make_plant(referee.track_map_name(track),1,seed)
    try:
        pose=rv.initial_pose(plant.env.unwrapped.track.raceline,0)[None,:]; obs,_=plant.reset(pose)
        c=float(plant.chassis_map_clearance(0)); contact=bool(obs['agent_0']['collision'])
        m=None
        for t in range(10):
            m=geom(obs['agent_0']['scan'],wrapper)
            if m is not None:break
            obs,*_=plant.step([(0.,0.)]); contact|=bool(obs['agent_0']['collision'])
        valid=(not contact and c>.2 and m is not None and m>.2)
        return dict(valid=bool(valid),spawn_map_m=c,first_stationary_scan_m=m,first_scan_s=t*.025,spawn_contact=contact,pose=pose.tolist())
    finally:plant.close()

def run(name,seed):
    track=str(D/'maps'/name) if name.startswith('new_') else name
    startwrap=SafetyWrapper(WrapperConfig(max_speed=5.))
    start=preflight(track,seed,startwrap)
    mdl=Path(os.environ['MODEL']); score=json.loads(Path(os.environ['SCORECARD']).read_text())['date_split']
    assert model_store.weights_hash(mdl/'weights.npz')==score['model_hash']==S['model_hash']
    assert score['cap_mps']==5.
    w,card=model_store.load_dir(mdl)
    with np.load(D/'steering.npz') as f: head={k:f[k] for k in f}
    dest=D/'trials';dest.mkdir(exist_ok=True)
    for mode in ('original','corrected','learned','gap'):
        rec=[]; chosen={}
        def factory(plant):
            drv=referee.GapDriver() if mode=='gap' else referee.NeuralDriver(w,card,v_cap=5.,imu=False,
                steering_gains=score['steering_gains'] if mode=='corrected' else None)
            if mode!='gap':
                orig_step=drv.policy.step; orig_wrapper=drv.wrapper.apply
                def policy(o):
                    v=orig_step(o)
                    if mode=='learned':v=replace(v,steering=predict(head,chosen['scan']))
                    chosen.update(net_steer=v.steering,net_speed=v.speed);return v
                def wrap(*args,**kw):
                    r=orig_wrapper(*args,**kw)
                    chosen.update(binding=r.binding,wrapper_clear_m=r.clearance)
                    return r
                drv.policy.step=policy; drv.wrapper.apply=wrap
            command=drv.command
            def observed(obs,truth,t):
                chosen.clear();chosen['scan']=obs['scan']
                cmd=command(obs,truth,t); m=geom(obs['scan'],startwrap)
                if mode!='gap' and m is not None and chosen.get('binding')!='stop':
                    assert np.isclose(chosen['wrapper_clear_m'],m,atol=1e-8)
                rec.append(dict(time_s=t*.025,clear_m=m,x=truth[0],y=truth[1],true_v=truth[3],command_steer=cmd[0],command_speed=cmd[1],
                                net_steer=chosen.get('net_steer'),net_speed=chosen.get('net_speed'),binding=chosen.get('binding')))
                return cmd
            drv.command=observed;return drv
        if start['spawn_contact']:
            result=dict(outcome='void_start',margin_entry_s=None)
        else:
            race=referee.race(track,{mode:factory},seed,laps=1,timeout_s=90.)
            tail=rec[-200:]; xy=np.array([[s['x'],s['y']] for s in tail]); span=float(np.linalg.norm(np.ptp(xy,axis=0)))
            vmax=float(max(abs(s['true_v']) for s in tail))
            stalled=race.finish_time[0] is None and not race.contact[0] and len(tail)==200 and vmax<.05 and span<.05
            end=('contact' if race.contact[0] else 'finish' if race.finish_time[0] is not None else 'stall' if stalled else 'other_timeout')
            margin=[r for r in rec if r['clear_m'] is not None and r['clear_m']<=.2]
            result=dict(outcome=end,finish_s=race.finish_time[0],contact=race.contact[0],stall=stalled,progress_m=race.progress[0],
                        margin_entry_s=margin[0]['time_s'] if margin else None,min_clear_m=min(r['clear_m'] for r in rec if r['clear_m'] is not None),
                        margin_ticks=len(margin),ticks=len(rec),tail_max_abs_v=vmax,tail_span_m=span,lap_count=race.laps[0])
            with gzip.open(dest/f'{name}-{seed}-{mode}.csv.gz','wt') as f:
                out=csv.DictWriter(f,fieldnames=list(rec[0]));out.writeheader();out.writerows(rec)
        result.update(track=name,seed=seed,policy=mode,start=start)
        (dest/f'{name}-{seed}-{mode}.json').write_text(json.dumps(result)+'\n')
        print(name,seed,mode,'valid',start['valid'],result['outcome'],result.get('progress_m'),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='maps':write_maps()
    else:run(sys.argv[1],int(sys.argv[2]))
