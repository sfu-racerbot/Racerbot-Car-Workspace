"""Local-only read-only replay. Run at the frozen evaluator checkout with one BLAS thread."""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
WORK = Path.cwd()
DATA = Path(os.environ["DATA_WORKSPACE"])
MODEL = Path(os.environ["MODEL"])
sys.path.insert(0, str(WORK / "tools/neural_racer"))
import real_runs as rr
import real_run_dataset as rd
from base import predict

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(p):
    return json.loads((HERE / p).read_text())

PLAN = load("november-reservation.json")
assert digest(HERE/"november-reservation.json") == "3ee4dfc735458b5b6c124713f2d3e7f4188923294815d4dfff1a32b4b1cba03c"
assert digest(HERE/"steering.npz") == PLAN["head_sha256"]
assert digest(HERE/"base.py") == PLAN["inference_sha256"]
assert rr.model_store.weights_hash(MODEL/"weights.npz") == PLAN["base_model"]
for repo, field in [(WORK,"evaluator_commit"),(DATA,"data_commit")]:
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip() == PLAN[field]
    assert not subprocess.check_output(["git","diff","--name-only","HEAD"],cwd=repo,text=True)

def run(row):
    car, runid = row['car'],row['run']
    bag = DATA/f"data/{car}/racerbot_runs/{runid}/bag"
    result = dict(id=f"{car}/{runid}",cohort=row["cohort"],groups={})
    out = HERE/"scores"/f"{car}-{runid}.json"
    def write():
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(result["id"],result.get("counts"),result.get("quarantine"),flush=True)
    c,seen=Counter(),set()
    try:
        info=rd.bag_info(bag)
        topics={v["topic_metadata"]["name"]:v["message_count"] for v in info["topics_with_message_count"]}
        result["topic_counts"]={k:topics.get(k,0) for k in [*sorted(rd.TOPICS),"/imu/data","/imu/status","/drive_intent","/steering/measured"]}
        ticks=list(rd.aligned_ticks(rd.messages(bag,info),c,seen))
    except (ValueError, FileNotFoundError, KeyError) as e:
        result["quarantine"]=f"{type(e).__name__}: {e}";result["partial_counts_not_scored"]=dict(c);write();return
    result["counts"]=dict(c)
    result["missing_decoded_topics"]=sorted(rd.TOPICS.keys()-seen)
    assert c['scans']==sum(v for k,v in c.items() if k!='scans')
    if not ticks:
        write();return
    t,lag,speed,ranges,geom,action,reset=zip(*ticks)
    data=dict(time_ns=np.asarray(t),speed=np.float32(speed),ranges=np.stack(ranges),geometry=np.float32(geom),action=np.float32(action),reset=np.array(reset))
    assert np.all(np.diff(t)>0) and min(lag)>=0 and max(lag)<=rd.LABEL_NS and reset[0]
    weights, card=rr.model_store.load_dir(MODEL)
    gains=load("input-scorecard.json")["date_split"]["steering_gains"]
    with np.load(HERE/"steering.npz",allow_pickle=False) as f:head=dict(f)
    for name in ("original","corrected","learned"):
        policy=rr.PolicyNumpy(weights,card)
        if name=="learned":
            scans=iter(data["ranges"])
            original=policy.step
            def step(state):
                return replace(original(state),steering=predict(head,next(scans)))
            policy.step=step
        groups=defaultdict(rr.Metrics)
        for tick in rr.replay(data,policy,rr.wrapper_for(car,PLAN["cap_mps"]),calibration=gains if name=="corrected" else None):
            r=tick.result
            for key in ["all",rr.speed_bucket(tick.speed)]:
                groups[key].add((r.steering,r.speed),tick.target,r.intervened,r.stopped)
        if name=="learned":assert next(scans,None) is None
        assert groups["all"].n==len(ticks)==c["accepted"]
        result["groups"][name]={k:m.report() for k,m in groups.items()}
    write()

if __name__=="__main__":
    (HERE/"scores").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("JOBS",4))) as pool:
        list(pool.map(run,PLAN["runs"]))
