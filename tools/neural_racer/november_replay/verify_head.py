"""Independent cross-check of the adapter using a scan iterator and a different integration point."""
import numpy as np
from dataclasses import replace
from evaluate_november import HERE,MODEL,PLAN,rr,predict
data=dict(np.load(HERE/'qa-ticks/rb1-20261116-095652.npz',allow_pickle=False))
w,card=rr.model_store.load_dir(MODEL)
with np.load(HERE/'steering.npz',allow_pickle=False) as f:head=dict(f)
def trace(d):
    # Supply the frozen steering to the wrapper at the raw-scan argument instead
    # of monkeypatching the policy's step; commands retained by replay still come
    # from the real wrapper.
    wrapper=rr.wrapper_for('rb1',PLAN['cap_mps'])
    apply=wrapper.apply
    def wrapped(steering,speed,path,ranges,*args):
        return apply(predict(head,ranges),speed,path,ranges,*args)
    wrapper.apply=wrapped
    return np.asarray([[t.result.steering,t.result.speed] for t in rr.replay(d,rr.PolicyNumpy(w,card),wrapper)])
a=trace(data)
np.testing.assert_array_equal(a,trace(dict(data,action=-data['action'][::-1])))
import json
expected=json.loads((HERE/'scores/rb1-20261116-095652.json').read_text())['groups']['learned']['all']
np.testing.assert_allclose(abs(a-data['action']).mean(axis=0),[expected['steer_mae_rad'],expected['speed_mae_mps']],rtol=0,atol=1e-12)
print('Learned head: second integration point matches both errors; predictions unchanged with shuffled/sign-flipped labels (60 pairs).')
