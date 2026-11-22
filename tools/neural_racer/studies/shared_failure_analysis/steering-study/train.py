"""Fit steering to center-seeking targets on unrelated synthetic courses only."""
from base import D, A, tracks, np, encode, json, sha
from numba import njit
import torch

@njit(cache=True)
def ranges(grid,res,ox,oy,pose,angles):
    X=np.empty((len(pose),len(angles)),np.float32)
    for i,(x,y,th) in enumerate(pose):
        sx=x+.26*np.cos(th); sy=y+.26*np.sin(th)
        for j,ang in enumerate(angles):
            vx=np.cos(th+ang); vy=np.sin(th+ang); z=.025
            while z<10.:
                col=int((sx+z*vx-ox)/res); row=int((sy+z*vy-oy)/res)
                if row<0 or col<0 or row>=grid.shape[0] or col>=grid.shape[1] or grid[row,col]: break
                z+=.035
            X[i,j]=z
    return X

def data(seed,n=1400):
    r=tracks.rasterize(tracks.generate(seed,kind=['smooth_loop','filleted_polygon'][seed%2]))
    s=r.spec; center=s.centre+tracks.normals(s.centre)*((s.w_left-s.w_right)/2)[:,None]
    rng=np.random.default_rng(seed+884); i=rng.integers(len(center),size=n); direction=rng.choice([-1,1],size=n)
    unit=(center[(i+1)%len(center)]-center[i-1])*direction[:,None]
    yaw=np.arctan2(unit[:,1],unit[:,0]); side=np.c_[-np.sin(yaw),np.cos(yaw)]
    lateral=rng.uniform(-1,1,n)*np.clip((s.w_left[i]+s.w_right[i])/2-.22,.03,.75)
    xy=center[i]+side*lateral[:,None]
    heading=yaw+rng.uniform(-.60,.60,n)
    # Aim ~1 m ahead; labels alone know the training path.
    v=center[(i+direction*10)%len(center)]-xy
    y=np.clip(np.arctan2(2*.36*(-np.sin(heading)*v[:,0]+np.cos(heading)*v[:,1]),(v*v).sum(axis=1)),-.33,.33)
    X=ranges(r.occupancy,r.resolution,*r.origin,np.c_[xy,heading],A)
    X+=rng.normal(0,.014,X.shape).astype('f')
    X[rng.random(X.shape)<.004]=10.
    return encode(X),y.astype('f'),r.digest

def main():
    import torch.nn as nn
    torch.set_num_threads(2); torch.manual_seed(1367)
    ds=[data(s) for s in range(70100,70140)]
    X=torch.from_numpy(np.vstack([v[0] for v in ds])); y=torch.from_numpy(np.concatenate([v[1] for v in ds]))[:,None]
    X=torch.cat((X,X.flip([1])),dim=0); y=torch.cat((y,-y),dim=0)
    model=nn.Sequential(nn.Linear(109,96),nn.ReLU(),nn.Linear(96,64),nn.ReLU(),nn.Linear(64,1))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    for ep in range(30):
        if ep==22:
            for g in opt.param_groups: g['lr']=.0003
        for ids in torch.randperm(len(X)).split(512):
            pred=model(X[ids]); loss=nn.functional.smooth_l1_loss(pred,y[ids],beta=.04)
            opt.zero_grad(); loss.backward(); opt.step()
    w={}
    for k,layer in enumerate(model[::2]):
        w[f'w{k}']=layer.weight.detach().numpy();w[f'b{k}']=layer.bias.detach().numpy()
    path=D/'steering.npz'; np.savez(path,**w)
    from base import predict
    test=np.linspace(.3,4.,1081); a=torch.from_numpy(encode(test[::10]))
    assert np.isclose(predict(w,test),np.clip(.5*(model(a)-model(a.flip([0]))).item(),-.33,.33),atol=1e-6)
    assert np.isclose(predict(w,test),-predict(w,test[::-1]),atol=1e-6)
    meta=dict(train_seeds=list(range(70100,70140)),track_digests=[v[2] for v in ds],n_poses=len(y)//2,
              n_examples=len(y),candidate_sha256=sha(path),reservation_sha256=sha(D/'reservation.json'),training_code=sha(D/'train.py'),inference_code=sha(D/'base.py'))
    (D/'freeze.json').write_text(json.dumps(meta,indent=2));print(meta['candidate_sha256'],flush=True)
if __name__=='__main__':main()
