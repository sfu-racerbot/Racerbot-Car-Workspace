import numpy as np
from nlos_proto import *
W=1.5; sc=Scene(W=W); angles=np.linspace(-np.radians(135),np.radians(135),1081)[300:781]
def prior(pose, err=(0,0,0)):
    x,y,yaw=pose[0]+err[0],pose[1]+err[1],pose[2]+err[2]
    R=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    return R@(sc.P0-[x,y]), R@(sc.P1-[x,y])
fp_k,maxp=0.012,6.0
def learn(xmax,ys,nrep,dil,rng):
    bg=Background()
    for rep in range(nrep):
        for xe in np.arange(-7,xmax,0.05):
            for ye in ys:
                pose=(xe,ye+rng.normal(0,0.03),rng.normal(0,0.05)); r,_=trace_scan(sc,pose,angles,rng=rng,fp_k=fp_k,max_ghost_path=maxp)
                res=unfold_scan(r,angles,*prior(pose,(rng.normal(0,0.15),rng.normal(0,0.15),rng.normal(0,np.radians(3)))),rng=rng)
                if res is not None:
                    P=to_panel_frame(res,res['p'][res['ghost']]); bg.add(P[P[:,1]>0])
    bg.finalize(min_hits=2,dilate=dil); return bg
def test(bg,xmax,seed=999):
    rng2=np.random.default_rng(seed); nscan=ncl=nconf=0; where=[]
    for rep in range(4):
        ye=rng2.uniform(0.4,1.1); hist=[]
        for xe in np.arange(-7,xmax,0.1):
            pose=(xe,ye+rng2.normal(0,0.02),rng2.normal(0,0.05))
            r,_=trace_scan(sc,pose,angles,rng=rng2,fp_k=fp_k,max_ghost_path=maxp)
            res=unfold_scan(r,angles,*prior(pose,(rng2.normal(0,0.15),rng2.normal(0,0.15),rng2.normal(0,np.radians(3)))),rng=rng2)
            nscan+=1; det=False
            if res is not None:
                P=to_panel_frame(res,res['p'][res['ghost']]); P=P[P[:,1]>0]; fg=~bg.is_static(P)
                cl=[c for c in cluster(P[fg]) if np.ptp(P[fg][c],axis=0).max()<0.8]
                det=len(cl)>0
                if det: where.append(round(xe,1))
            ncl+=det; hist.append(det); nconf+= sum(hist[-6:])>=4
    return nscan,ncl,nconf,where
for (label,xmax_l,ys,nrep,dil,xmax_t) in [
    ('base: learn x<-0.2, 3 lanes, dil 2; test x<0.9',-0.2,(0.525,0.75,0.975),1,2,0.9),
    ('test only x<-0.2',-0.2,(0.525,0.75,0.975),1,2,-0.2),
    ('learn x<0.9, 3 lanes',0.9,(0.525,0.75,0.975),1,2,0.9),
    ('learn x<0.9, 7 lanes',0.9,(0.4,0.52,0.64,0.75,0.86,0.98,1.1),1,2,0.9),
    ('learn x<0.9, 7 lanes, dil 4',0.9,(0.4,0.52,0.64,0.75,0.86,0.98,1.1),1,4,0.9),
    ('learn x<0.9, 7 lanes x2 reps, dil 4',0.9,(0.4,0.52,0.64,0.75,0.86,0.98,1.1),2,4,0.9)]:
    bg=learn(xmax_l,ys,nrep,dil,np.random.default_rng(5)); n,c,f,w=test(bg,xmax_t)
    print(f'{label}: scans {n}, with cluster {c}, 4-of-6 {f}; ego x of clusters: {sorted(set(w))[:12]}{"..." if len(set(w))>12 else ""}')
