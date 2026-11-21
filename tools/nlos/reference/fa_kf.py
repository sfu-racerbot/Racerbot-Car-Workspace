import numpy as np
from nlos_proto import *
# (a) false alarms on an empty approach with an independently learned background
W=1.5; sc=Scene(W=W); angles=np.linspace(-np.radians(135),np.radians(135),1081)[300:781]
def prior(pose, err=(0,0,0)):
    x,y,yaw=pose[0]+err[0],pose[1]+err[1],pose[2]+err[2]
    R=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    return R@(sc.P0-[x,y]), R@(sc.P1-[x,y])
for fp_k,maxp in ((0.012,6.0),(0.0,1e9)):
    rng=np.random.default_rng(100)
    for min_hits in (2,3):
        bg=Background()
        for xe in np.arange(-7,-0.2,0.05):
            for ye in (0.35*W,0.5*W,0.65*W):
                pose=(xe,ye,rng.normal(0,0.03)); r,_=trace_scan(sc,pose,angles,rng=rng,fp_k=fp_k,max_ghost_path=maxp)
                res=unfold_scan(r,angles,*prior(pose),rng=rng)
                if res is not None: bg.add(to_panel_frame(res,res['p'][res['ghost']]))
        bg.finalize(min_hits=min_hits)
        rng2=np.random.default_rng(999); nscan=0; ncl=0; hist=[]; nconf=0; nfg=0
        for rep in range(4):
            ye=rng2.uniform(0.4,1.1); hist=[]
            for xe in np.arange(-7,0.9,0.1):
                pose=(xe,ye+rng2.normal(0,0.02),rng2.normal(0,0.05))
                r,_=trace_scan(sc,pose,angles,rng=rng2,fp_k=fp_k,max_ghost_path=maxp)
                res=unfold_scan(r,angles,*prior(pose,(rng2.normal(0,0.15),rng2.normal(0,0.15),rng2.normal(0,np.radians(3)))),rng=rng2)
                nscan+=1; det=False
                if res is not None:
                    P=to_panel_frame(res,res['p'][res['ghost']]); P=P[P[:,1]>0]; fg=~bg.is_static(P); nfg+=fg.sum()
                    cl=[c for c in cluster(P[fg]) if np.ptp(P[fg][c],axis=0).max()<0.8]
                    det=len(cl)>0
                ncl+=det; hist.append(det); nconf+= sum(hist[-6:])>=4
        print(f'fp_k={fp_k} maxpath={maxp} bg_min_hits={min_hits}: scans {nscan}, scans with a cluster {ncl}, scans in 4-of-6 state {nconf}, foreground pts {nfg}')
# (b) KF thresholds
def kf_run(seed,sig_meas=0.02,R=0.02,sa=4.0,dt=0.025,T=2.0,v=-1.0):
    rng=np.random.default_rng(seed); F=np.array([[1,dt],[0,1]]); Q=sa**2*np.array([[dt**4/4,dt**3/2],[dt**3/2,dt**2]]); H=np.array([[1.0,0]])
    x=None; out=[]
    for k in range(int(T/dt)):
        z=3.0+v*k*dt+rng.normal(0,sig_meas)
        if x is None: x=np.array([z,0.0]); P=np.diag([R**2,9.0]); out.append((x[0]-(3.0+v*k*dt),x[1]-v)); continue
        x=F@x; P=F@P@F.T+Q; S=H@P@H.T+R**2; K=P@H.T/S; x=x+(K*(z-H@x)).ravel(); P=(np.eye(2)-K@H)@P
        out.append((x[0]-(3.0+v*k*dt),x[1]-v))
    return np.array(out)
E=np.array([kf_run(s) for s in range(300)])
k1=int(1.0/0.025)
print('KF after 1 s: |vel err| p50 %.3f p95 %.3f max %.3f ; |pos err| p95 %.3f max %.3f'%(np.median(np.abs(E[:,k1:,1])),np.percentile(np.abs(E[:,k1:,1]),95),np.abs(E[:,k1:,1]).max(),np.percentile(np.abs(E[:,k1:,0]),95),np.abs(E[:,k1:,0]).max()))
