import numpy as np, sys
from nlos_proto import *
PY=np.random.default_rng(7)
W=float(sys.argv[1]) if len(sys.argv)>1 else 1.5
fp_k=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
sc=Scene(W=W)
angles=np.linspace(-np.radians(135),np.radians(135),1081)
t_true=sc.t; n_true=sc.n
def prior(pose, err=(0.0,0.0,0.0)):
    x,y,yaw=pose[0]+err[0],pose[1]+err[1],pose[2]+err[2]
    R=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    return R@(sc.P0-[x,y]), R@(sc.P1-[x,y])
# background from a solo pass
bg=Background()
for xe in np.arange(-7,-0.2,0.05):
    for ye in (0.35*W,0.5*W,0.65*W):
        pose=(xe,ye,PY.normal(0,0.03))
        r,_=trace_scan(sc,pose,angles,rng=PY,fp_k=fp_k)
        res=unfold_scan(r,angles,*prior(pose),rng=PY)
        if res is None: continue
        bg.add(to_panel_frame(res,res['p'][res['ghost']]))
bg.finalize()
print(f'W={W} panel L={sc.L:.3f} P0={sc.P0} P1={sc.P1} fp_k={fp_k}; background static cells {bg.static.sum()}')
car_L,car_W=0.50,0.28
hid=[(xh,yh) for xh in np.arange(0.25,W-0.2,0.25) for yh in np.arange(W+0.4,W+6.01,0.4)]
print('  D[m]  ego_x | hidden_from_LOS  NLOS_det(of hidden)  LOS_det | outline RMSE[m]  centre err med[m]  fitfail | max depth y-W seen [m]')
for xe in (-6,-5,-4,-3,-2,-1.5,-1,-0.5):
    pose=(xe,0.5*W,0.0)
    D=np.linalg.norm((sc.P0+sc.P1)/2-np.array(pose[:2]))
    nh=nd=nl=ff=0; errs=[]; cen=[]; depth=[]
    for (xh,yh) in hid:
        car=(xh,yh,-np.pi/2,car_L,car_W)
        r,truth=trace_scan(sc,pose,angles,car=car,rng=PY,fp_k=fp_k)
        los=(truth==4).sum()>=3
        nl+=los
        if los: continue
        nh+=1
        perr=(PY.normal(0,0.15),PY.normal(0,0.15),PY.normal(0,np.radians(3)))
        res=unfold_scan(r,angles,*prior(pose,perr),rng=PY)
        if res is None: ff+=1; continue
        P=to_panel_frame(res,res['p'][res['ghost']])
        fg=~bg.is_static(P)
        cl=cluster(P[fg])
        cl=[c for c in cl if np.ptp(P[fg][c],axis=0).max()<0.8]
        if not cl: continue
        best=max(cl,key=len); Q=P[fg][best]
        Qm=sc.P0+Q[:,[0]]*t_true+Q[:,[1]]*n_true
        d=outline_distance(Qm,car)
        if np.median(d)<0.3:
            nd+=1; errs.append(d); cen.append(np.linalg.norm(Qm.mean(0)-np.array(car[:2]))); depth.append(yh-W)
    e=np.concatenate(errs) if errs else np.array([np.nan])
    print(f' {D:5.2f} {xe:5.1f}  | {nh:3d}/{len(hid)}   {nd:3d} ({(nd/max(nh,1)):.2f})   {nl:3d} | {np.sqrt(np.nanmean(e**2)):.3f}   {np.median(cen) if cen else np.nan:.3f}   {ff} | {max(depth) if depth else 0:.1f}')
