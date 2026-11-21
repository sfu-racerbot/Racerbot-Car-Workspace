import numpy as np, sys
from nlos_proto import *
W=float(sys.argv[1]); fp_k=float(sys.argv[2]); v=float(sys.argv[3]) if len(sys.argv)>3 else 4.0; MAXP=float(sys.argv[4]) if len(sys.argv)>4 else 1e9
rng=np.random.default_rng(3)
sc=Scene(W=W); angles=np.linspace(-np.radians(135),np.radians(135),1081)
def prior(pose, err=(0,0,0)):
    x,y,yaw=pose[0]+err[0],pose[1]+err[1],pose[2]+err[2]
    R=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    return R@(sc.P0-[x,y]), R@(sc.P1-[x,y])
bg=Background()
for xe in np.arange(-7,-0.2,0.05):
    for ye in (0.35*W,0.5*W,0.65*W):
        pose=(xe,ye,rng.normal(0,0.03)); r,_=trace_scan(sc,pose,angles,rng=rng,fp_k=fp_k,max_ghost_path=MAXP)
        res=unfold_scan(r,angles,*prior(pose),rng=rng)
        if res is not None: bg.add(to_panel_frame(res,res['p'][res['ghost']]))
bg.finalize()
dx=v/40.0
print(f'W={W} fp_k={fp_k} v={v} m/s; stationary car in B; ego on centreline of A; x_corner(inner)=0, ego stops being hidden-relevant at x~0')
print(' car (x_h, depth beyond inner corner) | first NLOS confirm at ego x | first LOS at ego x | warning gain [m] / [s] | NLOS beams at confirm')
for xh in (W-0.25, W-0.5, W/2):
    for depth in (0.5,1.0,2.0,3.0):
        car=(xh,W+depth,np.pi/2,0.50,0.28)
        hist=[]; x_n=None; x_l=None; nb=0; lh=[]
        for xe in np.arange(-7,W-0.6,dx):
            pose=(xe,0.5*W,0.0)
            r,truth=trace_scan(sc,pose,angles,car=car,rng=rng,fp_k=fp_k,max_ghost_path=MAXP)
            lh.append((truth==4).sum()>=3)
            if x_l is None and sum(lh[-6:])>=4: x_l=xe
            det=False; k=0
            res=unfold_scan(r,angles,*prior(pose,(rng.normal(0,0.15),rng.normal(0,0.15),rng.normal(0,np.radians(3)))),rng=rng)
            if res is not None:
                P=to_panel_frame(res,res['p'][res['ghost']]); fg=~bg.is_static(P)
                cl=[c for c in cluster(P[fg]) if np.ptp(P[fg][c],axis=0).max()<0.8]
                if cl:
                    Q=P[fg][max(cl,key=len)]; Qm=sc.P0+Q[:,[0]]*sc.t+Q[:,[1]]*sc.n
                    if np.median(outline_distance(Qm,car))<0.3: det=True; k=len(Q)
            hist.append(det)
            if x_n is None and sum(hist[-6:])>=4: x_n=xe; nb=k
            if x_n is not None and x_l is not None: break
        gain=(x_l-x_n) if (x_n is not None and x_l is not None) else float('nan')
        print(f'  ({xh:.2f}, {depth:.1f}) | {x_n if x_n is None else round(x_n,2)} | {x_l if x_l is None else round(x_l,2)} | {gain:.2f} m / {gain/v:.2f} s | {nb}')
