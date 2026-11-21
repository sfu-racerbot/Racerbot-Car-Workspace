import numpy as np
from nlos_proto import *
rng=np.random.default_rng(11); sc=Scene(W=1.5); angles=np.linspace(-np.radians(135),np.radians(135),1081)
true_ang=np.arctan2(sc.t[1],sc.t[0])
print('D_lo-D_hi | n | fail | angle err median / p95 [rad] | origin err along t median / p95 [m] | span err p95 [m] | inliers median')
for (lo,hi) in [(1.5,2.5),(2.5,3.5),(3.5,4.5),(4.5,5.5),(5.5,6.5)]:
    ae=[];oe=[];se=[];ni=[];fail=0;n=0
    for _ in range(150):
        D=rng.uniform(lo,hi); ye=rng.uniform(0.3,1.2); 
        # ego x such that distance to panel centre ~ D
        pc=(sc.P0+sc.P1)/2; xe=pc[0]-np.sqrt(max(D**2-(pc[1]-ye)**2,0.01)); yaw=rng.normal(0,0.1)
        pose=(xe,ye,yaw); r,_=trace_scan(sc,pose,angles,rng=rng,fp_k=0.012)
        e=(rng.normal(0,0.15),rng.normal(0,0.15),rng.normal(0,np.radians(3)))
        x,y,yw=pose[0]+e[0],pose[1]+e[1],pose[2]+e[2]
        R=np.array([[np.cos(yw),np.sin(yw)],[-np.sin(yw),np.cos(yw)]])
        res=unfold_scan(r,angles,R@(sc.P0-[x,y]),R@(sc.P1-[x,y]),rng=rng); n+=1
        if res is None: fail+=1; continue
        Rt=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        tm=Rt@res['t']; a=np.arctan2(tm[1],tm[0]); ae.append(abs((a-true_ang+np.pi)%(2*np.pi)-np.pi))
        om=Rt@(res['c']+res['s_lo']*res['t'])+np.array(pose[:2]); oe.append(abs((om-sc.P0)@sc.t)); se.append(abs(res['s_hi']-res['s_lo']-sc.L)); ni.append(res['ninl'])
    print(f'{lo}-{hi} | {n} | {fail} | {np.median(ae):.4f} / {np.percentile(ae,95):.4f} | {np.median(oe):.3f} / {np.percentile(oe,95):.3f} | {np.percentile(se,95):.3f} | {np.median(ni):.0f}')
