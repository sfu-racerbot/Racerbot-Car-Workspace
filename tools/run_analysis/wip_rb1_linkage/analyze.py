import numpy as np, math, sys, glob, os
from pathlib import Path
L=0.36; DT=0.025
runs=['20261026-174337','20261026-180228','20261026-194514','20261026-194714','20261026-194913','20261030-175418','20261030-175701','20261030-180531','20261030-180746','20261030-190734']
def grid(name):
    d=np.load(Path.home()/'work/series'/f'{name}.npz')
    dr,od,gy=d['drive'],d['odom'],d['sensors_imu_raw']
    t=np.arange(max(dr[0,0],od[0,0],gy[0,0]),min(dr[-1,0],od[-1,0],gy[-1,0]),DT)
    j=np.clip(np.searchsorted(dr[:,0],t,side='right')-1,0,len(dr)-1)
    g=np.interp(t,gy[:,0],gy[:,1]); 
    return dict(t=t,steer=dr[j,1],cmd=dr[j,2],v=np.interp(t,od[:,0],od[:,1]),oyaw=np.interp(t,od[:,0],od[:,2]),gyro=g)
for r in runs:
    s=grid(r); g=s['gyro']
    still=(np.abs(s['v'])<0.02)&(s['cmd']==0)
    bias=np.median(g[still]) if still.sum()>40 else np.nan
    # lag-shift steer 0.14s
    lag=int(0.14/DT); st=np.r_[np.full(lag,s['steer'][0]),s['steer'][:-lag]]
    gr=np.radians(g-bias) if abs(np.nanmax(np.abs(g)))>20 else g-bias
    mov=s['v']>1.0
    imp=np.arctan(L*gr/np.maximum(s['v'],1e-3))
    straight=mov&(np.abs(st)<0.02)&(np.abs(np.gradient(st))<0.005)
    print(r,'moving s',round(mov.sum()*DT,1),'bias',round(bias,3),'gyro max',round(np.abs(g).max(),2),'trim',round(np.median(imp[straight]),4) if straight.sum()>20 else None,'n',straight.sum(),
      'ratio gyro/odomyaw', round(np.median(gr[mov&(np.abs(s['oyaw'])>0.3)]/s['oyaw'][mov&(np.abs(s['oyaw'])>0.3)]),3) if (mov&(np.abs(s['oyaw'])>0.3)).sum() else None)
