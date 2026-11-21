import numpy as np
from pathlib import Path
L=0.36; DT=0.025
def grid(name):
    d=np.load(Path.home()/'work/series'/f'{name}.npz')
    dr,od,gy=d['drive'],d['odom'],d['sensors_imu_raw']
    t=np.arange(max(dr[0,0],od[0,0],gy[0,0]),min(dr[-1,0],od[-1,0],gy[-1,0]),DT)
    j=np.clip(np.searchsorted(dr[:,0],t,side='right')-1,0,len(dr)-1)
    g=np.interp(t,gy[:,0],gy[:,1]); v=np.interp(t,od[:,0],od[:,1])
    still=(np.abs(v)<0.02)&(dr[j,2]==0)
    return dict(t=t,steer=dr[j,1],cmd=dr[j,2],v=v,oyaw=np.interp(t,od[:,0],od[:,2]),gyro=g,bias=float(np.median(g[still])))
