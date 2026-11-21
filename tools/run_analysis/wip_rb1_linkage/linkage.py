import numpy as np, sys
sys.path.insert(0, str(__import__('pathlib').Path.home()/'work'))
from analyze_lib import grid, DT, L
runs=sys.argv[1:]
edges=np.array([-0.6,-0.42,-0.34,-0.26,-0.2,-0.14,-0.09,-0.05,-0.02,0.02,0.05,0.09,0.14,0.2,0.26,0.34,0.42,0.6])
C=[];A=[];AY=[]
for r in runs:
    s=grid(r); lag=int(0.14/DT)
    st=np.r_[np.full(lag,s['steer'][0]),s['steer'][:-lag]]
    steady=np.convolve(np.abs(np.gradient(st)),np.ones(9)/9,'same')<0.003
    gr=np.radians(s['gyro']-s['bias'])
    k=(s['v']>1.0)&steady
    C.append(st[k]);A.append(np.arctan(L*gr[k]/s['v'][k]));AY.append(np.abs(gr[k]*s['v'][k]))
C=np.concatenate(C);A=np.concatenate(A);AY=np.concatenate(AY)
for lo_ay,hi_ay in [(0,2.0),(2.0,4),(4,99)]:
    print('lat accel',lo_ay,hi_ay)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(C>=lo)&(C<hi)&(AY>=lo_ay)&(AY<hi_ay)
        if m.sum()>=15: print(f'  cmd {np.median(C[m]):+.3f} actual {np.median(A[m]):+.3f} n {m.sum()}')
