import numpy as np, sys, json, itertools
sys.path.insert(0, '/home/ubuntu/work')
from replay import job
from concurrent.futures import ProcessPoolExecutor
FIT = ['20261026-174337','20261026-180228','20261026-194514','20261026-194714','20261026-194913']
def link(r, l, trim=0.0):
    return ((-0.6, -0.6 * r + trim), (0.0, trim), (0.6, 0.6 * l + trim))
if __name__ == '__main__':
    settings = [None] + [dict(steering_linkage=link(r, l), steering_delay_ticks=d)
                for r, l, d in itertools.product([0.80, 0.84, 0.87, 0.90, 0.94, 1.0], [0.95, 1.0, 1.05], [2, 3, 4])]
    jobs = [(run, s or {}) for s in settings for run in FIT]
    with ProcessPoolExecutor(8) as p:
        res = list(p.map(job, jobs))
    out = []
    for i, s in enumerate(settings):
        ch = res[i*len(FIT):(i+1)*len(FIT)]
        out.append(dict(setting=s, yaw=float(np.mean([c[1] for c in ch])), speed=float(np.mean([c[2] for c in ch]))))
    print('baseline', out[0])
    for o in sorted(out[1:], key=lambda o: o['yaw'])[:10]: print(o)
    json.dump(out, open('/home/ubuntu/work/grid_out.json', 'w'))
