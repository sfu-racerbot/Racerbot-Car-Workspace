import numpy as np, sys, json
from dataclasses import replace
sys.path.insert(0, '/home/ubuntu/work')
from analyze_lib import grid, DT
from sim_fidelity.calibration import calibration_for_car
from run_analysis.fit_sim import simulate_commands
from concurrent.futures import ProcessPoolExecutor
def job(a):
    r, over = a
    cal = replace(calibration_for_car('rb1'), **over)
    s = grid(r); g = np.radians(s['gyro'] - s['bias'])
    start = int(np.argmax(s['v'] > 0.5)); start = max(0, start - 80); end = min(len(s['t']), start + int(90 / DT))
    sp, yw = simulate_commands(np.column_stack([s['steer'][start:end], s['cmd'][start:end]]), cal)
    v = s['v'][start:end]; gg = g[start:end]
    k = (v > 1.0) & (sp > 1.0)
    return r, float(np.sqrt(np.mean((yw[k] - gg[k])**2))), float(np.sqrt(np.mean((sp[k] - v[k])**2))), int(k.sum())
if __name__ == '__main__':
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    runs = sys.argv[2:]
    with ProcessPoolExecutor(5) as p:
        for row in p.map(job, [(r, over) for r in runs]): print(row, flush=True)
