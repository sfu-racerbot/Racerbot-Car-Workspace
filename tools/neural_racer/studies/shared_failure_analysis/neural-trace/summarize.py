"""Recompute the summary from the unrounded per-tick trace and compare with the supplied scorecard."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
score = json.loads(Path(sys.argv[1]).read_text())
rows, series, controls = [], {}, []
originals = {}
for group in score['simulations']:
    for race in group['races']:
        originals[(race['track'], group['grid'], race['seed'], race['names'][-1])] = race
assert len(originals) == 24
seen = set()
for p in sorted((D/'traces').glob('*.json')):
    r = json.loads(p.read_text())
    name = r['names'][-1]
    if name not in ('base', 'adapted'):
        controls.append(r)
        continue
    key = (r['track'],r['grid'],r['seed'],name)
    if key in originals:
        assert all(r[k] == v for k,v in originals[key].items()), key
        seen.add(key)
    d = pd.read_csv(p.with_suffix('.csv.gz'))
    assert np.array_equal(d.step, np.arange(len(d)))
    assert np.allclose(d.time_s, d.step*.025)
    assert len(d) == round(r['timeout_s']*40)
    assert (d.binding == 'stop').equals(d.reason.notna())
    braking = d.binding.eq('braking')
    # Verify the binding from the measured clearance, independently of its label.
    expected_cap = np.minimum(5., np.sqrt(.75**2+6*np.maximum(0.,d.wrapper_clear_m-.20))-.75)
    assert np.allclose(d.command_speed_mps[braking], expected_cap[braking], atol=1e-12,rtol=0)
    first = d[braking & d.command_speed_mps.eq(0)].iloc[0]
    moving = d.index[d.true_v_mps.abs()>=.05]
    stall = int(moving[-1]+1 if len(moving) else 0)
    # No quiet non-finishes: test against positions, not just referee lap flags.
    tail = d.tail(400)
    span = np.hypot(np.ptp(tail.x_m.to_numpy()), np.ptp(tail.y_m.to_numpy()))
    row = dict(track=r['track'],grid=r['grid'],seed=r['seed'],policy=name,
               split='original' if key in originals else 'new-sim',ticks=len(d),
               progress_m=r['progress'][-1],contact=r['contact'][-1],finish_s=r['finish_time'][-1],
               startup_invalid_ticks=int(d.iloc[:2].reason.eq('invalid_beams').sum()),
               total_hard_stops=int(d.binding.eq('stop').sum()),
               max_measured_mps=d.measured_v_mps.max(),high_band_ticks=int((d.measured_v_mps>=3).sum()),
               braking_ticks=int(braking.sum()),first_braking_zero_s=first.time_s,
               first_clear_m=first.wrapper_clear_m,first_beam_deg=first.closest_beam_deg,
               first_forward10_clear_m=first.forward10_clear_m,first_map_clear_m=first.map_gap_m,
               first_raw_speed_mps=first.network_speed_mps,
               first_command_steer_rad=first.command_steer_rad,
               slow_through_timeout_s=d.time_s.iloc[stall] if stall<len(d) else None,
               tail_max_displacement_bound_m=span,tail_median_network_mps=tail.network_speed_mps.median(),
               tail_median_command_mps=tail.command_speed_mps.median(),tail_median_clear_m=tail.wrapper_clear_m.median(),
               tail_braking_fraction=tail.binding.eq('braking').mean())
    rows.append(row)
    if r['seed']==12345 and r['grid']=='solo' and r['track']=='course_2026_10_30':
        display=d.iloc[:641:4][['time_s','network_speed_mps','command_speed_mps','true_v_mps','wrapper_clear_m']].copy()
        display['reserve_m']=.2
        series[name]=json.loads(display.replace([np.inf,-np.inf],np.nan).round(4).to_json(orient='records'))
assert seen == set(originals)
assert len(rows)==36 and len(controls)==7
with (D/'trial-summary.csv').open('w') as fp:
    writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
(D/'summary.json').write_text(json.dumps(dict(trials=rows,controls=controls,illustration=series), allow_nan=False, indent=2))
print('All 24 original race outcomes matched every supplied race field; 12 new runs and 7 controls.')
print('Max high-band ticks:',max(r['high_band_ticks'] for r in rows),
      'max final-10s bounding-box diagonal (m):',max(r['tail_max_displacement_bound_m'] for r in rows))
