"""Extract rb1 series from bags to ~/work/series/<run>.npz (ROS-free, rosbags)."""
import sys, math
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
out = Path.home()/'work'/'series'; out.mkdir(exist_ok=True)
T = ['/drive','/odom','/sensors/imu/raw','/steering/measured','/imu/vesc/data']
for run in sys.argv[1:]:
    run = Path(run); rows = {t: [] for t in T}
    try:
        with AnyReader([run/'bag']) as r:
            cons = [c for c in r.connections if c.topic in T]
            for c, ts, raw in r.messages(connections=cons):
                m = r.deserialize(raw, c.msgtype); t = ts/1e9
                if c.topic == '/drive': rows[c.topic].append((t, m.drive.steering_angle, m.drive.speed))
                elif c.topic == '/odom': rows[c.topic].append((t, m.twist.twist.linear.x, m.twist.twist.angular.z))
                elif c.topic == '/sensors/imu/raw': rows[c.topic].append((t, m.angular_velocity.z))
                elif c.topic == '/imu/vesc/data': rows[c.topic].append((t, m.angular_velocity.z))
                else: rows[c.topic].append((t, m.position[0] if len(m.position) else float('nan')))
    except Exception as e:
        print(run.name, 'ERR', e); continue
    np.savez(out/f'{run.name}.npz', **{k.strip('/').replace('/','_'): np.array(v) for k, v in rows.items()})
    d = np.array(rows['/drive']); o = np.array(rows['/odom'])
    print(run.name, {k: len(v) for k, v in rows.items()}, 'maxcmd', d[:,2].max() if len(d) else None, 'maxv', o[:,1].max() if len(o) else None, flush=True)
