"""Decode a published rb1 window without ROS or a localization cache."""
import json,os,sys
from pathlib import Path
import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
root=Path(os.environ['DATA_WORKSPACE']).resolve()
sys.path.insert(0,str(root/'tools/run_analysis'))
import steer_stages as s
from run_analysis.runlog import Run

run=root/'data/rb1/racerbot_runs/20261111-155542'
bag=run/'bag'
info=yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
scan_t=[];scan_stamp=[];scan=[];odom=[];drive=[];joy=[];gyro=[];origin=None
for name in info['relative_file_paths']:
    with (bag/name).open('rb') as f:
        for schema,ch,rec,m in make_reader(f,decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=['/scan','/odom','/drive','/imu/data','/drive_intent','/joy']):
            t=rec.log_time/1e9
            if origin is None and ch.topic in ['/joy','/odom','/drive_intent']:origin=t
            if ch.topic=='/scan':
                scan_t.append(t);scan_stamp.append(m.header.stamp.sec+m.header.stamp.nanosec/1e9);scan.append(np.float32(m.ranges))
                meta=m.angle_min,m.angle_increment,m.range_min,m.range_max
            elif ch.topic=='/odom':
                p,q=m.pose.pose.position,m.pose.pose.orientation
                odom.append([t,m.twist.twist.linear.x,m.twist.twist.angular.z,p.x,p.y,q.z,q.w])
            elif ch.topic=='/joy':joy.append([t,m.buttons[4] if len(m.buttons)>4 else 0])
            elif ch.topic=='/drive':drive.append([t,m.header.stamp.sec+m.header.stamp.nanosec/1e9,m.drive.steering_angle,m.drive.speed])
            elif ch.topic=='/imu/data':gyro.append([t,m.angular_velocity.z])
r=Run(run,np.array(scan_t),np.array(scan_stamp),np.stack(scan),*meta,np.array(odom),np.array(drive),np.array(joy))
rows=s.replay_window(r,s.recorded_params(run),origin+84.5,origin+88.2,np.array(gyro).T)
e=np.array([a['steering']-a['real_steering'] for a in rows])
res=dict(bag=run.name,window_s=[84.5,88.2],absolute_origin=origin,scans=len(rows),paired=int(np.isfinite(e).sum()),max_rad=float(abs(e).max()),rms_rad=float(np.sqrt((e**2).mean())),within_point02_mps=sum(abs(a['speed_cmd']-a['real_speed'])<=.02 for a in rows))
print(json.dumps(res,indent=2))
