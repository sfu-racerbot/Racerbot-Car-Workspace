"""Causal scan inputs and subsequent mux labels from recorded ROS 2 MCAPs."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from neural_racer import observation as obs

TOPICS = {'/scan': 'sensor_msgs/msg/LaserScan', '/joy': 'sensor_msgs/msg/Joy',
          '/odom': 'nav_msgs/msg/Odometry', '/ackermann_cmd': 'ackermann_msgs/msg/AckermannDriveStamped'}
MS = 1_000_000
# Node freshness limits. rb2's LiDAR header can lead the recorder by ~3 ms.
JOY_NS, ODOM_NS, SCAN_NS = 500 * MS, 100 * MS, 75 * MS
LABEL_NS, CLOCK_SKEW_NS = 25 * MS, 5 * MS


def bag_info(bag):
    return yaml.safe_load((bag / 'metadata.yaml').read_text())['rosbag2_bagfile_information']


def messages(bag, info):
    """Read split files in rosbag order (bag_10 is not the third chunk)."""
    paths = info['relative_file_paths']
    if not paths or set(paths) != {p.name for p in bag.glob('*.mcap')} or len(paths) != len(set(paths)):
        raise ValueError(f'{bag}: MCAP chunks do not match metadata')
    for name in paths:
        with (bag / name).open('rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, record, msg in reader.iter_decoded_messages(topics=TOPICS):
                if schema.name != TOPICS[channel.topic]:
                    raise ValueError(f'{bag}: unexpected {schema.name} on {channel.topic}')
                yield channel.topic, record.log_time, msg


def aligned_ticks(events, counts, seen):
    """Inputs are frozen on scan receipt. Never use the target as its own input."""
    joy = odom = pending = None
    prev_scan = None
    last_header = {}
    last_log = -1
    reset = True
    for topic, t, msg in events:
        seen.add(topic)
        if t < last_log:
            raise ValueError('MCAP log time went backwards across chunks')
        last_log = t
        stamp = msg.header.stamp
        h = stamp.sec * 1_000_000_000 + stamp.nanosec
        ordered = h > last_header.get(topic, -1)
        if ordered:
            last_header[topic] = h
        age = t - h
        valid_clock = ordered and -CLOCK_SKEW_NS <= age

        if topic == '/joy':
            held = valid_clock and age < JOY_NS and len(msg.buttons) > 4 and msg.buttons[4] == 1
            joy = min(t, h) if held else None
            if not held:
                if pending is not None:
                    counts['released_before_label'] += 1
                pending = None
                reset = True
        elif topic == '/odom':
            speed = msg.twist.twist.linear.x
            odom = (min(t, h), speed) if valid_clock and age < ODOM_NS and math.isfinite(speed) else None
        elif topic == '/scan':
            counts['scans'] += 1
            if pending is not None:
                counts['unmatched'] += 1
                pending = None
                reset = True
            if prev_scan is not None and t - prev_scan > SCAN_NS:
                reset = True
            prev_scan = t
            reason = None
            if not valid_clock or age > SCAN_NS:
                reason = 'scan_time'
            elif joy is None or t - joy >= JOY_NS:
                reason = 'deadman'
            elif odom is None or t - odom[0] >= ODOM_NS:
                reason = 'odom'
            elif (len(msg.ranges) != obs.N_BEAMS or not all(math.isfinite(x) for x in
                  (msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max))
                  or abs(msg.angle_min - math.radians(-135)) > math.radians(.125)
                  or abs(msg.angle_min + (obs.N_BEAMS - 1) * msg.angle_increment
                         - math.radians(135)) > math.radians(.125)
                  or not 0 <= msg.range_min < msg.range_max):
                reason = 'scan_geometry'
            if reason:
                counts[reason] += 1
                reset = True
            else:
                pending = (t, h, odom[1], msg)
        elif topic == '/ackermann_cmd' and pending is not None:
            ts, hs, speed, scan = pending
            action = (msg.drive.steering_angle, msg.drive.speed)
            if (not valid_clock or h < hs or t - ts > LABEL_NS or age > LABEL_NS
                    or joy is None or t - joy >= JOY_NS or not all(map(math.isfinite, action))):
                continue
            geometry = (scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max)
            yield ts, t - ts, speed, np.asarray(scan.ranges, np.float32), geometry, action, reset
            counts['accepted'] += 1
            pending = None
            reset = False
    if pending is not None:
        counts['unmatched'] += 1


def extract(data, out, context_file):
    bags = sorted(data.glob('*/racerbot_runs/*/bag'))
    if not bags:
        raise ValueError(f'no car/racerbot_runs/run/bag directories in {data}')
    out.mkdir(parents=True, exist_ok=False)
    with context_file.open() as f:
        context = {(r['car'], r['run']): (r['course'], r['traffic']) for r in csv.DictReader(f)}
    runs = []
    for bag in bags:
        car, _, run, _ = bag.relative_to(data).parts
        course, traffic = context.get((car, run), ('unknown', 'unknown'))
        counts, seen = Counter(), set()
        info = bag_info(bag)
        topic_counts = {v['topic_metadata']['name']: v['message_count']
                        for v in info.get('topics_with_message_count', [])}
        rows = list(aligned_ticks(messages(bag, info), counts, seen))
        filename = f'{car}-{run}.npz'
        if rows:
            t, lag, speed, ranges, geom, action, reset = zip(*rows)
            np.savez_compressed(out / filename, time_ns=t, label_delay_ns=lag,
                                speed=np.float32(speed), ranges=np.stack(ranges),
                                geometry=np.float32(geom), action=np.float32(action), reset=reset)
        runs.append(dict(car=car, run=run, course=course, traffic=traffic, counts=dict(counts),
                         imu_data_messages=topic_counts.get('/imu/data', 0),
                         imu_status_messages=topic_counts.get('/imu/status', 0),
                         missing_topics=sorted(TOPICS.keys() - seen), file=filename if rows else None))
        print(f'{car}/{run}: {len(rows)}/{counts["scans"]} scans', flush=True)
    (out / 'manifest.json').write_text(json.dumps(dict(version=1, runs=runs), indent=2) + '\n')
