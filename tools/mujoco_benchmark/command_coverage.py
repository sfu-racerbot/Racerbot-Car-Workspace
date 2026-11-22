"""Independent command-only coverage cross-check. Does not decode measurements."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('data_root', type=Path)
root = parser.parse_args().data_root
out = csv.writer(sys.stdout)
out.writerow(['car', 'split', 'run', 'full_length_blocks', 'fresh_command_blocks',
              'rejected_stale_input', 'fresh_command_steps'])
for car, split, date, ids in [
    ('rb1', 'development', '20261111', '153409 153436 153721 155542'),
    ('rb1', 'evaluation', '20261116', '095058 095242 095652'),
    ('rb2', 'development', '20261111', '135934 142254 152132'),
    ('rb2', 'evaluation', '20261111', '135453 140322 141227 142036 142604 152844 154204'),
]:
    for run in ids.split():
        name = f'{date}-{run}'
        path = root / car / 'racerbot_runs' / name / 'bag'
        chunks = sorted(path.glob('*.mcap'), key=lambda p:int(p.stem.rsplit('_', 1)[1]))
        if not chunks:
            raise RuntimeError(f'missing bag {path}')
        times = []
        for file in chunks:
            with file.open('rb') as f:
                times.extend(msg.log_time for _,_,msg in
                             make_reader(f).iter_messages(topics=['/ackermann_cmd']))
        times = np.sort(np.asarray(times, dtype='i8'))
        blocks = good = 0
        if len(times):
            for b in range(times[0], times[-1] - 12_000_000_000 + 1, 10_000_000_000):
                t = b + np.arange(480,dtype='i8') * 25_000_000
                ages = t - times[np.searchsorted(times,t,side='right')-1]
                blocks += 1
                good += int(np.all(ages <= 100_000_000))
        out.writerow([car,split,name,blocks,good,blocks-good,good*480])
        sys.stdout.flush()
