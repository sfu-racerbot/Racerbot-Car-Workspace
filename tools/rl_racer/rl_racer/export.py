"""Export trusted trainer checkpoints to an inference-only NumPy archive."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .train import make_env, sha256


def export(checkpoint, output):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    for key, digest in saved['assets'].items():
        if sha256(config[key]) != digest:
            raise ValueError(f'{key} changed since training; refusing export')
    env = make_env(config)
    weights = {key: value.detach().cpu().numpy() for key, value in saved['model'].items()
               if key.startswith('actor.')}
    vehicle = yaml.safe_load(Path(config['vehicle']).read_text())
    metadata = {'format_version': 1, 'config_hash': saved['config_hash'], 'config': config,
                'assets': saved['assets'], 'vehicle': vehicle, 'smoke': saved['smoke'],
                'negative_slope': config['negative_slope'], 'dt': config['dt'],
                'preview_points': env.k, 'steps': saved['steps']}
    output = Path(output)
    if output.suffix != '.npz':
        raise ValueError('export output must end in .npz')
    np.savez_compressed(output, **weights, observation_scale=env.scales,
                        action_scale=np.array([.5, 5.]),
                        steering_limits=np.array([env.model.params.steer_min, env.model.params.steer_max]),
                        v_max=np.array(env.model.params.v_max), metadata=np.array(json.dumps(metadata)))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    print(export(args.checkpoint, args.output))


if __name__ == '__main__':
    main()
