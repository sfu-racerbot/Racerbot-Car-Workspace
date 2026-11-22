"""Steering-only experiment. No output is published to the car."""
import numpy as np
from gap_follow import gap_logic
from neural_racer import observation
from neural_racer.safety_wrapper import scan_angles, WrapperConfig
def choose(scan,prior):
    cfg=WrapperConfig(max_speed=5.)
    scan=np.asarray(scan,dtype=float)
    valid=observation.beam_validity(scan,.05,25.)
    if scan.shape!=(1081,) or 1.-valid.mean()>cfg.max_invalid_fraction:
        return prior  # original wrapper handles this scan
    angles=scan_angles(-np.radians(135.),np.radians(270.)/(len(scan)-1),len(scan))
    return gap_logic.arc_guard_steering(scan,valid,angles,prior,
               required_length=1.,half_width=cfg.car_width/2+cfg.reserve_distance,
               car_length=cfg.car_length,wheelbase=cfg.wheelbase,laser_offset_x=cfg.laser_offset_x,
               laser_offset_y=0.,max_steering_angle=cfg.max_steering_angle,
               max_correction=.1745,candidates=29,blocked_fallback='request')
