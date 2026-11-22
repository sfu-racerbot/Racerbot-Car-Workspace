"""LiDAR mount conversion for the pinned Gym single-track model."""
from dataclasses import replace


def measured_st_lidar(config, *, car_id: str, model, lr: float):
    """Express the measured rear-axle mount in Gym's center-of-gravity frame."""
    from f1tenth_gym.envs.dynamic_models import DynamicModel

    if car_id != 'rb2':
        raise ValueError('the measured LiDAR mount is only verified for rb2')
    if model != DynamicModel.ST:
        raise ValueError('the measured LiDAR conversion requires the ST state frame')
    x, y, yaw = config.base_link_to_lidar_tf
    return replace(config, base_link_to_lidar_tf=(x - lr, y, yaw))
