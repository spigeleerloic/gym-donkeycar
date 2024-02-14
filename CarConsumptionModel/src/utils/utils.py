import numpy as np
import logging
from typing import Any, Dict


def supply_defaults(conf: Dict[str, Any]) -> None:
    """
    Update the config dictonnary
    with defaults when values are missing.

    :param conf: The user defined config dict,
        passed to the environment constructor.
    """
    defaults = [
        ("start_delay", 5.0),
        ("max_cte", 8.0),
        ("frame_skip", 1),
        ("cam_resolution", (120, 160, 3)),
        ("log_level", logging.INFO),
        ("host", "localhost"),
        ("port", 9091),
        ("steer_limit", 1.0),
        ("throttle_min", 0.0),
        ("throttle_max", 1.0),
    ]

    for key, val in defaults:
        if key not in conf:
            conf[key] = val
            print(f"Setting default: {key} {val}")



def compute_VSP(self):
        
    speed = self.speed

    acceleration_x, acceleration_y, acceleration_z = self.accel_x, self.accel_y, self.accel_z
    acceleration = np.sqrt(acceleration_x ** 2 + acceleration_y ** 2 + acceleration_z ** 2)

    gravity = 9.81
    road_grade = self.gyro_z
    rolling_resistance = 0.132
    aerodynamic_drag = 0.000302

    return (speed * (1.1 * acceleration + gravity * np.sin(road_grade) + rolling_resistance) + aerodynamic_drag * speed ** 3)
