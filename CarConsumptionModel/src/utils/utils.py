import numpy as np
import logging
from typing import Any, Dict
import wandb


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

class LogToWandb():

    def __init__(self) -> None:
        self.total_episode_reward = 0.0
        self.episode_length = 0.0
        self.episode_count = 0.0
        wandb.define_metric("episode_reward", step_metric="episode_count")
        wandb.define_metric("episode_length", step_metric="episode_count")
    
    def on_step(self, obs : np.array, reward : float, done : bool, info : Dict) -> None:

        wandb.log({
            "rewards": reward,
            "next_marker": info["next_marker"],
            "distance_towards_objective": info["distance_towards_objective"],
            "distance_to_next_marker": info["distance_to_next_marker"],
            "maximal_distance": info["maximal_distance"],
            "collision_reward": info["collision_reward"],
            "coef_collision_reward": info["coef_collision_reward"],
            "done_reward": info["done_reward"],
            "checkpoint_reward": info["checkpoint_reward"],
            "avoid_collision_reward": info["avoid_collision_reward"],
        })

        if done:
            wandb.log({"episode_reward": self.total_episode_reward, "episode_count":self.episode_count})
            wandb.log({"episode_length" : self.episode_length, "episode_count": self.episode_count})
            self.total_episode_reward = 0.0
            self.episode_length = 0.0
            self.episode_count += 1.0
        else:
            self.total_episode_reward += reward
            self.episode_length += 1.0



def compute_VSP(self):
        
    speed = self.speed

    acceleration_x, acceleration_y, acceleration_z = self.accel_x, self.accel_y, self.accel_z
    acceleration = np.sqrt(acceleration_x ** 2 + acceleration_y ** 2 + acceleration_z ** 2)

    gravity = 9.81
    road_grade = self.gyro_z
    rolling_resistance = 0.132
    aerodynamic_drag = 0.000302

    return (speed * (1.1 * acceleration + gravity * np.sin(road_grade) + rolling_resistance) + aerodynamic_drag * speed ** 3)