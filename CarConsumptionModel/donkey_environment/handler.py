
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimHandler
import logging
from typing import Any, Dict, Tuple
import os
import numpy as np
import math

from donkey_environment.rewards import positive_centering, positive_centering_consumption, negative_centering, negative_centering_consumption, positive_centering_distance

logger = logging.getLogger(__name__)

COLLISION_REWARD = -1000.0
COEF_COLLISION_REWARD = 10000.0
DONE_REWARD = 0.0
CHECKPOINT_REWARD = 0.0
AVOID_COLLISION_REWARD = 0.0
CENTERING_COEF_REWARD = - 1.0

# Math helpers added by CireNeikual (222464)
def euler_to_quat(e):
    cx = np.cos(e[0] * 0.5)
    sx = np.sin(e[0] * 0.5)
    cy = np.cos(e[1] * 0.5)
    sy = np.sin(e[1] * 0.5)
    cz = np.cos(e[2] * 0.5)
    sz = np.sin(e[2] * 0.5)

    x = sz * cx * cy - cz * sx * sy
    y = cz * sx * cy + sz * cx * sy
    z = cz * cx * sy - sz * sx * cy
    w = cz * cx * cy + sz * sx * sy

    return [x, y, z, w]


def cross(v0, v1):
    return [v0[1] * v1[2] - v0[2] * v1[1], v0[2] * v1[0] - v0[0] * v1[2], v0[0] * v1[1] - v0[1] * v1[0]]


def rotate_vec(q, v):
    uv = cross(q[0:3], v)
    uuv = cross(q[0:3], uv)

    scaleUv = 2.0 * q[3]

    uv[0] *= scaleUv
    uv[1] *= scaleUv
    uv[2] *= scaleUv

    uuv[0] *= 2.0
    uuv[1] *= 2.0
    uuv[2] *= 2.0

    return [v[0] + uv[0] + uuv[0], v[1] + uv[1] + uuv[1], v[2] + uv[2] + uuv[2]]


class DonkeyHandler(DonkeyUnitySimHandler):

    def __init__(self, conf: Dict[str, Any]):
        super().__init__(conf)
        self.distance_to_middle_line = 0.0
        self.raycast = []
        self.cumulative_consumption = 0.0
        self.objective_reached = False
        self.reward_function = "positive_centering"


    def reset(self) -> None:
        super().reset()
        self.objective_reached = False

    def determine_episode_over(self):

        if self.hit != "none":
            logger.debug("collision")
            self.over = True

        elif self.objective_reached:
            logger.debug("lap complete")
            self.over = True

        elif self.missed_checkpoint:
            logger.debug("missed checkpoint")
            self.over = True
        elif self.dq:
            logger.debug("disqualified")
            self.over = True

        # Disable reset
        if os.environ.get("RACE") == "True":
            self.over = False

    def on_telemetry(self, message: Dict[str, Any]) -> None:

        if "distance_to_objective" in message:
            # has already been normalized
            self.objective_distance = float(message["distance_to_objective"])

        if "rayCastFences" in message:
            # not really needed
            self.raycast = message["rayCastFences"]

        if "distance_to_middle_line" in message:
            # distance / 10.0f -> not normalized
            self.distance_to_middle_line = float(message["distance_to_middle_line"])

        if "vsp" in message:
            self.vsp = float(message["vsp"])
            self.cumulative_consumption += self.vsp
        
        if "objective_reached" in message:
            self.objective_reached = True

        if "next_marker" in message:
            self.next_marker = float(message["next_marker"])

        super().on_telemetry(message)


    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        observation, reward, done, info = super().observe()
        # add new information
        info["distance_to_middle_line"] = self.distance_to_middle_line
        #info["raycast"] = self.raycast
        info["objective_reached"] = self.objective_reached
        info["vsp"] = self.vsp
        info["next_marker"] = self.next_marker
        info["cumulative_consumption"] = self.cumulative_consumption
        info["objective_distance"] = self.objective_distance

        return observation, reward, done, info
    
    def send_pause(self) -> None:
        msg = {"msg_type": "pause"}
        self.blocking_send(msg)

    def send_resume(self) -> None:
        msg = {"msg_type": "resume"}
        self.blocking_send(msg)
    
    def compute_distance_to_objective(self) -> float:
        """
        Compute the distance to the objective and normalize it.
        updates the distance towards the markers and the objective along with the current marker.
        
        Returns:
            float: normalized distance to the objective
        """
        position = np.array([self.x, self.y, self.z])        
        distance_to_next_marker = np.linalg.norm(position - self.markers[self.next_marker])

        self.distance_to_next_marker = distance_to_next_marker
        self.distance_towards_objective = distance_to_next_marker + self.cumulated_distance_objective[self.next_marker]
        
        threshold = 1.0
        if distance_to_next_marker < threshold:
            self.next_marker = (self.next_marker + 1) % len(self.markers)
                
        normalized_distance = self.distance_towards_objective / self.maximal_distance
        # ensure 0 <= normalized_distance <= 1
        if normalized_distance > 1.0:
            normalized_distance = 1.0
        return normalized_distance

    def calc_reward(self, done: bool) -> float:

        return positive_centering(self, done)

    
    