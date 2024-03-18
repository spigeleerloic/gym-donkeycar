
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimHandler
import logging
from typing import Any, Dict, Tuple
import os
import numpy as np

logger = logging.getLogger(__name__)

COLLISION_REWARD = -1000.0
COEF_COLLISION_REWARD = 10000.0
DONE_REWARD = 0.0
CHECKPOINT_REWARD = 0.0
AVOID_COLLISION_REWARD = 0.0

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

    def reset(self) -> None:
        super().reset()
        self.next_marker = 1

    def determine_episode_over(self):

        if self.hit != "none":
            logger.debug("collision")
            self.over = True

        elif self.next_marker == 0:
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
        
    def on_car_loaded(self, message: Dict[str, Any]) -> None:
        logger.debug("car loaded message: %s", message)
        self.load_information(message)
        self.loaded = True
        # Enable hand brake, so the car doesn't move
        self.send_control(0, 0, 1.0)
        self.on_need_car_config({})

    def load_information(self, message: Dict[str, Any]) -> None:
        self.markers = []
        if "LocationMarker" in message:
            for marker in message["LocationMarker"]:
                self.markers.append(np.array([float(marker[0]), float(marker[1]), float(marker[2])]))
        
            self.maximal_distance = 0.0
            self.distance = []

            for i in range(len(self.markers)):
                distance = np.linalg.norm(self.markers[i] - self.markers[(i + 1) % len(self.markers)])
                self.maximal_distance += distance
                self.distance.append(distance)
            self.cumulated_distance_objective = [0.0 for _ in range(len(self.markers))]
            for i in range(len(self.markers)):
                self.cumulated_distance_objective[i] = sum(self.distance[i:])

            # set initial marker
            self.next_marker = 1

    def on_telemetry(self, message: Dict[str, Any]) -> None:
        if "distanceToLocationMarker" in message:
            self.objective_distance = message["distanceToLocationMarker"]
        
        if "rayCastFences" in message:
            self.raycast = message["rayCastFences"]
        
        if "distanceToLeftFence" in message:
            distance_to_left_fence = message["distanceToLeftFence"]
            distance_to_right_fence = message.get("distanceToRightFence", None)
            if distance_to_right_fence is not None:
                self.distance_to_middle_line = (float(distance_to_left_fence) - float(distance_to_right_fence))
            
        super().on_telemetry(message)

    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        observation, reward, done, info = super().observe()
        # add new information
        info["distance_to_middle_line"] = self.distance_to_middle_line
        info["raycast"] = self.raycast

        return observation, reward, done, info
    

    def calc_reward(self, done: bool) -> float:

        logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
        if self.hit != "none":
            reward = COLLISION_REWARD - COEF_COLLISION_REWARD * self.objective_distance
            logger.debug(f"reward : {reward}")
            return reward
        if done:
            logger.debug("done reward")
            return DONE_REWARD

        return - self.objective_distance




