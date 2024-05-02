
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimHandler
import logging
from typing import Any, Dict, Tuple
import os
import numpy as np
import math

logger = logging.getLogger(__name__)

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
        self.has_reached_checkpoint = False
        self.next_marker = 1.0
        self.destination_marker = None

    def reset(self) -> None:
        super().reset()
        self.objective_reached = False
        self.cumulative_consumption = 0.0
        self.has_reached_checkpoint = False

    
    def on_car_loaded(self, message: Dict[str, Any]) -> None:
        super().on_car_loaded(message)

        if "destination_marker" in message:
            self.destination_marker = int(message["destination_marker"])

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
            current_marker = self.next_marker
            self.next_marker = float(message["next_marker"])
            self.has_reached_checkpoint = False
            if current_marker != self.next_marker:
                self.has_reached_checkpoint = True
            

        super().on_telemetry(message)


    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        observation, reward, done, info = super().observe()
        # add new information
        #info["raycast"] = self.raycast
        info["distance_to_middle_line"] = self.distance_to_middle_line
        info["objective_reached"] = self.objective_reached
        info["vsp"] = self.vsp
        info["next_marker"] = self.next_marker
        info["cumulative_consumption"] = self.cumulative_consumption
        info["objective_distance"] = self.objective_distance
        info["destination_marker"] = self.destination_marker

        return observation, reward, done, info
    
    def send_pause(self) -> None:
        msg = {"msg_type": "pause"}
        self.blocking_send(msg)

    def send_resume(self) -> None:
        msg = {"msg_type": "resume"}
        self.blocking_send(msg)


    
    