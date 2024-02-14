import logging
import os

from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimHandler

import base64
import logging
import time
import types
from io import BytesIO
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

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
        print("Starting DonkeyHandler") 
        super().__init__(conf)

    def determine_episode_over(self):

        if self.hit != "none":
            logger.debug(f"game over: hit {self.hit}")
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
    
    @staticmethod
    def extract_keys(dict_: Dict[str, Any], list_: List[str]) -> Dict[str, Any]:
        return_dict = {}
        for key in list_:
            if key in dict_:
                return_dict[key] = dict_[key]
        print(f"{return_dict=}")
        return return_dict
    
    def on_telemetry(self, message: Dict[str, Any]) -> None:
        
        attributes = {
        "image_b": None, "pos_x": None, "pos_y": None, "pos_z": None,
        "speed": None, "LocationMarker": [], "gyro_x": None, "gyro_y": None, "gyro_z": None,
        "accel_x": None, "accel_y": None, "accel_z": None, "vel_x": None, "vel_y": None, "vel_z": None,
        "roll": None, "pitch": None, "yaw": None, "cte": None, "lidar": None, "hit": None
        }

        img_string = message["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))

        # always update the image_array as the observation loop will hang if not changing.
        self.image_array = np.asarray(image)
        self.time_received = time.time()

        # Loop over the attributes
        for attr, default in attributes.items():
            if attr in message:
                if attr == "image_b":
                    img_string = message[attr]
                    image = Image.open(BytesIO(base64.b64decode(img_string)))
                    self.image_array_b = np.asarray(image)
                    
                elif attr == "LocationMarker":
                    self.markers = [tuple() for _ in range(len(message[attr]))]
                    for key, value in message[attr].items():
                        self.markers[int(key)] = (value["x"], value["y"], value["z"])
                    for marker in self.markers:
                        print(f"{marker=}")
                else:
                    setattr(self, attr, message[attr])
            else:
                setattr(self, attr, default)

        # Handle the special cases
        self.time_received = time.time()
        e = [self.pitch * np.pi / 180.0, self.yaw * np.pi / 180.0, self.roll * np.pi / 180.0]
        q = euler_to_quat(e)
        forward = rotate_vec(q, [0.0, 0.0, 1.0])
        self.forward_vel = forward[0] * self.vel_x + forward[1] * self.vel_y + forward[2] * self.vel_z

        # Don't update hit once session over
        if self.over:
            return

        self.determine_episode_over()