import logging
import os
import math 
import base64
import logging
import time
import types
import numpy as np

from PIL import Image
from io import BytesIO
from typing import Any, Callable, Dict, List, Tuple, Union

from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimHandler
from gym_donkeycar.core.fps import FPSTimer

from utils.utils import compute_VSP

COLLISION_REWARD = -1000.0
COEF_COLLISION_REWARD = 10000.0
DONE_REWARD = 0.0
CHECKPOINT_REWARD = 0.0
AVOID_COLLISION_REWARD = 0.0

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
        print("DonkeyHandler init")
        super().__init__(conf)
        # simulation
        self.markers = []
        self.next_marker = 1
        # compute time in YYYY-MM-DD HH:MM:SS format using local time
        self.timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.distance_towards_objective = 0.0
        self.distance_to_next_marker = 0.0
        self.normalized_distance = 0.0

    def reset(self) -> None:
        super().reset()
        self.next_marker = 1
        self.normalized_distance = 0.0


    def determine_episode_over(self):

        if self.hit != "none":
            logger.debug(f"game over: hit {self.hit}")
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
        if "LocationMarker" in message:
            self.markers = []
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


    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        while self.last_received == self.time_received:
            time.sleep(0.001)

        self.last_received = self.time_received
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)

        info = {
            "pos": (self.x, self.y, self.z),
            "cte": self.cte,
            "speed": self.speed,
            "forward_vel": self.forward_vel,
            "hit": self.hit,
            "gyro": (self.gyro_x, self.gyro_y, self.gyro_z),
            "accel": (self.accel_x, self.accel_y, self.accel_z),
            "vel": (self.vel_x, self.vel_y, self.vel_z),
            "lidar": (self.lidar),
            "car": (self.roll, self.pitch, self.yaw),
            "last_lap_time": self.last_lap_time,
            "lap_count": self.lap_count,
            # more information for environment
            "LocationMarker": self.markers,
            "next_marker": self.next_marker,
            "maximal_distance": self.maximal_distance,
            "distance_towards_objective": self.distance_towards_objective,
            "distance_to_next_marker": self.distance_to_next_marker,
            # reward shaping
            "collision_reward": COLLISION_REWARD,
            "coef_collision_reward": COEF_COLLISION_REWARD,
            "done_reward": DONE_REWARD,
            "checkpoint_reward": CHECKPOINT_REWARD,
            "avoid_collision_reward": AVOID_COLLISION_REWARD,
            # timestamp
            "timestamp": self.timestamp,
        }

        # Add the second image to the dict
        if self.image_array_b is not None:
            info["image_b"] = self.image_array_b

        # self.timer.on_frame()
        logger.debug("sending obs, reward, done info with timestamp : " + str(self.timestamp))
        return observation, reward, done, info
    
    def on_telemetry(self, message: Dict[str, Any]) -> None:
        
        img_string = message["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))

        # always update the image_array as the observation loop will hang if not changing.
        self.image_array = np.asarray(image)
        self.time_received = time.time()

        if "image_b" in message:
            img_string_b = message["image_b"]
            image_b = Image.open(BytesIO(base64.b64decode(img_string_b)))
            self.image_array_b = np.asarray(image_b)

        if "pos_x" in message:
            self.x = message["pos_x"]
            self.y = message["pos_y"]
            self.z = message["pos_z"]

        if "speed" in message:
            self.speed = message["speed"]

        e = [self.pitch * np.pi / 180.0, self.yaw * np.pi / 180.0, self.roll * np.pi / 180.0]
        q = euler_to_quat(e)

        forward = rotate_vec(q, [0.0, 0.0, 1.0])

        # dot
        self.forward_vel = forward[0] * self.vel_x + forward[1] * self.vel_y + forward[2] * self.vel_z

        if "gyro_x" in message:
            self.gyro_x = message["gyro_x"]
            self.gyro_y = message["gyro_y"]
            self.gyro_z = message["gyro_z"]
        if "accel_x" in message:
            self.accel_x = message["accel_x"]
            self.accel_y = message["accel_y"]
            self.accel_z = message["accel_z"]
        if "vel_x" in message:
            self.vel_x = message["vel_x"]
            self.vel_y = message["vel_y"]
            self.vel_z = message["vel_z"]

        if "roll" in message:
            self.roll = message["roll"]
            self.pitch = message["pitch"]
            self.yaw = message["yaw"]

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 4 scenes available now.
        if "cte" in message:
            self.cte = message["cte"]

        if "lidar" in message:
            self.lidar = self.process_lidar_packet(message["lidar"])

        # don't update hit once session over
        if self.over:
            return

        if "hit" in message:
            self.hit = message["hit"]

        if "timestamp" in message:
            self.timestamp = message["timestamp"]

        self.determine_episode_over()

    def calc_reward(self, done: bool) -> float:


        if self.hit != "none":
            reward = COLLISION_REWARD - COEF_COLLISION_REWARD * self.normalized_distance
            logger.debug(f"reward : {reward}")
            return reward
        if done:
            logger.debug("done reward")
            return DONE_REWARD
        
        if len(self.markers) <= 0:
            logger.debug("No markers found")
            return 0.0
        
        position = np.array([self.x, self.y, self.z]) 
        logger.debug(f"position : {position}")
        self.distance_to_next_marker = np.linalg.norm(position - self.markers[self.next_marker])
        logger.debug(f"distance_to_next_marker : {self.distance_to_next_marker}")
        self.distance_towards_objective = self.distance_to_next_marker + self.cumulated_distance_objective[self.next_marker] 
        logger.debug(f"distance_towards_objective : {self.distance_towards_objective}")

        threshold = 1.0
        if self.distance_to_next_marker < threshold:
            logger.debug(f"checkpoint {self.next_marker} reached")
            self.next_marker = (self.next_marker + 1) % len(self.markers)
        
        normalized_distance = self.distance_towards_objective / self.maximal_distance
        logger.debug(f"normalized_distance : {normalized_distance}")
        # ensure 0 <= normalized_distance <= 1
        self.normalized_distance = normalized_distance
        if normalized_distance > 1:
            self.normalized_distance = 1.0
        logger.debug(f"normalized_distance after bound checking : {self.normalized_distance}")

        logger.debug(f"reward : { - self.normalized_distance}")
        return - self.normalized_distance
