from gym_donkeycar.envs.donkey_env import DonkeyEnv
from typing import Tuple, Dict, Any, Optional
import numpy as np
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess
import time 
from gym import spaces
from utils.utils import supply_defaults

from donkey_environment.controller import DonkeyController

logger = logging.getLogger(__name__)

class ConsumptionWrapper(DonkeyEnv):

    def __init__(self, level: str, conf: Optional[Dict[str, Any]] = None):
        print("starting DonkeyGym env")
        self.viewer = None
        self.proc = None

        if conf is None:
            conf = {}

        conf["level"] = level

        # ensure defaults are supplied if missing.
        supply_defaults(conf)

        # set logging level
        logging.basicConfig(level=conf["log_level"])

        logger.debug("DEBUG ON")
        logger.debug(conf)

        # start Unity simulation subprocess
        self.proc = None
        if "exe_path" in conf:
            self.proc = DonkeyUnityProcess()
            # the unity sim server will bind to the host ip given
            self.proc.start(conf["exe_path"], host="0.0.0.0", port=conf["port"])

            # wait for simulator to startup and begin listening
            time.sleep(conf["start_delay"])

        # start simulation com
        self.viewer = DonkeyController(conf=conf)

        # Note: for some RL algorithms, it would be better to normalize the action space to [-1, 1]
        # and then rescale to proper limtis
        # steering and throttle
        self.action_space = spaces.Box(
            low=np.array([-float(conf["steer_limit"]), float(conf["throttle_min"])]),
            high=np.array([float(conf["steer_limit"]), float(conf["throttle_max"])]),
            dtype=np.float32,
        )

        # camera sensor data
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = conf["frame_skip"]

        # wait until the car is loaded in the scene
        self.viewer.wait_until_loaded()


    def _compute_VSP(self, info):
        
        speed = info['speed']

        acceleration_x, acceleration_y, acceleration_z = info['accel']
        acceleration = np.sqrt(acceleration_x ** 2 + acceleration_y ** 2 + acceleration_z ** 2)

        gravity = 9.81
        road_grade = info['gyro'][2]
        rolling_resistance = 0.132
        aerodynamic_drag = 0.000302

        return (speed * (1.1 * acceleration + gravity * np.sin(road_grade) + rolling_resistance) + aerodynamic_drag * speed ** 3)
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        # Call the original step function
        observation, reward, done, info = super().step(action)
        # Compute the VSP
        vsp = self._compute_VSP(info)
        #print(f"{info=}")
        #print(f"{done=}")
        modified_reward = reward - vsp

        # Log the resulting reward
        #print(f"Original Reward: {reward}")
        #print(f"Modified Reward: {modified_reward}")

        return observation, modified_reward, done, info
    