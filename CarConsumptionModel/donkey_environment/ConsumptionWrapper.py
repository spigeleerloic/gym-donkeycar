from gym_donkeycar.envs.donkey_env import DonkeyEnv
from typing import Tuple, Dict, Any, Optional
import numpy as np
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess
import time 
from gym import spaces
#from utils.utils import supply_defaults
import os

from donkey_environment.controller import DonkeyController

logger = logging.getLogger(__name__)


def supply_defaults(conf: Dict[str, Any]) -> None:
    """
    Update the config dictonnary
    with defaults when values are missing.

    :param conf: The user defined config dict,
        passed to the environment constructor.
    """
    defaults = [
        ("start_delay", 3.0),
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

class ConsumptionWrapper(DonkeyEnv):

    """
    OpenAI Gym Environment for Donkey

    :param level: name of the level to load
    :param conf: configuration dictionary
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    ACTION_NAMES: List[str] = ["steer", "throttle"]
    VAL_PER_PIXEL: int = 255

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
        script_dir = os.path.dirname(__file__)
        logging.basicConfig(filename=os.path.join(script_dir, '../logs/wrapper.log'),
                            level=conf["log_level"],
                            filemode="w")
        logging.getLogger().setLevel("DEBUG")

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