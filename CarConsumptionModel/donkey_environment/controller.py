
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
import logging
from typing import Any, Dict

from donkey_environment.handler import DonkeyHandler
from gym_donkeycar.core.sim_client import SimClient


logger = logging.getLogger(__name__)

class DonkeyController(DonkeyUnitySimContoller):
    
    def __init__(self, conf: Dict[str, Any]):
        logger.setLevel(conf["log_level"])

        self.address = (conf["host"], conf["port"])

        self.handler = DonkeyHandler(conf=conf)

        self.client = SimClient(self.address, self.handler)