import logging
import math
import os

from typing import Any, Dict

from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimHandler



logger = logging.getLogger(__name__)

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
    