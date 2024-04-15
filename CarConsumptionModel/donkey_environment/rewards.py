import logging 
import math

logger = logging.getLogger(__name__)

COLLISION_REWARD = -1000.0
COEF_COLLISION_REWARD = 10000.0
DONE_REWARD = 0.0
CHECKPOINT_REWARD = 0.0
AVOID_COLLISION_REWARD = 0.0
CENTERING_COEF_REWARD = - 1.0



def positive_centering(self, done: bool) -> float:

    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return -1.0
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    centering_reward_term = math.fabs(self.distance_to_middle_line)
    centering_reward_term = min(centering_reward_term, 1.0)
  
    reward =  (1.0 - centering_reward_term) * self.forward_vel
    logger.debug(f"reward: {reward}")
    return reward

def positive_centering_consumption(self, done: bool) -> float:


    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return -1.0
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    centering_reward_term = math.fabs(self.distance_to_middle_line)
    reward =  (1.0 - centering_reward_term) * self.forward_vel

    reward -= self.vsp

    logger.debug(f"reward: {reward}")
    return reward

def negative_centering(self, done: bool) -> float:

    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return COLLISION_REWARD - COEF_COLLISION_REWARD * self.objective_distance
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    centering_reward_term = math.fabs(self.distance_to_middle_line)
    reward =  - (centering_reward_term * self.forward_vel)
    logger.debug(f"reward: {reward}")
    return -reward

def negative_centering_consumption(self, done: bool) -> float:

    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return COLLISION_REWARD - COEF_COLLISION_REWARD * self.objective_distance
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    centering_reward_term = math.fabs(self.distance_to_middle_line)
    reward =  - (centering_reward_term * self.forward_vel)

    reward -= self.vsp

    logger.debug(f"reward: {reward}")
    return -reward

def positive_centering_distance(self, done : bool) -> float:

    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return -1.0
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD
    
    centering_reward_term = math.fabs(self.distance_to_middle_line)
    centering_reward_term = min(centering_reward_term, 1.0)

    reward = (1.0 - self.objective_distance) * (1.0 - centering_reward_term) * self.forward_vel
    logger.debug(f"reward: {reward}")
    return reward

def positive_distance_only(self, done: bool) -> float:

    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return -1.0
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    reward = (1.0 - self.objective_distance)
    logger.debug(f"reward: {reward}")
    return reward