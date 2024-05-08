import logging 
import math

logger = logging.getLogger(__name__)

COLLISION_REWARD = -500.0
COEF_COLLISION_REWARD = 1000.0
DONE_REWARD = 500.0
CHECKPOINT_REWARD = 0.0
AVOID_COLLISION_REWARD = 0.0
CENTERING_COEF_REWARD = - 1.0

def distance_based_reward(self, done: bool) -> float:
            
    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return COLLISION_REWARD
    
    # will need to add checkpoint bonus
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    objective_distance_term = - self.objective_distance
    distance_from_center_term = - math.fabs(self.distance_to_middle_line)
    weights = [0.5, 0.5]

    reward = weights[0] * objective_distance_term + weights[1] * distance_from_center_term
    logger.debug(f"reward: {reward}")
    return reward

def area_reward(self, done: bool) -> float:
        
    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return COLLISION_REWARD
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    reward = 1.0 - self.objective_distance
    logger.debug(f"reward: {reward}")
    return reward

def sparse_reward(self, done: bool) -> float:
        
    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return COLLISION_REWARD
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    reward = 0.0
    logger.debug(f"reward: {reward}")
    return reward    

def positive_centering(self, done: bool) -> float:

    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return -1.0
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    centering_reward_term = math.fabs(self.distance_to_middle_line)
    max_distance_to_middle_line = 0.6

    # normalize the distance to the middle line
    centering_reward_term = centering_reward_term / max_distance_to_middle_line
    centering_reward_term = min(centering_reward_term, 1.0)
    
    #forward_velocity_term = self.forward_vel / 10.0

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

    maximal_velocity = 4.0

    forward_velocity_term = self.forward_vel / maximal_velocity

    if forward_velocity_term > 1.0:
        # no penalty for going too fast
        forward_velocity_term = 0.0
    else:
        # if backward velocity, penalize harder
        # else penalize only the difference with the maximal velocity
        forward_velocity_term = 1.0 - forward_velocity_term

    centering_reward_term = math.fabs(self.distance_to_middle_line)
    reward =  - centering_reward_term - forward_velocity_term
    
    logger.debug(f"reward: {reward}")
    return reward


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
        return COLLISION_REWARD
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    reward = (1.0 - self.objective_distance)
    logger.debug(f"reward: {reward}")
    return reward


def hybrid_centering_velocity(self, done: bool) -> float:

    logger.debug(f"calc_reward : {self.hit} \t {done} \t {self.objective_distance}")
    if self.hit != "none":
        logger.debug(f"collision reward: {COLLISION_REWARD}")
        return COLLISION_REWARD - COEF_COLLISION_REWARD * self.objective_distance
    
    elif done:
        logger.debug(f"done reward: {DONE_REWARD}")
        return DONE_REWARD

    maximal_velocity = 4.0

    forward_velocity_term = self.forward_vel / maximal_velocity
    if forward_velocity_term < 0.0:
        forward_velocity_term = (-forward_velocity_term) * 2.0
    elif forward_velocity_term > 1.0:
        forward_velocity_term = 1.0

    centering_reward_term = math.fabs(self.distance_to_middle_line)
    reward =  - (centering_reward_term) + forward_velocity_term

    logger.debug(f"reward: {reward}")
    return reward