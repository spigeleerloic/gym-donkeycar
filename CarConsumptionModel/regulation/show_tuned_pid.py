from gym_donkeycar.envs.donkey_env import DonkeyEnv
import logging
import time
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import datetime 

from PIDController import PIDController

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
from utils import compute_distance_from_center_line, compute_forward_speed, ultimate_gain_to_controller
logger = logging.getLogger(__name__)

KP_MAX_CONVERGENCE_ITERATION = 100
TARGET_SPEED = 0.25
TARGET_DISTANCE = 0.0

def loop_action(speed_controller : PIDController, steering_controller: PIDController, log: bool = False, log_file: str = None):
    
    env.reset()
    done = False

    observed_forward_velocity = 0.0
    observed_distance = 0.6

    if log and log_file is not None:
        log_file = open(f"../data/ziegler_nichols/best_parameters.csv", "w")
        log_file.write("time,observed_speed,target_speed,observed_distance\n")


    while not done:
        steering = steering_controller.update(TARGET_DISTANCE, observed_distance)
        speed = speed_controller.update(TARGET_SPEED, observed_forward_velocity)

        action = np.array([steering, speed])
        _, _, done, info = env.step(action)
        current_time = (datetime.datetime.now().timestamp())

        observed_forward_velocity = compute_forward_speed(info)
        observed_distance = compute_distance_from_center_line(info)
        print(f"Distance: {observed_distance}, Speed: {observed_forward_velocity}")

        if log and log_file is not None:
            log_file.write(f"{current_time},{observed_forward_velocity},{TARGET_SPEED},{observed_distance}\n")
    
    if log and log_file is not None:
        log_file.close()
    env.close()

if __name__ == "__main__":

    level = "steep-ascent"
    env = ConsumptionWrapper(level=level)

    # 18 (speed) 0.1 (steering) -> able to make the track !
    json_file = open(f"../data/pid_best_parameters/pid_controller.json", "r")

    best_parameters = json.load(json_file)

    speed_parameters = best_parameters["speed_controller"]
    steering_parameters = best_parameters["steering_controller"]

    speed_controller = PIDController(
        kp = 0.5 * speed_parameters["ultimate_gain"],
        ki = 0.0,
        kd = 0.0,
        min_output = speed_parameters["min_output"],
        max_output = speed_parameters["max_output"],
    )
    # speed_controller = ultimate_gain_to_controller(
    #     ultimate_gain      = speed_parameters["ultimate_gain"],
    #     oscillation_period = speed_parameters["oscillation_period"],
    #     min_output         = speed_parameters["min_output"],
    #     max_output         = speed_parameters["max_output"],
    # )

    # steering_controller = ultimate_gain_to_controller(
    #     ultimate_gain      = steering_parameters["ultimate_gain"],
    #     oscillation_period = steering_parameters["oscillation_period"],
    #     min_output         = steering_parameters["min_output"],
    #     max_output         = steering_parameters["max_output"],
    # )

    steering_controller = PIDController(
        kp = 0.5 * steering_parameters["ultimate_gain"],
        ki = 0.0,
        kd = 0.0,
        min_output = -5.0,
        max_output = 5.0,
    )


    loop_action(
        speed_controller, 
        steering_controller,
        log=True, 
        log_file=f"../data/ziegler_nichols/best_parameters.csv"
    )

    json_file.close()