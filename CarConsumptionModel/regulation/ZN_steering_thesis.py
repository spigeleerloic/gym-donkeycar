from gym_donkeycar.envs.donkey_env import DonkeyEnv
import logging
import time
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from PIDController import PIDController
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
from metrics import compute_forward_speed, compute_distance_from_center_line, ultimate_gain_to_controller


logger = logging.getLogger(__name__)

TARGET_SPEED = 0.05


def apply_ZN(env : DonkeyEnv, steering_controller : PIDController = None, log_file : str = None):

    print(steering_controller)
    _ = env.reset()
    done = False
    forward_speed = 0.0
    distance_from_center_line = 0.0
    iteration = 0
    while not done:
        speed = TARGET_SPEED
        steering = steering_controller.update(0.0, distance_from_center_line)
        action = [steering, speed]
        _, _, done, info = env.step(action)

        iteration += 1
        current_time = (datetime.datetime.now().timestamp()) 
        forward_speed = compute_forward_speed(info)
        distance_from_center_line= compute_distance_from_center_line(info)

        if log_file is not None:
            with open(log_file, "a") as file:
                file.write(f"{steering_controller.kp},{steering_controller.ki},{steering_controller.kd},{forward_speed},{TARGET_SPEED},{distance_from_center_line},{current_time},{iteration}\n")
                

if __name__ == "__main__":

    # create environment
    level = "steep-ascent"
    env = ConsumptionWrapper(level=level)
    
    json_file = "../data/pid_best_parameters/ziegler-nichols.json"
    with open(json_file, "r") as file:
        data = json.load(file)

        steering_controller_json = data["steering_controller"]

        steering_controller = PIDController(
            kp=steering_controller_json["kp"],
            ki=steering_controller_json["ki"],
            kd=steering_controller_json["kd"],
            min_output=steering_controller_json["min_output"],
            max_output=steering_controller_json["max_output"]
        )

        log_file = "../data/ziegler_nichols/pid_table_coefficients_steering.csv"
        with open(log_file, "w") as file:
            file.write("kp,ki,kd,forward_speed,target_speed,distance_from_center_line,timestamps,steps\n")

        
        apply_ZN(
            env=env,
            steering_controller=steering_controller,
            log_file=log_file
        )