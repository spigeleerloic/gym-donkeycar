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

TARGET_SPEED = 1.5
TARGET_STEERING = 0.035


def apply_ZN(env : DonkeyEnv, speed_controller : PIDController, steering_controller : PIDController = None, log_file : str = None):

    print(steering_controller)
    _ = env.reset()
    done = False
    forward_speed = 0.0
    distance_from_center_line = 0.0
    iteration = 0

    if log_file is not None:
        with open(log_file, "w") as file:
            file.write("kp,ki,kd,forward_speed,target_speed,distance_from_center_line,timestamps,steps\n")

    while not done:
        speed = speed_controller.update(TARGET_SPEED, forward_speed)
        if steering_controller is not None:
            steering = steering_controller.update(TARGET_STEERING, distance_from_center_line)
        else:
            steering = TARGET_STEERING
        action = [steering, speed]
        _, _, done, info = env.step(action)

        iteration += 1
        current_time = (datetime.datetime.now().timestamp()) 
        forward_speed = compute_forward_speed(info)
        distance_from_center_line= compute_distance_from_center_line(info)

        if log_file is not None:
            with open(log_file, "a") as file:
                file.write(f"{speed_controller.kp},{speed_controller.ki},{speed_controller.kd},{forward_speed},{TARGET_SPEED},{distance_from_center_line},{current_time},{iteration}\n")
                

if __name__ == "__main__":

    # create environment
    level = "steep-ascent"
    env = ConsumptionWrapper(level=level)
    
    json_file = "../data/pid_best_parameters/ziegler-nichols.json"
    with open(json_file, "r") as file:
        data = json.load(file)

        speed_controller_json = data["speed_controller"]

        speed_controller = PIDController(
            kp=speed_controller_json["kp"],
            ki=speed_controller_json["ki"],
            kd=speed_controller_json["kd"],
            min_output=speed_controller_json["min_output"],
            max_output=speed_controller_json["max_output"]
        )

        # steering_controller_json = data["steering_controller"]

        # steering_controller = PIDController(
        #     kp=steering_controller_json["kp"],
        #     ki=steering_controller_json["ki"],
        #     kd=steering_controller_json["kd"],
        #     min_output=steering_controller_json["min_output"],
        #     max_output=steering_controller_json["max_output"]
        # )

        log_file = "../data/ziegler_nichols/pid_table_coefficients.csv"
        apply_ZN(
            env=env,
            speed_controller=speed_controller,
            steering_controller=None,
            log_file=log_file
        )