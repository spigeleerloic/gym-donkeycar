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
from utils import compute_forward_speed, compute_distance_from_center_line, ultimate_gain_to_controller


logger = logging.getLogger(__name__)

KP_MAX_CONVERGENCE_ITERATION = 100
TARGET_SPEED = 0.25
TARGET_STEERING = 0.035


def ziegler_nichols(env : ConsumptionWrapper, forward_speed_controller : PIDController , use_logs=True, log_file="ziegler_nichols_speed_iterations.csv"):
    """
    ziegler_nichols method to tune PID controller for steering based on a pretrained speed controller

    :param env: donkey environment

    :return: an instance of the tuned PID controller used for line following steering along with some metrics (times, distances)
    """

    line_follower_controller = PIDController(0.2810243684806432, 0.0, 0.0, min_output=-5.0, max_output=5.0)
    ultimate_gain            = None # ku
    oscillation_period       = None  # tu
    converged                = False
    increase_factor          = 1.1
    epsilon                  = 0.1
    iteration                = 0

    times = []
    distances = []
    consecutive_actions_close_to_middle_line = 0

    if use_logs:
        file = open(log_file, "a")
        #file.write("kp,observed_speed,target_speed,observed_distance,target_distance,time,iterations\n")


    # find ultimate gain
    while ultimate_gain is None:
        env.reset()

        line_follower_controller.kp *= increase_factor
        converged = False
        observed_distance = 0.6
        observed_forward_velocity = 0.0
        done = False
        kp_iteration = 0
        
        while not done and kp_iteration < 1000:

            steering = line_follower_controller.update(0.0, observed_distance)
            speed = forward_speed_controller.update(TARGET_SPEED, observed_forward_velocity)
            action = [steering, speed]

            _, _, done, info = env.step(action)
            current_time = (datetime.datetime.now().timestamp()) 


            distance_from_middle_line = compute_distance_from_center_line(info)
            #print(f"Distance from middle line: {distance_from_middle_line}")

            observed_forward_velocity = compute_forward_speed(info)
            #print(f"Forward velocity: {observed_forward_velocity}")

            # interested in oscillations around the middle line -> taking absolute value
            if (abs(distance_from_middle_line) < epsilon):
                consecutive_actions_close_to_middle_line += 1
            else:
                consecutive_actions_close_to_middle_line = 0
            
            if consecutive_actions_close_to_middle_line > 100:
                converged = True
            
            observed_distance = distance_from_middle_line
            iteration += 1
            kp_iteration += 1
        
            # write to csv kp value with the number of iterations
            if use_logs:
                file.write(f"{line_follower_controller.kp},\
                           {observed_forward_velocity},\
                           {TARGET_SPEED},\
                           {observed_distance},\
                           {0.0},\
                           {current_time}\
                           {kp_iteration}\n")

        if converged:
            # ultimate gain = ratio of the amplitude at stable oscillation
            
            continue
    env.close()
    if use_logs:
        file.close()
    return line_follower_controller, times, distances

if __name__ == "__main__":

    # create environment
    level = "steep-ascent"
    env = ConsumptionWrapper(level=level)
    
    # read json in ../data/pid_best_parameters/pid_controller.json
    json_file = "../data/pid_best_parameters/pid_controller.json"
    with open(json_file, "r") as file:
        data = json.load(file)

        speed_controller_json = data["speed_controller"]

        speed_controller = ultimate_gain_to_controller(
            ultimate_gain      = speed_controller_json["ultimate_gain"], 
            oscillation_period = speed_controller_json["oscillation_period"],
            min_output         = speed_controller_json["min_output"],
            max_output         = speed_controller_json["max_output"]
        )

        log_file = "../data/ziegler_nichols/pid_tuning.csv"
        tuned_pid_controller, times, distances = ziegler_nichols(
            env, 
            speed_controller, 
            use_logs=True, 
            log_file=log_file
        )