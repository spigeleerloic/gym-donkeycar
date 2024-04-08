from gym_donkeycar.envs.donkey_env import DonkeyEnv
import logging
import time
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import datetime

from PIDController import PIDController
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ..donkey_environment.ConsumptionWrapper import ConsumptionWrapper
from metrics import compute_forward_speed, compute_distance_from_center_line, ultimate_gain_to_controller
logger = logging.getLogger(__name__)

KP_MAX_CONVERGENCE_ITERATION = 100
TARGET_SPEED = 0.25
TARGET_STEERING = 0.035

def tune_speed_ziegler_nichols(env : ConsumptionWrapper, use_logs=True, log_file="ziegler_nichols_speed_iterations.csv"):
    """
    ziegler_nichols method to tune PID controller for speed only (not steering)

    :param env: donkey environment

    :return: an instance of the tuned PID controller (for speed only) along with some metrics (times, distances)
    """

    forward_speed_controller = PIDController(0.1, 0.0, 0.0, min_output=0, max_output=1.0)
    ultimate_gain            = None # ku
    oscillation_period       = None  # tu
    converged                = False
    increase_factor          = 1.1
    epsilon                  = 0.05
    iteration                = 0

    consecutive_actions_close_to_middle_line = 0

    number_of_kp_increases = 0

    consecutive_useless_actions = 0
    previous_position = (0.0, 0.0, 0.0)

    if use_logs:
        file = open(log_file, "w")
        file.write("kp,observed_speed,target_speed,delta_time,iterations\n")

    # find ultimate gain
    while ultimate_gain is None:
        env.reset()

        print(f"next kp : {number_of_kp_increases} with kp : {forward_speed_controller.kp}")
        forward_speed_controller.kp *= increase_factor
        converged = False
        observed_forward_velocity = 0.0
        done = False
        kp_iteration = 0

        while not done:

            speed = forward_speed_controller.update(TARGET_SPEED, observed_forward_velocity)
            action = [TARGET_STEERING, speed]

            _, _, done, info = env.step(action)

            current_time = (datetime.datetime.now().timestamp()) 

            observed_forward_velocity = compute_forward_speed(info)
            print(f"Forward velocity: {observed_forward_velocity}")

            # interested in oscillations around the middle line -> taking absolute value
            if (abs(observed_forward_velocity - TARGET_SPEED) < epsilon):
                consecutive_actions_close_to_middle_line += 1
            else:
                consecutive_actions_close_to_middle_line = 0
            
            if consecutive_actions_close_to_middle_line > 100:
                converged = True

            # if the car is not moving, we can stop the iteration
            if (np.linalg.norm(np.array(info["pos"]) - np.array(previous_position)) < 0.003):
                consecutive_useless_actions += 1
            else:
                consecutive_useless_actions = 0
            
            if consecutive_useless_actions > 50:
                # stop the iteration as it will not converge 
                break

            
            previous_position = info["pos"]
            iteration += 1
            kp_iteration += 1
            # write to csv kp value with the number of iterations
            if use_logs:
                file.write(f"{forward_speed_controller.kp},{observed_forward_velocity},{TARGET_SPEED},{current_time},{kp_iteration}\n")

        if converged:
            # ultimate gain = ratio of the amplitude at stable oscillation
            
            ultimate_gain = forward_speed_controller.kp
            oscillation_period = 1.0 / ultimate_gain
            
            print(f"Iteration {iteration}: Potential ultimate gain ({ultimate_gain})")
            # using table from ziegler nichols method
            forward_speed_controller.kp = 0.6 * ultimate_gain
            forward_speed_controller.ki = 2.0 * forward_speed_controller.kp / oscillation_period
            forward_speed_controller.kd = forward_speed_controller.kp * oscillation_period / 8.0
    
    env.close()
    if use_logs:
        file.close()
    return forward_speed_controller, times, distances


if __name__ == "__main__":

    # create environment
    level = "steep-ascent"
    env = ConsumptionWrapper(level=level)

    tuned_speed_pid_controller, times, distances = tune_speed_ziegler_nichols(env, use_logs=True, log_file=f"ziegler_nichols_speed_iterations_{TARGET_SPEED}.csv")
    