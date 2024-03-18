from gym_donkeycar.envs.donkey_env import DonkeyEnv
import logging
import time
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

from PIDController import PIDController
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
logger = logging.getLogger(__name__)
KP_MAX_CONVERGENCE_ITERATION = 100
TARGET_SPEED = 0.5
TARGET_STEERING = 0.0

def compute_distance_from_center_line(info: Dict[str, Any]) -> float:
    """
    Compute the distance from the center line

    :param info: info dictionary from the environment

    :return: the distance from the center line
    """
    #print(info)
    return info["distance_to_middle_line"]

def ziegler_nichols(env : ConsumptionWrapper):
    """
    ziegler_nichols method to tune PID controller

    :param env: donkey environment

    :return: (Kp, Ki, Kd) the tuned PID controller parameters
    """

    line_follower_controller = PIDController(0.015, 0.0, 0.0)
    ultimate_gain            = None # ku
    oscillation_period       = None  # tu
    converged                = False
    increase_factor          = 1.05
    epsilon                  = 1.0
    iteration                = 0

    times = []
    distances = []
    consecutive_actions_close_to_middle_line = 0

    number_of_kp_increases = 0

    file = open("kp_iterations.csv", "w")
    file.write("kp,iterations\n")

    # find ultimate gain
    while ultimate_gain is None:
        env.reset()

        print(f"next kp : {number_of_kp_increases} with kp : {line_follower_controller.kp}")
        line_follower_controller.kp *= increase_factor
        converged = False
        observed_distance = 0.6
        done = False
        kp_iteration = 0
        
        while not done:

            start_time = time.monotonic()
            steering = line_follower_controller.update(TARGET_STEERING, observed_distance)
            action = [steering, TARGET_SPEED]

            _, _, done, info = env.step(action)

            end_time = time.monotonic()
            times.append(end_time - start_time)
            distances.append(steering)

            distance_from_middle_line = compute_distance_from_center_line(info)
            print(f"Distance from middle line: {distance_from_middle_line}")

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
        
        file.write(f"{line_follower_controller.kp},{kp_iteration}\n")

        if converged:
            # ultimate gain = ratio of the amplitude at stable oscillation
            
            ultimate_gain = line_follower_controller.kp
            oscillation_period = 1.0 / ultimate_gain
            
            print(f"Iteration {iteration}: Potential ultimate gain ({ultimate_gain})")
            # using table from ziegler nichols method
            line_follower_controller.kp = 0.6 * ultimate_gain
            line_follower_controller.ki = 2.0 * line_follower_controller.kp / oscillation_period
            line_follower_controller.kd = line_follower_controller.kp * oscillation_period / 8.0
    
    env.close()
    file.close()
    return line_follower_controller, times, distances

if __name__ == "__main__":

    # create environment
    level = "steep-ascent"
    env = ConsumptionWrapper(level=level)
    tuned_pid_controller, times, distances = ziegler_nichols(env)
    
    print(f"kp: {tuned_pid_controller.kp} ki: {tuned_pid_controller.ki} kd: {tuned_pid_controller.kd}")

    print(f"times : {times}")
    print(f"distances : {distances}")
