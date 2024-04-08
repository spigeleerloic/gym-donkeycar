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
import pandas as pd 

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
from metrics import compute_forward_speed, compute_distance_from_center_line, ultimate_gain_to_controller


logger = logging.getLogger(__name__)

TARGET_SPEED = 1.0
TARGET_STEERING = 0.035

def apply_factorial(env : DonkeyEnv, log_file: str = "../data/pid/factorial.csv"):
    """
    Apply a factorial design to the PID controller controlling the speed of the car
    It will try different values of kp and ki and log the results in a csv file
    Parameters
    ----------
    env : DonkeyEnv
        The Donkey car environment
    log_file : str
        The file to log the results of the factorial design in csv format
    -------
    Returns
        None
    """
    with open(log_file, "w") as file:
        file.write("kp,ki,kd,observed_speed,target_speed,time,iteration\n")

    kp_values = np.linspace(0.1, 3.0, num=10)
    ki_values = np.linspace(0.0, 0.1, num=100)
    kd = 0.0


    speed_controller = PIDController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        min_output=-1.0,
        max_output=1.0,
    )
    iteration = 0

    speed_controller.kd = kd
    for kp in kp_values:
        for ki in ki_values:
            speed_controller.kp = kp
            speed_controller.ki = ki
            iteration = make_actions(env, speed_controller,iteration=iteration, log_file=log_file)

    env.close()


def apply_factorial_pretrained_speed(env : DonkeyEnv, speed_controller : PIDController, log_file: str = "../data/pid/factorial.csv"):
    """
    Apply a factorial design to the PID controller controlling the steering of the car
    It will try different values of kp and ki and log the results in a csv file
    Parameters
    ----------
    env : DonkeyEnv
        The Donkey car environment
    log_file : str
        The file to log the results of the factorial design in csv format
    -------
    Returns
        None
    """
    with open(log_file, "w") as file:
        file.write("kp,ki,kd,observed_speed,target_speed,observed_distance,target_distance,time,iteration\n")
    

    kp_values = np.linspace(0.0, 1.0, num=10)
    ki_values = np.linspace(1e-4, 1e-3, num=10)
    kd_values = np.linspace(0.0, 1.0, num=10)

    steering_controller = PIDController(
        kp = 0.0,
        ki = 0.0,
        kd = 0.0,
        min_output=-5.0,
        max_output=5.0,
    )
    df = pd.read_csv(log_file)

    # last_kp = -1.0
    # last_ki = -1.0
    # if df.shape[0] != 0:
    #     # get last row of the dataframe
    #     last_row = df.iloc[-1]
    #     last_kp = last_row["kp"]
    #     last_ki = last_row["ki"]
    # print(f"last_kp: {last_kp}, last_ki: {last_ki}")
    iteration = 0

    steering_controller.kp = 1.0
    #for kp in kp_values:
    for kd in kd_values:
        for ki in ki_values:
            # if kp < last_kp or (kp == last_kp and ki <= last_ki):
            #     continue
            steering_controller.kd = kd
            steering_controller.ki = ki

            iteration = make_actions_pretrained_speed(env, speed_controller, steering_controller, iteration=iteration, log_file=log_file)

    env.close()

def make_actions(env : DonkeyEnv, speed_controller : PIDController, iteration: int = 0, log_file: str = "../data/pid/factorial.csv"):

    print(f"kp: {speed_controller.kp}, ki: {speed_controller.ki}, kd: {speed_controller.kd}")

    env.reset()
    done = False

    observed_speed = 0.0

    file = open(log_file, "a")
    
    kp =  speed_controller.kp
    ki =  speed_controller.ki
    kd =  speed_controller.kd

    inner_iteration = 0
    
    while not done and inner_iteration < 500:

        speed = speed_controller.update(TARGET_SPEED, observed_speed)
        
        action = [TARGET_STEERING, speed]
        _, _, done, info = env.step(action)

        observed_speed = compute_forward_speed(info)
        print(f"observed_speed: {observed_speed} \t action : {action}")
        
        current_time = datetime.datetime.now().timestamp()

        file.write(f"{kp},{ki},{kd},{observed_speed},{TARGET_SPEED},{current_time}{iteration}\n")
        inner_iteration += 1
        iteration += 1
    
    file.close()
    return iteration

def make_actions_pretrained_speed(env : DonkeyEnv, speed_controller : PIDController, steering_controller : PIDController, iteration: int = 0, log_file: str = "../data/pid/factorial.csv"):
    """
    This assumes a speed pid controller to be trained
    """
    print(f"kp: {steering_controller.kp}, ki: {steering_controller.ki}, kd: {steering_controller.kd}")

    env.reset()
    done = False

    observed_speed = 0.0
    observed_distance = 0.6

    file = open(log_file, "a")
    
    kp =  steering_controller.kp
    ki =  steering_controller.ki
    kd =  steering_controller.kd

    inner_iteration = 0
    
    while not done and inner_iteration < 500:

        speed = speed_controller.update(TARGET_SPEED, observed_speed)
        steering = steering_controller.update(0.0, observed_distance)

        action = [steering, speed]
        _, _, done, info = env.step(action)

        observed_speed = compute_forward_speed(info)
        observed_distance = compute_distance_from_center_line(info)
        print(f"observed_speed: {observed_speed} \t observed_distance : {observed_distance} \t action : {action}")
        
        current_time = datetime.datetime.now().timestamp()

        file.write(f"{kp},{ki},{kd},{observed_speed},{TARGET_SPEED},{observed_distance},{0.0},{current_time},{iteration}\n")
        inner_iteration += 1
        iteration += 1
    
    file.close()
    return iteration

if __name__ == "__main__":


    log_file = "../data/pid/PI_controller.csv"

    level = "steep-ascent"
    env = ConsumptionWrapper(level=level)

    speed_controller = PIDController(
        kp = 1.388888888888,
        ki = 0.1,
        kd = 0.0,
        min_output = -1.0,
        max_output = 1.0,
    )

    apply_factorial_pretrained_speed(env, speed_controller=speed_controller, log_file=log_file)