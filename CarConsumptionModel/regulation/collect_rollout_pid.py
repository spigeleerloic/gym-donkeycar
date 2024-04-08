from gym_donkeycar.envs.donkey_env import DonkeyEnv
import logging
import time
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import datetime 
import h5py

from PIDController import PIDController

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
from metrics import compute_distance_from_center_line, compute_forward_speed, ultimate_gain_to_controller
import logging

from utils.ExpertDataset import ExpertDataSet

logger = logging.getLogger(__name__)

MAX_ITER = 1_000_000

def loop_action(speed_controller : PIDController, steering_controller: PIDController, dataset_file: str = None, expert_dataset: ExpertDataSet = None):
    
    obs = env.reset()
    done = False

    observed_forward_velocity = 0.0
    observed_distance = 0.6

    #dataset_file = open(dataset_file, "w")
    #dataset_file.write("obs,action,reward,next_state\n")
    
    iteration = 0

    while iteration < MAX_ITER:

        previous_obs = obs

        steering = steering_controller.update(TARGET_DISTANCE, observed_distance)
        speed = speed_controller.update(TARGET_SPEED, observed_forward_velocity)

        action = np.array([steering, speed])
        obs, reward, done, info = env.step(action)

        observed_forward_velocity = compute_forward_speed(info)
        observed_distance = compute_distance_from_center_line(info)
        #print(f"Distance: {observed_distance}, Speed: {observed_forward_velocity}")

        #dataset_file.write(f"{previous_obs},{action},{reward},{obs}\n")
        expert_dataset.add_to_dataset(previous_obs, action)

        iteration += 1


        if done:
            env.reset()
    #dataset_file.close()
    env.close()

if __name__ == "__main__":

    level = "steep-ascent"
    env = ConsumptionWrapper(level=level)

    # 18 (speed) 0.1 (steering) -> able to make the track !

    # retrieve best parameters
    json_file = open(f"../data/pid_best_parameters/pid_controller.json", "r")
    best_parameters = json.load(json_file)

    TARGET_SPEED = best_parameters["target_speed"]
    TARGET_DISTANCE = best_parameters["target_distance"]

    speed_parameters = best_parameters["speed_controller"]
    steering_parameters = best_parameters["steering_controller"]

    speed_controller = PIDController(**speed_parameters)

    steering_controller = PIDController(**steering_parameters)

    try : 
        expertDataset = ExpertDataSet()
        loop_action(
            speed_controller, 
            steering_controller,
            dataset_file=f"../data/rollout/dataset.npz",
            expert_dataset=expertDataset
        )
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        expertDataset.store_dataset()
        json_file.close()