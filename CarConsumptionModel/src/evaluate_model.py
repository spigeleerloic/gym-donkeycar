import os
import argparse
import uuid
import wandb
import numpy as np
import re 
import pickle
import zipfile
import torch 

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ddpg.policies import CnnPolicy, MlpPolicy
import stable_baselines3.sac.policies as sac_policies
from stable_baselines3.common.policies import BasePolicy

import sys

sys.path.insert(0, r"c:\Users\spige\memoire\gym-donkeycar-retry\gym-donkeycar\CarConsumptionModel")

from utils.environment import change_env_space, change_model_action_space
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
from agent.EpsilonGreedyDDPG import EpsilonGreedyDDPG
from agent.EpsilonGreedySAC import EpsilonGreedySAC

script_dir = os.path.dirname(__file__)
#default_path = os.path.join(script_dir, '../../../simulator/linux_build.x86_64')
default_path = os.path.join(script_dir, '../../../../pid_controller_simulator/donkey_sim.exe')
if not os.path.exists(default_path):
    raise ValueError(f"Default path '{default_path}' does not exist or is inaccessible.")
else:
    print(f"Using default simulator path: {default_path}")


parser = argparse.ArgumentParser(description='RL algorithm with consumption model applied to donkey car')

parser.add_argument("--env_name", help="name of the donkey car environment", 
                    type=str, dest='environment', default="steep-ascent")
parser.add_argument("-p", "--path" , help="path to the simulator if it is not running", 
                    type=str, dest="path", default=default_path)

parser.add_argument("--port", help="port in use for TCP connections",
                    default=9091, type=int, dest="port")

parser.add_argument("--logs", help="Whether to use logs for the training",
                     action="store_false", dest="logs")

parser.add_argument("--model_name", help="Path to the model to load", type=str, dest="model_name", required=True)

parser.add_argument("--n_runs", help="Number of runs to do", type=int, dest="n_runs", default=1)

args = parser.parse_args()

environment = args.environment

env = ConsumptionWrapper(environment)



model_directory = "..\models"

model_path = os.path.join(model_directory, args.model_name)
try:
    #re = re.compile("checkpoint_\d+_steps.zip")
    re = re.compile("checkpoint_policy_\d+_steps.pth")
    for file in os.listdir(model_path):
        # if "checkpoint_%d_steps.zip" in file then load the model
        # use regex to match the pattern
        if re.match(file):
            print(f"running model : {file}")
            
            path_to_model = os.path.join(model_path, file)
            # file = open(path_to_model, "rb")
            # policy = sac_policies.SACPolicy.load_state_dict(file)
            # file.close()
            # ppo_agent = SAC("CnnPolicy", env, verbose=1)
            # ppo_agent.policy = policy

            # load saved model from zip file
            try:
                model = EpsilonGreedySAC(

                    expert_policy = None,
                    speed_controller = None, 
                    steering_controller = None, 
                    target_speed = None, 
                    target_steering = None,
                    epsilon = None,

                    policy = "CnnPolicy",
                    env = env,
                    verbose=1,
                    buffer_size=20_000,
                )
                # load policy from pth file
                model.policy.load_state_dict(torch.load(path_to_model))
                # policy = torch.load(path_to_model)
                # model.policy = policy
                for i in range(args.n_runs):
                    obs = env.reset()
                    rewards = []
                    done = False

                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        # show what the neural network is predicting
                        # neural_obs = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float()
                        # neural_action = model(neural_obs)
                        # print(f"neural action : {neural_action}")

                        obs, reward, done, info = env.step(action)
                        print(f"action : {action}") 
                        print(f"forward velocity : {info['forward_vel']}")
                        print(f"distance to middle line : {info['distance_to_middle_line']}")

                        rewards.append(reward)
                    print(f"Total steps: {len(rewards)}")
                    print(f"Total reward: {sum(rewards)}")
                    print(f"rewards : {rewards}")
            except Exception as e:
                print(f"Error loading model: {e}")
                continue
except PermissionError as e:
    print("Permission denied error:", e)

