import os
import argparse
import uuid
import wandb
import numpy as np
import torch 

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import sys

sys.path.insert(0, r"c:\Users\spige\memoire\gym-donkeycar-retry\gym-donkeycar\CarConsumptionModel")

from utils.environment import change_env_space, change_model_action_space
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper

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

parser.add_argument("--model_name", help="Path to the model to load", type=str, dest="model_name", default="pretrained_ppo_1.zip")

args = parser.parse_args()

environment = args.environment

# conf = {
#     "exe_path": args.path,
#     "host": "127.0.0.1",
#     "port": args.port,
#     "body_style": "donkey",
#     "body_rgb": (128, 128, 128),
#     "car_name": args.model_name,
#     "font_size": 100,
#     "racer_name": "PPO",
#     "country": "BEL",
#     "bio": "Learning from experiences",
#     "guid": str(uuid.uuid4()),
#     "max_cte": 10,
# }

env = ConsumptionWrapper(environment)
# if args.logs:
#     wandb.login()
#     wandb.init(project="donkey-car", name=args.model_name)


model_directory = "..\models"

model_path = os.path.join(model_directory, args.model_name)
try:
    # Check if the path exists
    if os.path.exists(model_path):
        print(f"path exists: {model_path}")
        # Load your model here

        ppo_agent = PPO.load(model_path)
        ppo_agent = change_model_action_space(ppo_agent)

        #model = ppo_agent.policy.to("cpu")

        for i in range(10):
            obs = env.reset()
            rewards = []
            done = False

            while not done:
                action, _ = ppo_agent.predict(obs, deterministic=True)
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
    else:
        print("Model path does not exist.")
except PermissionError as e:
    print("Permission denied error:", e)

