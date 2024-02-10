import os
import gym
import gym_donkeycar
import numpy as np
import argparse
import wandb
import uuid

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from utils.ConsumptionWrapper import ConsumptionWrapper
from utils.callbacks import LogCallback, SaveModelCallback

wandb.login()

mean_reward_episode = 0.0
rewards = []

nbr_of_episode = 100
time_steps = 100000

script_dir = os.path.dirname(__file__)
default_path = os.path.join(script_dir, '../../../simulator/linux_build.x86_64')

parser = argparse.ArgumentParser(description='RL algorithm with consumption model applied to donkey car')

parser.add_argument("--env_name", help="name of the donkey car environment", 
                    type=str, dest='environment', default="steep-ascent")
parser.add_argument("-p", "--path" , help="path to the simulator if it is not running", 
                    type=str, dest="path", default=default_path)

parser.add_argument("--multi", action="store_true", help="start multiple sims at once")

parser.add_argument("-n", "--name" , help="name of the model to train", type=str, dest="model_name", required=True)

parser.add_argument("--port", help="port in use for TCP connections",
                    default=9091, type=int, dest="port")

parser.add_argument("--logs", help="Whether to use logs for the training",
                     action="store_false", dest="logs")

parser.add_argument("--cte", help="Maximum CTE for the environment",
                    action="store", type=float, dest="cte", default=10)

args = parser.parse_args()
environment = args.environment

if args.path == "sim_path" and args.multi:
    print("you must supply the sim path with --sim when running multiple environments")
    exit(1)

conf = {
    "exe_path": args.path,
    "host": "127.0.0.1",
    "port": args.port,
    "body_style": "donkey",
    "body_rgb": (128, 128, 128),
    "car_name": args.model_name,
    "font_size": 100,
    "racer_name": "PPO",
    "country": "BEL",
    "bio": "Learning from experiences",
    "guid": str(uuid.uuid4()),
    "max_cte": args.cte,
}

env = ConsumptionWrapper(environment, conf=conf)

model = PPO("CnnPolicy", env, verbose=1, batch_size=256)

callback = []
if args.logs:
    print("Using logs")
    run = wandb.init(
        # Set the project where this run will be logged
        project="donkey_car"
    )
    callback.append(LogCallback())
callback.append(SaveModelCallback(model, args.model_name))

model.learn(total_timesteps=time_steps, callback=callback, progress_bar=True)
    
model.save(f"../models/{args.model_name}/model")