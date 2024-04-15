import os
import numpy as np
import argparse
import wandb
import uuid

import datetime

import torch

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gym_donkeycar.envs.donkey_env import DonkeyEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.callbacks import CustomProgressBarCallback, CustomWandbCallback, UnityInteractionCallback

from donkey_environment.ConsumptionWrapper import ConsumptionWrapper

script_dir = os.path.dirname(__file__)
#default_path = os.path.join(script_dir, '../../../simulator/linux_build.x86_64')
default_path = os.path.join(script_dir, '../../../../pid_controller_simulator/donkey_sim.exe')
if not os.path.exists(default_path):
    raise ValueError(f"Default path '{default_path}' does not exist or is inaccessible.")
else:
    print(f"Using default simulator path: {default_path}")

parser = argparse.ArgumentParser(description='RL algorithm with consumption model applied to donkey car')

parser.add_argument(
  "--env_name", 
  help="name of the donkey car environment", 
  type=str, 
  dest='environment', 
  default="steep-ascent"
)

parser.add_argument(
  "--load_model_name", 
  help="Path to the model to load", 
  type=str, dest="model_name", 
  default="pretrained_ppo_1"
)

args = parser.parse_args()


model_directory = "../models/"

#env = gym.make("donkey-steep-ascent-track-v0")
env = ConsumptionWrapper(level=args.environment)

current_datetime = datetime.datetime.now()
name = f"positive_centering_{current_datetime.strftime('%Y-%m-%d-%H-%M')}"


checkpoint_callback = CheckpointCallback(
  save_freq=10_000,
  save_path="../models/{name}",
  name_prefix=f"checkpoint",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

custom_progress_bar_callback = CustomProgressBarCallback()

config = {
    "learning_rate" : 3e-4,
    "n_steps" : 2048,
    "gamma": 0.9,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    #"use_sde": True,
    "sde_sample_freq": -1,
    "seed": 42,
    #pretrained_model.clip_range = 0.1 # need to create a Schedule -> float not callable
}

run = wandb.init(
    # Set the project where this run will be logged
    project="donkey_car",
    config=config,
    name=name,
    sync_tensorboard=True,
    save_code=True,
)

wandbcallback = CustomWandbCallback(gradient_save_freq=100, verbose=2)
unityInteractionCallback = UnityInteractionCallback(env=env)
callback = CallbackList(
    [
        checkpoint_callback, 
        custom_progress_bar_callback, 
        wandbcallback, 
        unityInteractionCallback
    ]
)

pretrained_model = PPO.load(f"../models/{args.model_name}.zip", 
    env=env,
    verbose=1, 
    **config
)

env.reset()
pretrained_model.learn(total_timesteps=10*100_000, callback=callback)
pretrained_model.save(f"../models/{name}")

run.finish()