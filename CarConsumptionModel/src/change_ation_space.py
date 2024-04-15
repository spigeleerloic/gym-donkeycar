import os
import numpy as np
import argparse
import wandb
import uuid

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gym_donkeycar.envs.donkey_env import DonkeyEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.callbacks import CustomProgressBarCallback

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

print(args)

model_directory = "../models/"

#env = gym.make("donkey-steep-ascent-track-v0")
env = ConsumptionWrapper(level=args.environment)

name = "default_reward"

checkpoint_callback = CheckpointCallback(
  save_freq=10_000,
  save_path="../models/",
  name_prefix=f"{name}",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

custom_progress_bar_callback = CustomProgressBarCallback()


callback = CallbackList([checkpoint_callback, custom_progress_bar_callback])

pretrained_model = PPO.load(f"{model_directory}{args.model_name}")
pretrained_model.action_space.low = np.array([-1.0, -1.0])
pretrained_model.action_space.high = np.array([1.0, 1.0])

pretrained_model.save(f"{model_directory}{args.model_name}")

