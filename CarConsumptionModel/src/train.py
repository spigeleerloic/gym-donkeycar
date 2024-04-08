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

from utils.callbacks import CustomProgressBarCallback, SaveObservations
from gym import spaces
import gym


from donkey_environment.ConsumptionWrapper import ConsumptionWrapper

#env = gym.make("donkey-steep-ascent-track-v0")
env = ConsumptionWrapper(level="steep-ascent")
#env = ConsumptionWrapper(level="mountain_track")

name = "default_reward"

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="../models/",
  name_prefix=f"{name}",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

custom_progress_bar_callback = CustomProgressBarCallback()

save_observation_callback = SaveObservations(save_file="../dataset/observation.csv")


callback = CallbackList([checkpoint_callback, custom_progress_bar_callback, save_observation_callback])

#model = PPO("CnnPolicy", env, verbose=1)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100_000, callback=callback)
model.save(f"../models/{name}")

