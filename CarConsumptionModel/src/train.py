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
from gym import spaces
import gym


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper

#env = gym.make("donkey-steep-ascent-track-v0")
env = ConsumptionWrapper(level="steep-ascent")

model = PPO("CnnPolicy", env, verbose=1, batch_size=256)
model.learn(total_timesteps=100000, progress_bar=True)
model.save("ppo_steep_ascent")