import os
import gym
import gym_donkeycar
import numpy as np
import random as rd 
from gym_donkeycar.envs.donkey_env import DonkeyEnv
from utils.ConsumptionWrapper import ConsumptionWrapper
from stable_baselines3 import PPO


env = ConsumptionWrapper("steep-ascent")
# PLAY
obs = env.reset()
done = False
while not done:
    steering = 0.05
    throttle = 1.0
    action = np.array([steering, throttle])
    obs, reward, done, info = env.step(action)
    # obs shape: (120, 160, 3)
