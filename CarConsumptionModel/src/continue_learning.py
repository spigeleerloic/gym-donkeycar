import os
import numpy as np
import argparse
import wandb
import uuid
import json
import datetime

import torch

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import TQC

import stable_baselines3 as sb3
import sb3_contrib as sb3_contrib


# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.callbacks import retrieve_callbacks
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
import donkey_environment.rewards as rewards
from agent.CustomPPO import CustomPPO

if __name__ == "__main__":

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
        default="PPO"
    )

    args = parser.parse_args()


    model_directory = "../models/"

    current_datetime = datetime.datetime.now()
    current_reward = rewards.negative_centering

    name = f"{current_reward.__name__}_{args.model_name}_{current_datetime.strftime('%Y-%m-%d-%H-%M')}"

    #env = gym.make("donkey-steep-ascent-track-v0")
    #env = ConsumptionWrapper(level=args.environment)

    # create vectorized environment
    env = make_vec_env(
        ConsumptionWrapper, 
        n_envs=1, 
        env_kwargs={"level": args.environment}, 
        seed=42,
        vec_env_cls=DummyVecEnv,
        monitor_dir=f"../models/{name}",
    )

    # set reward fn for env
    for env in env.unwrapped.envs:
        env.set_reward_fn(current_reward)

    config = open(f"../hyperparams/{args.model_name}.json", "r").read()
    config = json.loads(config)[current_reward.__name__]
    print(config)

    try:
        algo = getattr(sb3, args.model_name)
    except Exception as e:
        algo = getattr(sb3_contrib, args.model_name)


    run = wandb.init(
        # Set the project where this run will be logged
        project="donkey_car",
        config=config,
        name=name,
        sync_tensorboard=True,
        save_code=True,
    )

    callback = retrieve_callbacks(env, name, config)

    # TODO : add wrapper for car that does not move at all for a number of steps (can't move uphill or does not move at all)
    pretrained_model = algo.load(f"../models/pretrained_{args.model_name}_1.zip", 
        env=env,
        verbose=1, 
        **config
    )

    #print(pretrained_model.policy)

    env.reset()
    pretrained_model.learn(total_timesteps=10*100_000, callback=callback)
    pretrained_model.save(f"../models/{name}")

    run.finish()