import os
import numpy as np
import argparse
import wandb
import uuid

import datetime
import torch
import json

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor

import stable_baselines3 as sb3
import sb3_contrib as sb3_contrib
from sb3_contrib import TQC


# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.callbacks import retrieve_callbacks
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
import donkey_environment.rewards as rewards
from agent.CustomPPO import CustomPPO

def get_eval_frequency(config):


    steps_before_update = config.get("n_steps", None)
    default_frequency = 500

    if steps_before_update is None:
        steps_before_update = config.get("timesteps", None)

    return steps_before_update if steps_before_update is not None else 500
    
    

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='RL algorithm with consumption model applied to donkey car')

    parser.add_argument(
        "--env_name", 
        help="name of the donkey car environment", 
        type=str, 
        dest='environment', 
        default="steep-ascent"
    )

    parser.add_argument(
        "--wandb",
        help="log to wandb",
        action="store_false",
        dest="wandb"
    )

    parser.add_argument(
        "--forward-action-space",
        help="clip action space to avoid moving backwards",
        action="store_false",
        dest="action_space"
    )

    args = parser.parse_args()

    current_datetime = datetime.datetime.now()
    current_reward = rewards.distance_based_reward_positive
    name = f"{current_reward.__name__}_{current_datetime.strftime('%Y-%m-%d-%H-%M')}"

    
    env_kwargs = {
        "level": args.environment,
    }

    if args.action_space:
        conf = {
            "throttle_min" : 0.0,
            "throttle_max" : 1.0
        }

        env_kwargs.update({"conf": conf})
    
    env = make_vec_env(
            ConsumptionWrapper, 
            n_envs=1, 
            env_kwargs=env_kwargs, 
            seed=42,
            vec_env_cls=DummyVecEnv,
            monitor_dir=f"../models/{name}",
        )
    # set reward fn for env
    for env in env.unwrapped.envs:
        env.set_reward_fn(current_reward)
    

    model_type = "PPO"
    # instantiate a model "PPO" by the name of model

    try:
        algo = getattr(sb3, model_type)
    except Exception as e:
        algo = getattr(sb3_contrib, model_type)
    
    # create a variable config that reads the parameters of ../hyperparams/tqc.json
    config = open(f"../hyperparams/{model_type}.json", "r").read()
    config = json.loads(config)[current_reward.__name__]
    if "train_freq" in config and type(config["train_freq"]) == list:
        config["train_freq"] = tuple(config["train_freq"])
    print(config)
    
    if args.wandb :
        run = wandb.init(
            # Set the project where this run will be logged
            project="donkey_msi",
            config=config,
            name=name,
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
        )
    os.makedirs(f"../models/{name}", exist_ok=True)

    eval_frequency = get_eval_frequency(config=config)    
    eval_frequency = 500

    callback = retrieve_callbacks(
        env=env, 
        name=name, 
        config=config, 
        use_wandb=args.wandb,
        save_frequency=eval_frequency,
        eval_frequency=eval_frequency
    )
    model = algo("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/" , **config)
    model.learn(total_timesteps=100_000, callback=callback)
    model.save(f"../models/{name}")

    if args.wandb:
        run.finish()
