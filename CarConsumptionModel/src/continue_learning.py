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

from utils.callbacks import retrieve_callbacks
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
import donkey_environment.rewards as rewards
from agent.CustomPPO import CustomPPO
from agent.EpsilonGreedyPPO import EpsilonGreedyPPO
import agent as agent

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
        "--load_model_name", 
        help="Path to the model to load", 
        type=str, dest="model_name", 
        default="PPO"
    )

    parser.add_argument(
        "--wandb",
        help="Whether to log using wandb",
        dest="wandb",
        action="store_false"
    )

    parser.add_argument(
        "--forward-action-space",
        help="forward action space",
        action="store_false",
        dest='forward_action_space',
    )
    
    args = parser.parse_args()
    model_directory = "../models/"

    current_datetime = datetime.datetime.now()
    current_reward = rewards.distance_based_reward

    name = f"{current_reward.__name__}_{args.model_name}_{current_datetime.strftime('%Y-%m-%d-%H-%M')}"

    env_kwargs = {
        "level": args.environment,
    }

    if args.forward_action_space:
        conf = {
            "throttle_min" :  0.0,
            "throttle_max" : 1.0,
        }
        env_kwargs.update({"conf": conf})
    
    # create vectorized environment
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
    
    epsilonGreedy = False 
    model_name = args.model_name
    try:
        algo = getattr(sb3, args.model_name)
    except Exception as e:
        try :
            print("why not ppo wtf")
            algo = getattr(sb3_contrib, args.model_name)
        except Exception as e:
            epsilonGreedy = True 

            class_name = ["EpsilonGreedySAC", "EpsilonGreedyDDPG", "EpsilonGreedyPPO"]
            module = __import__("agent", class_name)
            algo = getattr(module, args.model_name) 
            model_name = args.model_name.strip()[13:]

    config = open(f"../hyperparams/{args.model_name}.json", "r").read()
    config = json.loads(config)[current_reward.__name__]
    if "train_freq" in config and type(config["train_freq"]) == list:
        config["train_freq"] = tuple(config["train_freq"])
    print(config)

    if args.wandb:
        run = wandb.init(
            # Set the project where this run will be logged
            project="donkey_pretrained",
            config=config,
            name=name,
            sync_tensorboard=True,
            save_code=True,
        )
    eval_frequency = 500
    callback = retrieve_callbacks(
        env=env,
        name=name,
        config=config,
        save_frequency=eval_frequency,
        eval_frequency=eval_frequency,
        use_wandb=args.wandb)
    
    # TODO : add wrapper for car that does not move at all for a number of steps (can't move uphill or does not move at all)
    pretrained_model =  algo.load(f"../models/pretrained_{model_name}_1.zip", 
        env=env,
        verbose=1, 
        **config
    )
    #print(pretrained_model.policy)

    env.reset()
    pretrained_model.learn(total_timesteps=100_000, callback=callback)
    pretrained_model.save(f"../models/{name}")

    run.finish()