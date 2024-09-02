import os
import gym.spaces
import numpy as np
import argparse
import wandb
import uuid
import json
import datetime

import torch

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
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
import gym


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
        help="clip action space to avoid moving backwards",
        action="store_false",
        dest="action_space"
    )
    
    args = parser.parse_args()
    model_directory = "../models/"

    current_datetime = datetime.datetime.now()
    current_reward = rewards.negative_centering

    name = f"{current_reward.__name__}_{args.model_name}_{current_datetime.strftime('%Y-%m-%d-%H-%M')}"

    env_kwargs = {
        "level": args.environment,
    }

    if args.action_space:
        conf = {
            "throttle_min" : 0.0,
            "throttle_max" : 1.0
        }

        env_kwargs.update({"conf": conf})

    # #create vectorized environment
    env = make_vec_env(
        ConsumptionWrapper, 
        n_envs=1, 
        env_kwargs=env_kwargs, 
        seed=42,
        vec_env_cls=DummyVecEnv,
        monitor_dir=f"../models/{name}",
    )


        # base_env = make_vec_env(
        #     lambda: ConsumptionWrapper(**env_kwargs), 
        #     n_envs=1, 
        #     seed=42
        # )

    # Wrap the base environment with VecTransposeImage if needed
    #env = VecTransposeImage(base_env)
    

    # env = ConsumptionWrapper(
    #     level=args.environment,
    #     conf = {
    #         "throttle_min" : 0.0,
    #         "throttle_max" : 1.0
    #     }
    # )
    # print(env.observation_space)
    # obs = env.observation_space

    # new_obs = gym.spaces.Box(
    #     low=obs.low.transpose((2, 0, 1)),
    #     high=obs.high.transpose((2, 0, 1)),
    #     shape=(3, 120, 160),
    #     dtype=np.uint8
    # )

    # env.observation_space = new_obs


    # env=TransposeObservationWrapper(env)
    # env = Monitor(env, filename=f"../models{name}")
    # env.set_reward_fn(current_reward)


    #set reward fn for env
    for env in env.unwrapped.envs:
        env.set_reward_fn(current_reward)
    
    epsilonGreedy = False 
    model_name = args.model_name
    try:
        algo = getattr(sb3, args.model_name)
    except Exception as e:
        try :
            algo = getattr(sb3_contrib, args.model_name)
        except Exception as e:
            epsilonGreedy = True 

            class_name = ["EpsilonGreedySAC", "EpsilonGreedyDDPG", "EpsilonGreedyPPO"]
            module = __import__("agent", class_name)
            algo = getattr(module, args.model_name) 
            model_name = args.model_name.strip()[13:]

    config = open(f"../hyperparams/{model_name}.json", "r").read()
    #print(config)
    config = json.loads(config)[current_reward.__name__]
    if "train_freq" in config and type(config["train_freq"]) == list:
        config["train_freq"] = tuple(config["train_freq"])
    #print(config)

    if args.wandb:
        run = wandb.init(
            # Set the project where this run will be logged
            project="donkey_continue_learning",
            config=config,
            name=name,
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
        )
    
    eval_frequency = get_eval_frequency(config=config)    
    eval_frequency = 500

    os.makedirs(f"../models/{name}", exist_ok=True)

    callback = retrieve_callbacks(
        env=env, 
        name=name, 
        config=config, 
        use_wandb=args.wandb,
        save_frequency=eval_frequency,
        eval_frequency=eval_frequency
    )   

    #print(env.action_space)
    # TODO : add wrapper for car that does not move at all for a number of steps (can't move uphill or does not move at all)
    pretrained_model =  algo.load(f"../models/pretrained_{model_name}_1.zip", 
        env=env,
        verbose=1, 
        **config
    )
    # pretrained_model =  algo.load(f"../models/ppo_low_lr.zip", 
    #     env=env,
    #     verbose=1, 
    #     **config
    # )

    env.reset()
    pretrained_model.learn(total_timesteps=50_000, callback=callback)
    pretrained_model.save(f"../models/{name}")
    if args.wandb:
        run.finish()