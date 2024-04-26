import os
import numpy as np
import argparse
import wandb
import uuid
import json
import datetime
from typing import Tuple, Dict, Any, Optional, Union

import torch

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy

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
from agent.EpsilonGreedyDDPG import EpsilonGreedyDDPG
from agent.EpsilonGreedySAC import EpsilonGreedySAC
from regulation.PIDController import PIDController

# class EpsilonGreedyPolicy(PolicyWrapper):
#     def __init__(self, policy, epsilon=0.1):        
#         super(EpsilonGreedyPolicy, self).__init__(policy)
#         self.policy = policy
#         self.epsilon = epsilon

#     # def predict(
#     #     self,
#     #     observation: Union[np.ndarray, Dict[str, np.ndarray]],
#     #     state: Optional[Tuple[np.ndarray, ...]] = None,
#     #     episode_start: Optional[np.ndarray] = None,
#     #     deterministic: bool = True,
#     # ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
#     #     if deterministic or np.random.rand() > self.epsilon:
#     #         # Follow expert behavior (80% of the time)
#     #         action, _ = self.policy.predict(observation, deterministic)
#     #     else:
#     #         # Explore by choosing a random action (20% of the time)
#     #         action = np.random.choice(self.policy.action_space.n)
#     #     return action, state

#     def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
#         if deterministic or np.random.rand() > self.epsilon:
#             # Follow expert behavior (80% of the time)
#             action, _ = self.policy.predict(observation, deterministic=deterministic)
#         else:
#             # Explore by choosing a random action (20% of the time)
#             action = np.random.choice(self.policy.action_space.n)
#         return action

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
        default="pretrained_PPO_1"
    )

    args = parser.parse_args()


    model_directory = "../models/"

    current_datetime = datetime.datetime.now()
    current_reward = rewards.negative_centering

    name = f"{current_reward.__name__}_{current_datetime.strftime('%Y-%m-%d-%H-%M')}"

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

    model_type = "DDPG"

    #config = open(f"../hyperparams/{model_type}.json", "r").read()
    #config = json.loads(config)[current_reward.__name__]
    #print(config)

    try:
        algo = getattr(sb3, model_type)
    except Exception as e:
        algo = getattr(sb3_contrib, model_type)


    run = wandb.init(
        # Set the project where this run will be logged
        project="donkey_car",
        #config=config,
        name=name,
        sync_tensorboard=True,
        save_code=True,
    )
    # TODO : add wrapper for car that does not move at all for a number of steps (can't move uphill or does not move at all)
    pretrained_model = PPO.load(f"../models/{args.model_name}.zip", 
        env=env,
        verbose=1, 
        buffer_size=20_000,
    )

    json_file = open(f"../data/pid_best_parameters/pid_controller.json", "r")

    best_parameters = json.load(json_file)

    target_speed = best_parameters["target_speed"]
    target_distance = best_parameters["target_distance"]

    speed_parameters = best_parameters["speed_controller"]
    steering_parameters = best_parameters["steering_controller"]


    speed_controller = PIDController(
        kp = speed_parameters["kp"],
        ki = speed_parameters["ki"],
        kd = speed_parameters["kd"],
        min_output = speed_parameters["min_output"],
        max_output = speed_parameters["max_output"]
    )
    steering_controller = PIDController(
        kp = steering_parameters["kp"],
        ki = steering_parameters["ki"],
        kd = steering_parameters["kd"],
        min_output = steering_parameters["min_output"],
        max_output = steering_parameters["max_output"],
    )

    model = EpsilonGreedySAC(
        pretrained_model,
        speed_controller,
        steering_controller,
        target_speed,
        target_distance,
        epsilon=0.1,
        policy="CnnPolicy",
        env=env,
        verbose=1,
        buffer_size=20_000,
        learning_starts=0,
        train_freq=(1, "episode"),
    )

    callback = retrieve_callbacks(env, name, config=None, save_frequency=10)

    env.reset()
    model.learn(total_timesteps=10*100_000, callback=callback)
    model.save(f"../models/{name}")

    run.finish()