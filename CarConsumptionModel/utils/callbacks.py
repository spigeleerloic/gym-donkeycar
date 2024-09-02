from tqdm.rich import tqdm
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy

from wandb.integration.sb3 import WandbCallback
import wandb

import torch
import numpy as np
import logging
import utils.eval_policy as eval_policy
import os


logger = logging.getLogger(__name__)

class CustomProgressBarCallback(ProgressBarCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    def __init__(self) -> None: 
        super().__init__()

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(
            total=self.locals["total_timesteps"] - self.model.num_timesteps, 
            unit="timesteps",
            desc="Agent training progress: ",
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            unit_scale=False,
            leave=False,
            dynamic_ncols=True,
        )

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()

class CustomWandbCallback(WandbCallback):
        
    def __init__(self,config, name,  **kwargs) -> None:
        super().__init__(**kwargs)

        self.config = config
        self.name = name

        self.total_episode_reward = 0
        self.episode_length = 0
        self.episode_count = 0

        wandb.define_metric("episode_reward", step_metric="episode_count")
        wandb.define_metric("episode_length", step_metric="episode_count")



    def _on_step(self) -> bool:
        super()._on_step()

        self.total_episode_reward += self.locals['rewards'][0]
        self.episode_length += 1

        wandb.log({
            "rewards": self.locals['rewards'][0],
            "next_marker": self.locals["infos"][0]["next_marker"],
            "objective_distance": self.locals["infos"][0]["objective_distance"],
            "cumulative_consumption": self.locals["infos"][0]["cumulative_consumption"],
            "vsp": self.locals["infos"][0]["vsp"],
            "objective_reached": self.locals["infos"][0]["objective_reached"],
            "distance_to_middle_line" : self.locals["infos"][0]["distance_to_middle_line"],
        }) 

        if self.locals['dones'][0]:
            wandb.log({"episode_reward": self.total_episode_reward, "episode_count":self.episode_count})
            wandb.log({"episode_length" : self.episode_length, "episode_count": self.episode_count})
            
            self.total_episode_reward = 0
            self.episode_length = 0
            self.episode_count += 1

        return True

    
class UnityInteractionCallback(BaseCallback):

    def __init__(self, env, **kwargs) -> None:
        super().__init__(**kwargs)
        self.env = env
        self.pause_message = {"msg_type": "pause"}
        self.resume_message = {"msg_type": "resume"}
        self.vectorized = False
        # if env is vectorized
        if hasattr(self.env, "envs"):
            self.vectorized = True
    
    def send_message(self, vectorized=False, message=None):
        if vectorized:
            for env in self.env.envs:
                env.viewer.handler.blocking_send(message)
        else:
            self.env.viewer.handler.blocking_send(message)
    
    def _on_step(self) -> bool:
        return super()._on_step()

    def on_rollout_start(self) -> None:
        self.send_message(self.vectorized, self.resume_message)
    
    def on_rollout_end(self) -> None: 
        self.send_message(self.vectorized, self.pause_message)

class LogCallback(BaseCallback):

    def __init__(self, log_path="../logs/training.csv", **kwargs) -> None:
        super().__init__(**kwargs)

        self.log_path = log_path
        # set logger to some destination files
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_path, mode="w")
        logger.addHandler(file_handler)
        
        self.episode_count = 0
        self.steps = 1
        #logger.info(f"Episode number : {self.episode_count}\n")
        logger.info("reward,forward_velocity,distance_to_middle_line,objective_distance,done,checkpoint,collision,episode_count,step")

    def _on_step(self) -> bool:
        # log the rewards
        # logger.info(
        #     f"""
        #     Reward: {self.locals['rewards'][0]}
        #     forward velocity : {self.locals['infos'][0]['forward_vel']} 
        #     \t distance to middle line : {self.locals['infos'][0]['distance_to_middle_line']} 
        #     \t objective distance : {self.locals['infos'][0]['objective_distance']} 
        #     \t vsp : {self.locals['infos'][0]['vsp']} 
        #     \t done : {self.locals['dones'][0]}\n"""
        # )
        reward = self.locals['rewards'][0]
        info = self.locals['infos'][0]
        forward_velocity = info['forward_vel']
        distance_to_middle_line = info['distance_to_middle_line']
        objective_distance = info['objective_distance']
        done = self.locals['dones'][0]
        checkpoint = info['checkpoint']
        collision = info["collision"]

        logger.info(f"{reward},{forward_velocity},{distance_to_middle_line},{objective_distance},{done},{checkpoint},{collision},{self.episode_count},{self.steps}")
        
        if self.locals['dones'][0]:
            self.episode_count += 1
        self.steps += 1
        return True
    
class CheckpointWithUnityInteractionCallback(CheckpointCallback):

    def __init__(self, env, **kwargs) -> None:
        super().__init__(**kwargs)
        self.env = env
        self.pause_message = {"msg_type": "pause"}
        self.resume_message = {"msg_type": "resume"}
        self.vectorized = False
        # if env is vectorized
        if hasattr(self.env, "envs"):
            self.vectorized = True

    def send_message(self, vectorized=False, message=None):
        if vectorized:
            for env in self.env.envs:
                env.viewer.handler.blocking_send(message)
        else:
            self.env.viewer.handler.blocking_send(message)

    def _on_step(self) -> bool:
        self.send_message(self.vectorized, self.pause_message)

        model_path = "../models/ppo_low_lr.zip"
        self.model.policy.save(model_path)
        if self.verbose >= 2:
            print(f"Saving model to {model_path}")
        #if self.n_calls % self.save_freq == 0:
            #self.send_message(self.vectorized, self.pause_message)
            
            #model_path="../models/ppo_low_lr.zip"
            # save current policy
            #self.model.policy.save(model_path)
            #if self.verbose >= 2:
            #    print(f"Saving model checkpoint to {model_path}")

            #policy_path = self._checkpoint_path("policy_", extension="pth")
            #torch.save(self.model.policy.state_dict(), policy_path)
            #if self.verbose >= 2:
            #    print(f"Saving policy model to {policy_path}")

            #if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
            #    replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
            #    self.model.save_replay_buffer(replay_buffer_path)
            #    if self.verbose > 1:
            #        print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            #if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
            #    vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
            #    self.model.get_vec_normalize_env().save(vec_normalize_path)
            #    if self.verbose >= 2:
            #        print(f"Saving model VecNormalize to {vec_normalize_path}")
        self.send_message(self.vectorized, self.resume_message)
        return True
    



def retrieve_callbacks(
    env,
    name: str,
    config : Dict,
    save_frequency : int = 10_000,
    eval_frequency : int = 10_000,
    use_wandb : bool = True) -> CallbackList:
    
    """
    Create a list of callbacks to be used during training
    The list of callbacks includes:
    - CheckpointWithUnityInteractionCallback             -> Save model and replay buffer
    - CustomProgressBarCallback                          -> Display progress bar
    - CustomWandbCallback                                -> Log metrics to wandb
    - UnityInteractionCallback                           -> Pause and resume Unity environment
    - LogCallback                                        -> Log training metrics to a file

    returns:
        callback : CallbackList 
    A list of callbacks to be used during training
    """
    checkpoint_callback = CheckpointWithUnityInteractionCallback(
        env=env,
        save_freq=save_frequency,
        save_path=f"../models/{name}",
        name_prefix=f"checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    unityInteractionCallback = UnityInteractionCallback(env=env)
    custom_progress_bar_callback = CustomProgressBarCallback()

    logCallback = LogCallback(log_path=f"../models/{name}/training.csv")
    
    # evalCallback = CustomEvalCallback(
    #     env,
    #     best_model_save_path=f"../eval/{name}/",
    #     log_path=f"../eval/{name}/",
    #     eval_freq=eval_frequency,
    #     deterministic=True,
    #     render=False,
    #     n_eval_episodes=5,
    #     name=name
    # )

    evalCallback = EvalCallback(
        env,
        best_model_save_path=f"../eval/{name}/",
        log_path=f"../eval/{name}/",
        eval_freq=eval_frequency,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    
    list_of_callbacks = [
        checkpoint_callback, 
        unityInteractionCallback, 
        custom_progress_bar_callback, 
        logCallback,
        evalCallback,
    ]

    if use_wandb:
        wandbcallback = CustomWandbCallback(name=name, config=config, gradient_save_freq=100, verbose=2)
        list_of_callbacks.append(wandbcallback)
    
    callback = CallbackList(list_of_callbacks)
    

    return callback