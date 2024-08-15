from tqdm.rich import tqdm
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy

from wandb.integration.sb3 import WandbCallback
import wandb


import gym
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
        #if self.n_calls % self.save_freq == 0:
            #self.send_message(self.vectorized, self.pause_message)
            #model_path = self._checkpoint_path(extension="zip")

            # save current policy
            #self.model.policy.save(model_path)
            # if self.verbose >= 2:
            #     print(f"Saving model checkpoint to {model_path}")

            # policy_path = self._checkpoint_path("policy_", extension="pth")
            # torch.save(self.model.policy.state_dict(), policy_path)
            # if self.verbose >= 2:
            #     print(f"Saving policy model to {policy_path}")

            # if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            #     # If model has a replay buffer, save it too
            #     replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
            #     self.model.save_replay_buffer(replay_buffer_path)
            #     if self.verbose > 1:
            #         print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            # if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            #     # Save the VecNormalize statistics
            #     vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
            #     self.model.get_vec_normalize_env().save(vec_normalize_path)
            #     if self.verbose >= 2:
            #         print(f"Saving model VecNormalize to {vec_normalize_path}")
            #self.send_message(self.vectorized, self.resume_message)
        return True


class CustomEvalCallback(EvalCallback):
    
    def __init__(self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        name: str = "ppo_model_sample"):

        super(CustomEvalCallback, self).__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn
        )
        self.name = name
        self.center_informations = {}
        self.velocity_informations = {}
        self.objective_informations = {}
        self.eval_nbr = 0
        self.pause_message = {"msg_type": "pause"}
        self.resume_message = {"msg_type": "resume"}

        self.vectorized = False
        # if env is vectorized
        if hasattr(self.eval_env, "envs"):
            self.vectorized = True
        

    def send_message(self, vectorized=False, message=None):
        if vectorized:
            for env in self.eval_env.envs:
                env.viewer.handler.blocking_send(message)
        else:
            self.env.viewer.handler.blocking_send(message)
    
    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, center_info, velocity_info, objective_info  = eval_policy.custom_evaluation_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback
            )

            if self.log_path is not None:
                
                self.send_message(self.vectorized, self.pause_message)


                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)



                self.center_informations[self.eval_nbr] = center_info
                self.velocity_informations[self.eval_nbr] = velocity_info
                self.objective_informations[self.eval_nbr] = objective_info
                self.eval_nbr += 1
                

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

                np.savez(
                    f"../eval/{self.name}/evaluation_results.npz",
                    center=self.center_informations,
                    velocity=self.velocity_informations,
                    objective=self.objective_informations
                )

                self.send_message(self.vectorized, self.resume_message)


            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training




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
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    unityInteractionCallback = UnityInteractionCallback(env=env)
    custom_progress_bar_callback = CustomProgressBarCallback()

    logCallback = LogCallback(log_path=f"../models/{name}/training.csv")
    
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