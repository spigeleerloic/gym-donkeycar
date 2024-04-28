from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Callable
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
import torch as th
from gymnasium import spaces

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

import random
from gym import Env

random.seed(42)
SelfCustomPPO = TypeVar("SelfCustomPPO", bound="CustomPPO")


class CustomPPO(PPO):

    def __init__(
        self, 
        policy: str | type[ActorCriticPolicy], 
        env: Env | VecEnv | str, 
        learning_rate: float | Callable[[float], float] = 0.0003, 
        n_steps: int = 2048, 
        batch_size: int = 64,
        n_epochs: int = 10, 
        gamma: float = 0.99, 
        gae_lambda: float = 0.95, 
        clip_range: float | Callable[[float], float] = 0.2, 
        clip_range_vf: None | float | Callable[[float], float] = None, 
        normalize_advantage: bool = True, 
        ent_coef: float = 0, 
        vf_coef: float = 0.5, 
        max_grad_norm: float = 0.5, 
        use_sde: bool = False, 
        sde_sample_freq: int = -1, 
        target_kl: float | None = None, 
        stats_window_size: int = 100, 
        tensorboard_log: str | None = None, 
        policy_kwargs: Dict[str, Any] | None = None, 
        verbose: int = 0, seed: int | None = None, 
        device: th.device | str = "auto", 
        _init_setup_model: bool = True,
        deterministic_action_probability: float = 0.6,
        ):
        super().__init__(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, normalize_advantage, ent_coef, vf_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)

        self.deterministic_action_probability = deterministic_action_probability

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                deterministic = False
                if random.random() < self.deterministic_action_probability:
                    deterministic = True
                
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor, deterministic=deterministic)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
