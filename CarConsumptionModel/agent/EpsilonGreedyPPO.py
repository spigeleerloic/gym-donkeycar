from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Callable
import numpy as np
import random
from gym import Env
import torch as th

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


random.seed(42)
SelfEpsilonGreedyPPO = TypeVar("SelfEpsilonGreedyPPO", bound="EpsilonGreedyPPO")


class EpsilonGreedyPPO(PPO):

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
        epsilon_schedule: Schedule = lambda x: 0.1,
        ):
        super().__init__(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, normalize_advantage, ent_coef, vf_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)
        
        self.epsilon_schedule = epsilon_schedule

    def predict(self, 
        observation: np.ndarray | Dict[str, np.ndarray], 
        state: th.Tuple[np.ndarray] | None = None, 
        episode_start: np.ndarray | None = None, 
        deterministic: bool = False
    ) -> th.Tuple[np.ndarray | th.Tuple[np.ndarray] | None]:
        
        epsilon = self.epsilon_schedule(self._current_progress_remaining)
        
        if random.random() < epsilon:
            return self.env.action_space.sample(), state
        
        return super().predict(observation, state, episode_start, deterministic)

    
