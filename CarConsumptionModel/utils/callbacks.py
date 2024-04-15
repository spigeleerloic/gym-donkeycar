from tqdm.rich import tqdm
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from wandb.integration.sb3 import WandbCallback
import wandb

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
        
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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
    
    def _on_step(self) -> bool:
        return super()._on_step()


    def on_rollout_start(self) -> None:
        self.env.viewer.handler.blocking_send(self.resume_message)

    
    def on_rollout_end(self) -> None: 
        self.env.viewer.handler.blocking_send(self.pause_message)