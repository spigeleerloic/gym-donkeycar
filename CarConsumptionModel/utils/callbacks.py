from tqdm.rich import tqdm
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback


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
            unit_scale=True,
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
        self.pbar.close()


class SaveObservations(BaseCallback):
    """
    Callback for saving observations from the environment.
    """

    def __init__(self, verbose: int = 0, save_file="../dataset/observation.csv"):
        super(SaveObservations, self).__init__(verbose)

        # open file for writing
        self.save_file = save_file
        self.save_file = open(self.save_file, "w")
        self.save_file.write("obs,reward,done,info\n")
        

    def _on_step(self) -> bool:
        # Save observations
        obs = self.model.env.get_observations()
        reward = self.model.env.get_reward()
        done = self.model.env.get_done()
        info = self.model.env.get_info()

        self.save_file.write(f"{obs},{reward},{done},{info}\n")
        return True
    