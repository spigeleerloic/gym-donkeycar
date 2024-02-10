from stable_baselines3.common.callbacks import BaseCallback
import wandb
import datetime
import os


class LogCallback(BaseCallback):
    def __init__(self, verbose=0, use_wandb=True, use_logs=True):
        super(LogCallback, self).__init__(verbose)
        self.total_episode_reward = 0
        self.episode_length = 0
        self.episode_count = 0

        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.define_metric("episode_reward", step_metric="episode_count")
            wandb.define_metric("episode_length", step_metric="episode_count")

        self.use_logs = use_logs

        self.logs_dir = "./logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        current_datetime = datetime.datetime.now()
        self.filename = f"{self.logs_dir}/{current_datetime.strftime('%Y%m%d%H%M')}.csv"
        
        if self.use_logs:
            print(f"Logging to {self.filename}")
            with open(self.filename, "w") as file:
                ...
                #file.write("episode_count,episode_reward,episode_length\n")


    def _on_step(self) -> bool:
        # Accumulate the reward obtained at each step to calculate total episode reward
        reward = self.locals['rewards'][0]
        if self.use_wandb:
            wandb.log({"rewards": reward})

        if self.locals['dones'][0]:

            if self.use_wandb:
                wandb.log({"episode_reward": self.total_episode_reward, "episode_count":self.episode_count})
                wandb.log({"episode_length" : self.episode_length, "episode_count": self.episode_count})

            self.total_episode_reward = 0
            self.episode_length = 0
            self.episode_count += 1
        else:
            self.total_episode_reward += reward
            self.episode_length += 1

        # Accumulate the reward obtained at each step to calculate total episode reward
        return True
    
    def _write_to_csv(self) -> None:
        with open(self.filename, "a") as file:
            ...

class SaveModelCallback(BaseCallback):
    def __init__(self, model, verbose=0, save_freq=10, model_name="ppo_consumption_model"):  # 'save_freq' determines after how many episodes to save
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.episode_count = 0
        self.model_name = model_name
        self.model = model

    def _on_step(self) -> bool:

        if self.locals['dones'][0]:
            self.episode_count += 1
        if self.episode_count % self.save_freq == 0:
            # Save the model after every 'save_freq' episodes
            self.model.save(f"../models/{self.model_name}/checkpoint_{self.episode_count}")

        return True
