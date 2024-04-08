from torch.utils.data.dataset import Dataset
import numpy as np 
import pathlib


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations = [], expert_actions = [], reward = [], next_state = []):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)
    
    def add_to_dataset(self, obs, action):

        self.observations.append(obs)
        self.actions.append(action)        
    
    def store_dataset(self, file_path : str = "../data/rollout/dataset.npz"):
        
        with open(file_path, "w") as _:
            np.savez_compressed(
                file_path, 
                obs=self.observations, 
                action=self.actions, 
            )
    
    def load_dataset(self, file_path : str = "../data/rollout/dataset.npz"):
        
        with np.load(file_path) as data:
            self.observations = data['obs']
            self.actions = data['action']
