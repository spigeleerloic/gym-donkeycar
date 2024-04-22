import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

class EpsilonGreedyWrapperClass(VecEnvWrapper):

    def __init__(self, env, epsilon, **kwargs):
        super(EpsilonGreedyWrapperClass, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.env = env

    def set_pretrained_model(self, pretrained_model):
        self.pretrained_model = pretrained_model

    def reset(self):
        return self.env.reset()
    
    def step_wait(self):
        return self.env.step_wait()

    def step(self, action):
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.pretrained_model.predict(self.env.get_observation())
        return self.env.step(action)