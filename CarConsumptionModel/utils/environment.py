import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def change_env_space(env, new_low = [-1.0, -1.0], new_high = [1.0, 1.0]):
    """
    Change the action space of the environment to a new one
    with new_low and new_high as the new bounds.
    """
    env.action_space.low = np.array(new_low)
    env.action_space.high = np.array(new_high)
    return env

def change_model_action_space(model, new_low = [-1.0, -1.0], new_high = [1.0, 1.0]):
    """
    Change the action space of the model to a new one
    with new_low and new_high as the new bounds.
    """
    model.action_space.low = np.array(new_low)
    model.action_space.high = np.array(new_high)
    return model

def epsilon_greedy_actions(model, state, epsilon):
    """
    Choose an action using epsilon-greedy strategy
    """
    if np.random.rand() < epsilon:
        return model.action_space.sample()
    else:
        return model.predict(state)