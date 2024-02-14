import numpy as np
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper

env = ConsumptionWrapper("steep-ascent")
# PLAY
obs = env.reset()
done = False
while not done:
    steering = 0.05
    throttle = 1.0
    action = np.array([steering, throttle])
    obs, reward, done, info = env.step(action)
    # obs shape: (120, 160, 3)
