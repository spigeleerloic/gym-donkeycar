import os
import argparse
import uuid

from stable_baselines3 import PPO
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper


script_dir = os.path.dirname(__file__)
default_path = os.path.join(script_dir, '../../../simulator/linux_build.x86_64')

if not os.path.exists(default_path):
    raise ValueError(f"Default path '{default_path}' does not exist or is inaccessible.")
else:
    print(f"Using default simulator path: {default_path}")


parser = argparse.ArgumentParser(description='RL algorithm with consumption model applied to donkey car')

parser.add_argument("--env_name", help="name of the donkey car environment", 
                    type=str, dest='environment', default="steep-ascent")
parser.add_argument("-p", "--path" , help="path to the simulator if it is not running", 
                    type=str, dest="path", default=default_path)

parser.add_argument("-n", "--name" , help="name of the model to train", type=str, dest="model_name", required=True)

parser.add_argument("--port", help="port in use for TCP connections",
                    default=9091, type=int, dest="port")


args = parser.parse_args()

environment = args.environment


conf = {
    "exe_path": args.path,
    "host": "127.0.0.1",
    "port": args.port,
    "body_style": "donkey",
    "body_rgb": (128, 128, 128),
    "car_name": args.model_name,
    "font_size": 100,
    "racer_name": "PPO",
    "country": "BEL",
    "bio": "Learning from experiences",
    "guid": str(uuid.uuid4()),
    "max_cte": 10,
}

env = ConsumptionWrapper(environment, conf=conf)

model = PPO.load()
# PLAY
obs = env.reset(f"../models/{args.model_name}")
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    # obs shape: (120, 160, 3)
