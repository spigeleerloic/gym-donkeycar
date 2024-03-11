import os
import argparse
import uuid
import wandb

from stable_baselines3 import PPO
from donkey_environment.ConsumptionWrapper import ConsumptionWrapper
from utils.utils import LogToWandb

script_dir = os.path.dirname(__file__)
#default_path = os.path.join(script_dir, '../../../simulator/linux_build.x86_64')
default_path = os.path.join(script_dir, '../../../testing_gym/donkey_sim.exe')
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

parser.add_argument("--logs", help="Whether to use logs for the training",
                     action="store_false", dest="logs")


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
if args.logs:
    wandb.login()
    wandb.init(project="donkey-car", name=args.model_name)
logToWandb = LogToWandb()

model_directory = "..\models"

model_path = os.path.join(model_directory, args.model_name)
model_path = os.path.join(model_path, "model.zip")
try:
    # Check if the path exists
    if os.path.exists(model_path):
        print(f"path exists: {model_path}")
        # Load your model here
        model = PPO.load(model_path)
        for i in range(10):
            obs = env.reset()
            rewards = []
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                if args.logs:
                    logToWandb.on_step(obs, reward, done, info)
                rewards.append(reward)
                # obs shape: (120, 160, 3)
            print(f"Total steps: {len(rewards)}")
            print(f"Total reward: {sum(rewards)}")
            print(f"rewards : {rewards}")
    else:
        print("Model path does not exist.")
except PermissionError as e:
    print("Permission denied error:", e)

