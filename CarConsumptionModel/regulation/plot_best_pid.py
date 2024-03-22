import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

df = pd.read_csv("../data/ziegler_nichols/best_parameters.csv")

target_speed = df["target_speed"].values[0]

# mkdir -p ../plots/ziegler_nichols/speed_controller_08 if not exists
directory_path = "../plots/ziegler_nichols/pid_controller"
if not Path(directory_path).exists():
    Path(directory_path).mkdir(parents=True)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
dataframe_length = len(df["time"])

ax[0].plot(df["time"], df["observed_speed"], label="observed_speed")
ax[0].plot(df["time"], np.ones(dataframe_length) * target_speed, label="target speed")
ax[0].plot(df["time"], np.zeros(dataframe_length), label="zero speed")
ax[0].set_title(f"observed speed")
ax[0].legend()

ax[1].plot(df["time"], df["observed_distance"], label="observed_distance")
ax[1].plot(df["time"], np.zeros(len(df["time"])), label="target distance")
ax[1].set_title(f"observed distance from middle line")
ax[1].legend()

plt.savefig(f"{directory_path}/best_pid.png")
plt.show()