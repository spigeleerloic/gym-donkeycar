import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

df = pd.read_csv("../data/ziegler_nichols/pid_tuning.csv")

target_speed = df["target_speed"].values[0]
unique_kp = df["kp"].unique()

# mkdir -p ../plots/ziegler_nichols/speed_controller_08 if not exists
directory_path = "../plots/ziegler_nichols/pid_controller"
if not Path(directory_path).exists():
    Path(directory_path).mkdir(parents=True)

for i, kp in enumerate(unique_kp):
    df_kp = df[df["kp"] == kp]
    # subplot with 2 rows and 1 column
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    dataframe_length = len(df_kp["time"])

    ax[0].plot(df_kp["time"], df_kp["observed_speed"], label="observed_speed")
    ax[0].plot(df_kp["time"], np.ones(dataframe_length) * target_speed, label="target speed")
    ax[0].plot(df_kp["time"], np.zeros(dataframe_length), label="zero speed")
    ax[0].set_title(f"observed speed during tuning")
    ax[0].legend()

    ax[1].plot(df_kp["time"], df_kp["observed_distance"], label="observed_distance")
    ax[1].plot(df_kp["time"], np.zeros(len(df_kp["time"])), label="target distance")
    ax[1].set_title(f"observed distance from middle line with kp = {kp}")
    ax[1].legend()

    plt.savefig(f"{directory_path}/pid_tuning_{kp}.png")
    plt.show()