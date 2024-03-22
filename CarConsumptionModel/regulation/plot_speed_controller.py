import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

df = pd.read_csv("../donkey_environment/ziegler_nichols_speed_iterations_0.25.csv")


target_speed = df["target_speed"].values[0]
unique_kp = df["kp"].unique()

# mkdir -p ../plots/ziegler_nichols/speed_controller_08 if not exists
directory_path = "../plots/ziegler_nichols/speed_controller_0.25"
if not Path(directory_path).exists():
    Path(directory_path).mkdir(parents=True)

for i, kp in enumerate(unique_kp):
    df_kp = df[df["kp"] == kp]
    plt.plot(df_kp["iterations"], df_kp["observed_speed"], label="observed_speed")
    plt.plot(df_kp["iterations"], np.ones(len(df_kp["iterations"])) * target_speed, label="target speed")
    plt.title(f"kp = {kp}")
    plt.legend()
    plt.savefig(f"{directory_path}/observed_speed_{kp}.png")
    plt.show()