import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

df = pd.read_csv("../data/ziegler_nichols/pid_tuning.csv")


target_speed = df["target_speed"].values[0]
unique_kp = df["kp"].unique()

# for each target speed find peaks
with open("../data/ziegler_nichols/kp_oscillation_period_best_pid.csv", "w") as f:
    f.write("kp,oscillation_period\n")
    for i, kp in enumerate(unique_kp):
        df_kp = df[df["kp"] == kp]

        local_maximums = find_peaks( df_kp["observed_speed"].values)[0]
        local_maximums = local_maximums[1:] # remove first peak

        local_minimums = find_peaks( -df_kp["observed_speed"].values)[0]
        local_minimums = local_minimums[1:] # remove first peak

        # take timestamps between each local maximum and take the mean
        oscillation_periods = []
        for i in range(len(local_maximums)-1):

            first_index = local_maximums[i]
            next_index = local_maximums[i+1]

            start = df.iloc[first_index]
            end = df.iloc[next_index]

            # bad csv formatting (forgot ",")
            start_time = float(start["time"].strip().split(" ")[0])
            end_time = float(end["time"].strip().split(" ")[0])

            oscillation_periods.append(end_time - start_time)

        #print(oscillation_periods)
        rounded_kp = round(kp, 4)
        rounded_mean_oscillation_period = round(np.mean(oscillation_periods), 4)
        print(f"kp : {rounded_kp} with oscillation period of : {rounded_mean_oscillation_period}s")
        f.write(f"{kp},{np.mean(oscillation_periods)}\n")


