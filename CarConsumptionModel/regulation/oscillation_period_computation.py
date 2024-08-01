import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

#df = pd.read_csv("../data/ziegler_nichols/pid_tuning.csv")
df = pd.read_csv("../data/ziegler_nichols/ziegler_nichols_speed_iterations_1.0.csv")

target_speed = df["target_speed"].values[0]
unique_kp = df["kp"].unique()


# for each target speed find peaks
# df['time'] = df['time'].str.strip()  # Remove leading and trailing spaces
# df['time'] = df['time'].str.split().str[0]  # Take the first part of the split string
# # Convert the cleaned 'time' column to datetime
# df['time'] = df['time'].astype(float)

for i, kp in enumerate(unique_kp):
    df_kp = df[df["kp"] == kp]

    smoothed_amplitude = savgol_filter(df_kp["observed_speed"].values, window_length=51, polyorder=3)
    #smoothed_amplitude = df_kp["observed_distance"].values
    peaks, _ = find_peaks(smoothed_amplitude)
    # remove first peak
    peaks = peaks[1:]

    minimal_time = df_kp["time"].min()
    df_kp["time"] = df_kp["time"] - minimal_time

    peak_times = df_kp["time"].values[peaks]
    print(*zip(peaks, peak_times))
    periods = np.diff(peak_times)   

    average_period = np.mean(periods)

    print(f"{kp=},{average_period=}\n")

    if kp > 7.0:
        break


