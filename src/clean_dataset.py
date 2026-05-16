import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# ----------------------------------------
# load dataset
# ----------------------------------------

df = pd.read_csv("data/dataset_cleaned.csv")

# ----------------------------------------
# rename columns
# ----------------------------------------

df.columns = [
    "timestamp",
    "ecg",
    "lead_status",
    "ir",
    "body_temp",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z"
]

# ----------------------------------------
# convert timestamp
# ----------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["time_sec"] = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds()

# ----------------------------------------
# remove duplicate timestamps
# ----------------------------------------

df = df.drop_duplicates(subset="timestamp")

# ----------------------------------------
# remove disconnected leads
# ----------------------------------------

df = df[df["lead_status"] == 1]

# ----------------------------------------
# remove invalid temperature values
# ----------------------------------------

df = df[
    (df["body_temp"] > 20) &
    (df["body_temp"] < 45)
]

# ----------------------------------------
# remove missing values
# ----------------------------------------

df = df.dropna()

# ----------------------------------------
# remove outliers using IQR
# ----------------------------------------

numeric_cols = [
    "ecg",
    "ir",
    "body_temp",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z"
]

for col in numeric_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df = df[
        (df[col] >= lower) &
        (df[col] <= upper)
    ]

# ----------------------------------------
# estimate sampling frequency
# ----------------------------------------

sampling_intervals = np.diff(df["time_sec"])

fs = 1 / np.mean(sampling_intervals)

print("sampling frequency:", fs)

# ----------------------------------------
# bandpass filter functions
# ----------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return b, a


def apply_filter(signal, lowcut, highcut, fs):

    b, a = butter_bandpass(
        lowcut,
        highcut,
        fs
    )

    filtered = filtfilt(b, a, signal)

    return filtered

# ----------------------------------------
# filter ecg
# ----------------------------------------

df["ecg_filtered"] = apply_filter(
    df["ecg"],
    lowcut=0.5,
    highcut=40,
    fs=fs
)

# ----------------------------------------
# filter ppg
# ----------------------------------------

df["ppg_filtered"] = apply_filter(
    df["ir"],
    lowcut=0.5,
    highcut=8,
    fs=fs
)

# ----------------------------------------
# motion magnitude
# ----------------------------------------

df["accel_magnitude"] = np.sqrt(
    df["accel_x"]**2 +
    df["accel_y"]**2 +
    df["accel_z"]**2
)

df["gyro_magnitude"] = np.sqrt(
    df["gyro_x"]**2 +
    df["gyro_y"]**2 +
    df["gyro_z"]**2
)

# ----------------------------------------
# normalize signals
# ----------------------------------------

def normalize(x):

    return (
        (x - np.mean(x)) /
        np.std(x)
    )

df["ecg_norm"] = normalize(
    df["ecg_filtered"]
)

df["ppg_norm"] = normalize(
    df["ppg_filtered"]
)

# ----------------------------------------
# save cleaned dataset
# ----------------------------------------

df.to_csv(
    "cleaned_output.csv",
    index=False
)

print("cleaning complete")
print(df.head())