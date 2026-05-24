import pandas as pd
import numpy as np

from scipy.signal import butter, filtfilt

from pathlib import Path

# ====================================================
# BASE DATA DIRECTORY
# ====================================================

BASE_DIR = Path("data")

# ====================================================
# FIND ALL DATASET FOLDERS
# ====================================================

dataset_dirs = [

    folder for folder in BASE_DIR.iterdir()

    if (
        folder.is_dir() and
        folder.name.startswith("driver_twin_dataset")
    )
]

print(f"\nfound {len(dataset_dirs)} dataset folders\n")

# ====================================================
# FILTER FUNCTIONS
# ====================================================

def butter_bandpass(
    lowcut,
    highcut,
    fs,
    order=4
):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    if high >= 1:
        high = 0.99

    if low >= high:
        raise ValueError(
            "invalid filter frequencies"
        )

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return b, a


def apply_filter(
    signal,
    lowcut,
    highcut,
    fs
):

    b, a = butter_bandpass(
        lowcut,
        highcut,
        fs
    )

    return filtfilt(
        b,
        a,
        signal
    )


def normalize(x):

    std = np.std(x)

    if std == 0:
        return np.zeros(len(x))

    return (
        (x - np.mean(x)) / std
    )

# ====================================================
# PROCESS EACH DATASET FOLDER
# ====================================================

for dataset_dir in dataset_dirs:

    print("\n===================================")
    print(f"dataset folder: {dataset_dir.name}")
    print("===================================")

    # ------------------------------------------------
    # RAW DIRECTORY
    # ------------------------------------------------

    raw_dir = dataset_dir / "raw"

    if not raw_dir.exists():

        print("raw folder missing")
        continue

    # ------------------------------------------------
    # PROCESSED DIRECTORY
    # ------------------------------------------------

    processed_dir = (
        dataset_dir / "processed_raw"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------
    # FIND CSV FILES
    # ------------------------------------------------

    csv_files = list(
        raw_dir.glob("*.csv")
    )

    print(
        f"found {len(csv_files)} csv files"
    )

    # ------------------------------------------------
    # PROCESS FILES
    # ------------------------------------------------

    for csv_file in csv_files:

        try:

            print("\n-----------------------")
            print(
                "processing:",
                csv_file.name
            )

            # ========================================
            # LOAD
            # ========================================

            df = pd.read_csv(
                csv_file
            )

            # ========================================
            # SORT TIMESTAMPS
            # ========================================

            df = df.sort_values(
                "timestamp_us"
            )

            # ========================================
            # REMOVE DUPLICATES
            # ========================================

            df = df.drop_duplicates(
                subset="timestamp_us"
            )

            # ========================================
            # TIME IN SECONDS
            # ========================================

            df["time_sec"] = (

                df["timestamp_us"] -
                df["timestamp_us"].iloc[0]

            ) / 1_000_000

            # ========================================
            # SAMPLING RATE
            # ========================================

            sampling_intervals = np.diff(
                df["time_sec"]
            )

            sampling_intervals = (

                sampling_intervals[
                    sampling_intervals > 0
                ]
            )

            if len(sampling_intervals) == 0:

                print(
                    "invalid timestamps"
                )

                continue

            fs = 1 / np.mean(
                sampling_intervals
            )

            print(
                "sampling frequency:",
                round(fs, 2),
                "Hz"
            )

            # ========================================
            # REMOVE BAD LEADS
            # ========================================

            df = df[
                df["leads"] == 1
            ]

            # ========================================
            # REMOVE BAD TEMP
            # ========================================

            df = df[
                (df["temp"] > 20) &
                (df["temp"] < 45)
            ]

            # ========================================
            # DROP NaNs
            # ========================================

            df = df.dropna()

            # ========================================
            # NUMERIC COLUMNS
            # ========================================

            numeric_cols = [

                "ecg",
                "ir",
                "temp",

                "accX",
                "accY",
                "accZ",

                "gyroX",
                "gyroY",
                "gyroZ"
            ]

            # ========================================
            # OUTLIER CLIPPING
            # ========================================

            for col in numeric_cols:

                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)

                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                df[col] = df[col].clip(
                    lower,
                    upper
                )

            # ========================================
            # ENSURE ENOUGH SAMPLES
            # ========================================

            if len(df) < 20:

                print(
                    "too few samples"
                )

                continue

            # ========================================
            # FILTER SETTINGS
            # ========================================

            if fs > 80:

                ecg_highcut = min(
                    40,
                    fs / 2 - 1
                )

                ppg_highcut = min(
                    8,
                    fs / 2 - 1
                )

            else:

                ecg_highcut = min(
                    10,
                    fs / 2 - 1
                )

                ppg_highcut = min(
                    4,
                    fs / 2 - 1
                )

            # ========================================
            # FILTER ECG
            # ========================================

            df["ecg_filtered"] = apply_filter(

                df["ecg"],

                lowcut=0.5,

                highcut=ecg_highcut,

                fs=fs
            )

            # ========================================
            # FILTER PPG
            # ========================================

            df["ppg_filtered"] = apply_filter(

                df["ir"],

                lowcut=0.5,

                highcut=ppg_highcut,

                fs=fs
            )

            # ========================================
            # MOTION FEATURES
            # ========================================

            df["accel_magnitude"] = np.sqrt(

                df["accX"]**2 +
                df["accY"]**2 +
                df["accZ"]**2
            )

            df["gyro_magnitude"] = np.sqrt(

                df["gyroX"]**2 +
                df["gyroY"]**2 +
                df["gyroZ"]**2
            )

            # ========================================
            # NORMALIZATION
            # ========================================

            df["ecg_norm"] = normalize(
                df["ecg_filtered"]
            )

            df["ppg_norm"] = normalize(
                df["ppg_filtered"]
            )

            # ========================================
            # SAVE OUTPUT
            # ========================================

            output_path = (

                processed_dir /

                f"processed_{csv_file.name}"
            )

            df.to_csv(
                output_path,
                index=False
            )

            print(
                "saved:",
                output_path
            )

            print(
                "final rows:",
                len(df)
            )

        except Exception as e:

            print(
                "\nfailed:",
                csv_file.name
            )

            print(e)

print("\n===================================")
print("all dataset processing complete")
print("===================================")