import pandas as pd
import numpy as np

from scipy.signal import (
    find_peaks,
    savgol_filter
)

from pathlib import Path

# =====================================================
# BASE DIRECTORY
# =====================================================

BASE_DIR = Path("data")

# =====================================================
# DATASET FOLDERS
# =====================================================

dataset_dirs = [

    folder for folder in BASE_DIR.iterdir()

    if (
        folder.is_dir() and
        folder.name.startswith(
            "driver_twin_dataset"
        )
    )
]

print(f"\nfound {len(dataset_dirs)} datasets\n")

# =====================================================
# PROCESS DATASETS
# =====================================================

for dataset_dir in dataset_dirs:

    print("\n================================")
    print(dataset_dir.name)
    print("================================")

    processed_raw_dir = (
        dataset_dir / "processed_raw"
    )

    if not processed_raw_dir.exists():
        continue

    output_dir = (
        dataset_dir / "feature_data"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_files = list(
        processed_raw_dir.glob("*.csv")
    )

    for processed_file in processed_files:

        try:

            print(
                f"\nprocessing: "
                f"{processed_file.name}"
            )

            # =========================================
            # LOAD
            # =========================================

            df = pd.read_csv(
                processed_file
            )

            if len(df) < 120:
                continue

            # =========================================
            # SIGNALS
            # =========================================

            time = df[
                "time_sec"
            ].values

            ecg = df[
                "ecg_filtered"
            ].values

            ppg = df[
                "ppg_filtered"
            ].values

            accel = df[
                "accel_magnitude"
            ].values

            gyro = df[
                "gyro_magnitude"
            ].values

            temp_signal = df[
                "temp"
            ].values

            # =========================================
            # SAMPLING RATE
            # =========================================

            fs = 1 / np.mean(
                np.diff(time)
            )

            print(
                f"fs = {fs:.2f} Hz"
            )

            # =========================================
            # REJECT EXTREMELY LOW FS
            # =========================================

            if fs < 8:

                print(
                    "sampling rate too low"
                )

                continue

            # =========================================
            # SMOOTH SIGNALS
            # =========================================

            smooth_window = max(
                5,
                int(fs * 0.12)
            )

            if smooth_window % 2 == 0:
                smooth_window += 1

            ecg = savgol_filter(
                ecg,
                smooth_window,
                3
            )

            ppg = savgol_filter(
                ppg,
                smooth_window,
                3
            )

            # =========================================
            # WINDOW SETTINGS
            # =========================================

            window_sec = 6

            step_sec = 3

            window_samples = int(
                window_sec * fs
            )

            step_samples = int(
                step_sec * fs
            )

            feature_rows = []

            # =========================================
            # WINDOW LOOP
            # =========================================

            for start in range(

                0,

                len(df) - window_samples,

                max(1, step_samples)
            ):

                end = (
                    start +
                    window_samples
                )

                t_win = time[start:end]

                ecg_win = ecg[start:end]

                ppg_win = ppg[start:end]

                accel_win = accel[start:end]

                gyro_win = gyro[start:end]

                temp_win = temp_signal[start:end]

                # =====================================
                # MOTION FILTERING
                # =====================================

                accel_std = np.std(
                    accel_win
                )

                gyro_std = np.std(
                    gyro_win
                )

                if accel_std > 1.5:
                    continue

                if gyro_std > 1.0:
                    continue

                # =====================================
                # SIGNAL QUALITY
                # =====================================

                if np.std(ecg_win) < 3:
                    continue

                if np.std(ppg_win) < 5:
                    continue

                # =====================================
                # ECG PEAKS
                # =====================================

                ecg_peaks, _ = find_peaks(

                    ecg_win,

                    distance=max(
                        1,
                        int(fs * 0.4)
                    ),

                    prominence=np.std(
                        ecg_win
                    ) * 0.8
                )

                if len(ecg_peaks) < 3:
                    continue

                ecg_times = t_win[
                    ecg_peaks
                ]

                # =====================================
                # RR INTERVALS
                # =====================================

                rr_ms = np.diff(
                    ecg_times
                ) * 1000

                rr_ms = rr_ms[
                    (
                        rr_ms > 450
                    ) &
                    (
                        rr_ms < 1500
                    )
                ]

                if len(rr_ms) < 2:
                    continue

                # =====================================
                # REMOVE RR OUTLIERS
                # =====================================

                rr_med = np.median(
                    rr_ms
                )

                rr_ms = rr_ms[
                    (
                        rr_ms >
                        rr_med * 0.7
                    ) &
                    (
                        rr_ms <
                        rr_med * 1.3
                    )
                ]

                if len(rr_ms) < 2:
                    continue

                # =====================================
                # HEART RATE
                # =====================================

                heart_rate = (
                    60000 /
                    np.mean(rr_ms)
                )

                if (
                    heart_rate < 45 or
                    heart_rate > 130
                ):
                    continue

                # =====================================
                # HRV
                # =====================================

                sdnn = np.std(
                    rr_ms
                )

                rmssd = np.sqrt(

                    np.mean(
                        np.diff(rr_ms) ** 2
                    )
                )

                if sdnn > 180:
                    continue

                if rmssd > 250:
                    continue

                # =====================================
                # PPG PEAKS
                # =====================================

                ppg_peaks, _ = find_peaks(

                    ppg_win,

                    distance=max(
                        1,
                        int(fs * 0.4)
                    ),

                    prominence=np.std(
                        ppg_win
                    ) * 0.4
                )

                if len(ppg_peaks) < 3:
                    continue

                ppg_times = t_win[
                    ppg_peaks
                ]

                # =====================================
                # PAT EXTRACTION
                # =====================================

                pat_values = []

                used_ppg = set()

                for r_time in ecg_times:

                    valid = np.where(

                        (
                            ppg_times >
                            r_time + 0.08
                        ) &

                        (
                            ppg_times <
                            r_time + 0.35
                        )

                    )[0]

                    if len(valid) == 0:
                        continue

                    idx = valid[0]

                    if idx in used_ppg:
                        continue

                    used_ppg.add(idx)

                    pat = (

                        ppg_times[idx] -
                        r_time

                    ) * 1000

                    if (
                        80 <= pat <= 300
                    ):
                        pat_values.append(
                            pat
                        )

                if len(pat_values) < 2:
                    continue

                pat_values = np.array(
                    pat_values
                )

                pat_mean = np.mean(
                    pat_values
                )

                pat_std = np.std(
                    pat_values
                )

                if pat_std < 0.5:
                    continue

                if pat_std > 50:
                    continue

                # =====================================
                # PPG AMPLITUDE
                # =====================================

                ppg_amp = (
                    np.max(ppg_win) -
                    np.min(ppg_win)
                )

                if (
                    ppg_amp < 100 or
                    ppg_amp > 20000
                ):
                    continue

                # =====================================
                # PULSE WIDTH
                # =====================================

                pulse_width_ms = (
                    np.mean(rr_ms) * 0.35
                )

                if (
                    pulse_width_ms < 60 or
                    pulse_width_ms > 450
                ):
                    continue

                # =====================================
                # STORE FEATURES
                # =====================================

                feature_rows.append({

                    "pat_mean_ms":
                        float(pat_mean),

                    "pat_std_ms":
                        float(pat_std),

                    "heart_rate_bpm":
                        float(heart_rate),

                    "sdnn":
                        float(sdnn),

                    "rmssd":
                        float(rmssd),

                    "ppg_amplitude":
                        float(ppg_amp),

                    "pulse_width_ms":
                        float(
                            pulse_width_ms
                        ),

                    "accel_motion":
                        float(
                            np.mean(
                                accel_win
                            )
                        ),

                    "gyro_motion":
                        float(
                            np.mean(
                                gyro_win
                            )
                        ),

                    "temperature":
                        float(
                            np.mean(
                                temp_win
                            )
                        )
                })

            # =========================================
            # SAVE
            # =========================================

            features_df = pd.DataFrame(
                feature_rows
            )

            # =========================================
            # REMOVE DUPLICATES
            # =========================================

            features_df = (
                features_df
                .drop_duplicates()
            )

            if len(features_df) == 0:

                print(
                    "no valid rows"
                )

                continue

            output_file = (

                output_dir /

                f"features_{processed_file.name}"
            )

            features_df.to_csv(
                output_file,
                index=False
            )

            print(
                f"saved "
                f"{len(features_df)} rows"
            )

        except Exception as e:

            print(
                f"failed: "
                f"{processed_file.name}"
            )

            print(e)

print("\n================================")
print("feature extraction complete")
print("================================")