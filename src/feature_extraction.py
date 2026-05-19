import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# ----------------------------------------
# load dataset
# ----------------------------------------

df = pd.read_csv("data/cleaned_output.csv")

# ----------------------------------------
# signals
# ----------------------------------------

time = df["time_sec"].values

ecg = df["ecg_filtered"].values
ppg = df["ppg_filtered"].values

accel = df["accel_magnitude"].values
gyro = df["gyro_magnitude"].values

temp_signal = df["temp"].values

# ----------------------------------------
# estimate sampling frequency
# ----------------------------------------

fs = 1 / np.mean(np.diff(time))

print(f"sampling frequency: {fs:.2f} Hz")

# ----------------------------------------
# window settings
# ----------------------------------------

window_sec = 5
overlap = 0.5

window_samples = int(window_sec * fs)

step_size = int(
    window_samples * (1 - overlap)
)

# ----------------------------------------
# feature rows
# ----------------------------------------

feature_rows = []

# ----------------------------------------
# process each window
# ----------------------------------------

for start in range(
    0,
    len(df) - window_samples,
    step_size
):

    end = start + window_samples

    # ----------------------------------------
    # extract window
    # ----------------------------------------

    t_win = time[start:end]

    ecg_win = ecg[start:end]
    ppg_win = ppg[start:end]

    accel_win = accel[start:end]
    gyro_win = gyro[start:end]

    temp_win = temp_signal[start:end]

    # ----------------------------------------
    # ECG peak detection
    # ----------------------------------------

    ecg_distance = int(fs * 0.25)

    ecg_peaks, _ = find_peaks(
        ecg_win,
        distance=max(1, ecg_distance),
        prominence=np.std(ecg_win) * 0.5
    )

    # ----------------------------------------
    # PPG peak detection
    # ----------------------------------------

    ppg_distance = int(fs * 0.4)

    ppg_peaks, _ = find_peaks(
        ppg_win,
        distance=max(1, ppg_distance),
        prominence=np.std(ppg_win) * 0.3
    )

    # ----------------------------------------
    # skip bad windows
    # ----------------------------------------

    if len(ecg_peaks) < 2:
        continue

    if len(ppg_peaks) < 2:
        continue

    # ----------------------------------------
    # peak times
    # ----------------------------------------

    ecg_peak_times = t_win[ecg_peaks]
    ppg_peak_times = t_win[ppg_peaks]

    # ----------------------------------------
    # PAT calculation
    # ----------------------------------------

    pat_values = []

    for r_time in ecg_peak_times:

        future_ppg = ppg_peak_times[
            ppg_peak_times > r_time
        ]

        if len(future_ppg) == 0:
            continue

        ppg_time = future_ppg[0]

        pat = (
            ppg_time - r_time
        ) * 1000

        if 50 < pat < 400:
            pat_values.append(pat)

    # ----------------------------------------
    # skip if no valid PAT
    # ----------------------------------------

    if len(pat_values) == 0:
        continue

    # ----------------------------------------
    # heart rate
    # ----------------------------------------

    rr_intervals = np.diff(
        ecg_peak_times
    )

    if len(rr_intervals) == 0:
        continue

    heart_rates = 60 / rr_intervals

    heart_rate = np.mean(
        heart_rates
    )

    # ----------------------------------------
    # HRV
    # ----------------------------------------

    rr_ms = rr_intervals * 1000

    sdnn = np.std(rr_ms)

    if len(rr_ms) > 1:

        rmssd = np.sqrt(
            np.mean(
                np.diff(rr_ms) ** 2
            )
        )

    else:
        rmssd = 0

    # ----------------------------------------
    # PPG amplitude
    # ----------------------------------------

    ppg_amplitudes = ppg_win[
        ppg_peaks
    ]

    ppg_amp_mean = np.mean(
        ppg_amplitudes
    )

    # ----------------------------------------
    # pulse width
    # ----------------------------------------

    pulse_widths = []

    for peak in ppg_peaks:

        half_height = (
            ppg_win[peak] / 2
        )

        left = peak
        right = peak

        while (
            left > 0 and
            ppg_win[left] > half_height
        ):
            left -= 1

        while (
            right < len(ppg_win) - 1 and
            ppg_win[right] > half_height
        ):
            right += 1

        width = (
            right - left
        ) / fs * 1000

        pulse_widths.append(width)

    pulse_width_mean = np.mean(
        pulse_widths
    )

    # ----------------------------------------
    # motion features
    # ----------------------------------------

    accel_motion = np.mean(
        accel_win
    )

    gyro_motion = np.mean(
        gyro_win
    )

    # ----------------------------------------
    # temperature
    # ----------------------------------------

    temp_mean = np.mean(
        temp_win
    )

    # ----------------------------------------
    # final feature row
    # ----------------------------------------

    feature_row = {

        "window_start_sec":
            t_win[0],

        "window_end_sec":
            t_win[-1],

        "pat_mean_ms":
            np.mean(pat_values),

        "pat_std_ms":
            np.std(pat_values),

        "heart_rate_bpm":
            heart_rate,

        "sdnn":
            sdnn,

        "rmssd":
            rmssd,

        "ppg_amplitude":
            ppg_amp_mean,

        "pulse_width_ms":
            pulse_width_mean,

        "accel_motion":
            accel_motion,

        "gyro_motion":
            gyro_motion,

        "temperature":
            temp_mean
    }

    feature_rows.append(
        feature_row
    )

# ----------------------------------------
# final dataframe
# ----------------------------------------

features_df = pd.DataFrame(
    feature_rows
)

# ----------------------------------------
# save dataset
# ----------------------------------------

features_df.to_csv(
    "training_features.csv",
    index=False
)

# ----------------------------------------
# output
# ----------------------------------------

print("\nfeature extraction complete\n")

print(features_df.head())

print(
    f"\nTotal feature rows: "
    f"{len(features_df)}"
)