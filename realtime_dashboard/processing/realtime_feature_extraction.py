import numpy as np

from scipy.signal import find_peaks

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(

    processed_window,

    fs=250
):

    ecg = processed_window[
        "ecg_filtered"
    ]

    ppg = processed_window[
        "ppg_filtered"
    ]

    red = processed_window[
        "red_filtered"
    ]

    accel = processed_window[
        "accel_magnitude"
    ]

    gyro = processed_window[
        "gyro_magnitude"
    ]

    temp = processed_window[
        "temperature"
    ]

    timestamps = processed_window[
        "timestamp"
    ]

    # =============================================
    # ECG PEAKS
    # =============================================

    ecg_peaks, _ = find_peaks(

        ecg,

        distance=int(fs * 0.4),

        prominence=np.std(ecg) * 0.3
    )

    if len(ecg_peaks) < 2:
        return None

    ecg_times = (

        timestamps[ecg_peaks] -
        timestamps[0]

    ) / 1_000_000

    rr_ms = np.diff(
        ecg_times
    ) * 1000

    rr_ms = rr_ms[
        (
            rr_ms > 400
        ) &
        (
            rr_ms < 1500
        )
    ]

    if len(rr_ms) < 1:
        return None

    # =============================================
    # HEART RATE
    # =============================================

    heart_rate = (

        60000 /

        np.mean(rr_ms)
    )

    # =============================================
    # HRV
    # =============================================

    sdnn = np.std(
        rr_ms
    )

    rmssd = np.sqrt(

        np.mean(
            np.diff(rr_ms) ** 2
        )
    )

    # =============================================
    # PPG PEAKS
    # =============================================

    ppg_peaks, _ = find_peaks(

        ppg,

        distance=int(fs * 0.4),

        prominence=np.std(ppg) * 0.3
    )

    if len(ppg_peaks) < 2:
        return None

    ppg_times = (

        timestamps[ppg_peaks] -
        timestamps[0]

    ) / 1_000_000

    # =============================================
    # PAT
    # =============================================

    pat_values = []

    for r_time in ecg_times:

        valid = np.where(

            (
                ppg_times >
                r_time + 0.08
            ) &

            (
                ppg_times <
                r_time + 0.4
            )

        )[0]

        if len(valid) == 0:
            continue

        pat = (

            ppg_times[valid[0]] -
            r_time

        ) * 1000

        pat_values.append(
            pat
        )

    if len(pat_values) < 1:
        return None

    pat_values = np.array(
        pat_values
    )

    pat_mean = np.mean(
        pat_values
    )

    pat_std = np.std(
        pat_values
    )

    # =============================================
    # PPG AMPLITUDE
    # =============================================

    ppg_amplitude = (

        np.max(ppg) -
        np.min(ppg)
    )

    # =============================================
    # PULSE WIDTH
    # =============================================

    pulse_width_ms = (
        np.mean(rr_ms) * 0.35
    )

    # =============================================
    # REAL SPO2
    # =============================================

    dc_red = np.mean(red)

    dc_ir = np.mean(ppg)

    ac_red = np.std(red)

    ac_ir = np.std(ppg)

    ratio = (

        (ac_red / dc_red) /

        (ac_ir / dc_ir)
    )

    spo2 = 110 - (25 * ratio)

    spo2 = np.clip(
        spo2,
        70,
        100
    )

    # =============================================
    # RETURN FEATURES
    # =============================================

    return {

        "pat_mean_ms":
            float(pat_mean),

        "pat_std_ms":
            float(pat_std),

        "heart_rate_bpm":
            float(heart_rate),

        "spo2":
            float(spo2),

        "sdnn":
            float(sdnn),

        "rmssd":
            float(rmssd),

        "ppg_amplitude":
            float(ppg_amplitude),

        "pulse_width_ms":
            float(
                pulse_width_ms
            ),

        "accel_motion":
            float(
                np.mean(accel)
            ),

        "gyro_motion":
            float(
                np.mean(gyro)
            ),

        "temperature":
            float(
                np.mean(temp)
            )
    }