import numpy as np

from scipy.signal import (
    butter,
    filtfilt
)

# =====================================================
# BUTTER BANDPASS
# =====================================================

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

# =====================================================
# APPLY FILTER
# =====================================================

def apply_bandpass(

    signal,

    lowcut,

    highcut,

    fs,

    order=4
):

    try:

        b, a = butter_bandpass(

            lowcut,

            highcut,

            fs,

            order
        )

        filtered = filtfilt(

            b,

            a,

            signal
        )

        return filtered

    except Exception as e:

        print(
            "filter error:",
            e
        )

        return signal

# =====================================================
# ECG FILTER
# =====================================================

def filter_ecg(

    ecg_signal,

    fs=250
):

    return apply_bandpass(

        signal=ecg_signal,

        lowcut=0.5,

        highcut=40,

        fs=fs,

        order=4
    )

# =====================================================
# PPG FILTER
# =====================================================

def filter_ppg(

    ppg_signal,

    fs=250
):

    return apply_bandpass(

        signal=ppg_signal,

        lowcut=0.5,

        highcut=8,

        fs=fs,

        order=4
    )

# =====================================================
# NORMALIZE
# =====================================================

def normalize_signal(signal):

    signal = np.array(signal)

    std = np.std(signal)

    if std == 0:

        return np.zeros(
            len(signal)
        )

    return (

        signal -

        np.mean(signal)

    ) / std

# =====================================================
# MAGNITUDES
# =====================================================

def accel_magnitude(

    accX,

    accY,

    accZ
):

    return np.sqrt(

        accX**2 +

        accY**2 +

        accZ**2
    )

def gyro_magnitude(

    gyroX,

    gyroY,

    gyroZ
):

    return np.sqrt(

        gyroX**2 +

        gyroY**2 +

        gyroZ**2
    )

# =====================================================
# FULL PROCESSING
# =====================================================

def process_window(

    window,

    fs=250
):

    try:

        # =============================================
        # FILTER
        # =============================================

        ecg_filtered = filter_ecg(

            window["ecg"],

            fs
        )

        ir_filtered = filter_ppg(

            window["ir"],

            fs
        )

        red_filtered = filter_ppg(

            window["red"],

            fs
        )

        # =============================================
        # NORMALIZE
        # =============================================

        ecg_norm = normalize_signal(
            ecg_filtered
        )

        ir_norm = normalize_signal(
            ir_filtered
        )

        red_norm = normalize_signal(
            red_filtered
        )

        # =============================================
        # MOTION
        # =============================================

        accel_mag = accel_magnitude(

            window["accX"],

            window["accY"],

            window["accZ"]
        )

        gyro_mag = gyro_magnitude(

            window["gyroX"],

            window["gyroY"],

            window["gyroZ"]
        )

        return {

            "timestamp":
                window["timestamp"],

            "ecg_filtered":
                ecg_filtered,

            "ppg_filtered":
                ir_filtered,

            "red_filtered":
                red_filtered,

            "ecg_norm":
                ecg_norm,

            "ppg_norm":
                ir_norm,

            "red_norm":
                red_norm,

            "accel_magnitude":
                accel_mag,

            "gyro_magnitude":
                gyro_mag,

            "temperature":
                window["temp"]
        }

    except Exception as e:

        print(
            "processing error:",
            e
        )

        return None