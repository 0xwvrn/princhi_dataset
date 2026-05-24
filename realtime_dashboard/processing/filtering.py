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
# ACCEL MAGNITUDE
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

# =====================================================
# GYRO MAGNITUDE
# =====================================================

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
# FULL REALTIME PROCESSING
# =====================================================

def process_window(

    window,

    fs=250
):

    try:

        ecg_filtered = filter_ecg(

            window["ecg"],

            fs
        )

        ppg_filtered = filter_ppg(

            window["ir"],

            fs
        )

        ecg_norm = normalize_signal(
            ecg_filtered
        )

        ppg_norm = normalize_signal(
            ppg_filtered
        )

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
                ppg_filtered,

            "ecg_norm":
                ecg_norm,

            "ppg_norm":
                ppg_norm,

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

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    fs = 250

    t = np.linspace(

        0,

        8,

        fs * 8
    )

    fake_ecg = (

        np.sin(

            2 * np.pi * 1.2 * t
        ) * 1000
    )

    fake_ppg = (

        np.sin(

            2 * np.pi * 1.2 * t
        ) * 5000
    )

    fake_window = {

        "timestamp": t,

        "ecg": fake_ecg,

        "ir": fake_ppg,

        "accX":
            np.random.randn(len(t)),

        "accY":
            np.random.randn(len(t)),

        "accZ":
            np.random.randn(len(t)),

        "gyroX":
            np.random.randn(len(t)),

        "gyroY":
            np.random.randn(len(t)),

        "gyroZ":
            np.random.randn(len(t)),

        "temp":
            np.ones(len(t)) * 33.5
    }

    processed = process_window(
        fake_window
    )

    print(
        "processed keys:"
    )

    print(
        processed.keys()
    )