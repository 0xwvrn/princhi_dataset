import numpy as np

from scipy.signal import find_peaks

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(

    processed_window,

    fs=250
):

    try:

        # =============================================
        # SIGNALS
        # =============================================

        ecg = processed_window[
            "ecg_filtered"
        ]

        ppg = processed_window[
            "ppg_filtered"
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
        # SIGNAL QUALITY
        # =============================================

        if np.std(ecg) < 5:
            return None

        if np.std(ppg) < 20:
            return None

        # =============================================
        # ECG PEAKS
        # =============================================

        ecg_peaks, _ = find_peaks(

            ecg,

            distance=int(fs * 0.45),

            prominence=np.std(ecg) * 0.8
        )

        if len(ecg_peaks) < 5:
            return None

        # =============================================
        # ECG TIMES
        # =============================================

        ecg_times = (

            timestamps[ecg_peaks] -
            timestamps[0]

        ) / 1_000_000

        rr_ms = np.diff(
            ecg_times
        ) * 1000

        rr_ms = rr_ms[
            (
                rr_ms > 500
            ) &
            (
                rr_ms < 1400
            )
        ]

        if len(rr_ms) < 4:
            return None

        # =============================================
        # HEART RATE
        # =============================================

        heart_rate = (

            60000 /

            np.mean(rr_ms)
        )

        if (
            heart_rate < 45 or
            heart_rate > 140
        ):
            return None

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

            distance=int(fs * 0.45),

            prominence=np.std(ppg) * 0.5
        )

        if len(ppg_peaks) < 5:
            return None

        # =============================================
        # PPG TIMES
        # =============================================

        ppg_times = (

            timestamps[ppg_peaks] -
            timestamps[0]

        ) / 1_000_000

        # =============================================
        # PAT
        # =============================================

        pat_values = []

        used_ppg = set()

        for r_time in ecg_times:

            valid = np.where(

                (
                    ppg_times >
                    r_time + 0.10
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
                100 <= pat <= 300
            ):
                pat_values.append(
                    pat
                )

        if len(pat_values) < 3:
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
        # MOTION
        # =============================================

        accel_motion = np.mean(
            accel
        )

        gyro_motion = np.mean(
            gyro
        )

        # =============================================
        # TEMPERATURE
        # =============================================

        temperature = np.mean(
            temp
        )

        # =============================================
        # FINAL FEATURES
        # =============================================

        features = {

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
                float(ppg_amplitude),

            "pulse_width_ms":
                float(
                    pulse_width_ms
                ),

            "accel_motion":
                float(
                    accel_motion
                ),

            "gyro_motion":
                float(
                    gyro_motion
                ),

            "temperature":
                float(
                    temperature
                )
        }

        return features

    except Exception as e:

        print(
            "feature extraction error:",
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

    ecg = (

        np.sin(
            2 * np.pi * 1.2 * t
        ) * 1000
    )

    ppg = (

        np.sin(
            2 * np.pi * 1.2 * (t - 0.2)
        ) * 5000
    )

    fake_processed = {

        "timestamp":
            np.arange(
                len(t)
            ) * 4000,

        "ecg_filtered":
            ecg,

        "ppg_filtered":
            ppg,

        "accel_magnitude":
            np.ones(len(t)) * 9.8,

        "gyro_magnitude":
            np.ones(len(t)) * 0.04,

        "temperature":
            np.ones(len(t)) * 33.5
    }

    features = extract_features(
        fake_processed
    )

    print("\nfeatures:\n")

    print(features)