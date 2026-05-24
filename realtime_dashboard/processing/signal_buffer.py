from collections import deque

import numpy as np

# =====================================================
# SIGNAL BUFFER
# =====================================================

class SignalBuffer:

    def __init__(

        self,

        window_seconds=8,

        sampling_rate=250
    ):

        self.window_seconds = (
            window_seconds
        )

        self.sampling_rate = (
            sampling_rate
        )

        self.max_samples = int(

            window_seconds *
            sampling_rate
        )

        # =============================================
        # BUFFERS
        # =============================================

        self.timestamps = deque(
            maxlen=self.max_samples
        )

        self.ecg = deque(
            maxlen=self.max_samples
        )

        self.ir = deque(
            maxlen=self.max_samples
        )

        self.red = deque(
            maxlen=self.max_samples
        )

        self.accX = deque(
            maxlen=self.max_samples
        )

        self.accY = deque(
            maxlen=self.max_samples
        )

        self.accZ = deque(
            maxlen=self.max_samples
        )

        self.gyroX = deque(
            maxlen=self.max_samples
        )

        self.gyroY = deque(
            maxlen=self.max_samples
        )

        self.gyroZ = deque(
            maxlen=self.max_samples
        )

        self.temp = deque(
            maxlen=self.max_samples
        )

    # =================================================
    # ADD SAMPLE
    # =================================================

    def add_sample(

        self,

        sample
    ):

        try:

            self.timestamps.append(
                sample["timestamp"]
            )

            self.ecg.append(
                sample["ecg"]
            )

            self.ir.append(
                sample["ir"]
            )

            self.red.append(
                sample["red"]
            )

            self.accX.append(
                sample["accX"]
            )

            self.accY.append(
                sample["accY"]
            )

            self.accZ.append(
                sample["accZ"]
            )

            self.gyroX.append(
                sample["gyroX"]
            )

            self.gyroY.append(
                sample["gyroY"]
            )

            self.gyroZ.append(
                sample["gyroZ"]
            )

            self.temp.append(
                sample["temp"]
            )

        except Exception as e:

            print(
                "buffer add error:",
                e
            )

    # =================================================
    # BUFFER READY
    # =================================================

    def is_ready(self):

        return (
            len(self.ecg) >=
            self.max_samples
        )

    # =================================================
    # GET WINDOW
    # =================================================

    def get_window(self):

        if not self.is_ready():

            return None

        return {

            "timestamp":
                np.array(
                    self.timestamps
                ),

            "ecg":
                np.array(
                    self.ecg
                ),

            "ir":
                np.array(
                    self.ir
                ),

            "red":
                np.array(
                    self.red
                ),

            "accX":
                np.array(
                    self.accX
                ),

            "accY":
                np.array(
                    self.accY
                ),

            "accZ":
                np.array(
                    self.accZ
                ),

            "gyroX":
                np.array(
                    self.gyroX
                ),

            "gyroY":
                np.array(
                    self.gyroY
                ),

            "gyroZ":
                np.array(
                    self.gyroZ
                ),

            "temp":
                np.array(
                    self.temp
                )
        }

    # =================================================
    # CLEAR
    # =================================================

    def clear(self):

        self.timestamps.clear()

        self.ecg.clear()

        self.ir.clear()

        self.red.clear()

        self.accX.clear()

        self.accY.clear()

        self.accZ.clear()

        self.gyroX.clear()

        self.gyroY.clear()

        self.gyroZ.clear()

        self.temp.clear()

    # =================================================
    # STATUS
    # =================================================

    def current_size(self):

        return len(self.ecg)

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    buffer = SignalBuffer()

    fake_sample = {

        "timestamp": 1,

        "ecg": 2048,

        "ir": 120000,

        "red": 118000,

        "accX": 0.1,

        "accY": 0.2,

        "accZ": 9.8,

        "gyroX": 0.01,

        "gyroY": 0.02,

        "gyroZ": 0.01,

        "temp": 33.5
    }

    for _ in range(2500):

        buffer.add_sample(
            fake_sample
        )

    print(
        "buffer ready:",
        buffer.is_ready()
    )

    print(
        "samples:",
        buffer.current_size()
    )

    window = buffer.get_window()

    print(
        "window keys:",
        window.keys()
    )