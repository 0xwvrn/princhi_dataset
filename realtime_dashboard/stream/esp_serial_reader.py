import serial
import json
import time

# =====================================================
# CONFIG
# =====================================================

SERIAL_PORT = "COM5"

BAUD_RATE = 921600

# =====================================================
# CONNECT
# =====================================================

print("\nconnecting to ESP32...\n")

ser = serial.Serial(

    SERIAL_PORT,

    BAUD_RATE,

    timeout=1
)

time.sleep(2)

print("connected\n")

# =====================================================
# READ LOOP
# =====================================================

while True:

    try:

        # =============================================
        # READ LINE
        # =============================================

        raw_line = ser.readline()

        if not raw_line:
            continue

        # =============================================
        # DECODE
        # =============================================

        line = raw_line.decode(

            "utf-8",

            errors="ignore"

        ).strip()

        if len(line) == 0:
            continue

        # =============================================
        # PARSE JSON
        # =============================================

        data = json.loads(line)

        # =============================================
        # ACCESS VALUES
        # =============================================

        timestamp = data["timestamp"]

        ecg = data["ecg"]

        ir = data["ir"]

        temp = data["temp"]

        # =============================================
        # PRINT
        # =============================================

        print(
            f"ECG: {ecg} | "
            f"IR: {ir} | "
            f"Temp: {temp}"
        )

    except json.JSONDecodeError:

        print(
            "invalid json packet"
        )

    except KeyboardInterrupt:

        print("\nstopped")

        ser.close()

        break

    except Exception as e:

        print(
            "error:",
            e
        )