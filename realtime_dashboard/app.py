import streamlit as st

import pandas as pd
import numpy as np

from collections import deque

# =====================================================
# IMPORTS
# =====================================================

from stream.esp_serial_reader import (
    connect_serial,
    read_sensor_packet
)

from processing.signal_buffer import (
    SignalBuffer
)

from processing.filtering import (
    process_window
)

from processing.realtime_feature_extraction import (
    extract_features
)

from processing.realtime_predictor import (
    predict_bp
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="Realtime BP Dashboard",

    layout="wide"
)

st.title(
    "Realtime Blood Pressure Monitoring"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(

    "Model",

    [

        "xgboost",

        "random_forest",

        "gradient_boosting"
    ]
)

serial_port = st.sidebar.text_input(

    "Serial Port",

    "COM5"
)

start_button = st.sidebar.button(
    "Start Streaming"
)

# =====================================================
# PLACEHOLDERS
# =====================================================

bp_placeholder = st.empty()

status_placeholder = st.empty()

feature_placeholder = st.empty()

chart_placeholder = st.empty()

# =====================================================
# START
# =====================================================

if start_button:

    # =============================================
    # SERIAL
    # =============================================

    ser = connect_serial(

        port=serial_port,

        baud_rate=921600
    )

    # =============================================
    # BUFFER
    # =============================================

    buffer = SignalBuffer(

        window_seconds=8,

        sampling_rate=250
    )

    # =============================================
    # CHART BUFFERS
    # =============================================

    ecg_chart = deque(
        maxlen=1000
    )

    ppg_chart = deque(
        maxlen=1000
    )

    # =============================================
    # LOOP
    # =============================================

    while True:

        # =========================================
        # READ SENSOR PACKET
        # =========================================

        sample = read_sensor_packet(
            ser
        )

        if sample is None:
            continue

        # =========================================
        # UPDATE BUFFER
        # =========================================

        buffer.add_sample(sample)

        # =========================================
        # UPDATE CHART DATA
        # =========================================

        ecg_chart.append(
            sample["ecg"]
        )

        ppg_chart.append(
            sample["ir"]
        )

        # =========================================
        # WAIT FOR FULL WINDOW
        # =========================================

        if not buffer.is_ready():
            continue

        # =========================================
        # GET WINDOW
        # =========================================

        window = buffer.get_window()

        # =========================================
        # FILTER
        # =========================================

        processed = process_window(
            window
        )

        if processed is None:
            continue

        # =========================================
        # FEATURES
        # =========================================

        features = extract_features(
            processed
        )

        if features is None:
            continue

        # =========================================
        # PREDICT BP
        # =========================================

        prediction = predict_bp(

            features,

            model_type=model_choice
        )

        if prediction is None:
            continue

        # =========================================
        # DISPLAY BP
        # =========================================

        bp_placeholder.metric(

            "Estimated Blood Pressure",

            f"{prediction['sbp']} / "
            f"{prediction['dbp']} mmHg"
        )

        # =========================================
        # STATUS
        # =========================================

        status_placeholder.success(

            f"Status: "
            f"{prediction['status']}"
        )

        # =========================================
        # FEATURES TABLE
        # =========================================

        feature_df = pd.DataFrame(

            [features]
        )

        feature_placeholder.dataframe(
            feature_df
        )

        # =========================================
        # CHARTS
        # =========================================

        chart_df = pd.DataFrame({

            "ECG":
                list(ecg_chart),

            "PPG":
                list(ppg_chart)
        })

        chart_placeholder.line_chart(
            chart_df
        )