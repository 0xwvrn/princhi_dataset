import pandas as pd

import joblib

from pathlib import Path

# =====================================================
# MODEL PATHS
# =====================================================

MODEL_DIR = Path("../models")

# =====================================================
# LOAD MODELS
# =====================================================

print("\nloading models...\n")

# =============================================
# RANDOM FOREST
# =============================================

rf_sbp_model = joblib.load(
    MODEL_DIR / "sbp_model.pkl"
)

rf_dbp_model = joblib.load(
    MODEL_DIR / "dbp_model.pkl"
)

rf_scaler = joblib.load(
    MODEL_DIR / "scaler.pkl"
)

# =============================================
# XGBOOST
# =============================================

xgb_sbp_model = joblib.load(
    MODEL_DIR / "xgb_sbp_model.pkl"
)

xgb_dbp_model = joblib.load(
    MODEL_DIR / "xgb_dbp_model.pkl"
)

xgb_scaler = joblib.load(
    MODEL_DIR / "xgb_scaler.pkl"
)

# =============================================
# GRADIENT BOOSTING
# =============================================

gbr_sbp_model = joblib.load(
    MODEL_DIR / "gbr_sbp_model.pkl"
)

gbr_dbp_model = joblib.load(
    MODEL_DIR / "gbr_dbp_model.pkl"
)

gbr_scaler = joblib.load(
    MODEL_DIR / "gbr_scaler.pkl"
)

print("models loaded\n")

# =====================================================
# FEATURE ORDER
# =====================================================

FEATURE_COLUMNS = [

    "pat_mean_ms",

    "pat_std_ms",

    "heart_rate_bpm",

    "sdnn",

    "rmssd",

    "ppg_amplitude",

    "pulse_width_ms",

    "accel_motion",

    "gyro_motion",

    "temperature"
]

# =====================================================
# BP STATUS
# =====================================================

def get_bp_status(

    sbp,

    dbp
):

    if sbp < 90 or dbp < 60:

        return "Low BP"

    elif sbp < 120 and dbp < 80:

        return "Normal"

    elif sbp < 130 and dbp < 85:

        return "Elevated"

    elif sbp < 140 or dbp < 90:

        return "Stage 1 Hypertension"

    else:

        return "Stage 2 Hypertension"

# =====================================================
# PREDICT
# =====================================================

def predict_bp(

    features,

    model_type="xgboost"
):

    try:

        # =============================================
        # DATAFRAME
        # =============================================

        input_df = pd.DataFrame(
            [features]
        )

        input_df = input_df[
            FEATURE_COLUMNS
        ]

        # =============================================
        # MODEL SELECT
        # =============================================

        if model_type == "random_forest":

            scaler = rf_scaler

            sbp_model = rf_sbp_model

            dbp_model = rf_dbp_model

        elif model_type == "gradient_boosting":

            scaler = gbr_scaler

            sbp_model = gbr_sbp_model

            dbp_model = gbr_dbp_model

        else:

            scaler = xgb_scaler

            sbp_model = xgb_sbp_model

            dbp_model = xgb_dbp_model

        # =============================================
        # SCALE
        # =============================================

        scaled_input = scaler.transform(
            input_df
        )

        # =============================================
        # PREDICT
        # =============================================

        sbp = sbp_model.predict(
            scaled_input
        )[0]

        dbp = dbp_model.predict(
            scaled_input
        )[0]

        # =============================================
        # ROUND
        # =============================================

        sbp = round(float(sbp), 1)

        dbp = round(float(dbp), 1)

        # =============================================
        # STATUS
        # =============================================

        status = get_bp_status(
            sbp,
            dbp
        )

        # =============================================
        # RETURN
        # =============================================

        return {

            "sbp": sbp,

            "dbp": dbp,

            "status": status,

            "model": model_type
        }

    except Exception as e:

        print(
            "prediction error:",
            e
        )

        return None

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    fake_features = {

        "pat_mean_ms": 180,

        "pat_std_ms": 22,

        "heart_rate_bpm": 78,

        "sdnn": 55,

        "rmssd": 42,

        "ppg_amplitude": 2200,

        "pulse_width_ms": 260,

        "accel_motion": 9.8,

        "gyro_motion": 0.04,

        "temperature": 33.5
    }

    prediction = predict_bp(

        fake_features,

        model_type="xgboost"
    )

    print("\nprediction:\n")

    print(prediction)