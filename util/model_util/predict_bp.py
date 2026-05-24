import pandas as pd

import joblib

# =====================================================
# LOAD MODELS
# =====================================================

print("\nloading models...\n")

# ==========================================
# RANDOM FOREST
# ==========================================

rf_sbp = joblib.load(
    "models/sbp_model.pkl"
)

rf_dbp = joblib.load(
    "models/dbp_model.pkl"
)

rf_scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================================
# XGBOOST
# ==========================================

xgb_sbp = joblib.load(
    "models/xgb_sbp_model.pkl"
)

xgb_dbp = joblib.load(
    "models/xgb_dbp_model.pkl"
)

xgb_scaler = joblib.load(
    "models/xgb_scaler.pkl"
)

# ==========================================
# GRADIENT BOOSTING
# ==========================================

gbr_sbp = joblib.load(
    "models/gbr_sbp_model.pkl"
)

gbr_dbp = joblib.load(
    "models/gbr_dbp_model.pkl"
)

gbr_scaler = joblib.load(
    "models/gbr_scaler.pkl"
)

# =====================================================
# FEATURES
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
# SAMPLE INPUT
# =====================================================

sample = {

    "pat_mean_ms": 180,

    "pat_std_ms": 22,

    "heart_rate_bpm": 78,

    "sdnn": 55,

    "rmssd": 42,

    "ppg_amplitude": 2200,

    "pulse_width_ms": 260,

    "accel_motion": 9.95,

    "gyro_motion": 0.04,

    "temperature": 33.5
}

# =====================================================
# DATAFRAME
# =====================================================

input_df = pd.DataFrame(
    [sample]
)

input_df = input_df[
    FEATURE_COLUMNS
]

# =====================================================
# RANDOM FOREST
# =====================================================

rf_scaled = rf_scaler.transform(
    input_df
)

rf_sbp_pred = rf_sbp.predict(
    rf_scaled
)[0]

rf_dbp_pred = rf_dbp.predict(
    rf_scaled
)[0]

# =====================================================
# XGBOOST
# =====================================================

xgb_scaled = xgb_scaler.transform(
    input_df
)

xgb_sbp_pred = xgb_sbp.predict(
    xgb_scaled
)[0]

xgb_dbp_pred = xgb_dbp.predict(
    xgb_scaled
)[0]

# =====================================================
# GBR
# =====================================================

gbr_scaled = gbr_scaler.transform(
    input_df
)

gbr_sbp_pred = gbr_sbp.predict(
    gbr_scaled
)[0]

gbr_dbp_pred = gbr_dbp.predict(
    gbr_scaled
)[0]

# =====================================================
# OUTPUT
# =====================================================

print("\n================================")
print("BP PREDICTIONS")
print("================================")

print("\nRandomForest")

print(
    f"SBP: {rf_sbp_pred:.1f} mmHg"
)

print(
    f"DBP: {rf_dbp_pred:.1f} mmHg"
)

print("\nXGBoost")

print(
    f"SBP: {xgb_sbp_pred:.1f} mmHg"
)

print(
    f"DBP: {xgb_dbp_pred:.1f} mmHg"
)

print("\nGradientBoosting")

print(
    f"SBP: {gbr_sbp_pred:.1f} mmHg"
)

print(
    f"DBP: {gbr_dbp_pred:.1f} mmHg"
)