import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import (
    StandardScaler
)

from xgboost import XGBRegressor

import joblib

# =====================================================
# PATHS
# =====================================================

DATA_PATH = (
    "data/final_ml_dataset/"
    "combined_bp_features.csv"
)

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

print("\nloading dataset...\n")

df = pd.read_csv(DATA_PATH)

print(df.head())

print("\nrows:", len(df))

# =====================================================
# REMOVE MISSING
# =====================================================

df = df.dropna()

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
# TARGETS
# =====================================================

X = df[FEATURE_COLUMNS]

y_sbp = df["sbp"]

y_dbp = df["dbp"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

(
    X_train,
    X_test,

    y_sbp_train,
    y_sbp_test,

    y_dbp_train,
    y_dbp_test

) = train_test_split(

    X,

    y_sbp,

    y_dbp,

    test_size=0.1,

    random_state=42
)

print("\ntrain rows:", len(X_train))

print("test rows:", len(X_test))

# =====================================================
# SCALE
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =====================================================
# SBP MODEL
# =====================================================

print("\ntraining SBP XGBoost...\n")

sbp_model = XGBRegressor(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42
)

sbp_model.fit(

    X_train_scaled,

    y_sbp_train
)

# =====================================================
# DBP MODEL
# =====================================================

print("\ntraining DBP XGBoost...\n")

dbp_model = XGBRegressor(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42
)

dbp_model.fit(

    X_train_scaled,

    y_dbp_train
)

# =====================================================
# PREDICTIONS
# =====================================================

sbp_preds = sbp_model.predict(
    X_test_scaled
)

dbp_preds = dbp_model.predict(
    X_test_scaled
)

# =====================================================
# METRICS
# =====================================================

def evaluate_model(

    y_true,

    y_pred,

    label
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print("\n========================")
    print(label)
    print("========================")

    print(
        f"MAE  : {mae:.2f}"
    )

    print(
        f"RMSE : {rmse:.2f}"
    )

    print(
        f"R2   : {r2:.2f}"
    )

# =====================================================
# RESULTS
# =====================================================

evaluate_model(

    y_sbp_test,

    sbp_preds,

    "SBP XGBOOST"
)

evaluate_model(

    y_dbp_test,

    dbp_preds,

    "DBP XGBOOST"
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({

    "feature": FEATURE_COLUMNS,

    "importance":
        sbp_model.feature_importances_
})

importance = importance.sort_values(

    by="importance",

    ascending=False
)

print("\n========================")
print("FEATURE IMPORTANCE")
print("========================\n")

print(importance)

# =====================================================
# SAVE
# =====================================================

joblib.dump(

    sbp_model,

    MODEL_DIR / "xgb_sbp_model.pkl"
)

joblib.dump(

    dbp_model,

    MODEL_DIR / "xgb_dbp_model.pkl"
)

joblib.dump(

    scaler,

    MODEL_DIR / "xgb_scaler.pkl"
)

print("\n========================")
print("XGBOOST MODELS SAVED")
print("========================")