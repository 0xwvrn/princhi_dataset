import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import (
    GradientBoostingRegressor
)

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

df = df.dropna()

print(df.head())

print("\nrows:", len(df))

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
# INPUT / TARGETS
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
# FEATURE SCALING
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

print("\ntraining SBP GBR...\n")

sbp_model = GradientBoostingRegressor(

    n_estimators=300,

    learning_rate=0.03,

    max_depth=4,

    subsample=0.8,

    random_state=42
)

sbp_model.fit(

    X_train_scaled,

    y_sbp_train
)

# =====================================================
# DBP MODEL
# =====================================================

print("\ntraining DBP GBR...\n")

dbp_model = GradientBoostingRegressor(

    n_estimators=300,

    learning_rate=0.03,

    max_depth=4,

    subsample=0.8,

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
# METRICS FUNCTION
# =====================================================

def evaluate_model(

    true_values,

    predictions,

    label
):

    mae = mean_absolute_error(

        true_values,

        predictions
    )

    rmse = np.sqrt(

        mean_squared_error(

            true_values,

            predictions
        )
    )

    r2 = r2_score(

        true_values,

        predictions
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
# EVALUATE
# =====================================================

evaluate_model(

    y_sbp_test,

    sbp_preds,

    "SBP GBR RESULTS"
)

evaluate_model(

    y_dbp_test,

    dbp_preds,

    "DBP GBR RESULTS"
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

print("\n========================")
print("FEATURE IMPORTANCE")
print("========================\n")

importance = pd.DataFrame({

    "feature": FEATURE_COLUMNS,

    "importance":
        sbp_model.feature_importances_
})

importance = importance.sort_values(

    by="importance",

    ascending=False
)

print(importance)

# =====================================================
# SAVE MODELS
# =====================================================

joblib.dump(

    sbp_model,

    MODEL_DIR / "gbr_sbp_model.pkl"
)

joblib.dump(

    dbp_model,

    MODEL_DIR / "gbr_dbp_model.pkl"
)

joblib.dump(

    scaler,

    MODEL_DIR / "gbr_scaler.pkl"
)

print("\n========================")
print("MODELS SAVED")
print("========================")

print(
    "\nmodels/"
)

print(
    "- gbr_sbp_model.pkl"
)

print(
    "- gbr_dbp_model.pkl"
)

print(
    "- gbr_scaler.pkl"
)