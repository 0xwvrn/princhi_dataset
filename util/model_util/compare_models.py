import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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
    RandomForestRegressor
)

from sklearn.ensemble import (
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

# =====================================================
# LOAD DATA
# =====================================================

DATA_PATH = (
    "data/final_ml_dataset/"
    "combined_bp_features.csv"
)

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

X = df[FEATURE_COLUMNS]

y_sbp = df["sbp"]

y_dbp = df["dbp"]

# =====================================================
# SPLIT
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
# MODELS
# =====================================================

models = {

    "RandomForest": {

        "sbp": RandomForestRegressor(

            n_estimators=300,

            max_depth=10,

            min_samples_split=4,

            min_samples_leaf=2,

            random_state=42
        ),

        "dbp": RandomForestRegressor(

            n_estimators=300,

            max_depth=10,

            min_samples_split=4,

            min_samples_leaf=2,

            random_state=42
        )
    },

    "XGBoost": {

        "sbp": XGBRegressor(

            n_estimators=300,

            max_depth=5,

            learning_rate=0.03,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="reg:squarederror",

            random_state=42
        ),

        "dbp": XGBRegressor(

            n_estimators=300,

            max_depth=5,

            learning_rate=0.03,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="reg:squarederror",

            random_state=42
        )
    },

    "GradientBoosting": {

        "sbp": GradientBoostingRegressor(

            n_estimators=300,

            learning_rate=0.03,

            max_depth=4,

            subsample=0.8,

            random_state=42
        ),

        "dbp": GradientBoostingRegressor(

            n_estimators=300,

            learning_rate=0.03,

            max_depth=4,

            subsample=0.8,

            random_state=42
        )
    }
}

# =====================================================
# RESULTS STORAGE
# =====================================================

results = []

# =====================================================
# TRAIN + EVALUATE
# =====================================================

for model_name, model_pair in models.items():

    print("\n================================")
    print(model_name)
    print("================================")

    # =============================================
    # TRAIN
    # =============================================

    model_pair["sbp"].fit(

        X_train_scaled,

        y_sbp_train
    )

    model_pair["dbp"].fit(

        X_train_scaled,

        y_dbp_train
    )

    # =============================================
    # PREDICT
    # =============================================

    sbp_preds = model_pair["sbp"].predict(
        X_test_scaled
    )

    dbp_preds = model_pair["dbp"].predict(
        X_test_scaled
    )

    # =============================================
    # METRICS
    # =============================================

    sbp_mae = mean_absolute_error(
        y_sbp_test,
        sbp_preds
    )

    sbp_rmse = np.sqrt(
        mean_squared_error(
            y_sbp_test,
            sbp_preds
        )
    )

    sbp_r2 = r2_score(
        y_sbp_test,
        sbp_preds
    )

    dbp_mae = mean_absolute_error(
        y_dbp_test,
        dbp_preds
    )

    dbp_rmse = np.sqrt(
        mean_squared_error(
            y_dbp_test,
            dbp_preds
        )
    )

    dbp_r2 = r2_score(
        y_dbp_test,
        dbp_preds
    )

    # =============================================
    # PRINT
    # =============================================

    print("\nSBP")

    print(
        f"MAE  : {sbp_mae:.2f}"
    )

    print(
        f"RMSE : {sbp_rmse:.2f}"
    )

    print(
        f"R2   : {sbp_r2:.2f}"
    )

    print("\nDBP")

    print(
        f"MAE  : {dbp_mae:.2f}"
    )

    print(
        f"RMSE : {dbp_rmse:.2f}"
    )

    print(
        f"R2   : {dbp_r2:.2f}"
    )

    # =============================================
    # SAVE
    # =============================================

    results.append({

        "model": model_name,

        "sbp_mae": sbp_mae,

        "sbp_rmse": sbp_rmse,

        "sbp_r2": sbp_r2,

        "dbp_mae": dbp_mae,

        "dbp_rmse": dbp_rmse,

        "dbp_r2": dbp_r2
    })

# =====================================================
# RESULTS DATAFRAME
# =====================================================

results_df = pd.DataFrame(
    results
)

print("\n================================")
print("FINAL COMPARISON")
print("================================\n")

print(results_df)

# =====================================================
# SBP R2 PLOT
# =====================================================

plt.figure(figsize=(8, 5))

plt.bar(

    results_df["model"],

    results_df["sbp_r2"]
)

plt.ylabel("SBP R2")

plt.title(
    "SBP Model Comparison"
)

plt.grid(True)

plt.show()

# =====================================================
# DBP R2 PLOT
# =====================================================

plt.figure(figsize=(8, 5))

plt.bar(

    results_df["model"],

    results_df["dbp_r2"]
)

plt.ylabel("DBP R2")

plt.title(
    "DBP Model Comparison"
)

plt.grid(True)

plt.show()

# =====================================================
# SAVE RESULTS
# =====================================================

results_df.to_csv(

    "model_comparison_results.csv",

    index=False
)

print("\nresults saved:")
print(
    "model_comparison_results.csv"
)