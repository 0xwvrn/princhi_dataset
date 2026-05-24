import pandas as pd

from pathlib import Path

import re

# =====================================================
# BASE DATA DIRECTORY
# =====================================================

BASE_DIR = Path("data")

# =====================================================
# FIND DATASET FOLDERS
# =====================================================

dataset_dirs = [

    folder for folder in BASE_DIR.iterdir()

    if (
        folder.is_dir() and
        folder.name.startswith(
            "driver_twin_dataset"
        )
    )
]

print(
    f"\nfound {len(dataset_dirs)} dataset folders\n"
)

# =====================================================
# STORE ALL DATAFRAMES
# =====================================================

all_feature_dataframes = []

# =====================================================
# PROCESS EACH DATASET
# =====================================================

for dataset_dir in dataset_dirs:

    print("\n====================================")
    print(
        f"dataset: {dataset_dir.name}"
    )
    print("====================================")

    # -------------------------------------------------
    # BP LABEL FILE
    # -------------------------------------------------

    bp_label_file = (
        dataset_dir / "bp_labels.csv"
    )

    if not bp_label_file.exists():

        print(
            "bp_labels.csv missing"
        )

        continue

    # -------------------------------------------------
    # FEATURE DATA FOLDER
    # -------------------------------------------------

    feature_data_dir = (
        dataset_dir / "feature_data"
    )

    if not feature_data_dir.exists():

        print(
            "feature_data folder missing"
        )

        continue

    # -------------------------------------------------
    # LOAD BP LABELS
    # -------------------------------------------------

    bp_df = pd.read_csv(
        bp_label_file
    )

    # -------------------------------------------------
    # NORMALIZE SESSION IDS
    # -------------------------------------------------

    def normalize_session_id(x):

        numbers = re.findall(
            r"\d+",
            str(x)
        )

        if len(numbers) == 0:
            return None

        return (
            f"{int(numbers[-1]):03d}"
        )

    bp_df["session_suffix"] = (

        bp_df["session_id"]

        .apply(
            normalize_session_id
        )
    )

    # -------------------------------------------------
    # KEEP ONLY TARGETS
    # -------------------------------------------------

    bp_df = bp_df[[
        "session_suffix",
        "sbp",
        "dbp"
    ]]

    # -------------------------------------------------
    # FIND FEATURE FILES
    # -------------------------------------------------

    feature_files = list(
        feature_data_dir.glob("*.csv")
    )

    print(
        f"found {len(feature_files)} feature files"
    )

    # -------------------------------------------------
    # PROCESS FEATURE FILES
    # -------------------------------------------------

    for feature_file in feature_files:

        try:

            print("\n--------------------------")
            print(
                f"processing: "
                f"{feature_file.name}"
            )

            # =========================================
            # SKIP EMPTY FILES
            # =========================================

            if (
                feature_file.stat().st_size == 0
            ):

                print(
                    "empty file"
                )

                continue

            # =========================================
            # LOAD FEATURES
            # =========================================

            features_df = pd.read_csv(
                feature_file
            )

            if len(features_df) == 0:

                print(
                    "empty dataframe"
                )

                continue

            # =========================================
            # EXTRACT SESSION NUMBER
            # =========================================

            numbers = re.findall(
                r"\d+",
                feature_file.stem
            )

            if len(numbers) == 0:

                print(
                    "no session number found"
                )

                continue

            session_suffix = (
                f"{int(numbers[-1]):03d}"
            )

            # =========================================
            # ADD SESSION SUFFIX
            # =========================================

            features_df[
                "session_suffix"
            ] = session_suffix

            # =========================================
            # MERGE TARGETS
            # =========================================

            merged_df = pd.merge(

                features_df,

                bp_df,

                on="session_suffix",

                how="left"
            )

            # =========================================
            # REMOVE UNMATCHED
            # =========================================

            merged_df = merged_df.dropna(
                subset=["sbp", "dbp"]
            )

            if len(merged_df) == 0:

                print(
                    "no BP match"
                )

                continue

            # =========================================
            # REMOVE METADATA COLUMNS
            # =========================================

            columns_to_remove = [

                "session_id",
                "window_start_sec",
                "window_end_sec",
                "session_suffix",
                "condition",
                "notes",
                "dataset_source"
            ]

            existing_columns = [

                col for col in columns_to_remove

                if col in merged_df.columns
            ]

            merged_df = merged_df.drop(
                columns=existing_columns
            )

            # =========================================
            # STORE
            # =========================================

            all_feature_dataframes.append(
                merged_df
            )

            print(
                f"final rows: "
                f"{len(merged_df)}"
            )

        except Exception as e:

            print(
                f"\nfailed: "
                f"{feature_file.name}"
            )

            print(e)

# =====================================================
# FINAL DATASET
# =====================================================

if len(all_feature_dataframes) == 0:

    print(
        "\nno datasets merged"
    )

else:

    final_dataset = pd.concat(

        all_feature_dataframes,

        ignore_index=True
    )

    # -------------------------------------------------
    # OUTPUT DIRECTORY
    # -------------------------------------------------

    output_dir = (
        BASE_DIR / "final_ml_dataset"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    output_file = (

        output_dir /

        "combined_bp_features.csv"
    )

    final_dataset.to_csv(
        output_file,
        index=False
    )

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------

    print("\n====================================")
    print("FINAL ML DATASET CREATED")
    print("====================================")

    print(
        f"\nsaved:\n{output_file}"
    )

    print(
        f"\nrows: "
        f"{len(final_dataset)}"
    )

    print(
        f"\ncolumns:\n"
    )

    print(
        list(final_dataset.columns)
    )

    print("\npreview:\n")

    print(
        final_dataset.head()
    )