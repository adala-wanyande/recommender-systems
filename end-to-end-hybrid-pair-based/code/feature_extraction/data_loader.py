# data_loader.py

import pandas as pd
import numpy as np
import os
import textwrap # For pretty printing info() output

# Import configuration from config.py
import config

def load_and_merge_data(train_csv_path: str, metadata_csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train data and item metadata, merges them, and performs initial NaN handling.

    Args:
        train_csv_path (str): Path to the train.csv file.
        metadata_csv_path (str): Path to the cleaned item metadata CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing
        (train_df, clean_item_meta_df, train_merged_df) on success.

    Raises:
        FileNotFoundError: If input CSV files are not found.
        ValueError: If critical columns are missing from input files.
        Exception: For other errors during loading or merging.
    """
    print("\n--- Step 1 & 2: Load Core Data and Initial Merge ---")
    train_df = None
    clean_item_meta_df = None
    train_merged_df = None

    # --- Load User-Item Interaction Data (train.csv) ---
    print(f"\nLoading User-Item Interaction Data from {train_csv_path}...")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at {train_csv_path}")

    try:
        train_df = pd.read_csv(train_csv_path)
        print("Train data loaded successfully.")
        print(f"Train dataset shape: {train_df.shape}")

        # Ensure critical columns exist
        required_train_cols = [config.USER_ID_COL, config.ITEM_ID_COL, config.TIMESTAMP_COL]
        if not all(col in train_df.columns for col in required_train_cols):
             missing = [col for col in required_train_cols if col not in train_df.columns]
             raise ValueError(f"train.csv is missing required columns: {missing}")

        # Ensure user_id and item_id are string type
        train_df[config.USER_ID_COL] = train_df[config.USER_ID_COL].astype(str)
        train_df[config.ITEM_ID_COL] = train_df[config.ITEM_ID_COL].astype(str)
        print(f"Ensured '{config.USER_ID_COL}' and '{config.ITEM_ID_COL}' in train_df are string type.")

    except Exception as e:
        raise Exception(f"An error occurred during loading train.csv: {e}") from e


    # --- Loading Clean Item Metadata ---
    print(f"\nLoading clean item metadata from {metadata_csv_path}...")
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(f"Clean item metadata file not found at {metadata_csv_path}. Please ensure it is created first.")

    try:
        clean_item_meta_df = pd.read_csv(metadata_csv_path)
        print("Clean item metadata loaded successfully.")
        print(f"Clean item metadata shape: {clean_item_meta_df.shape}")

        # Ensure item_id in metadata is string and exists
        if config.ITEM_ID_COL not in clean_item_meta_df.columns:
             raise ValueError(f"'{config.ITEM_ID_COL}' column not found in clean item metadata.")
        clean_item_meta_df[config.ITEM_ID_COL] = clean_item_meta_df[config.ITEM_ID_COL].astype(str)
        print(f"Ensured '{config.ITEM_ID_COL}' in clean_item_meta_df is string type.")

    except Exception as e:
        raise Exception(f"An error occurred during metadata loading: {e}") from e


    # --- Merge metadata onto train data ---
    print("\nMerging clean item metadata onto train data...")
    try:
        train_merged_df = pd.merge(
            train_df, # Left DataFrame
            clean_item_meta_df, # Right DataFrame with item features
            on=config.ITEM_ID_COL, # Key column for merging
            how='left' # Keep all rows from train_df
        )
        print("Merge onto train data complete.")
        print(f"Merged train dataset shape: {train_merged_df.shape}")

        # Ensure 'interaction' column exists - it's the target variable.
        # If not, create it assuming all rows in train_df are positive interactions (before negative sampling).
        if config.INTERACTION_COL not in train_merged_df.columns:
             print(f"Warning: '{config.INTERACTION_COL}' column not found in merged data. Assuming all initial rows are positive (interaction=1).")
             train_merged_df[config.INTERACTION_COL] = 1 # Add the target column

    except Exception as e:
        raise Exception(f"An error occurred during merging train data and metadata: {e}") from e


    # --- Identify Feature Types and Initial NaN Handling on Merged Data ---
    # This step handles NaNs *before* generating user/item features or creating train_pairs.
    # NaNs in item features occur if an item_id from train_df was NOT present in the item_meta data.
    # 1. Numerical Columns: Impute NaNs using median from the *entire* merged data.
    # 2. Binary Columns: Impute NaNs with 0 (assuming missing means the feature is absent).
    # 3. Text Columns: Fill NaN with empty string "".
    # 4. Categorical/Details Columns: Fill NaN with a placeholder string ('Missing') *before* encoding.

    print("\nChecking and handling NaNs in merged train data...")

    # Identify details_cols dynamically in the merged dataframe
    details_cols = [col for col in train_merged_df.columns if col.startswith('details_')]

    # For Numerical Columns: Impute NaNs using median
    numerical_cols_in_df = [col for col in config.ITEM_NUMERICAL_COLS if col in train_merged_df.columns]
    for col in numerical_cols_in_df:
         if train_merged_df[col].isnull().sum() > 0:
              # Calculate median only if there's non-NaN data, otherwise use 0 as fallback
              median_val = train_merged_df[col].median() if not train_merged_df[col].isnull().all() else 0
              train_merged_df[col] = train_merged_df[col].fillna(median_val)
              # print(f"  Imputed NaNs in numerical column '{col}' using median ({median_val:.4f}).") # Optional print
         train_merged_df[col] = train_merged_df[col].astype(float) # Ensure float type

    # For Binary Columns: Impute NaNs with 0 and ensure int type
    binary_cols_in_df = [col for col in config.ITEM_BINARY_COLS if col in train_merged_df.columns]
    for col in binary_cols_in_df:
        if train_merged_df[col].isnull().sum() > 0:
            train_merged_df[col] = train_merged_df[col].fillna(0)
            # print(f"  Imputed NaNs in binary column '{col}' with 0.") # Optional print
        train_merged_df[col] = train_merged_df[col].astype(int)


    # For Text Columns ('title'): Fill NaN with empty string and ensure string type
    text_cols_in_df = [col for col in config.ITEM_TEXT_COLS if col in train_merged_df.columns]
    for col in text_cols_in_df:
         if train_merged_df[col].isnull().sum() > 0:
             train_merged_df[col] = train_merged_df[col].fillna('')
             # print(f"  Filled NaNs in text column '{col}' with empty string.") # Optional print
         train_merged_df[col] = train_merged_df[col].astype(str) # Ensure string type


    # For Categorical/Details Columns (store, details_*): Fill NaN with 'Missing' string and ensure string type
    categorical_or_details_cols_to_fill_missing = [col for col in config.ITEM_NOMINAL_CATEGORICAL_COLS + details_cols if col in train_merged_df.columns]
    for col in categorical_or_details_cols_to_fill_missing:
         if train_merged_df[col].isnull().sum() > 0:
              train_merged_df[col] = train_merged_df[col].fillna('Missing')
              # print(f"  Filled NaNs in column '{col}' with 'Missing'.") # Optional print
         train_merged_df[col] = train_merged_df[col].astype(str) # Ensure string type


    # Print NaN counts again for verification across relevant columns
    print("\nNaN counts after initial handling in merged train data:")
    feature_cols_to_check_nan = numerical_cols_in_df + binary_cols_in_df + text_cols_in_df + categorical_or_details_cols_to_fill_missing
    print(train_merged_df[feature_cols_to_check_nan].isnull().sum().sort_values(ascending=False).head())

    print("\nInitial NaN handling on merged data complete.")
    print("Step 1 & 2 execution complete.")

    return train_df, clean_item_meta_df, train_merged_df

# Optional: Add a main block to test this script standalone
if __name__ == "__main__":
    print("Running data_loader.py as standalone script...")
    try:
        train_df_loaded, meta_df_loaded, merged_df_loaded = load_and_merge_data(
            config.TRAIN_CSV_PATH,
            config.CLEANED_METADATA_CSV_PATH
        )
        print("\n--- Standalone Data Loading and Merging Complete ---")
        if merged_df_loaded is not None:
            print(f"Final merged_train_df shape: {merged_df_loaded.shape}")
            print("\nFirst 5 rows of merged data:")
            # Display some key columns
            cols_to_display_head = [config.ITEM_ID_COL, config.USER_ID_COL, config.TIMESTAMP_COL, config.INTERACTION_COL]
            preview_cols = [col for col in config.ITEM_NUMERICAL_COLS + config.ITEM_BINARY_COLS + config.ITEM_NOMINAL_CATEGORICAL_COLS if col in merged_df_loaded.columns]
            details_cols_preview = [col for col in merged_df_loaded.columns if col.startswith('details_')]
            cols_to_display_head.extend(preview_cols[:5]) # Add up to 5 preview cols
            cols_to_display_head.extend(details_cols_preview[:3]) # Add up to 3 details cols
            cols_to_display_existing = list(dict.fromkeys([col for col in cols_to_display_head if col in merged_df_loaded.columns]))
            print(textwrap.indent(merged_df_loaded[cols_to_display_existing].head().__str__(), '  '))

            print("\nNaN counts in merged data after handling:")
            # Re-check NaNs for verification in a few columns
            check_nan_cols = [config.ITEM_ID_COL, config.USER_ID_COL, config.INTERACTION_COL] + preview_cols + details_cols_preview
            check_nan_cols_existing = [col for col in check_nan_cols if col in merged_df_loaded.columns]
            print(merged_df_loaded[check_nan_cols_existing].isnull().sum().sort_values(ascending=False).head(10))

        else:
            print("Data loading/merging function returned None for merged_df.")

    except FileNotFoundError as e:
         print(f"Script failed: {e}")
    except ValueError as e:
         print(f"Script failed due to missing column: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during script execution: {e}")