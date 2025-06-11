# train_catboost.py

import pandas as pd
import numpy as np
import os
import time
import joblib
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

import config
import traceback # Import traceback for debugging

print("--- Training CatBoost Model ---")
start_time = time.time()

# --- Configuration and Paths ---
X_train_hybrid_path = config.X_TRAIN_HYBRID_PATH
y_train_path = config.Y_TRAIN_PATH # This path should point to the y_train.csv file
output_model_path = os.path.join(config.TRAINED_MODELS_PATH, 'catboost_model.joblib') # CatBoost can be saved as joblib

# Get model-specific parameters from config
model_params = config.MODEL_CONFIGS.get('CatBoostClassifier', {})
early_stopping_rounds = model_params.pop('early_stopping_rounds', None) # Remove from params dict if used separately by fit
verbose_setting = model_params.pop('verbose', False) # Remove verbose from params dict

# Ensure output directory exists
os.makedirs(config.TRAINED_MODELS_PATH, exist_ok=True)

print(f"Loading data from: {X_train_hybrid_path}, {y_train_path}")
print(f"Saving model to: {output_model_path}")
print("-" * 50)
print(f"Using CatBoost parameters: {model_params}")
if early_stopping_rounds is not None:
    print(f"Using early stopping rounds: {early_stopping_rounds}")
print("-" * 50)


# --- Load Data ---
try:
    X_train_hybrid = sp.load_npz(X_train_hybrid_path)

    # *** REVISED CORRECTED LOADING FOR y_train CSV ***
    # Try reading with a header first. If it works and has a single column, use that.
    # Otherwise, try reading without a header.
    try:
        # Attempt to read with a header
        y_train_df = pd.read_csv(y_train_path)

        # Check if the DataFrame has columns after reading with header
        if not y_train_df.empty and y_train_df.shape[1] >= 1:
            # Assuming the labels are in the first column (index 0) after reading with header
            # or potentially a column named 'interaction' or similar
            # Let's try getting the first column by index (iloc[:, 0]) first
            # If there's a pandas index column saved, the label might be at index 1
            # For simplicity, let's check column names first, then fall back to iloc[:, 0]
            if config.INTERACTION_COL in y_train_df.columns:
                 y_train = y_train_df[config.INTERACTION_COL].values
                 print(f"Loaded y_train using column '{config.INTERACTION_COL}'.")
            elif y_train_df.shape[1] == 1:
                 y_train = y_train_df.iloc[:, 0].values # Single column case
                 print("Loaded y_train using first column (assuming no index, no header).")
            elif y_train_df.shape[1] > 1 and y_train_df.columns[0].lower() == 'unnamed: 0': # Check for common pandas index column name
                 y_train = y_train_df.iloc[:, 1].values # Assume labels are in the second column if first is index
                 print("Loaded y_train using second column (assuming first column is index).")
            else:
                 # Fallback to first column if no clear header or named column
                 y_train = y_train_df.iloc[:, 0].values
                 print("Loaded y_train using first column by index.")

            # Convert the loaded data to integer type
            y_train = y_train.astype(int)

            print("Successfully loaded y_train CSV assuming header or single column.")

        else:
            # If reading with header resulted in an empty or problematic DataFrame, fall back
             raise ValueError("Reading with header resulted in empty or no columns.") # Trigger fallback

    except Exception as e_header:
        # If reading with header failed, try reading without a header
        print(f"Could not load y_train CSV with header correctly ({e_header}). Attempting to read without header...")
        try:
            y_train_df = pd.read_csv(y_train_path, header=None) # Try reading without a header

            if y_train_df.shape[1] >= 1:
                 y_train = y_train_df.iloc[:, 0].values # Get the first column as NumPy array
                 y_train = y_train.astype(int) # Convert to int
                 print("Successfully loaded y_train CSV using header=None.")
            else:
                 raise ValueError("Reading without header resulted in no columns.")

        except Exception as e_noheader:
            raise RuntimeError(f"Failed to load y_train CSV using both header and no header attempts: {e_noheader}") # Re-raise as critical error


    print(f"Data loaded successfully.")
    print(f"X_train_hybrid shape: {X_train_hybrid.shape}")
    # Ensure y_train is a 1D numpy array with correct length
    if not isinstance(y_train, np.ndarray) or y_train.ndim != 1:
         # This should be caught by the conversion steps above, but double-check
         print(f"WARNING: y_train is not a 1D numpy array after loading (type: {type(y_train)}, shape: {getattr(y_train, 'shape', 'N/A')}). Attempting to convert.")
         y_train = np.asarray(y_train).flatten()


    print(f"y_train final shape: {y_train.shape}")


    if X_train_hybrid.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("Loaded data is empty.")
    # Check if shapes match AFTER successful loading and processing
    if X_train_hybrid.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train_hybrid row count ({X_train_hybrid.shape[0]}) does not match y_train length ({y_train.shape[0]}).")


except FileNotFoundError as e:
    print(f"FATAL ERROR: Data file not found: {e}")
    exit()
except pd.errors.EmptyDataError:
    print(f"FATAL ERROR: y_train CSV file is empty: {y_train_path}")
    exit()
except ValueError as e:
    print(f"FATAL ERROR: Data loading or shape mismatch error: {e}")
    traceback.print_exc()
    exit()
except RuntimeError as e:
     print(f"FATAL ERROR: Failed to load y_train CSV: {e}")
     traceback.print_exc()
     exit()
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during data loading: {e}")
    traceback.print_exc() # Add traceback
    exit()


# --- Split Data ---
print("\nSplitting data into training and validation sets...")
try:
    test_size = config.TRAIN_VALIDATION_SPLIT_SIZE

    # Ensure y_train is suitable for train_test_split (1D array or list)
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_hybrid, y_train, test_size=test_size, random_state=config.SEED, stratify=y_train
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Training target shape: {y_train_split.shape}")
    print(f"Validation target shape: {y_val.shape}")
    print("Data split successfully.")

except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during data splitting: {e}")
    traceback.print_exc() # Add traceback
    exit()


# --- Train Model ---
print("\nTraining CatBoost model...")
try:
    model = CatBoostClassifier(**model_params)

    train_pool = Pool(X_train, y_train_split)
    val_pool = Pool(X_val, y_val)

    fit_params = {
        'eval_set': val_pool,
        'verbose': verbose_setting,
    }
    if early_stopping_rounds is not None:
        fit_params['early_stopping_rounds'] = early_stopping_rounds

    model.fit(train_pool, **fit_params)

    end_train_time = time.time()
    print(f"Model training completed in {end_train_time - start_time:.2f} seconds.")

except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during model training: {e}")
    traceback.print_exc() # Add traceback
    exit()


# --- Evaluate Model ---
print("\nEvaluating model on validation set...")
try:
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    auc_score = roc_auc_score(y_val, y_pred_proba)

    print(f"Validation AUC: {auc_score:.4f}")
    print("Evaluation completed.")

except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during model evaluation: {e}")
    traceback.print_exc() # Add traceback
    exit()


# --- Save Model ---
print("\nSaving trained model...")
try:
    joblib.dump(model, output_model_path)
    print(f"Model saved successfully to {output_model_path}")

except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during model saving: {e}")
    traceback.print_exc() # Add traceback
    exit()


print("\n--- CatBoost Training Script Finished ---")
print(f"Total execution time: {time.time() - start_time:.2f} seconds.")