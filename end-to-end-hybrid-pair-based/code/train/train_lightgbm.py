# train_lightgbm.py

import pandas as pd
import numpy as np
import os
import time
import joblib
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import config
import traceback # Import traceback for debugging

print("--- Training LightGBM Model ---")
start_time = time.time()

# --- Configuration and Paths ---
X_train_hybrid_path = config.X_TRAIN_HYBRID_PATH
y_train_path = config.Y_TRAIN_PATH # This path should point to the y_train.csv file
output_model_path = os.path.join(config.TRAINED_MODELS_PATH, 'lightgbm_model.joblib')

# Get model-specific parameters from config
# Use .get() with an empty dictionary fallback in case the key is missing (safer)
model_params = config.MODEL_CONFIGS.get('LGBMClassifier', {})
early_stopping_rounds = model_params.pop('early_stopping_rounds', None) # Remove from params dict if used separately by fit
# LightGBM uses 'callbacks' for early stopping, not a direct parameter like CatBoost's fit
# The metric to monitor for early stopping is usually specified in 'metric' parameter of the constructor or 'eval_metric' in fit
# Let's assume 'metric' is in model_params or handled by eval_metric

# Ensure output directory exists
os.makedirs(config.TRAINED_MODELS_PATH, exist_ok=True)

print(f"Loading data from: {X_train_hybrid_path}, {y_train_path}")
print(f"Saving model to: {output_model_path}")
print("-" * 50)
print(f"Using LightGBM parameters: {model_params}")
if early_stopping_rounds is not None:
    print(f"Using early stopping rounds: {early_stopping_rounds}")
print("-" * 50)


# --- Load Data ---
try:
    X_train_hybrid = sp.load_npz(X_train_hybrid_path)

    # *** REVISED CORRECTED LOADING FOR y_train CSV ***
    try:
        # Attempt to read with a header
        y_train_df = pd.read_csv(y_train_path)

        if not y_train_df.empty and y_train_df.shape[1] >= 1:
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
                 y_train = y_train_df.iloc[:, 0].values
                 print("Loaded y_train using first column by index.")

            y_train = y_train.astype(int) # Convert to integer type
            print("Successfully loaded y_train CSV assuming header or single column.")

        else:
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
    # Use the train-validation split size from config
    test_size = config.TRAIN_VALIDATION_SPLIT_SIZE

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
print("\nTraining LightGBM model...")
try:
    # Instantiate the model using parameters from config
    model = lgb.LGBMClassifier(**model_params)

    # Prepare fit parameters, including eval_set and callbacks for early stopping
    fit_params = {
        'eval_set': [(X_val, y_val)],
        'eval_metric': model_params.get('metric', 'auc'), # Use metric from params or default to auc
        'callbacks': [], # Initialize callbacks list
    }

    # Add early stopping callback if early_stopping_rounds is configured
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
         fit_params['callbacks'].append(lgb.early_stopping(early_stopping_rounds, verbose=False)) # verbose=False to keep training output clean

    # Train the model
    model.fit(X_train, y_train_split, **fit_params)


    end_train_time = time.time()
    print(f"Model training completed in {end_train_time - start_time:.2f} seconds.")

except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during model training: {e}")
    traceback.print_exc() # Add traceback
    exit()


# --- Evaluate Model ---
print("\nEvaluating model on validation set...")
try:
    # Predict probabilities on the validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1] # Probability of the positive class (1)

    # Calculate AUC
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


print("\n--- LightGBM Training Script Finished ---")
print(f"Total execution time: {time.time() - start_time:.2f} seconds.")