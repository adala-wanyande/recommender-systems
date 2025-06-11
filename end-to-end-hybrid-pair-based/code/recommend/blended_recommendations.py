# generate_multimodel_blended_submission.py

import pandas as pd
import numpy as np
import os
import time
import joblib
import scipy.sparse as sp
import json
import pickle
import sys
import gc
import random
from collections import defaultdict

# Import transformer classes needed for loading
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
# Import model classes needed for loading
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


import config
import traceback

# Determine which blending script is being run (multi-model with blending)
print("--- Generating Multi-Model Blended Submission (RF, LGBM, Popular, Random) using Submission Users ---")
start_time = time.time()


# --- Configuration and Paths ---
# Use sample_submission.csv to get the list of users for prediction and output structure
sample_submission_path = config.SAMPLE_SUBMISSION_CSV_PATH
train_csv_path = config.TRAIN_CSV_PATH
test_csv_path = config.TEST_CSV_PATH # Keep test_csv_path for potential future use

clean_item_meta_path = config.CLEANED_METADATA_CSV_PATH

user_features_path = config.USER_FEATURES_CSV_PATH
user_numerical_scaler_path = config.USER_NUMERICAL_SCALER_PKL_PATH
user_categorical_encoder_path = config.USER_CATEGORICAL_ENCODER_PKL_PATH
user_categorical_svd_path = config.USER_CATEGORICAL_SVD_PKL_PATH
user_num_cols_path = config.USER_NUM_COLS_FOR_SCALING_JSON_PATH
user_cat_cols_path = config.USER_CAT_COLS_FOR_ENCODING_JSON_PATH

x_items_reduced_path = config.X_ITEMS_REDUCED_NPY_PATH # Or NPZ if you changed it
item_id_to_index_json_path = config.ITEM_ID_TO_INDEX_JSON_PATH
index_to_item_id_json_path = config.INDEX_TO_ITEM_ID_JSON_PATH

trained_models_dir = config.TRAINED_MODELS_PATH
submission_output_dir = config.SUBMISSION_OUTPUT_DIR

# Specify the model filenames to load from config
RF_MODEL_FILENAME = config.RF_MODEL_FILENAME
LGBM_MODEL_FILENAME = config.LGBM_MODEL_FILENAME

rf_model_path = os.path.join(trained_models_dir, RF_MODEL_FILENAME)
lgbm_model_path = os.path.join(trained_models_dir, LGBM_MODEL_FILENAME)


# The final submission path
# Use a more descriptive name that includes the models and blend strategy
blended_submission_path = os.path.join(submission_output_dir, f'{os.path.splitext(RF_MODEL_FILENAME)[0]}_{os.path.splitext(LGBM_MODEL_FILENAME)[0]}_popular_random_blended_submission.csv')


# Config for blending - LOAD ALL BLENDING CONFIG
RECOMMENDATION_COUNT = config.RECOMMENDATION_COUNT
NUM_POPULAR_ITEMS_TO_BLEND = config.NUM_POPULAR_ITEMS_TO_BLEND
NUM_RANDOM_ITEMS_TO_BLEND = config.NUM_RANDOM_ITEMS_TO_BLEND

# Define the target counts from each model type directly in this script
NUM_RF_ITEMS_TO_BLEND = 4
NUM_LGBM_ITEMS_TO_BLEND = 1
# Total target count from models: NUM_RF_ITEMS_TO_BLEND + NUM_LGBM_ITEMS_TO_BLEND = 6
# Total target blend count: NUM_POPULAR_ITEMS_TO_BLEND + NUM_RANDOM_ITEMS_TO_BLEND = 3 + 2 = 5
# Total desired unique items from targets: 6 + 5 = 11. The blending logic ensures
# uniqueness and stops at RECOMMENDATION_COUNT (10). The extra target slots
# simply mean it *tries* to pull that many from each source if available and unique.
# The final fill step from RF/Popular/Random handles cases where targets aren't met.


# Ensure output directory exists
os.makedirs(submission_output_dir, exist_ok=True)


# --- Load All Necessary Data and Pipeline Components ---
print("Loading data and pipeline components...")
X_items_reduced = None
item_id_to_index = None
index_to_item_id = None
sample_submission_df = None
train_df = None
test_df = None
clean_item_meta_df = None
user_features_df = None
user_numerical_scaler = None
user_categorical_encoder = None
user_categorical_svd = None
user_num_cols = []
user_cat_cols = []


try:
    sample_submission_df = pd.read_csv(sample_submission_path)
    sample_submission_df[config.USER_ID_COL] = sample_submission_df[config.USER_ID_COL].astype(str)
    print(f"Loaded sample submission ({len(sample_submission_df)} rows) - used for output structure and users to predict.")

    train_df = pd.read_csv(train_csv_path)
    train_df[config.USER_ID_COL] = train_df[config.USER_ID_COL].astype(str)
    train_df[config.ITEM_ID_COL] = train_df[config.ITEM_ID_COL].astype(str)
    print(f"Loaded training data ({len(train_df)} interactions) - used for popularity and seen items.")

    # --- Load test data (optional, but good practice if test features might be needed) ---
    try:
         test_df = pd.read_csv(test_csv_path)
         test_df[config.USER_ID_COL] = test_df[config.USER_ID_COL].astype(str)
         test_df[config.ITEM_ID_COL] = test_df[config.ITEM_ID_COL].astype(str)
         print(f"Loaded test data ({len(test_df)} interactions).")
    except FileNotFoundError:
         print(f"WARNING: Test data not found at {test_csv_path}. Proceeding without it.")
         test_df = pd.DataFrame({config.USER_ID_COL: [], config.ITEM_ID_COL: [], config.TIMESTAMP_COL: []})


    clean_item_meta_df = pd.read_csv(clean_item_meta_path)
    clean_item_meta_df[config.ITEM_ID_COL] = clean_item_meta_df[config.ITEM_ID_COL].astype(str)
    print(f"Loaded item metadata ({len(clean_item_meta_df)} items)")


    user_features_df = pd.read_csv(user_features_path, dtype={config.USER_ID_COL: str})
    print(f"Loaded user features ({len(user_features_df)} users)")


    # Load transformers
    user_numerical_scaler = joblib.load(user_numerical_scaler_path)
    user_categorical_encoder = joblib.load(user_categorical_encoder_path)
    print("Loaded user transformers (scaler, encoder).")

    if os.path.exists(user_categorical_svd_path):
        try:
            user_categorical_svd = joblib.load(user_categorical_svd_path)
            if not hasattr(user_categorical_svd, 'n_components') or user_categorical_svd.n_components <= 0:
                 user_categorical_svd = None
                 print("Loaded User Categorical SVD but n_components <= 0 or invalid. Will skip SVD transformation.")
            else:
                 print(f"Loaded User Categorical SVD with {user_categorical_svd.n_components} components.")
        except Exception as svd_e:
             print(f"Warning: Could not load User Categorical SVD from {user_categorical_svd_path}: {svd_e}. Will skip SVD transformation.")
             user_categorical_svd = None
    else:
        user_categorical_svd = None
        print(f"User Categorical SVD not found at {user_categorical_svd_path}. Will skip SVD transformation.")


    # Load column lists
    with open(user_num_cols_path, 'r') as f:
        user_num_cols = json.load(f)
    with open(user_cat_cols_path, 'r') as f:
        user_cat_cols = json.load(f)
    print("Loaded user column lists.")


    # --- Load processed item features (.npy or .npz) ---
    # Check if it's NPZ first (sparse)
    x_items_reduced_path_npz = x_items_reduced_path.replace('.npy', '.npz') # Assume NPZ is the preferred sparse format
    if os.path.exists(x_items_reduced_path_npz):
         try:
             print(f"Loading sparse item features from {x_items_reduced_path_npz} (.npz)...")
             X_items_reduced = sp.load_npz(x_items_reduced_path_npz)
             if not sp.issparse(X_items_reduced):
                  raise ValueError(f"Loaded data from {x_items_reduced_path_npz} is not a sparse matrix.")
             print(f"Successfully loaded sparse item features (.npz) shape: {X_items_reduced.shape}")

         except FileNotFoundError: # Should not happen due to exists() check, but defensive
             raise FileNotFoundError(f"FATAL ERROR: Sparse item features file not found at {x_items_reduced_path_npz} (.npz).")
         except Exception as npz_e:
             print(f"WARNING: Failed to load sparse item features from {x_items_reduced_path_npz} (.npz): {type(npz_e).__name__}: {npz_e}")
             print("Attempting to load dense .npy instead...")
             X_items_reduced = None # Ensure it's None if loading failed
    # Fallback to NPY (dense) if NPZ didn't exist or failed
    if X_items_reduced is None and os.path.exists(x_items_reduced_path):
        try:
            print(f"Loading dense item features from {x_items_reduced_path} (.npy)...")
            X_items_reduced = np.load(x_items_reduced_path)
            if not isinstance(X_items_reduced, np.ndarray):
                 raise ValueError(f"Loaded data from {x_items_reduced_path} is not a numpy array.")
            print(f"Successfully loaded dense item features (.npy) shape: {X_items_reduced.shape}")
            print("\n!!! WARNING: Loaded item features as a dense NumPy array (.npy). This will increase memory usage during prediction.")
            print("!!! Consider saving item features as a sparse matrix using scipy.sparse.save_npz and updating config.")
            print("-" * 80)

        except FileNotFoundError:
             raise FileNotFoundError(f"FATAL ERROR: Item features file not found at {x_items_reduced_path} (.npy). Please ensure this file exists.")
        except Exception as npy_e:
             raise RuntimeError(f"FATAL ERROR: Failed to load item features from {x_items_reduced_path} (.npy): {type(npy_e).__name__}: {npy_e}")
    elif X_items_reduced is None:
         raise FileNotFoundError(f"FATAL ERROR: Item features file not found at either {x_items_reduced_path_npz} (.npz) or {x_items_reduced_path} (.npy). Please ensure one of these files exists.")


    # --- Load item_id to index mappings from JSON ---
    try:
        print(f"Loading item ID to index mappings from {item_id_to_index_json_path} and {index_to_item_id_json_path} (JSON)...")
        with open(item_id_to_index_json_path, 'r') as f:
            item_id_to_index = json.load(f)
        print(f"Loaded item_id_to_index ({len(item_id_to_index)} items)")

        with open(index_to_item_id_json_path, 'r') as f:
            index_to_item_id_str_keys = json.load(f)
            index_to_item_id = {int(k): v for k, v in index_to_item_id_str_keys.items()}
        print(f"Loaded index_to_item_id ({len(index_to_item_id)} indices)")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"FATAL ERROR: Item ID mapping JSON file not found: {e}. Ensure item_processor.py was run successfully.")
    except json.JSONDecodeError as e:
         raise RuntimeError(f"FATAL ERROR: Failed to decode JSON mapping file: {e}")
    except Exception as e:
         raise RuntimeError(f"FATAL ERROR: An unexpected error occurred loading JSON mappings: {type(e).__name__}: {e}")

    # Basic validation after loading item features and mappings
    if X_items_reduced is None or X_items_reduced.shape[0] != len(item_id_to_index):
        raise RuntimeError(f"FATAL ERROR: Loaded item features matrix has {X_items_reduced.shape[0] if X_items_reduced is not None else 'None'} rows, but item ID map has {len(item_id_to_index)} items. These should match.")

    # Ensure all items in metadata are processable (have features)
    all_item_ids_meta = set(clean_item_meta_df[config.ITEM_ID_COL].unique())
    all_processable_item_ids = [item for item in all_item_ids_meta if item in item_id_to_index]
    print(f"Total items in metadata: {len(all_item_ids_meta)}")
    print(f"Total processable items (in item features): {len(all_processable_item_ids)}")

    if not all_processable_item_ids:
         print("FATAL ERROR: No processable items found after filtering against item features. Cannot proceed.")
         sys.exit(1)

    # Get item features for all processable items. Handle sparse/dense.
    # Create a mapping from item ID to its row index *within* the all_processable_item_ids list
    processable_item_id_to_local_index = {item_id: i for i, item_id in enumerate(all_processable_item_ids)}
    item_features_all_processable_indices_in_X_reduced = [item_id_to_index[item_id] for item_id in all_processable_item_ids]

    # Ensure item_features_all_processable uses the correct type (sparse or dense) and is indexed by the local index
    if sp.issparse(X_items_reduced):
        # Need to create a new sparse matrix subsetted by processable item indices
        item_features_all_processable = X_items_reduced[item_features_all_processable_indices_in_X_reduced, :]
        print(f"Shape of item features for all processable items (sparse): {item_features_all_processable.shape}")
    else: # Assume dense
        item_features_all_processable = X_items_reduced[item_features_all_processable_indices_in_X_reduced, :]
        print(f"Shape of item features for all processable items (dense): {item_features_all_processable.shape}")

    print(f"Type of item features for all processable items: {type(item_features_all_processable)}")


    print("All necessary data and components loaded successfully.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Required file not found during loading: {e}")
    traceback.print_exc()
    sys.exit(1)
except RuntimeError as e:
    print(e)
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during data/component loading: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Prepare Global Data Needed for Feature Generation and Prediction ---

print("\nPreparing global data structures for prediction...")

# *** Use users from sample_submission.csv for prediction ***
users_to_process = sample_submission_df[config.USER_ID_COL].unique().astype(str) # Get unique users from sample submission
print(f"Generating recommendations for {len(users_to_process)} unique users from sample submission file.")


# Create a set of (user, item) pairs from the training data for quick lookup
train_interactions = set(tuple(row) for row in train_df[[config.USER_ID_COL, config.ITEM_ID_COL]].values)
print(f"Loaded {len(train_interactions)} unique train interactions.")

# Store user train items in a dict for faster per-user lookup later
user_train_items_dict = defaultdict(set)
for user_id, item_id in train_interactions:
    user_train_items_dict[user_id].add(item_id)


# Process users from the sample submission file to get their features transformed ONCE
users_to_process_df_for_merge = pd.DataFrame({config.USER_ID_COL: users_to_process}).astype({config.USER_ID_COL: str})

user_features_for_prediction = pd.merge(
    users_to_process_df_for_merge,
    user_features_df,
    on=config.USER_ID_COL,
    how='left' # Keep all users from sample submission, potentially adding NaNs if user is new
)

# Handle potential NaNs in user features before transformation (fill with defaults)
print("Handling potential NaNs in user features before transformation...")
user_features_to_process_transformed_prep = user_features_for_prediction.copy()
for col in user_num_cols:
    if col in user_features_to_process_transformed_prep.columns:
        user_features_to_process_transformed_prep[col] = user_features_to_process_transformed_prep[col].fillna(0)

for col in user_cat_cols:
     if col in user_features_to_process_transformed_prep.columns:
          user_features_to_process_transformed_prep[col] = user_features_to_process_transformed_prep[col].fillna('__MISSING__')

print("Applying user feature transformers...")
# Apply numerical scaler
user_num_scaled = user_numerical_scaler.transform(user_features_to_process_transformed_prep[user_num_cols])
user_num_scaled_dense = user_num_scaled # Keep as dense

# Apply categorical encoder
user_cat_encoded_sparse = user_categorical_encoder.transform(user_features_to_process_transformed_prep[user_cat_cols])

# Apply categorical SVD if available and components > 0
user_cat_features_transformed = None # Initialize before conditional assignment
if user_categorical_svd is not None:
     user_cat_svd_reduced = user_categorical_svd.transform(user_cat_encoded_sparse)
     user_cat_features_transformed = user_cat_svd_reduced # Use dense SVD output
     print("User Categorical SVD applied. Result is dense.")
else:
     user_cat_features_transformed = user_cat_encoded_sparse # Keep as sparse if no SVD
     print("Note: User Categorical SVD was not applied during training or loading, using sparse one-hot encoded features.")

print("User feature transformation completed.")

# Map user_id (from sample submission) to their processed feature row index
user_id_to_feature_row_index = {user_id: i for i, user_id in enumerate(user_features_for_prediction[config.USER_ID_COL])}


# --- Calculate Item Popularity (Globally) ---
print("\nCalculating item popularity from training data...")
item_popularity = train_df[config.ITEM_ID_COL].value_counts()
# Sort and filter to include only processable items
popular_item_ids_full_list = item_popularity.index.tolist()
popular_item_ids = [item for item in popular_item_ids_full_list if item in all_processable_item_ids] # Filter to processable
print(f"Calculated popularity for {len(item_popularity)} unique items in train.")
print(f"Filtered popular items list to {len(popular_item_ids)} items that are in processed item features.")


print("Global data preparation complete.")


# --- Load Trained Models (Random Forest and LightGBM) ---
print(f"\nLoading trained models: {RF_MODEL_FILENAME} and {LGBM_MODEL_FILENAME}")
rf_model = None
lgbm_model = None

try:
    rf_model = joblib.load(rf_model_path)
    if not (hasattr(rf_model, 'predict_proba') or hasattr(rf_model, 'decision_function')):
         print(f"FATAL ERROR: Loaded RF model {RF_MODEL_FILENAME} does not have 'predict_proba' or 'decision_function'. Exiting.")
         sys.exit(1)
    # For predict_proba, check if it's a binary classifier with a positive class (1)
    if hasattr(rf_model, 'predict_proba') and (not hasattr(rf_model, 'classes_') or 1 not in rf_model.classes_):
         print(f"WARNING: RF model {RF_MODEL_FILENAME} has predict_proba but no 'classes_' attribute or does not contain class 1. Will attempt to use decision_function if available, otherwise fallback to predict_proba[:, 0].")

    print(f"Successfully loaded model: {RF_MODEL_FILENAME}")
except FileNotFoundError:
    print(f"FATAL ERROR: RF model file not found at {rf_model_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: Could not load RF model {RF_MODEL_FILENAME} from {rf_model_path}: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    lgbm_model = joblib.load(lgbm_model_path)
    if not (hasattr(lgbm_model, 'predict_proba') or hasattr(lgbm_model, 'decision_function')):
         print(f"FATAL ERROR: Loaded LGBM model {LGBM_MODEL_FILENAME} does not have 'predict_proba' or 'decision_function'. Exiting.")
         sys.exit(1)
    # For predict_proba, check if it's a binary classifier with a positive class (1)
    if hasattr(lgbm_model, 'predict_proba') and (not hasattr(lgbm_model, 'classes_') or 1 not in lgbm_model.classes_):
         print(f"WARNING: LGBM model {LGBM_MODEL_FILENAME} has predict_proba but no 'classes_' attribute or does not contain class 1. Will attempt to use decision_function if available, otherwise fallback to predict_proba[:, 0].")

    print(f"Successfully loaded model: {LGBM_MODEL_FILENAME}")
except FileNotFoundError:
    print(f"FATAL ERROR: LGBM model file not found at {lgbm_model_path}. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: Could not load LGBM model {LGBM_MODEL_FILENAME} from {lgbm_model_path}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Ensure both models are loaded
if rf_model is None or lgbm_model is None:
    print("FATAL ERROR: One or both models failed to load. Exiting.")
    sys.exit(1)


# --- Generate Features and Predict with Both Models in Batches ---
# Refactored to combine user and item features *before* batching to avoid sp.vstack error.

print("\nGenerating features and predicting scores with both models in batches...")

prediction_batch_size_pairs = 500000 # Adjust as needed based on memory

current_prediction_pairs_batch = [] # List to hold (user_id, item_id) for the current prediction batch
# Refactored: Single list to hold combined dense feature blocks per user
current_combined_features_batch = [] # List to hold combined (user+item) dense feature blocks for each user

all_multimodel_predictions_list = [] # List to store (user_id, item_id, rf_score, lgbm_score) tuples

total_pairs_processed_for_prediction = 0
batch_counter = 0

user_processing_start_time = time.time()
print(f"Starting pair generation and batching for {len(users_to_process)} users...")

try: # Outer try block for the entire prediction loop
    # Iterate through the users from the sample submission file
    for user_idx, user_id in enumerate(users_to_process):
         if (user_idx + 1) % 100 == 0:
              print(f"  Generating pairs for user {user_idx + 1}/{len(users_to_process)}...")

         # Get the processed user features for this user
         user_feat_row_index = user_id_to_feature_row_index.get(user_id)

         if user_feat_row_index is None:
              print(f"WARNING: User {user_id} from sample submission not found in processed user features. Will receive empty recommendation.")
              # This user will get an empty recommendation later as they won't be in `all_multimodel_predictions_list`
              continue # Skip pair generation for this user

         # --- Get user features for this user (keep as 2D shape (1, N)) ---
         user_num_feat_single = user_num_scaled_dense[user_feat_row_index:user_feat_row_index+1, :] # Shape (1, num_num_cols)

         # user_cat_features_transformed is either sparse OHE or dense SVD
         user_cat_feat_single = user_cat_features_transformed[user_feat_row_index:user_feat_row_index+1, :] # Shape (1, num_cat) or (1, num_svd)


         # Get items the user has already interacted with in training
         user_train_items = user_train_items_dict.get(user_id, set())

         # Define candidate items: all processable items NOT interacted with in training
         candidate_item_ids = [item for item in all_processable_item_ids if item not in user_train_items]

         if not candidate_item_ids:
             print(f"WARNING: User {user_id} has no valid candidate items (seen all processable items or none processable). Will receive empty recommendation.")
             # This user will get an empty recommendation later
             continue

         num_candidates = len(candidate_item_ids)

         # --- Generate combined features for all candidate items for THIS user ---

         # Repeat user numerical features for each candidate item
         user_num_feats_repeated = np.repeat(user_num_feat_single, num_candidates, axis=0) # Shape (num_candidates, num_num_cols)

         # Repeat user categorical features (handle sparse/dense)
         if sp.issparse(user_cat_feat_single):
             user_cat_feats_repeated = sp.vstack([user_cat_feat_single] * num_candidates) # Shape (num_candidates, num_cat_features) sparse
         else:
             user_cat_feats_repeated = np.repeat(user_cat_feat_single, num_candidates, axis=0) # Shape (num_candidates, num_svd) dense


         # Get item features for the candidate items (handle sparse/dense)
         # Need indices relative to the *all_processable_item_ids* list/matrix
         candidate_item_local_indices = [processable_item_id_to_local_index[item] for item in candidate_item_ids]
         item_feats_for_candidates = item_features_all_processable[candidate_item_local_indices, :] # Shape (num_candidates, num_item_features) sparse or dense


         # --- Combine user and item features *for this user's candidates* ---
         # Convert all components to dense *before* stacking if they aren't already.
         # This avoids sp.hstack issues with mixing sparse/dense and potential 1-column arrays.
         user_num_feats_repeated_dense = user_num_feats_repeated # Already dense
         user_cat_feats_repeated_dense = user_cat_feats_repeated.toarray() if sp.issparse(user_cat_feats_repeated) else user_cat_feats_repeated
         item_feats_for_candidates_dense = item_feats_for_candidates.toarray() if sp.issparse(item_feats_for_candidates) else item_feats_for_candidates


         # Use np.hstack now that all components are dense
         combined_features_for_user_candidates = np.hstack([
             user_num_feats_repeated_dense,
             user_cat_feats_repeated_dense,
             item_feats_for_candidates_dense
         ])

         # print(f"    User {user_id}: Combined features shape: {combined_features_for_user_candidates.shape}")


         # Add candidate pairs and the *combined dense* features to the batch lists
         current_prediction_pairs_batch.extend([(user_id, item) for item in candidate_item_ids])
         current_combined_features_batch.append(combined_features_for_user_candidates) # Append the combined DENSE block


         # Check if the current batch size is reached or exceeds, or if it's the last user
         current_batch_pair_count = sum(arr.shape[0] for arr in current_combined_features_batch)

         # Also need to check if it's the last user AND there are *any* pairs in the batch
         is_last_user = (user_idx == len(users_to_process) - 1)
         has_pairs_in_batch = bool(current_prediction_pairs_batch) # Or check current_batch_pair_count > 0


         if (current_batch_pair_count >= prediction_batch_size_pairs) or (is_last_user and has_pairs_in_batch):

             batch_counter += 1
             print(f"\n  Processing prediction batch {batch_counter} with {current_batch_pair_count} pairs...")

             try:
                 # Stack the combined dense feature blocks vertically
                 # Use np.vstack because current_combined_features_batch contains dense arrays
                 X_batch_for_models = np.vstack(current_combined_features_batch)

                 print(f"  Batch feature matrix shape (dense): {X_batch_for_models.shape}")

             except MemoryError:
                  print(f"FATAL ERROR: MemoryError while stacking batch features for batch {batch_counter}. Batch size {prediction_batch_size_pairs} is too large.")
                  print("Consider reducing 'prediction_batch_size_pairs'. Exiting prediction loop.")
                  # Clear batch data before raising
                  del current_prediction_pairs_batch
                  del current_combined_features_batch
                  if 'X_batch_for_models' in locals(): del X_batch_for_models
                  gc.collect()
                  raise MemoryError("Batch feature stacking failed.") # Re-raise to be caught by outer try


             # --- Predict scores for the current batch using BOTH loaded models ---
             try:
                 print(f"  Predicting batch with models: {RF_MODEL_FILENAME}, {LGBM_MODEL_FILENAME}")

                 # RF Prediction (expects dense input X_batch_for_models)
                 rf_scores = None
                 # Check predict_proba first, robustly find positive class index
                 if hasattr(rf_model, 'predict_proba') and hasattr(rf_model, 'classes_'):
                      try:
                           positive_class_idx = np.where(rf_model.classes_ == 1)[0][0]
                           rf_scores = rf_model.predict_proba(X_batch_for_models)[:, positive_class_idx]
                      except (IndexError, ValueError): # Class 1 not found or classes_ is weird
                           print(f"WARNING: RF model {RF_MODEL_FILENAME} predict_proba available but class 1 not found or classes_ is invalid. Trying decision_function.")
                           rf_scores = None # Reset to try decision_function
                 # If predict_proba failed or wasn't available, try decision_function
                 if rf_scores is None and hasattr(rf_model, 'decision_function'):
                      rf_scores = rf_model.decision_function(X_batch_for_models)
                      if rf_scores.ndim > 1 and rf_scores.shape[1] > 1:
                           print(f"WARNING: RF model decision_function returned multi-dimensional output {rf_scores.shape}. Using first column.")
                           rf_scores = rf_scores[:, 0]
                      elif rf_scores.ndim != 1 and rf_scores.shape[0] > 0: # Handle unexpected non-scalar shape
                           print(f"WARNING: RF model decision_function returned unexpected shape {rf_scores.shape}. Using zeros.")
                           rf_scores = np.zeros(X_batch_for_models.shape[0])

                 # Fallback if neither worked or shape is wrong
                 if rf_scores is None or not isinstance(rf_scores, np.ndarray) or rf_scores.shape[0] != X_batch_for_models.shape[0]:
                      print(f"WARNING: Could not get valid probability/decision score for RF model {RF_MODEL_FILENAME}. Using zeros.")
                      rf_scores = np.zeros(X_batch_for_models.shape[0])


                 # LGBM Prediction (expects dense input X_batch_for_models)
                 lgbm_scores = None
                 # Check predict_proba first, robustly find positive class index
                 if hasattr(lgbm_model, 'predict_proba') and hasattr(lgbm_model, 'classes_'):
                      try:
                           positive_class_idx = np.where(lgbm_model.classes_ == 1)[0][0]
                           lgbm_scores = lgbm_model.predict_proba(X_batch_for_models)[:, positive_class_idx]
                      except (IndexError, ValueError): # Class 1 not found or classes_ is weird
                            print(f"WARNING: LGBM model {LGBM_MODEL_FILENAME} predict_proba available but class 1 not found or classes_ is invalid. Trying decision_function.")
                            lgbm_scores = None # Reset to try decision_function
                 # If predict_proba failed or wasn't available, try decision_function
                 if lgbm_scores is None and hasattr(lgbm_model, 'decision_function'):
                      lgbm_scores = lgbm_model.decision_function(X_batch_for_models)
                      if lgbm_scores.ndim > 1 and lgbm_scores.shape[1] > 1:
                           print(f"WARNING: LGBM model decision_function returned multi-dimensional output {lgbm_scores.shape}. Using first column.")
                           lgbm_scores = lgbm_scores[:, 0]
                      elif lgbm_scores.ndim != 1 and lgbm_scores.shape[0] > 0: # Handle unexpected non-scalar shape
                           print(f"WARNING: LGBM model decision_function returned unexpected shape {lgbm_scores.shape}. Using zeros.")
                           lgbm_scores = np.zeros(X_batch_for_models.shape[0])

                 # Fallback if neither worked or shape is wrong
                 if lgbm_scores is None or not isinstance(lgbm_scores, np.ndarray) or lgbm_scores.shape[0] != X_batch_for_models.shape[0]:
                     print(f"WARNING: Could not get valid probability/decision score for LGBM model {LGBM_MODEL_FILENAME}. Using zeros.")
                     lgbm_scores = np.zeros(X_batch_for_models.shape[0])


                 # Append predictions to the main list
                 for j in range(len(current_prediction_pairs_batch)):
                     user_id, item_id = current_prediction_pairs_batch[j]
                     # Ensure scores are floats and handle potential non-scalar results if the above logic failed (defensive)
                     # Added check for scalar to prevent issues with potentially malformed arrays from models
                     rf_s = float(rf_scores[j]) if (isinstance(rf_scores, np.ndarray) and j < len(rf_scores) and np.isscalar(rf_scores[j])) else 0.0
                     lgbm_s = float(lgbm_scores[j]) if (isinstance(lgbm_scores, np.ndarray) and j < len(lgbm_scores) and np.isscalar(lgbm_scores[j])) else 0.0


                     all_multimodel_predictions_list.append((user_id, item_id, rf_s, lgbm_s))


                 total_pairs_processed_for_prediction += len(current_prediction_pairs_batch)


             except Exception as e:
                  print(f"FATAL ERROR: Prediction failed for models in batch {batch_counter}: {e}")
                  traceback.print_exc()
                  print("Exiting script due to prediction error.")
                  # Clean up batch data before raising
                  del current_prediction_pairs_batch
                  del current_combined_features_batch
                  if 'X_batch_for_models' in locals(): del X_batch_for_models
                  if 'rf_scores' in locals(): del rf_scores
                  if 'lgbm_scores' in locals(): del lgbm_scores
                  gc.collect()
                  raise # Re-raise to be caught by outer try


             print(f"  Batch {batch_counter} processed. Total pairs processed so far: {total_pairs_processed_for_prediction}")

             # --- Clean up batch variables to free memory ---
             del current_prediction_pairs_batch
             del current_combined_features_batch
             if 'X_batch_for_models' in locals(): del X_batch_for_models
             if 'rf_scores' in locals(): del rf_scores
             if 'lgbm_scores' in locals(): del lgbm_scores
             gc.collect() # Encourage garbage collection

             # Initialize new lists for the next batch
             current_prediction_pairs_batch = []
             current_combined_features_batch = []


except MemoryError as me:
     print(f"FATAL ERROR: Caught MemoryError during batch processing: {me}")
     traceback.print_exc()
     print("Exiting script due to MemoryError.")
     # Ensure models are deleted
     if 'rf_model' in locals() and rf_model is not None: del rf_model
     if 'lgbm_model' in locals() and lgbm_model is not None: del lgbm_model
     gc.collect()
     sys.exit(1) # Exit with error code
except Exception as e:
     print(f"FATAL ERROR: An unexpected error occurred during batch prediction: {e}")
     traceback.print_exc()
     print("Exiting script.")
     # Ensure models are deleted
     if 'rf_model' in locals() and rf_model is not None: del rf_model
     if 'lgbm_model' in locals() and lgbm_model is not None: del lgbm_model
     gc.collect()
     sys.exit(1)


user_processing_end_time = time.time()
print(f"\nFinished pair generation and batching for all users in {user_processing_end_time - user_processing_start_time:.2f} seconds.")
print(f"Total prediction pairs generated across all batches: {total_pairs_processed_for_prediction}")


# --- Final Ranking, Blending, and Formatting Submission ---
print("\nRanking items, blending, and formatting submission...")

# Use a dictionary to store user_id -> list of recommended item_ids for the users from sample submission
recommendations_for_submission_users = {}

# Get unique users from the predictions list
predicted_users = set(item[0] for item in all_multimodel_predictions_list)

if all_multimodel_predictions_list:
    multimodel_predictions_df = pd.DataFrame(all_multimodel_predictions_list, columns=[config.USER_ID_COL, config.ITEM_ID_COL, 'rf_score', 'lgbm_score'])
    print(f"Collected {len(multimodel_predictions_df)} predictions from the models for ranking.")

    # Group predictions by user for easier per-user processing
    user_predictions_groups = multimodel_predictions_df.groupby(config.USER_ID_COL)

    print(f"Blending targets: {NUM_POPULAR_ITEMS_TO_BLEND} Popular, {NUM_RANDOM_ITEMS_TO_BLEND} Random, {NUM_RF_ITEMS_TO_BLEND} RF, {NUM_LGBM_ITEMS_TO_BLEND} LGBM (Total {RECOMMENDATION_COUNT})...")

    # Ensure we have the list of *all* processable items available for random selection
    all_candidate_items_for_random = set(all_processable_item_ids) # Already filtered processable items


    # Iterate through each user that was processed and had candidates
    for user_id, group_df in user_predictions_groups:

         # Get model-ranked items for this user (based on scores)
         rf_ranked_items = group_df.sort_values('rf_score', ascending=False)[config.ITEM_ID_COL].tolist()
         lgbm_ranked_items = group_df.sort_values('lgbm_score', ascending=False)[config.ITEM_ID_COL].tolist()

         # Get items the user has already interacted with in TRAIN
         seen_items = user_train_items_dict.get(user_id, set())

         final_recs = []
         added_items = set(seen_items) # Start set with seen items to easily avoid adding them

         # --- Blending Logic ---

         # 1. Add Popular Items (Target: NUM_POPULAR_ITEMS_TO_BLEND)
         popular_items_added_count = 0
         # Iterate through popular items, skipping seen/already added, up to target count or max recs
         for item_id in popular_item_ids:
             if popular_items_added_count >= NUM_POPULAR_ITEMS_TO_BLEND:
                 break
             if len(final_recs) >= RECOMMENDATION_COUNT:
                 break

             # Check if it's unseen, not already added, AND is in the list of all processable candidates
             # The check `item_id in all_candidate_items_for_random` is implicitly true since popular_item_ids
             # is already filtered by `all_processable_item_ids`.
             if item_id not in added_items:
                 final_recs.append(item_id)
                 added_items.add(item_id)
                 popular_items_added_count += 1
         # print(f"  User {user_id}: Added {popular_items_added_count} popular items.")


         # 2. Add Random Items (Target: NUM_RANDOM_ITEMS_TO_BLEND)
         random_items_added_count = 0
         if NUM_RANDOM_ITEMS_TO_BLEND > 0:
             # Get available random candidates: all processable items NOT seen by user AND NOT already added
             available_random_candidates = list(all_candidate_items_for_random - added_items) # added_items includes seen_items and already blended items
             num_needed = RECOMMENDATION_COUNT - len(final_recs)
             num_to_sample = min(NUM_RANDOM_ITEMS_TO_BLEND, num_needed, len(available_random_candidates))

             if num_to_sample > 0:
                 try:
                     random_items = random.sample(available_random_candidates, num_to_sample)
                     for item_id in random_items:
                          # The checks below are slightly redundant because we sampled from `available_random_candidates`
                          # but included for robustness. `item_id not in added_items` should always be true here.
                          if len(final_recs) >= RECOMMENDATION_COUNT: break
                          if item_id not in added_items:
                               final_recs.append(item_id)
                               added_items.add(item_id)
                               random_items_added_count += 1
                 except ValueError as e:
                      print(f"WARNING: Could not sample {num_to_sample} random item(s) for user {user_id} from {len(available_random_candidates)} available candidates: {e}")
         # print(f"  User {user_id}: Added {random_items_added_count} random items.")


         # 3. Add RF Model Items (Target: NUM_RF_ITEMS_TO_BLEND)
         rf_items_added_count = 0
         for item_id in rf_ranked_items:
             if rf_items_added_count >= NUM_RF_ITEMS_TO_BLEND:
                 break
             if len(final_recs) >= RECOMMENDATION_COUNT:
                 break

             if item_id not in added_items: # added_items includes seen, pop, random, and already added RF items
                 final_recs.append(item_id)
                 added_items.add(item_id)
                 rf_items_added_count += 1
         # print(f"  User {user_id}: Added {rf_items_added_count} RF items.")


         # 4. Add LGBM Model Items (Target: NUM_LGBM_ITEMS_TO_BLEND)
         lgbm_items_added_count = 0
         for item_id in lgbm_ranked_items:
             if lgbm_items_added_count >= NUM_LGBM_ITEMS_TO_BLEND:
                 break
             if len(final_recs) >= RECOMMENDATION_COUNT:
                 break

             if item_id not in added_items: # added_items includes seen, pop, random, RF, and already added LGBM items
                 final_recs.append(item_id)
                 added_items.add(item_id)
                 lgbm_items_added_count += 1
         # print(f"  User {user_id}: Added {lgbm_items_added_count} LGBM items.")


         # 5. Fill remaining slots with any available items from the RF ranked list (highest score first)
         #    This handles cases where popular, random, and target model counts didn't reach 10
         if len(final_recs) < RECOMMENDATION_COUNT:
              # print(f"  User {user_id}: Total recs ({len(final_recs)}) < {RECOMMENDATION_COUNT}. Attempting to fill from RF ranked list.")
              for item_id in rf_ranked_items: # Iterate through the RF ranked list again
                   if len(final_recs) >= RECOMMENDATION_COUNT:
                       break
                   if item_id not in added_items: # Only add if not already in the list
                       final_recs.append(item_id)
                       added_items.add(item_id) # Mark as added


         # 6. If still not enough items, fill with any remaining popular items
         if len(final_recs) < RECOMMENDATION_COUNT:
             # print(f"  User {user_id}: Total recs ({len(final_recs)}) < {RECOMMENDATION_COUNT}. Attempting to fill from remaining popular items.")
             for item_id in popular_item_ids:
                 if len(final_recs) >= RECOMMENDATION_COUNT:
                      break
                 if item_id not in added_items: # added_items includes seen, and all items added so far
                     final_recs.append(item_id)
                     added_items.add(item_id)


         # 7. If still not enough items, fill with any remaining random processable items
         if len(final_recs) < RECOMMENDATION_COUNT:
             # print(f"  User {user_id}: Total recs ({len(final_recs)}) < {RECOMMENDATION_COUNT}. Attempting to fill from remaining random items.")
             available_random_candidates = list(all_candidate_items_for_random - added_items)
             num_needed = RECOMMENDATION_COUNT - len(final_recs)
             num_to_sample = min(num_needed, len(available_random_candidates))
             if num_to_sample > 0:
                  try:
                     random_items = random.sample(available_random_candidates, num_to_sample)
                     for item_id in random_items:
                          if len(final_recs) >= RECOMMENDATION_COUNT: break
                          # item_id not in added_items is guaranteed by sampling from available_random_candidates
                          final_recs.append(item_id)
                          added_items.add(item_id)
                  except ValueError as e:
                       print(f"WARNING: Could not sample {num_to_sample} random item(s) for user {user_id} from {len(available_random_candidates)} available candidates during fill step: {e}")


         # 8. Final Truncation to ensure exactly RECOMMENDATION_COUNT (if possible)
         #    If there weren't enough unique, unseen, processable items available *at all*,
         #    the list might be shorter than RECOMMENDATION_COUNT. This is expected.
         final_recs = final_recs[:RECOMMENDATION_COUNT]

         # Store the final recommendations for this submission user
         recommendations_for_submission_users[user_id] = ",".join(final_recs)

    print(f"\nGenerated blended recommendations for {len(recommendations_for_submission_users)} users who had candidates.")

else:
     print("No model predictions collected. Recommendations dictionary will rely solely on Popular/Random if applicable.")
     # If no model predictions, generate based purely on Popular/Random for all submission users
     print("Generating Popular/Random fallback recommendations for all submission users.")
     all_candidate_items_for_random = set(all_processable_item_ids)

     for user_id in users_to_process:
         seen_items = user_train_items_dict.get(user_id, set())
         final_recs = []
         added_items = set(seen_items)

         # Add Popular Items
         popular_items_added_count = 0
         for item_id in popular_item_ids:
             if popular_items_added_count >= NUM_POPULAR_ITEMS_TO_BLEND: break
             if len(final_recs) >= RECOMMENDATION_COUNT: break
             if item_id not in added_items:
                 final_recs.append(item_id)
                 added_items.add(item_id)
                 popular_items_added_count += 1

         # Add Random Items
         random_items_added_count = 0
         if NUM_RANDOM_ITEMS_TO_BLEND > 0:
             available_random_candidates = list(all_candidate_items_for_random - added_items)
             num_needed = RECOMMENDATION_COUNT - len(final_recs)
             num_to_sample = min(NUM_RANDOM_ITEMS_TO_BLEND, num_needed, len(available_random_candidates))
             if num_to_sample > 0:
                 try:
                     random_items = random.sample(available_random_candidates, num_to_sample)
                     for item_id in random_items:
                          if len(final_recs) >= RECOMMENDATION_COUNT: break
                          if item_id not in added_items: # Redundant due to sampling, but harmless
                               final_recs.append(item_id)
                               added_items.add(item_id)
                               random_items_added_count += 1
                 except ValueError as e:
                      print(f"WARNING: Could not sample {num_to_sample} random item(s) for user {user_id} (fallback): {e}")

         # Fill remaining with more popular if needed
         if len(final_recs) < RECOMMENDATION_COUNT:
             for item_id in popular_item_ids:
                 if len(final_recs) >= RECOMMENDATION_COUNT: break
                 if item_id not in added_items:
                      final_recs.append(item_id)
                      added_items.add(item_id)

         # Fill remaining with more random if needed
         if len(final_recs) < RECOMMENDATION_COUNT:
             available_random_candidates = list(all_candidate_items_for_random - added_items)
             num_needed = RECOMMENDATION_COUNT - len(final_recs)
             num_to_sample = min(num_needed, len(available_random_candidates))
             if num_to_sample > 0:
                  try:
                     random_items = random.sample(available_random_candidates, num_to_sample)
                     for item_id in random_items:
                          if len(final_recs) >= RECOMMENDATION_COUNT: break
                          final_recs.append(item_id)
                          added_items.add(item_id)
                  except ValueError as e:
                       print(f"WARNING: Could not sample {num_to_sample} random item(s) for user {user_id} (fallback fill): {e}")


         final_recs = final_recs[:RECOMMENDATION_COUNT]
         recommendations_for_submission_users[user_id] = ",".join(final_recs)

     print(f"Generated fallback recommendations for {len(recommendations_for_submission_users)} users.")


# --- Prepare the final submission DataFrame using sample_submission.csv structure ---
print("\nMapping recommendations to sample submission format...")
final_submission_df = sample_submission_df.copy()

# Ensure all users from sample_submission_df are present in the output.
# Use the recommendations_for_submission_users dictionary to get the generated recommendations
# for the users that are in sample_submission_df.
# Users in sample_submission_df for whom no predictions were generated (e.g., not in user features, or no candidates)
# or were skipped by `continue` will get an empty string recommendation (default in .get()).
# Important: Iterate through the original list of users_to_process to ensure all are covered.
final_submission_df[config.ITEM_ID_COL] = final_submission_df[config.USER_ID_COL].apply(
    lambda user_id: recommendations_for_submission_users.get(user_id, "") # Get blended recs, default to empty string if user was skipped or had no candidates
)


print(f"Final submission DataFrame prepared for {len(final_submission_df)} users.")

# --- Save Submission File ---
try:
    print(f"\nSaving blended submission file to {blended_submission_path}...")
    final_submission_df.to_csv(blended_submission_path, index=False)
    print("Blended submission file saved successfully.")
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred while saving the blended submission: {e}")
    traceback.print_exc()


print("\n--- Blended Submission Generation Finished ---")
print(f"Total script execution time: {time.time() - start_time:.2f} seconds.")