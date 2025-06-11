# generate_multiple_submissions.py

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
from collections import defaultdict # Needed for user recommendations dictionary

# Import transformer classes needed for loading
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
# Import model classes needed for loading (ensure all potential trained models are imported)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier # CatBoost needs its specific class
# Add other model imports if necessary (e.g., RandomForestClassifier, LogisticRegression etc.)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import config
import traceback

print("--- Generating Multiple Submission Files with Popularity Blending ---")
start_time = time.time()

# --- Configuration and Paths ---
sample_submission_path = config.SAMPLE_SUBMISSION_CSV_PATH
train_csv_path = config.TRAIN_CSV_PATH
clean_item_meta_path = config.CLEANED_METADATA_CSV_PATH

user_features_path = config.USER_FEATURES_CSV_PATH
user_numerical_scaler_path = config.USER_NUMERICAL_SCALER_PKL_PATH
user_categorical_encoder_path = config.USER_CATEGORICAL_ENCODER_PKL_PATH
user_categorical_svd_path = config.USER_CATEGORICAL_SVD_PKL_PATH
user_num_cols_path = config.USER_NUM_COLS_FOR_SCALING_JSON_PATH
user_cat_cols_path = config.USER_CAT_COLS_FOR_ENCODING_JSON_PATH

x_items_reduced_path = config.X_ITEMS_REDUCED_NPY_PATH

item_id_to_index_json_path = config.ITEM_ID_TO_INDEX_JSON_PATH
index_to_item_id_json_path = config.INDEX_TO_ITEM_ID_JSON_PATH

trained_models_dir = config.TRAINED_MODELS_PATH
submission_output_dir = config.SUBMISSION_OUTPUT_DIR

# New config for blending
RECOMMENDATION_COUNT = config.RECOMMENDATION_COUNT
NUM_POPULAR_ITEMS_TO_BLEND = config.NUM_POPULAR_ITEMS_TO_BLEND

# Ensure output directory exists
os.makedirs(submission_output_dir, exist_ok=True)


# --- Load All Necessary Data and Pipeline Components (excluding the giant prediction matrix) ---
print("Loading data and pipeline components...")
X_items_reduced = None
item_id_to_index = None
index_to_item_id = None
sample_submission_df = None
train_df = None
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
    print(f"Loaded sample submission ({len(sample_submission_df)} rows)")

    train_df = pd.read_csv(train_csv_path)
    train_df[config.USER_ID_COL] = train_df[config.USER_ID_COL].astype(str)
    train_df[config.ITEM_ID_COL] = train_df[config.ITEM_ID_COL].astype(str)
    print(f"Loaded training data ({len(train_df)} interactions)")


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


    # --- Load processed item features (.npy file) ---
    try:
        print(f"Loading dense item features from {x_items_reduced_path} (.npy)...")
        X_items_reduced = np.load(x_items_reduced_path)
        if not isinstance(X_items_reduced, np.ndarray):
             raise ValueError(f"Loaded data from {x_items_reduced_path} is not a numpy array (expected dense array).")
        print(f"Successfully loaded dense item features (.npy) shape: {X_items_reduced.shape}")

    except FileNotFoundError:
        raise FileNotFoundError(f"FATAL ERROR: Item features file not found at {x_items_reduced_path} (.npy). Please ensure this file exists.")
    except Exception as npy_e:
        raise RuntimeError(f"FATAL ERROR: Failed to load item features from {x_items_reduced_path} (.npy): {type(npy_e).__name__}: {npy_e}")

    print("\n!!! WARNING: Loaded item features as a dense NumPy array (.npy).")
    print("!!! This will cause the prediction matrix to be dense, significantly increasing memory usage compared to sparse (.npz).")
    print("!!! It is highly recommended to save item features as a sparse matrix using scipy.sparse.save_npz and update config.")
    print("-" * 80)


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


# --- Prepare Global Data Needed for Feature Generation ---

print("\nPreparing global data structures for prediction...")

submission_users = sample_submission_df[config.USER_ID_COL].unique()
print(f"Generating recommendations for {len(submission_users)} unique users from submission file.")

all_item_ids_meta = clean_item_meta_df[config.ITEM_ID_COL].unique()
if not item_id_to_index:
     print("FATAL ERROR: item_id to index map is empty after loading. Cannot proceed with feature generation.")
     sys.exit(1)
all_processable_item_ids = [item for item in all_item_ids_meta if item in item_id_to_index]
print(f"Total processable items (in item features): {len(all_processable_item_ids)}")

if not all_processable_item_ids:
     print("FATAL ERROR: No processable items found after filtering against item features. Cannot proceed with feature generation.")
     sys.exit(1)

if X_items_reduced is None or X_items_reduced.shape[0] != len(item_id_to_index):
    print(f"FATAL ERROR: Loaded item features matrix has {X_items_reduced.shape[0] if X_items_reduced is not None else 'None'} rows, but item ID map has {len(item_id_to_index)} items. These should match.")
    sys.exit(1)

item_features_all_processable_indices = [item_id_to_index[item_id] for item_id in all_processable_item_ids]
item_features_all_processable = X_items_reduced[item_features_all_processable_indices, :]
print(f"Shape of item features for all processable items: {item_features_all_processable.shape}")
print(f"Type of item features for all processable items: {type(item_features_all_processable)}")

# Create a set of (user, item) pairs from the training data for quick lookup
train_interactions = set(tuple(row) for row in train_df[[config.USER_ID_COL, config.ITEM_ID_COL]].values)
print(f"Loaded {len(train_interactions)} unique train interactions.")

# Process users from the submission file to get their features transformed ONCE
submission_users_df_for_merge = pd.DataFrame({config.USER_ID_COL: submission_users}).astype({config.USER_ID_COL: str})

submission_user_features = pd.merge(
    submission_users_df_for_merge,
    user_features_df,
    on=config.USER_ID_COL,
    how='left'
)

feature_cols_for_nan_check = [col for col in user_num_cols + user_cat_cols if col in submission_user_features.columns]
cold_start_users_in_submission = submission_user_features[submission_user_features[feature_cols_for_nan_check].isnull().all(axis=1)][config.USER_ID_COL].tolist()

if cold_start_users_in_submission:
     print(f"WARNING: {len(cold_start_users_in_submission)} users in submission were not found in training data's user features or have all feature columns as NaN.")


print("Handling potential NaNs in user features before transformation...")
user_features_to_process = submission_user_features.copy()
for col in user_num_cols:
    if col in user_features_to_process.columns:
        user_features_to_process[col] = user_features_to_process[col].fillna(0)

for col in user_cat_cols:
     if col in user_features_to_process.columns:
          user_features_to_process[col] = user_features_to_process[col].fillna('__MISSING__')

print("Applying user feature transformers...")
user_num_scaled = user_numerical_scaler.transform(user_features_to_process[user_num_cols])
user_num_scaled_dense = user_num_scaled

user_cat_encoded_sparse = user_categorical_encoder.transform(user_features_to_process[user_cat_cols])

if user_categorical_svd is not None:
     user_cat_svd_reduced = user_categorical_svd.transform(user_cat_encoded_sparse)
     user_cat_features_transformed_dense = user_cat_svd_reduced
     print("User Categorical SVD applied.")
else:
     user_cat_features_transformed_dense = user_cat_encoded_sparse.toarray()
     print("Note: User Categorical SVD was not applied during training or loading, using dense one-hot encoded features.")

print("User feature transformation completed.")

user_id_to_feature_row_index = {user_id: i for i, user_id in enumerate(submission_user_features[config.USER_ID_COL])}

# --- Calculate Item Popularity (Globally) ---
# We calculate popularity based on interactions in the training data
print("\nCalculating item popularity from training data...")
item_popularity = train_df[config.ITEM_ID_COL].value_counts()
# Get list of item IDs sorted by popularity (most popular first)
popular_item_ids = item_popularity.index.tolist()

# Filter popular items to include only those that made it into our processed item features
# This ensures we only try to blend in items we have features for
popular_item_ids = [item for item in popular_item_ids if item in all_processable_item_ids]
print(f"Calculated popularity for {len(item_popularity)} unique items in train.")
print(f"Filtered popular items list to {len(popular_item_ids)} items that are in processed item features.")


print("Global data preparation complete.")


# --- Iterate Through Trained Models, Predict in Batches, and Save Submission ---
print("\nProcessing each trained model...")

model_files = [f for f in os.listdir(trained_models_dir) if f.endswith('.joblib')]
print(f"Found {len(model_files)} model files in {trained_models_dir}")

if not model_files:
    print("No trained models found in the specified directory. Exiting.")
    sys.exit(1)

base_submission_df = sample_submission_df.copy()
base_submission_df[config.ITEM_ID_COL] = ""


for model_filename in model_files:
    model_name = os.path.splitext(model_filename)[0]
    current_model_path = os.path.join(trained_models_dir, model_filename)
    current_submission_path = os.path.join(submission_output_dir, f'{model_name}_submission.csv')

    print(f"\n--- Processing model: {model_name} ---")
    model_process_start_time = time.time()

    all_model_predictions_list = []

    try:
        print(f"Loading model from {current_model_path}")
        model = joblib.load(current_model_path)
        print("Model loaded.")

        prediction_batch_size_pairs = 500000 # Adjust as needed

        current_prediction_pairs_batch = []
        current_prediction_features_batch = []
        total_pairs_processed_for_prediction = 0
        batch_counter = 0

        user_processing_start_time = time.time()
        print(f"Starting pair generation and batching for {len(submission_users)} users...")

        for user_idx, user_id in enumerate(submission_users):
             if (user_idx + 1) % 100 == 0:
                  print(f"  Generating pairs for user {user_idx + 1}/{len(submission_users)}...")

             user_feat_row_index = user_id_to_feature_row_index.get(user_id)

             if user_feat_row_index is None:
                  continue # Skip users not in processed user features

             user_num_feat_single_dense = user_num_scaled_dense[user_feat_row_index, :].reshape(1, -1)
             user_cat_feat_single_transformed_dense = user_cat_features_transformed_dense[user_feat_row_index, :].reshape(1, -1)

             user_train_items = {item for u, item in train_interactions if u == user_id}

             # Define candidate items: all processable items NOT interacted with in training
             # IMPORTANT: We generate features/predictions for *all* unseen processable items first,
             # and then do the blending/ranking later based on model scores and popularity.
             candidate_item_ids = [item for item in all_processable_item_ids if (user_id, item) not in user_train_items]

             if not candidate_item_ids:
                 continue # No valid candidate items for this user

             candidate_item_indices = [item_id_to_index[item] for item in candidate_item_ids]
             item_feats_for_candidates_dense = item_features_all_processable[candidate_item_indices, :]

             num_candidates = len(candidate_item_ids)
             user_num_feats_repeated_dense = np.repeat(user_num_feat_single_dense, num_candidates, axis=0)
             user_cat_feats_repeated_transformed_dense = np.repeat(user_cat_feat_single_transformed_dense, num_candidates, axis=0)

             combined_user_item_features_dense = np.hstack([
                 user_num_feats_repeated_dense,
                 user_cat_feats_repeated_transformed_dense,
                 item_feats_for_candidates_dense
             ])

             current_prediction_pairs_batch.extend([(user_id, item) for item in candidate_item_ids])
             current_prediction_features_batch.append(combined_user_item_features_dense)

             if sum(len(arr) for arr in current_prediction_features_batch) >= prediction_batch_size_pairs or (user_idx == len(submission_users) - 1 and current_prediction_features_batch):
                 batch_counter += 1
                 print(f"\n  Processing prediction batch {batch_counter} with {len(current_prediction_pairs_batch)} pairs...")

                 try:
                     X_batch_dense = np.vstack(current_prediction_features_batch)
                     print(f"  Batch feature matrix shape: {X_batch_dense.shape}")
                 except MemoryError:
                      print(f"FATAL ERROR: MemoryError while stacking batch features for batch {batch_counter}. Batch size {prediction_batch_size_pairs} is too large.")
                      print("Consider reducing 'prediction_batch_size_pairs'. Skipping processing this model.")
                      all_model_predictions_list = [] # Clear any partial predictions
                      raise MemoryError("Batch feature stacking failed.")

                 try:
                     if hasattr(model, 'predict_proba'):
                          batch_scores = model.predict_proba(X_batch_dense)[:, 1]
                     elif hasattr(model, 'decision_function'):
                          batch_scores = model.decision_function(X_batch_dense)
                     else:
                          raise RuntimeError("Loaded model does not have predict_proba or decision_function.")

                     for j in range(len(current_prediction_pairs_batch)):
                         user_id, item_id = current_prediction_pairs_batch[j]
                         score = batch_scores[j]
                         all_model_predictions_list.append((user_id, item_id, score))

                     total_pairs_processed_for_prediction += len(current_prediction_pairs_batch)
                     print(f"  Batch {batch_counter} prediction completed. Total pairs processed for model {model_name}: {total_pairs_processed_for_prediction}")

                 except Exception as e:
                      print(f"FATAL ERROR: Error during prediction for batch {batch_counter} for model {model_name}: {e}")
                      traceback.print_exc()
                      print("Skipping processing this model.")
                      all_model_predictions_list = []
                      raise

                 del current_prediction_pairs_batch
                 del current_prediction_features_batch
                 del X_batch_dense
                 del batch_scores
                 gc.collect()

                 current_prediction_pairs_batch = []
                 current_prediction_features_batch = []

        user_processing_end_time = time.time()
        print(f"\nFinished pair generation and batching for all users for model {model_name} in {user_processing_end_time - user_processing_start_time:.2f} seconds.")
        print(f"Total prediction pairs generated across all batches: {total_pairs_processed_for_prediction}")


        # --- Ranking, Blending with Popular Items, and Formatting Submission ---
        print("\nRanking items, blending with popular items, and formatting submission...")

        submission_recommendations = {} # Use a dictionary to store user_id -> list of recommended item_ids

        if all_model_predictions_list:
            predictions_df_for_ranking = pd.DataFrame(all_model_predictions_list, columns=[config.USER_ID_COL, config.ITEM_ID_COL, 'score'])
            print(f"Collected {len(predictions_df_for_ranking)} predictions for ranking.")

            # Group predictions by user
            user_predictions_groups = predictions_df_for_ranking.groupby(config.USER_ID_COL)

            # Iterate through each user to blend and select top N
            print(f"Blending popular items and selecting top {RECOMMENDATION_COUNT} for each user...")

            # Pre-fetch all user train items into a dict for faster lookup
            user_train_items_dict = defaultdict(set)
            for user_id, item_id in train_interactions:
                user_train_items_dict[user_id].add(item_id)


            for user_id, group_df in user_predictions_groups:
                 # Sort model predictions for this user by score descending
                 model_ranked_predictions = group_df.sort_values(by='score', ascending=False)[config.ITEM_ID_COL].tolist()

                 # Get items the user has already interacted with
                 seen_items = user_train_items_dict[user_id]

                 # --- Blending Logic ---
                 final_recs = []
                 added_items = set() # Keep track of items added to avoid duplicates

                 # 1. Add top N popular items (that user hasn't seen and are processable)
                 popular_items_added_count = 0
                 for item_id in popular_item_ids:
                     if item_id not in seen_items and item_id not in added_items:
                         final_recs.append(item_id)
                         added_items.add(item_id)
                         popular_items_added_count += 1
                         if popular_items_added_count >= NUM_POPULAR_ITEMS_TO_BLEND:
                             break

                 # 2. Fill remaining slots with top items from the model's ranked list
                 for item_id in model_ranked_predictions:
                     if len(final_recs) >= RECOMMENDATION_COUNT:
                         break # Stop if we have enough recommendations

                     if item_id not in seen_items and item_id not in added_items:
                         final_recs.append(item_id)
                         added_items.add(item_id)

                 # 3. (Optional) If we still don't have RECOMMENDATION_COUNT items,
                 #    add more popular items to fill up the list. This ensures exactly 10 items
                 #    if possible, preferring popular items over a shorter list.
                 #    Remove this block if shorter lists are acceptable when few candidates exist.
                 if len(final_recs) < RECOMMENDATION_COUNT:
                     for item_id in popular_item_ids:
                          if len(final_recs) >= RECOMMENDATION_COUNT:
                             break
                          if item_id not in seen_items and item_id not in added_items:
                              final_recs.append(item_id)
                              added_items.add(item_id)

                 # Store the final recommendations for this user
                 # Convert the list of item_ids to a comma-separated string
                 submission_recommendations[user_id] = ",".join(final_recs)

            print(f"Generated and blended recommendations for {len(submission_recommendations)} users.")

        else:
             print("No valid predictions collected. Recommendations dictionary will be empty.")


        # Prepare the final submission DataFrame based on the original sample submission users
        final_submission_df = base_submission_df.copy()
        final_submission_df[config.ITEM_ID_COL] = final_submission_df[config.USER_ID_COL].apply(
            lambda user_id: submission_recommendations.get(user_id, "") # Get blended recs, default to empty string
        )


        # --- Save Submission File ---
        try:
            print(f"Saving submission file to {current_submission_path}...")
            final_submission_df.to_csv(current_submission_path, index=False)
            print("Submission file saved.")
        except Exception as e:
            print(f"FATAL ERROR: An unexpected error occurred while saving submission for model {model_name}: {e}")
            traceback.print_exc()


    except MemoryError as me:
         # Specific handling for MemoryError during batch processing
         print(f"Caught MemoryError for model {model_name}. This model's submission will not be generated.")
         print(me)
         # Clean up model and related batch data if possible
         del model # Explicitly delete the model
         gc.collect()
         # continue to the next model file in the loop
         continue # Jumps to the next iteration of the for model_filename loop


    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred while processing model {model_name}: {e}")
        traceback.print_exc()
        print(f"Skipping remaining processing for model {model_name}.")
        # Clean up model
        if 'model' in locals():
             del model
        gc.collect()
        # continue to the next model file in the loop
        continue # Jumps to the next iteration of the for model_filename loop


    model_process_end_time = time.time()
    print(f"Model {model_name} processing completed in {model_process_end_time - model_process_start_time:.2f} seconds.")
    print("-" * 50)

    # --- Clean up model-specific objects to free memory before next model ---
    # These del statements are outside the except block but inside the loop,
    # ensuring cleanup happens if no exception occurred *or* if the exception
    # was handled internally (like the MemoryError). The 'continue' skips these
    # if the exception handler decides to move to the next model.
    # Let's ensure deletion happens even after an error if the object was created.
    # We already added specific `del model` in error handlers.
    # Add checks for other objects if they might not be created before an error.
    if 'predictions_df_for_ranking' in locals(): del predictions_df_for_ranking
    # No need to delete group_df, it's just a view
    if 'user_predictions_groups' in locals(): del user_predictions_groups
    if 'user_train_items_dict' in locals(): del user_train_items_dict
    if 'submission_recommendations' in locals(): del submission_recommendations # The dict holding the final lists
    # final_submission_df might be large, ensure it's deleted
    if 'final_submission_df' in locals(): del final_submission_df
    gc.collect()


print("\n--- All Model Submissions Generation Attempted ---")
print(f"Total script execution time: {time.time() - start_time:.2f} seconds.")