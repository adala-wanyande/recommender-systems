# prepare_test_data.py

import pandas as pd
import numpy as np
import os
import time
import joblib
import scipy.sparse as sp
import json
import pickle # For loading pkl files
# Keep transformer classes for type checking or direct use if needed
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import traceback

# Import configuration
import config

print("--- Preparing Test Data for Prediction ---")
start_time = time.time()

# --- Configuration and Paths ---
raw_test_data_path = config.TEST_CSV_PATH
output_test_features_path = config.X_TEST_HYBRID_PATH

# Paths to load artifacts from training pipeline (ensure these match your config.py)
user_features_df_path = config.USER_FEATURES_CSV_PATH # User features pre-computed from training data
user_scaler_path = config.USER_NUMERICAL_SCALER_PKL_PATH
user_encoder_path = config.USER_CATEGORICAL_ENCODER_PKL_PATH
user_svd_path = config.USER_CATEGORICAL_SVD_PKL_PATH
user_num_cols_path = config.USER_NUM_COLS_FOR_SCALING_JSON_PATH
user_cat_cols_path = config.USER_CAT_COLS_FOR_ENCODING_JSON_PATH

# Item feature processing artifacts
# NOTE: Your config.py specifies X_ITEMS_REDUCED_NPY_PATH (.npy). Assuming it's a dense numpy array.
# If item_processor saved a sparse matrix (.npz), you must change the config path and use sp.load_npz
item_features_matrix_path = config.X_ITEMS_REDUCED_NPY_PATH
item_id_to_index_path = config.ITEM_ID_TO_INDEX_JSON_PATH
# Item column lists might be useful for understanding the feature space if you need to debug cold items
# item_num_cols_path = config.ITEM_NUM_COLS_JSON_PATH
# item_cat_cols_path = config.ITEM_CAT_COLS_JSON_PATH
# item_binary_cols_path = config.ITEM_BINARY_COLS_JSON_PATH

# Path to X_train_hybrid to get its shape (for verifying dimensions)
x_train_hybrid_ref_path = config.X_TRAIN_HYBRID_PATH

# Ensure output directory exists
os.makedirs(os.path.dirname(output_test_features_path), exist_ok=True)

print(f"Loading raw test data from: {raw_test_data_path}")
print(f"Loading training artifacts from: {config.INTERMEDIATE_DATA_PATH} etc.")
print(f"Saving test features to: {output_test_features_path}")
print("-" * 50)

# --- Load Raw Test Data ---
try:
    test_df = pd.read_csv(raw_test_data_path)
    # Ensure correct column types (especially user/item IDs matching how they were used in training prep)
    test_df[config.USER_ID_COL] = test_df[config.USER_ID_COL].astype(str)
    test_df[config.ITEM_ID_COL] = test_df[config.ITEM_ID_COL].astype(str)

    print(f"Raw test data loaded. Shape: {test_df.shape}")
    if test_df.empty:
        print("Warning: Test data is empty.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Raw test data file not found: {e}")
    exit()
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during raw test data loading: {e}")
    traceback.print_exc()
    exit()

# --- Load Training Artifacts ---
print("\nLoading fitted transformers and data from training pipeline...")
try:
    # Load User Artifacts
    user_features_df_train = pd.read_csv(user_features_df_path)
    user_features_df_train[config.USER_ID_COL] = user_features_df_train[config.USER_ID_COL].astype(str) # Ensure type consistency
    user_features_df_train = user_features_df_train.set_index(config.USER_ID_COL) # Set user_id as index for easy lookup

    user_numerical_scaler = joblib.load(user_scaler_path)
    user_categorical_encoder = joblib.load(user_encoder_path)
    user_categorical_svd = joblib.load(user_svd_path)

    with open(user_num_cols_path, 'r') as f:
        user_num_cols_for_scaling = json.load(f)
    with open(user_cat_cols_path, 'r') as f:
        user_cat_cols_for_encoding = json.load(f)

    # Load Item Artifacts
    # Assuming X_items_reduced was saved as a dense numpy array (.npy)
    X_items_reduced_train = np.load(item_features_matrix_path)
    print(f"Loaded X_items_reduced_train shape: {X_items_reduced_train.shape}")

    with open(item_id_to_index_path, 'r') as f:
        item_id_to_index = json.load(f) # String keys from json will be item IDs
    print(f"Loaded item_id_to_index mapping with {len(item_id_to_index)} items.")


    # Get the exact expected total feature dimension from the training data's shape
    print(f"Loading training hybrid matrix from {x_train_hybrid_ref_path} to get shape...")
    try:
        x_train_hybrid_shape_ref = sp.load_npz(x_train_hybrid_ref_path).shape
        total_hybrid_features_expected_dim = x_train_hybrid_shape_ref[1]
        print(f"Derived expected total feature dimension from X_train_hybrid: {total_hybrid_features_expected_dim}")
    except FileNotFoundError:
         print(f"FATAL ERROR: X_train_hybrid.npz not found at {x_train_hybrid_ref_path}. Cannot determine expected feature dimension.")
         exit()
    except Exception as e:
         print(f"FATAL ERROR: Could not load or inspect X_train_hybrid.npz from {x_train_hybrid_ref_path}: {e}")
         traceback.print_exc()
         exit()


    print("Training artifacts loaded successfully.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Training artifact file not found: {e}")
    traceback.print_exc()
    exit()
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during artifact loading: {e}")
    traceback.print_exc()
    exit()

# --- Calculate Feature Dimensions and Define Cold Start Vectors ---
print("\nCalculating feature dimensions and defining cold vectors...")
try:
    # 1. Calculate Warm User Feature Dimension
    # This is the dimension of the combined numerical (scaled) and categorical (SVD) features
    # The output of scaler.transform on N columns is N columns.
    # The output of svd.transform on encoded sparse matrix is svd.n_components columns.
    warm_user_feature_dim = len(user_num_cols_for_scaling) + user_categorical_svd.n_components

    # 2. Calculate Warm Item Feature Dimension
    # This is the dimension of the processed item features from item_processor (X_items_reduced)
    warm_item_feature_dim = X_items_reduced_train.shape[1]

    # 3. Verify Consistency (Crucial Check!)
    calculated_total_warm_dim = warm_user_feature_dim + warm_item_feature_dim
    print(f"Calculated warm user feature dim: {warm_user_feature_dim}")
    print(f"Calculated warm item feature dim: {warm_item_feature_dim}")
    print(f"Sum of warm dims: {calculated_total_warm_dim}")

    if calculated_total_warm_dim != total_hybrid_features_expected_dim:
        print(f"FATAL ERROR: Calculated total dimension from warm features ({calculated_total_warm_dim}) does NOT match expected total dimension from X_train_hybrid ({total_hybrid_features_expected_dim}).")
        print("This indicates an inconsistency in your training data preparation pipeline (hybrid_matrix_assembler.py or its inputs).")
        print("The feature dimension mismatch errors you were seeing stem from this underlying inconsistency.")
        exit() # Stop execution because the pipeline is fundamentally broken

    print("Training pipeline dimensions are consistent.")

    # 4. Define Cold Start Feature Vectors
    # A cold user needs features that, when combined with a *warm* item, result in the total dimension.
    # A cold item needs features that, when combined with a *warm* user, result in the total dimension.
    cold_user_vector_dim = total_hybrid_features_expected_dim - warm_item_feature_dim
    cold_item_vector_dim = total_hybrid_features_expected_dim - warm_user_feature_dim

    # Create sparse zero vectors for cold start
    # They must have shape (1, dimension) to be horizontally stacked with other sparse matrices
    cold_user_features_transformed = sp.csr_matrix((1, cold_user_vector_dim))
    cold_item_features_transformed = sp.csr_matrix((1, cold_item_vector_dim))

    print(f"Cold user vector dimension: {cold_user_vector_dim}")
    print(f"Cold item vector dimension: {cold_item_vector_dim}")

except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during dimension calculation or cold vector definition: {e}")
    traceback.print_exc()
    exit()


# --- Prepare Test Features ---
print("\nPreparing test feature matrix...")
try:
    test_features_list = [] # List to collect sparse row matrices

    # Iterate through each interaction in the test set
    print(f"Processing {len(test_df)} test interactions...")
    # Use tqdm for progress if installed: from tqdm import tqdm
    # for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
    for index, row in test_df.iterrows():
        user_id = str(row[config.USER_ID_COL]) # Ensure string type
        item_id = str(row[config.ITEM_ID_COL]) # Ensure string type
        # timestamp = row[config.TIMESTAMP_COL] # If you added timestamp features, process here

        # Use temporary variables for features for this row
        user_features_for_pair = None
        item_features_for_pair = None

        # --- Get and Transform User Features ---
        if user_id in user_features_df_train.index:
            user_features_row = user_features_df_train.loc[[user_id]] # Get DataFrame row for this user (maintains column names)

            # Select numerical and categorical columns used for transformation
            # Use .copy() to avoid SettingWithCopyWarning if you modify it later (not expected here)
            # Passing DataFrame slice might help with UserWarnings about feature names
            user_num_features_df = user_features_row[user_num_cols_for_scaling]
            user_cat_features_df = user_features_row[user_cat_cols_for_encoding]

            # Apply fitted transformers
            user_num_transformed = user_numerical_scaler.transform(user_num_features_df) # Returns numpy array
            user_cat_encoded = user_categorical_encoder.transform(user_cat_features_df) # Returns sparse matrix

            # Apply SVD (SVD input should be the OHE sparse output)
            user_svd_transformed = user_categorical_svd.transform(user_cat_encoded) # Returns numpy array

            # Combine transformed user features (matching the order/method in hybrid_matrix_assembler)
            # Ensure all parts are sparse before stacking
            user_features_for_pair = sp.hstack([
                 sp.csr_matrix(user_num_transformed), # Convert numerical numpy output to sparse
                 sp.csr_matrix(user_svd_transformed) # Convert SVD numpy output to sparse
                 # Make sure the order here is exactly the same as in your hybrid_matrix_assembler.py
                 # E.g., if hybrid_matrix_assembler stacks SVD first then numerical:
                 # sp.hstack([sp.csr_matrix(user_svd_transformed), sp.csr_matrix(user_num_transformed)])
            ])

            # Validate user feature dimension for this pair (Optional, but good for debugging)
            if user_features_for_pair.shape[1] != warm_user_feature_dim:
                 print(f"ERROR (Pair {index}): Warm user feature dimension mismatch for user {user_id}. Expected {warm_user_feature_dim}, got {user_features_for_pair.shape[1]}. Using cold user vector.")
                 user_features_for_pair = cold_user_features_transformed # Fallback to cold user vector on unexpected dimension

        else:
            # Cold User Strategy
            # print(f"Warning: Cold user encountered: {user_id}. Using default features.") # Can be noisy
            user_features_for_pair = cold_user_features_transformed # Use pre-defined cold user vector


        # --- Get Item Features ---
        if item_id in item_id_to_index:
            item_index = item_id_to_index[item_id]
            # Retrieve the row from the loaded item feature matrix
            # Ensure the slice returns a 1xN matrix (use [index:index+1, :])
            # Ensure it's a sparse matrix before stacking

            # X_items_reduced_train is dense (.npy), so slice returns dense numpy array
            item_features_dense_row = X_items_reduced_train[item_index:item_index+1, :]
            # Convert the dense numpy row slice to a sparse matrix
            item_features_for_pair = sp.csr_matrix(item_features_dense_row)


            # Validate item feature dimension for this pair (Optional)
            if item_features_for_pair.shape[1] != warm_item_feature_dim:
                 print(f"ERROR (Pair {index}): Warm item feature dimension mismatch for item {item_id}. Expected {warm_item_feature_dim}, got {item_features_for_pair.shape[1]}. Using cold item vector.")
                 item_features_for_pair = cold_item_features_transformed # Fallback to cold item vector

        else:
            # Cold Item Strategy
            # print(f"Warning: Cold item encountered: {item_id}. Using default features.") # Can be noisy
            item_features_for_pair = cold_item_features_transformed # Use pre-defined cold item vector

        # --- Assemble Hybrid Feature Vector for the Pair ---
        # Check that both parts are available (they should be due to cold start strategies)
        if user_features_for_pair is not None and item_features_for_pair is not None:
             # Concatenate user and item features horizontally
             # The order MUST match the assembly order in hybrid_matrix_assembler.py
             # Assuming user features are first, then item features:
             hybrid_features_row = sp.hstack([user_features_for_pair, item_features_for_pair])

             # --- Final Validation for the Pair (Crucial) ---
             if hybrid_features_row.shape[1] != total_hybrid_features_expected_dim:
                  # This indicates a serious issue in dimension calculation or stacking logic
                  print(f"FATAL ERROR (Pair {index}): Final hybrid feature dimension mismatch for pair ({user_id}, {item_id}). Expected {total_hybrid_features_expected_dim}, got {hybrid_features_row.shape[1]}")
                  print("This pair will be skipped to prevent downstream errors.")
                  # Do NOT append this row if dimension is wrong
                  continue # Skip this pair and proceed to the next

             test_features_list.append(hybrid_features_row)
        else:
             print(f"WARNING (Pair {index}): Could not assemble features for pair ({user_id}, {item_id}) due to missing components.")
             # Skip this pair if either component is missing

    # Stack all valid row matrices into a single sparse matrix
    if test_features_list:
        X_test_hybrid = sp.vstack(test_features_list)
        print(f"Assembled X_test_hybrid shape: {X_test_hybrid.shape}")
    else:
        print("No valid test interactions processed. X_test_hybrid is empty.")
        # Create an empty sparse matrix with the correct number of columns
        X_test_hybrid = sp.csr_matrix((0, total_hybrid_features_expected_dim if 'total_hybrid_features_expected_dim' in locals() else 0))


    # --- Save the Test Feature Matrix ---
    if X_test_hybrid.shape[0] > 0:
         # Always save test features as sparse (.npz) for consistency with training features
         sp.save_npz(output_test_features_path, X_test_hybrid)
         print(f"X_test_hybrid saved successfully to {output_test_features_path}")
    else:
         print("Skipping save: X_test_hybrid is empty or no valid rows were processed.")


except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during test feature preparation: {e}")
    traceback.print_exc()
    exit()


print("\n--- Test Data Preparation Script Finished ---")
print(f"Total execution time: {time.time() - start_time:.2f} seconds.")