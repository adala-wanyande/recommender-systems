# hybrid_matrix_assembler.py

import pandas as pd
import numpy as np
import os
import time
import glob
import shutil
import pickle
import json
import scipy.sparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD

# Import configuration from config.py
import config

def assemble_hybrid_matrix(
    train_pairs_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    user_numerical_scaler: StandardScaler | None,
    user_categorical_encoder: OneHotEncoder | None,
    user_categorical_svd: TruncatedSVD | None,
    user_num_cols_present_in_df: list, # List of numerical columns user_processor *actually* used
    user_cat_cols_present_in_df: list, # List of categorical columns user_processor *actually* used
    X_items_reduced: np.ndarray | scipy.sparse.csr_matrix | None, # Can be dense or sparse after SVD
    item_id_index_map_df: pd.DataFrame | None,
    stacking_batch_size: int,
    batch_temp_dir: str,
    x_train_hybrid_path: str,
    y_train_path: str
) -> tuple[scipy.sparse.csr_matrix | None, pd.Series | None]:
    """
    Assembles the final hybrid training feature matrix X_train_hybrid and target y_train.
    Merges user and item features onto training pairs, transforms user features,
    looks up item features, and stacks them horizontally and vertically in batches.

    Args:
        train_pairs_df (pd.DataFrame): DataFrame of training pairs (user_id, item_id, interaction).
        user_features_df (pd.DataFrame): DataFrame of raw user features (user_id, features...).
        user_numerical_scaler (StandardScaler | None): Fitted scaler for user numerical features.
        user_categorical_encoder (OneHotEncoder | None): Fitted encoder for user categorical features.
        user_categorical_svd (TruncatedSVD | None): Fitted SVD for encoded user categorical features.
        user_num_cols_present_in_df (list): List of numerical column names actually used by user_processor.
        user_cat_cols_present_in_df (list): List of categorical column names actually used by user_processor.
        X_items_reduced (np.ndarray | scipy.sparse.csr_matrix | None): Reduced item feature matrix (items as rows, features as columns).
        item_id_index_map_df (pd.DataFrame | None): DataFrame mapping item_id to its row index in X_items_reduced.
        stacking_batch_size (int): Batch size for processing and saving.
        batch_temp_dir (str): Temporary directory path for saving batch files.
        x_train_hybrid_path (str): Path to save the final X_train_hybrid sparse matrix (.npz).
        y_train_path (str): Path to save the final y_train Series (.csv).

    Returns:
        tuple[scipy.sparse.csr_matrix | None, pd.Series | None]: The final combined
        feature matrix X_train_hybrid and the target vector y_train on success,
        otherwise (None, None).

    Raises:
        ValueError: If essential input dataframes, matrices, or lists are None or invalid.
        Exception: For other errors during processing.
    """
    print("\n--- Step 6: Prepare Hybrid Training Matrix ---")

    # --- Validate Inputs ---
    if train_pairs_df is None or train_pairs_df.empty:
        raise ValueError("Input train_pairs_df is None or empty.")
    if user_features_df is None or user_features_df.empty:
        raise ValueError("Input user_features_df is None or empty.")
    if user_num_cols_present_in_df is None or not isinstance(user_num_cols_present_in_df, list):
         raise ValueError("Input user_num_cols_present_in_df must be a list.")
    if user_cat_cols_present_in_df is None or not isinstance(user_cat_cols_present_in_df, list):
         raise ValueError("Input user_cat_cols_present_in_df must be a list.")

    # Check item data inputs - they are essential for lookup
    if X_items_reduced is None or not hasattr(X_items_reduced, 'shape'):
         raise ValueError("Input X_items_reduced is None or invalid.")
    if X_items_reduced.shape[0] == 0 and X_items_reduced.shape[1] > 0:
         print("Warning: X_items_reduced has 0 rows but > 0 columns. This might indicate an issue.")
         # Decide if this is fatal. If there are train pairs but no item features, it's likely fatal.
         # If there are no train pairs, it might be okay. Let's check train_pairs_df size later.

    if item_id_index_map_df is None or item_id_index_map_df.empty:
         # Only fatal if train_pairs_df is not empty
         if not train_pairs_df.empty:
              raise ValueError("Input item_id_index_map_df is None or empty, but train_pairs_df is not empty. Cannot map items.")
         else:
              print("Warning: item_id_index_map_df is empty (but train_pairs_df is also empty). Proceeding, but will result in empty output.")
    elif config.ITEM_ID_COL not in item_id_index_map_df.columns or 'svd_index' not in item_id_index_map_df.columns:
         raise ValueError("Input item_id_index_map_df is missing required columns ('item_id', 'svd_index').")


    # Check transformers validity *if* they were needed based on column lists
    # If column list is non-empty, the corresponding transformer should ideally be non-None and fitted
    is_user_num_scaler_valid = True
    if len(user_num_cols_present_in_df) > 0 and (user_numerical_scaler is None or not hasattr(user_numerical_scaler, 'mean_')):
         print("FATAL ERROR: User numerical scaler is None or invalid despite numerical columns being present.")
         # This error is caught here rather than raised, so we can print detailed messages.
         # We will set a flag to stop processing.
         is_user_num_scaler_valid = False

    is_user_cat_encoder_valid = True
    if len(user_cat_cols_present_in_df) > 0 and (user_categorical_encoder is None or not hasattr(user_categorical_encoder, 'categories_')):
         print("FATAL ERROR: User categorical encoder is None or invalid despite categorical columns being present.")
         is_user_cat_encoder_valid = False

    is_user_cat_svd_valid = True
    n_components_expected = (user_categorical_svd.n_components if user_categorical_svd is not None and hasattr(user_categorical_svd, 'n_components') else 0)
    if len(user_cat_cols_present_in_df) > 0 and (user_categorical_encoder is not None and hasattr(user_categorical_encoder, 'categories_')): # SVD is expected if encoder was needed and valid
        if n_components_expected <= 0: # If encoder was valid but SVD components is <= 0
             print("Warning: User categorical SVD components is <= 0 despite categorical columns and valid encoder. SVD features will be 0-dimensional.")
             # Not a fatal error, just means no SVD features from this branch.
        elif user_categorical_svd is None or not (hasattr(user_categorical_svd, 'components_') and user_categorical_svd.n_components > 0):
             print("FATAL ERROR: User categorical SVD is None or invalid despite categorical columns and valid encoder, and SVD expected > 0 components.")
             is_user_cat_svd_valid = False

    # Overall check for transformer validity
    if not (is_user_num_scaler_valid and is_user_cat_encoder_valid and is_user_cat_svd_valid):
        print("FATAL ERROR: Essential user transformers are missing or invalid. Cannot proceed.")
        return None, None # Return None on fatal validation failure


    # Check if final output files already exist. If so, load and skip generation.
    final_matrix_is_ready = False # Assume not ready initially
    X_train_hybrid = None
    y_train = None

    if os.path.exists(x_train_hybrid_path) and os.path.exists(y_train_path):
         print(f"\nDetected final output files: {x_train_hybrid_path} and {y_train_path}. Attempting to load and skip processing.")
         try:
             # Attempt to load the final matrix and target
             X_train_hybrid = scipy.sparse.load_npz(x_train_hybrid_path)
             y_train_loaded = pd.read_csv(y_train_path)
             # Ensure y_train Series has the expected name
             if config.INTERACTION_COL in y_train_loaded.columns:
                 y_train = y_train_loaded[config.INTERACTION_COL]
             elif not y_train_loaded.empty: # Handle case where CSV might just be the series without header
                  y_train = y_train_loaded.iloc[:, 0] # Take first column
                  y_train.name = config.INTERACTION_COL # Assign name
             else: # Empty y_train file
                  y_train = pd.Series([], dtype=int, name=config.INTERACTION_COL)

             print(f"Loaded final X_train_hybrid shape: {X_train_hybrid.shape}")
             print(f"Loaded final y_train shape: {y_train.shape}")

             # Basic validation on loaded data
             if X_train_hybrid.shape[0] == len(y_train):
                  print("Loaded final data appears valid.")
                  final_matrix_is_ready = True # Set flag to indicate final matrix is ready
             else:
                  print("Warning: Loaded X_train_hybrid and y_train shapes do not match. Regenerating.")
                  X_train_hybrid = None; y_train = None # Ensure None on validation failure
                  final_matrix_is_ready = False # Loading failed, proceed with batch regeneration

         except Exception as e:
             print(f"Error loading final output files ({x_train_hybrid_path}, {y_train_path}): {e}. Proceeding with batch regeneration.")
             X_train_hybrid = None; y_train = None # Ensure None on error
             final_matrix_is_ready = False # Files not found, proceed with batching
    else:
         print("\nFinal output files not found. Proceeding with batch processing.")
         final_matrix_is_ready = False # Files not found, proceed with batching


    # Only perform batch processing if the final matrix wasn't loaded
    if not final_matrix_is_ready:
        start_time_batching = time.time()
        can_proceed_with_batching = True # Flag to stop batch loop on error

        # Ensure temporary batch directory exists and is empty
        if os.path.exists(batch_temp_dir):
            print(f"Clearing existing temporary batch directory: {batch_temp_dir}")
            try:
                shutil.rmtree(batch_temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temporary batch directory {batch_temp_dir}: {e}. Attempting to continue.")
        os.makedirs(batch_temp_dir, exist_ok=True)
        print(f"Created temporary batch directory: {batch_temp_dir}")

        num_samples = len(train_pairs_df) # Use length of training pairs as the number of samples
        num_batches = (num_samples + stacking_batch_size - 1) // stacking_batch_size
        print(f"\nProcessing {num_samples} samples in {num_batches} batches of size ~{stacking_batch_size}...")

        if num_samples == 0:
            print("Warning: No training pairs found. Creating empty output matrix and target.")
            # Skip batching loop, create empty outputs directly
            final_cols_for_empty = 0 # Will determine final columns later based on transformers
            # Need to estimate final columns even if no samples, for empty matrix shape
            num_user_num_cols_est = len(user_num_cols_present_in_df)
            num_user_cat_svd_cols_est = user_categorical_svd.n_components if user_categorical_svd is not None and hasattr(user_categorical_svd, 'n_components') else 0
            num_item_svd_cols_est = X_items_reduced.shape[1] if X_items_reduced is not None and hasattr(X_items_reduced, 'shape') else 0
            final_cols_for_empty = num_user_num_cols_est + num_user_cat_svd_cols_est + num_item_svd_cols_est
            final_cols_for_empty = max(0, final_cols_for_empty) # Ensure non-negative

            X_train_hybrid = scipy.sparse.csr_matrix((0, final_cols_for_empty))
            y_train = pd.Series([], dtype=int, name=config.INTERACTION_COL)
            can_proceed_with_batching = False # Skip loop
            final_matrix_is_ready = True # We have the empty output

        else: # num_samples > 0, proceed with batching
             # --- Prepare data for batching (merging raw features and item indices) ---
             print("\nMerging user features and item SVD index onto training pairs...")
             try:
                  # Merge item SVD index
                  # Ensure item_id columns are string for safe merge
                  train_pairs_with_svd_index = pd.merge(
                       train_pairs_df.astype({config.ITEM_ID_COL: str}),
                       item_id_index_map_df.astype({config.ITEM_ID_COL: str}), # Use the input map_df
                       on=config.ITEM_ID_COL,
                       how='left' # Keep all training pair rows
                   )

                  # Check if any item_ids in train_pairs were not found in item_id_index_map_df
                  if train_pairs_with_svd_index['svd_index'].isnull().sum() > 0:
                      missing_svd_count = train_pairs_with_svd_index['svd_index'].isnull().sum()
                      print(f"FATAL ERROR: {missing_svd_count} item_ids in training pairs were not found in item_id_index_map_df after merge. Cannot proceed as item features will be missing.")
                      can_proceed_with_batching = False # Set flag to stop
                  else:
                      # Convert svd_index to integer type
                      train_pairs_with_svd_index['svd_index'] = train_pairs_with_svd_index['svd_index'].astype(int)
                      print("svd_index column merged and converted to integer.")

                      # Merge user features
                      # Ensure user_id is string in both before merge
                      train_pairs_all = pd.merge(
                           train_pairs_with_svd_index.astype({config.USER_ID_COL: str}), # Ensure string
                           user_features_df.astype({config.USER_ID_COL: str}), # Use the input user_features_df
                           on=config.USER_ID_COL,
                           how='left' # Keep all pair rows, add user features
                       )
                      print(f"Merged training pairs (with item index) with user features. Shape: {train_pairs_all.shape}")
                      # Handle users in train_pairs_all that were NOT in user_features_df (Should not happen with proper pair generation)
                      # NaNs might appear in the *added* user feature columns if a user_id in train_pairs wasn't in user_features_df.
                      # Imputation for numerical user features was handled during user_processor.
                      # Categorical NaNs should be handled by the encoder's handle_unknown='ignore' or 'Missing' category (filled earlier).
                      # Let's just confirm no NaNs introduced from the merge itself on the core features we will use.
                      core_user_cols_to_check_nan = user_num_cols_present_in_df + user_cat_cols_present_in_df
                      if train_pairs_all[core_user_cols_to_check_nan].isnull().sum().sum() > 0:
                           print("Warning: NaNs found in core user feature columns after merging onto train_pairs. These should ideally be handled by the input user_features_df or transformers.")
                           # Impute numerical NaNs using median from the whole merged df as a fallback
                           for col in user_num_cols_present_in_df:
                                if col in train_pairs_all.columns and train_pairs_all[col].isnull().sum() > 0:
                                     median_val = train_pairs_all[col].median() if not train_pairs_all[col].isnull().all() else 0
                                     train_pairs_all[col] = train_pairs_all[col].fillna(median_val)
                           # Impute categorical NaNs with 'Missing' as a fallback
                           for col in user_cat_cols_present_in_df:
                                if col in train_pairs_all.columns and train_pairs_all[col].isnull().sum() > 0:
                                     train_pairs_all[col] = train_pairs_all[col].fillna('Missing')


                      # Separate Target Variable
                      if config.INTERACTION_COL not in train_pairs_all.columns:
                           print(f"FATAL ERROR: '{config.INTERACTION_COL}' column not found in train_pairs_all after merges. Cannot determine target variable.")
                           can_proceed_with_batching = False
                      else:
                          y_train_full = train_pairs_all[config.INTERACTION_COL] # Store full y before batching


                      # Select the raw user numerical and categorical columns
                      X_train_user_numerical_raw_all = train_pairs_all[user_num_cols_present_in_df].copy() # Use .copy()
                      X_train_user_categorical_raw_all = train_pairs_all[user_cat_cols_present_in_df].copy() # Use .copy()
                      svd_indices_all = train_pairs_all['svd_index'].values # Get numpy array of item SVD indices

                      # Explicitly delete intermediate/large DataFrames to free memory
                      del train_pairs_with_svd_index
                      del train_pairs_all
                      print("Deleted intermediate dataframes after extracting features and indices.")

             except Exception as e:
                  print(f"FATAL ERROR during initial merging and extraction for batching: {e}")
                  can_proceed_with_batching = False # Stop processing

             # --- Determine expected total columns based on available/fitted transformers ---
             # This must be consistent across all batches and match the dimension used in prediction.
             expected_total_cols_final = 0
             # Numerical user columns count (from scaler if fitted, otherwise 0 if list empty)
             if user_numerical_scaler is not None and hasattr(user_numerical_scaler, 'n_features_in_'):
                  expected_total_cols_final += user_numerical_scaler.n_features_in_[0] if isinstance(user_numerical_scaler.n_features_in_, tuple) else user_numerical_scaler.n_features_in_
             elif user_num_cols_present_in_df: # Fallback to list length if scaler wasn't fitted/available but list is non-empty
                  # This case indicates an error in user_processor, but we should still calculate the expected column count
                  print("Warning: Numerical scaler not available, but numerical columns list is non-empty. Using list length for expected columns.")
                  expected_total_cols_final += len(user_num_cols_present_in_df)
             # Note: If user_num_cols_present_in_df is None/empty, this correctly adds 0.

             # Categorical user SVD columns count (from SVD if fitted and >0 components, otherwise 0)
             if user_categorical_svd is not None and hasattr(user_categorical_svd, 'n_components') and user_categorical_svd.n_components > 0:
                  expected_total_cols_final += user_categorical_svd.n_components
             # Note: We don't add raw OneHotEncoder output dimension here, only the SVD reduced dimension


             # Item SVD/Reduced columns count (from Step 4, if available and >0 columns)
             if X_items_reduced is not None and hasattr(X_items_reduced, 'shape') and X_items_reduced.shape[1] > 0:
                   expected_total_cols_final += X_items_reduced.shape[1]

             print(f"Estimated final matrix column count: {expected_total_cols_final}")

             if expected_total_cols_final <= 0:
                  print("FATAL ERROR: Estimated total columns is <= 0 based on available/fitted transformers. Cannot create feature matrix.")
                  can_proceed_with_batching = False # Mark for failure


             # --- Batch Processing Loop ---
             if can_proceed_with_batching:
                  for batch_idx in range(num_batches):
                       start_idx = batch_idx * stacking_batch_size
                       end_idx = min((batch_idx + 1) * stacking_batch_size, num_samples)

                       # Check if processing should stop due to previous error
                       if not can_proceed_with_batching:
                           print(f"Skipping remaining batches due to previous error in batch {batch_idx+1}.")
                           break # Exit the batching loop

                       print(f"  Processing batch {batch_idx + 1}/{num_batches} (samples {start_idx} to {end_idx-1})...")

                       # Select data for the current batch from the full raw data
                       # Use .iloc[] which returns a view if possible
                       X_train_user_numerical_raw_batch = X_train_user_numerical_raw_all.iloc[start_idx:end_idx]
                       X_train_user_categorical_raw_batch = X_train_user_categorical_raw_all.iloc[start_idx:end_idx]
                       svd_indices_batch = svd_indices_all[start_idx:end_idx]
                       y_train_batch = y_train_full.iloc[start_idx:end_idx] # Collect target batch


                       feature_matrices_for_hstack_batch = [] # List to hold sparse matrices for horizontal stacking for this batch
                       current_batch_size = len(y_train_batch) # The actual number of samples in this batch


                       # Apply scaling using the *fitted* numerical user scaler
                       # Check if scaler is available AND fitted AND if there were numerical columns identified
                       if user_numerical_scaler is not None and hasattr(user_numerical_scaler, 'mean_') and user_num_cols_present_in_df: # Check list is non-empty
                            if X_train_user_numerical_raw_batch.shape[1] > 0: # Check if this batch slice has the numerical columns
                                 # Ensure data is numerical type before scaling
                                 try:
                                      # Use .loc to ensure column order matches scaler's fitted order if needed
                                      X_train_user_numerical_scaled_batch = user_numerical_scaler.transform(X_train_user_numerical_raw_batch.astype(float)) # TRANSFORM
                                      feature_matrices_for_hstack_batch.append(scipy.sparse.csr_matrix(X_train_user_numerical_scaled_batch))
                                 except Exception as e:
                                      print(f"    FATAL ERROR transforming user numerical features in batch {batch_idx+1}: {e}. Skipping remaining batches.")
                                      can_proceed_with_batching = False
                                      break # Exit loop
                            else: # Batch slice was empty of columns (should not happen if user_num_cols_present_in_df wasn't empty)
                                 # Add empty placeholder with dimension from fitted scaler
                                 scaler_output_cols = user_numerical_scaler.n_features_in_[0] if hasattr(user_numerical_scaler, 'n_features_in_') and isinstance(user_numerical_scaler.n_features_in_, tuple) else (user_numerical_scaler.n_features_in_ if hasattr(user_numerical_scaler, 'n_features_in_') else 0)
                                 print(f"    Warning: Numerical batch slice {batch_idx+1} resulted in 0 columns unexpectedly. Adding placeholder with {scaler_output_cols} columns.")
                                 feature_matrices_for_hstack_batch.append(scipy.sparse.csr_matrix((current_batch_size, scaler_output_cols)))

                       elif user_num_cols_present_in_df:
                             # Numerical columns were identified, but scaler is missing/not fitted - this indicates an error in user_processor or load
                             print(f"    FATAL ERROR: User numerical scaler not available/fitted despite numerical columns identified. Skipping batch {batch_idx+1}.")
                             can_proceed_with_batching = False
                             break # Exit loop
                       else:
                             # No numerical columns identified or list was empty - Add 0-column placeholder
                             feature_matrices_for_hstack_batch.append(scipy.sparse.csr_matrix((current_batch_size, 0)))


                       # Apply encoding and SVD using the *fitted* categorical user transformers
                       # Check if encoder and SVD are available AND fitted AND if there were categorical columns identified
                       is_cat_svd_available_and_fitted = (user_categorical_encoder is not None and hasattr(user_categorical_encoder, 'categories_')) and \
                                                         (user_categorical_svd is not None and hasattr(user_categorical_svd, 'components_') and user_categorical_svd.n_components > 0)

                       if is_cat_svd_available_and_fitted and user_cat_cols_present_in_df: # Check list is non-empty

                             if X_train_user_categorical_raw_batch.shape[1] > 0: # Check if this batch slice has the categorical columns
                                  # Ensure columns are strings before encoding
                                  X_train_user_categorical_raw_batch_str = X_train_user_categorical_raw_batch.astype(str).fillna('Missing')
                                  try:
                                       X_train_user_categorical_encoded_batch = user_categorical_encoder.transform(X_train_user_categorical_raw_batch_str) # TRANSFORM using fitted encoder
                                       # Check if encoded features have columns before SVD
                                       if X_train_user_categorical_encoded_batch.shape[1] > 0:
                                           X_train_user_categorical_reduced_batch = user_categorical_svd.transform(X_train_user_categorical_encoded_batch) # TRANSFORM using fitted SVD
                                           feature_matrices_for_hstack_batch.append(scipy.sparse.csr_matrix(X_train_user_categorical_reduced_batch))
                                       else: # Encoded features had 0 columns, add 0-column SVD placeholder
                                           print(f"    Warning: User categorical encoded features for batch {batch_idx+1} resulted in 0 columns. Adding 0-column SVD placeholder.")
                                           feature_matrices_for_hstack_batch.append(scipy.sparse.csr_matrix((current_batch_size, user_categorical_svd.n_components))) # Use SVD output dim
                                  except Exception as e:
                                        print(f"    FATAL ERROR transforming user categorical features in batch {batch_idx+1}: {e}. Skipping remaining batches.")
                                        can_proceed_with_batching = False
                                        break # Exit loop

                             else: # Batch slice was empty of columns (should not happen if user_cat_cols_present_in_df wasn't empty)
                                  # Add empty placeholder with dimension from fitted SVD
                                  svd_output_cols = user_categorical_svd.n_components # Should be > 0 if we are in this outer IF
                                  print(f"    Warning: Categorical batch slice {batch_idx+1} resulted in 0 columns unexpectedly. Adding placeholder with {svd_output_cols} columns.")
                                  feature_matrices_for_hstack_batch.append(scipy.sparse.csr_matrix((current_batch_size, svd_output_cols)))

                       elif user_cat_cols_present_in_df:
                              # Categorical columns were identified, but encoder/SVD is missing/not fitted correctly - error in user_processor or load
                              print(f"    FATAL ERROR: User categorical transformer(s) not available/fitted correctly despite categorical columns identified. Skipping batch {batch_idx+1}.")
                              can_proceed_with_batching = False
                              break # Exit loop
                       else:
                             # No categorical columns identified or SVD components is 0 - Add 0-column placeholder
                             # Determine the expected SVD output columns based on the user_cat_svd_n_components config *if* categorical columns were identified in input lists
                             # If user_cat_cols_present_in_df is empty, the expected SVD output is 0.
                             # If user_cat_cols_present_in_df is NOT empty, but SVD wasn't fitted successfully (e.g. error),
                             # the expected output *might* be the config value, but the transformer isn't available.
                             # Let's rely on the `expected_total_cols_final` check after stacking the batch.
                             # For now, add a 0-column matrix if no categorical columns were identified or SVD wasn't available.
                             svd_output_cols_placeholder = user_categorical_svd.n_components if user_categorical_svd is not None and hasattr(user_categorical_svd, 'n_components') and user_categorical_svd.n_components > 0 else 0
                             # print(f"    Warning: No categorical columns identified for user features or SVD not available. Adding {svd_output_cols_placeholder}-column placeholder for batch {batch_idx+1}.")
                             feature_matrices_for_hstack_batch.append(scipy.sparse.csr_matrix((current_batch_size, svd_output_cols_placeholder)))


                       # Item features from X_items_reduced (from Step 4)
                       # Ensure X_items_reduced exists and has >0 columns (checked via initial validation)
                       if X_items_reduced is not None and hasattr(X_items_reduced, 'shape') and X_items_reduced.shape[1] > 0:
                            try:
                                 # Select rows from X_items_reduced corresponding to the svd_index in this batch
                                 # svd_indices_batch must be valid indices for X_items_reduced
                                 if svd_indices_batch.max() >= X_items_reduced.shape[0] or svd_indices_batch.min() < 0:
                                     raise IndexError(f"svd_index out of bounds. Max index: {svd_indices_batch.max()}, X_items_reduced rows: {X_items_reduced.shape[0]}")

                                 X_train_item_features_svd_batch = X_items_reduced[svd_indices_batch] # This is dense numpy array
                                 feature_matrices_for_hstack_batch.append(scipy.sparse.csr_matrix(X_train_item_features_svd_batch))
                            except Exception as e:
                                 print(f"    FATAL ERROR selecting item features for batch {batch_idx+1} using svd_indices: {e}. This might indicate invalid svd_index values. Skipping remaining batches.")
                                 can_proceed_with_batching = False
                                 break # Exit loop
                       else:
                            # Reduced item features not available/have 0 columns - Add 0-column placeholder matching batch size
                            item_svd_output_cols = X_items_reduced.shape[1] if X_items_reduced is not None and hasattr(X_items_reduced, 'shape') else 0
                            # print(f"    Warning: Reduced item features (X_items_reduced) not available or have 0 columns. Adding {item_svd_output_cols}-column placeholder for batch {batch_idx+1}.")
                            feature_matrices_for_hstack_batch.append(scipy.sparse.csr_matrix((current_batch_size, item_svd_output_cols)))


                       # Combine the matrices for this batch horizontally
                       X_train_hybrid_batch = None
                       if feature_matrices_for_hstack_batch: # Only attempt stacking if list is not empty
                            try:
                                 # Check total cols in batch before hstacking to ensure it matches expected
                                 total_cols_in_batch_hstack = sum(m.shape[1] for m in feature_matrices_for_hstack_batch)

                                 # Handle cases where the result would be 0 columns but expected is > 0
                                 if total_cols_in_batch_hstack == 0 and expected_total_cols_final > 0:
                                     # Create a dummy matrix with the expected number of columns
                                     print(f"    Warning: Batch {batch_idx+1} resulted in 0 columns for stacking but expected {expected_total_cols_final}. Creating dummy matrix.")
                                     X_train_hybrid_batch = scipy.sparse.csr_matrix((current_batch_size, expected_total_cols_final))
                                 else:
                                     X_train_hybrid_batch = scipy.sparse.hstack(feature_matrices_for_hstack_batch)

                            except Exception as e:
                                 print(f"    FATAL ERROR: Could not horizontally stack feature matrices for batch {batch_idx+1}: {e}. Skipping remaining batches.")
                                 can_proceed_with_batching = False
                                 break # Exit loop

                       else: # No feature matrices were generated for this batch (e.g., if all column lists were empty)
                             # Add a matrix with the expected total columns, even if 0
                             print(f"    Warning: No feature matrices generated for batch {batch_idx+1}. Adding placeholder with {expected_total_cols_final} columns.")
                             X_train_hybrid_batch = scipy.sparse.csr_matrix((current_batch_size, expected_total_cols_final))


                       if X_train_hybrid_batch is not None: # If stacking succeeded or dummy was created
                            # Sanity check: Ensure batch has the expected number of columns
                            if X_train_hybrid_batch.shape[1] != expected_total_cols_final:
                                 print(f"    FATAL ERROR: Batch {batch_idx+1} created matrix with unexpected number of columns ({X_train_hybrid_batch.shape[1]}). Expected: {expected_total_cols_final}. Skipping remaining batches.")
                                 can_proceed_with_batching = False
                                 break # Exit the batching loop

                            # Save the horizontally stacked batch to disk
                            batch_filepath_x = os.path.join(batch_temp_dir, f'batch_X_train_hybrid_{batch_idx:05d}.npz') # Use 05d for sorting
                            scipy.sparse.save_npz(batch_filepath_x, X_train_hybrid_batch)
                            # Save the corresponding y batch
                            batch_filepath_y = os.path.join(batch_temp_dir, f'batch_y_train_{batch_idx:05d}.csv') # Use 05d for sorting
                            y_train_batch.to_csv(batch_filepath_y, index=False, header=True)
                            print(f"    Processed and saved batch {batch_idx + 1}/{num_batches}. X_batch shape: {X_train_hybrid_batch.shape}")

                       else: # Stacking failed and dummy was not created - should be caught by checks above, but safeguard
                             print(f"    FATAL ERROR: Horizontal stacking failed for batch {batch_idx+1} and dummy creation failed. X_train_hybrid_batch is None. Skipping remaining batches.")
                             can_proceed_with_batching = False
                             break # Exit loop


        end_time_batching = time.time()
        if can_proceed_with_batching:
             print(f"\nBatch processing and saving complete in {end_time_batching - start_time_batching:.2f} seconds.")
        else:
             print(f"\nBatch processing stopped early due to errors after {end_time_batching - start_time_batching:.2f} seconds.")


        # Explicitly delete raw feature DataFrames to free memory after batching loop (if they exist)
        if 'X_train_user_numerical_raw_all' in locals(): del X_train_user_numerical_raw_all
        if 'X_train_user_categorical_raw_all' in locals(): del X_train_user_categorical_raw_all
        if 'svd_indices_all' in locals(): del svd_indices_all
        if 'y_train_full' in locals(): del y_train_full
        print("Deleted raw feature data used for batching.")


        # --- Vertically Stack Batches from Disk ---
        X_train_hybrid = None # Reset before vstacking
        y_train = None # Reset before y concat

        if can_proceed_with_batching: # Only attempt vstack if batching was successful so far
            print("\nVertically stacking batches from disk...")
            start_time_vstack = time.time()
            can_proceed_with_vstack = True # Flag for vstack loop

            # Find all saved batch files (X matrices)
            batch_file_pattern_x = os.path.join(batch_temp_dir, 'batch_X_train_hybrid_*.npz')
            batch_files_x = sorted(glob.glob(batch_file_pattern_x)) # Sort to ensure correct order

            # Find all saved batch files (y targets)
            batch_file_pattern_y = os.path.join(batch_temp_dir, 'batch_y_train_*.csv')
            batch_files_y = sorted(glob.glob(batch_file_pattern_y)) # Sort to ensure correct order

            if not batch_files_x:
                print("FATAL ERROR: No batch files found to stack.")
                can_proceed_with_vstack = False
            elif len(batch_files_x) != len(batch_files_y):
                 print(f"FATAL ERROR: Mismatch between number of X batch files ({len(batch_files_x)}) and y batch files ({len(batch_files_y)}).")
                 can_proceed_with_vstack = False
            elif len(batch_files_x) != num_batches: # Check if we saved the expected number of batches
                 print(f"Warning: Mismatch between expected number of batches ({num_batches}) and found files ({len(batch_files_x)}). This might happen if a previous run crashed mid-batching.")
                 # Decide if this is fatal. If we only have partial batches, vstacking will fail or give wrong size.
                 # Let's treat as fatal for robustness.
                 print("FATAL ERROR: Incomplete batch files found.")
                 can_proceed_with_vstack = False
            else: # All files match, proceed with stacking
                 
                      # Vertically stack batches incrementally
                      if batch_files_x: # Ensure there is at least one file
                           print("  Loading and stacking batch 1/{}...".format(len(batch_files_x)))
                           X_train_hybrid_batches = [scipy.sparse.load_npz(batch_files_x[0])] # Start a list of sparse matrices
                           y_train_batches_list = [pd.read_csv(batch_files_y[0])] # Start a list of y dataframes

                           # Check if the first batch has the expected number of columns
                           if X_train_hybrid_batches[0].shape[1] != expected_total_cols_final:
                                print(f"FATAL ERROR: First batch has unexpected number of columns ({X_train_hybrid_batches[0].shape[1]}). Expected: {expected_total_cols_final}. Stopping vertical stacking.")
                                can_proceed_with_vstack = False
                           else:
                                # Stack remaining batches sequentially
                                for batch_idx in range(1, len(batch_files_x)):
                                    if not can_proceed_with_vstack: # Check flag again
                                          print(f"Skipping remaining vertical stacking due to previous error in batch {batch_idx+1}.")
                                          break # Exit the stacking loop

                                    print(f"  Loading and stacking batch {batch_idx + 1}/{len(batch_files_x)}...")
                                    X_batch = scipy.sparse.load_npz(batch_files_x[batch_idx])
                                    y_batch = pd.read_csv(batch_files_y[batch_idx])

                                    # Check if the batch has the expected number of columns before stacking
                                    if X_batch.shape[1] != expected_total_cols_final:
                                         print(f"FATAL ERROR: Batch {batch_idx+1} has unexpected number of columns ({X_batch.shape[1]}) for vertical stacking. Expected: {expected_total_cols_final}. Stopping.")
                                         can_proceed_with_vstack = False
                                         break # Exit stacking loop

                                    # Append batch to list
                                    X_train_hybrid_batches.append(X_batch)
                                    y_train_batches_list.append(y_batch) # Add y batch to list

                                # Perform vstack once after loading all batches into memory/list
                                if can_proceed_with_vstack and X_train_hybrid_batches:
                                     X_train_hybrid = scipy.sparse.vstack(X_train_hybrid_batches)
                                     print("  Vertically stacked X batches.")
                                else:
                                     print("Warning: No X batches to stack after loading. X_train_hybrid will be None.")
                                     X_train_hybrid = None
                                     can_proceed_with_vstack = False # Mark failure if X stacking list was empty despite files existing?

                                # Concatenate y batches once after vstacking X
                                if can_proceed_with_vstack and y_train_batches_list: # Only concatenate if no error during stacking X and list is not empty
                                     y_train = pd.concat(y_train_batches_list, ignore_index=True)
                                     print("  Concatenated y batches.")
                                elif can_proceed_with_vstack: # List was empty but X stacking didn't fail
                                     y_train = pd.Series([], dtype=int, name=config.INTERACTION_COL) # Create empty Series with correct name
                                     print("Warning: No y batch files loaded or y_batches_list is empty, y_train created as empty Series.")
                                else: # X stacking failed, y should also be None
                                     y_train = None
                                     print("Warning: Y batches concatenation skipped due to errors in X stacking.")


                        #  # This outer try covers potential errors during loading/initial stacking/appending
                        #  except Exception as e:
                        #       print(f"FATAL ERROR: An error occurred during vertical stacking setup or incremental processing: {e}")
                        #       X_train_hybrid = None; y_train = None # Ensure None on error
                        #       can_proceed_with_vstack = False # Mark for failure

            # Check after the loop and main try block if vstacking was successful
            if can_proceed_with_vstack and X_train_hybrid is not None and y_train is not None:
                 print("\nVertical stacking complete.")
            else:
                 print("\nVertical stacking failed.")
                 # Ensure outputs are None if vstacking failed
                 X_train_hybrid = None
                 y_train = None
                 can_proceed_with_batching = False # Ensure the overall batching process is marked as failed

        end_time_vstack = time.time()
        print(f"Vertical stacking complete in {end_time_vstack - start_time_vstack:.2f} seconds.")


        # Final shape check after vstacking (only if batching was attempted and finished without fatal errors)
        if can_proceed_with_batching and X_train_hybrid is not None and y_train is not None:
             # Check if X_train_hybrid and y_train exist and have the expected number of rows (num_samples)
             # and X has the expected number of columns (expected_total_cols_final)
             if X_train_hybrid.shape[0] == num_samples and X_train_hybrid.shape[1] == expected_total_cols_final and len(y_train) == num_samples:
                  print(f"\nFinal hybrid training feature matrix X_train_hybrid shape: {X_train_hybrid.shape}")
                  if isinstance(X_train_hybrid, scipy.sparse.csr_matrix):
                       # Calculate sparsity safely, avoiding division by zero
                       if X_train_hybrid.shape[0] * X_train_hybrid.shape[1] > 0:
                           sparsity = (1 - X_train_hybrid.nnz / (X_train_hybrid.shape[0] * X_train_hybrid.shape[1]))
                           print(f"X_train_hybrid sparsity: {sparsity:.4f}")
                       else:
                           print("X_train_hybrid is an empty matrix (0 area).")

                  print(f"\nFinal target y_train shape: {y_train.shape}")
                  print("\nTraining data matrix X_train_hybrid and target y_train prepared.")
                  final_matrix_is_ready = True # Set flag to True after successful vstack/concat

             else: # Final shape mismatch
                  print("FATAL ERROR: Final matrix/target shape mismatch after vertical stacking.")
                  if X_train_hybrid is not None and hasattr(X_train_hybrid, 'shape'): print(f"  X_train_hybrid final shape: {X_train_hybrid.shape}")
                  else: print("  X_train_hybrid is None or not available.")
                  if y_train is not None: print(f"  y_train final shape: {y_train.shape}")
                  else: print("  y_train is None or not available.")
                  print(f"  Expected number of samples: {num_samples}")
                  print(f"  Expected number of columns in X: {expected_total_cols_final}")

                  final_matrix_is_ready = False # Not ready
                  # Ensure outputs are None on failure
                  X_train_hybrid = None
                  y_train = None

        elif can_proceed_with_batching: # Batching finished without fatal errors, but X_train_hybrid or y_train is None after vstack
             print("FATAL ERROR: Batching finished without fatal errors, but final X_train_hybrid or y_train is None after stacking.")
             final_matrix_is_ready = False # Not ready
             # Outputs are already None
        # else: batching already failed earlier, final_matrix_is_ready is False and outputs are None


        # --- Clean up temporary batch files ---
        # Clean up only if batching was attempted and resulted in a final ready matrix OR if batching failed early
        # If batching completed successfully, clean up the temp directory.
        # If batching failed early (e.g., in batch loop), keep the directory for debugging.
        # Let's refine cleanup: clean up if batching completed *or* if the temp dir was created but no files were saved.
        # Keep if files were saved but stacking failed.
        if os.path.exists(batch_temp_dir):
             batch_files_saved_count = len(glob.glob(os.path.join(batch_temp_dir, 'batch_X_train_hybrid_*.npz')))
             if final_matrix_is_ready or batch_files_saved_count == 0: # Clean up if successful OR if temp dir is empty (nothing saved)
                 print("\nCleaning up temporary batch files...")
                 try:
                     shutil.rmtree(batch_temp_dir)
                     print(f"Temporary batch directory removed: {batch_temp_dir}")
                 except Exception as e:
                     print(f"Warning: Could not remove temporary batch directory {batch_temp_dir}: {e}")
             else:
                  print(f"\nSkipping temporary batch file cleanup ({batch_temp_dir}) due to vertical stacking errors or final matrix not ready, but batch files were saved ({batch_files_saved_count}).")
        # else: temp dir wasn't created or didn't exist initially


    # If final matrix is ready (either loaded or vstacked successfully) save it
    if final_matrix_is_ready:
        # --- Save Final X_train_hybrid and y_train ---
        print("\n--- Saving Final X_train_hybrid and y_train ---")
        # Ensure output directory exists (should be done by main, but safe)
        os.makedirs(os.path.dirname(x_train_hybrid_path), exist_ok=True)

        # Save X_train_hybrid (sparse matrix)
        if X_train_hybrid is not None:
            try:
                scipy.sparse.save_npz(x_train_hybrid_path, X_train_hybrid)
                print(f"X_train_hybrid (final) saved successfully to {x_train_hybrid_path}")
            except Exception as e:
                print(f"Error saving final X_train_hybrid to {x_train_hybrid_path}: {e}")

        # Save y_train
        if y_train is not None:
            try:
                # Ensure y_train is a DataFrame or Series before saving
                if not isinstance(y_train, (pd.Series, pd.DataFrame)):
                     # Attempt to convert numpy array back to DataFrame/Series if needed
                     if isinstance(y_train, np.ndarray):
                         if y_train.ndim == 1:
                              y_train = pd.Series(y_train, name=config.INTERACTION_COL)
                         else: # Handle multi-dimensional numpy array, though y_train should be 1D
                              y_train = pd.DataFrame(y_train)
                              print("Warning: y_train was multi-dimensional numpy array, saving as DataFrame.")
                     else: # Unknown type, try converting
                         y_train = pd.DataFrame(y_train)
                         print("Warning: y_train was not Series/DataFrame/ndarray, attempting save as DataFrame.")

                # Ensure header exists
                y_train.to_csv(y_train_path, index=False, header=True)
                print(f"y_train (final) saved successfully to {y_train_path}")
            except Exception as e:
                print(f"Error saving final y_train to {y_train_path}: {e}")
        else:
             print("Warning: y_train is None, skipping save.")

    else:
         print("\nSkipping final matrix/target save as preparation was not successful.")


    print("\nStep 6 execution finished.")

    # Return the final matrix and target (will be None if processing failed)
    return X_train_hybrid, y_train


# Optional: Add a main block to test this script standalone
if __name__ == "__main__":
    print("Running hybrid_matrix_assembler.py as standalone script...")
    # To run this standalone, you need to simulate *all* the required inputs
    # from previous steps: train_pairs_df, user_features_df,
    # user transformers + column lists, X_items_reduced, item_id_index_map_df.

    # --- Simulate Loading/Creating Required Input Data ---
    print("Simulating loading/creating input data for standalone test...")
    try:
        num_test_pairs = 1000
        num_test_users = 100
        num_test_items = 200
        item_svd_dim = 20
        user_cat_svd_dim = 50
        num_user_num_features = 3 # Simulate 3 numerical user features
        num_user_cat_features = 2 # Simulate 2 categorical user features

        # 1. Simulate train_pairs_df
        rng = np.random.RandomState(config.SEED)
        sim_train_pairs_df = pd.DataFrame({
            config.USER_ID_COL: rng.choice([f'user{i:03d}' for i in range(num_test_users)], size=num_test_pairs, replace=True),
            config.ITEM_ID_COL: rng.choice([f'item{i:03d}' for i in range(num_test_items)], size=num_test_pairs, replace=True),
            config.INTERACTION_COL: rng.randint(0, 2, size=num_test_pairs)
        }).astype(str)
        sim_train_pairs_df[config.INTERACTION_COL] = sim_train_pairs_df[config.INTERACTION_COL].astype(int)
        print(f"Simulated train_pairs_df shape: {sim_train_pairs_df.shape}")


        # 2. Simulate user_features_df
        sim_user_ids = sim_train_pairs_df[config.USER_ID_COL].unique()
        sim_user_features_df = pd.DataFrame({
             config.USER_ID_COL: sim_user_ids,
             'user_num_feat_1': rng.rand(len(sim_user_ids)),
             'user_num_feat_2': rng.randint(1, 100, len(sim_user_ids)),
             'user_num_feat_3': rng.randn(len(sim_user_ids)), # Can be negative/zero
             'user_cat_feat_1': rng.choice(['A', 'B', 'C', 'D'], len(sim_user_ids), p=[0.4, 0.3, 0.2, 0.1]),
             'user_cat_feat_2': rng.choice(['X', 'Y', 'Z'], len(sim_user_ids), p=[0.5, 0.3, 0.2]),
        }).astype({config.USER_ID_COL: str, 'user_cat_feat_1': str, 'user_cat_feat_2': str})
        # Introduce NaNs
        sim_user_features_df.loc[sim_user_features_df.sample(frac=0.05, random_state=config.SEED).index, 'user_num_feat_2'] = np.nan
        sim_user_features_df.loc[sim_user_features_df.sample(frac=0.05, random_state=config.SEED).index, 'user_cat_feat_1'] = np.nan
        print(f"Simulated user_features_df shape: {sim_user_features_df.shape}")
        sim_user_num_cols_final = ['user_num_feat_1', 'user_num_feat_2', 'user_num_feat_3']
        sim_user_cat_cols_final = ['user_cat_feat_1', 'user_cat_feat_2']


        # 3. Simulate fitted user transformers and used column lists (as if from user_processor)
        print("Simulating user transformer fitting...")
        sim_user_numerical_scaler = StandardScaler()
        # Fit on non-NaN data, imputing first
        sim_user_features_df_num = sim_user_features_df[sim_user_num_cols_final].copy()
        for col in sim_user_features_df_num.columns:
             median_val = sim_user_features_df_num[col].median() if not sim_user_features_df_num[col].isnull().all() else 0
             sim_user_features_df_num[col] = sim_user_features_df_num[col].fillna(median_val).astype(float)
        sim_user_numerical_scaler.fit(sim_user_features_df_num)

        sim_user_categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)
        sim_user_features_df_cat = sim_user_features_df[sim_user_cat_cols_final].copy().astype(str).fillna('Missing')
        sim_user_categorical_encoder.fit(sim_user_features_df_cat)

        # Simulate SVD (fit on encoded categorical features)
        sim_user_cat_encoded = sim_user_categorical_encoder.transform(sim_user_features_df_cat)
        # Ensure SVD components <= encoded features dimension
        actual_user_cat_svd_dim = min(user_cat_svd_dim, sim_user_cat_encoded.shape[1])
        if actual_user_cat_svd_dim > 0:
             sim_user_categorical_svd = TruncatedSVD(n_components=actual_user_cat_svd_dim, random_state=config.SEED)
             sim_user_categorical_svd.fit(sim_user_cat_encoded)
        else:
             sim_user_categorical_svd = None

        sim_user_num_cols_used = sim_user_num_cols_final # In this simulation, all were used
        sim_user_cat_cols_used = sim_user_cat_cols_final # In this simulation, all were used
        print("Simulated user transformer fitting complete.")


        # 4. Simulate X_items_reduced and item_id_index_map_df (as if from item_processor)
        sim_item_ids = sim_train_pairs_df[config.ITEM_ID_COL].unique() # Use items present in pairs for mapping
        sim_num_unique_items = len(sim_item_ids)

        # Simulate X_items_reduced (random data for placeholder)
        if sim_num_unique_items > 0 and item_svd_dim > 0:
             sim_X_items_reduced = rng.rand(sim_num_unique_items, item_svd_dim)
        else:
             sim_X_items_reduced = np.zeros((sim_num_unique_items, max(0, item_svd_dim))) # Handle 0 items or 0 components
        print(f"Simulated X_items_reduced shape: {sim_X_items_reduced.shape}")

        # Simulate item ID mappings for items *present in the simulated pairs*
        if sim_num_unique_items > 0:
            sim_item_id_to_index = {item_id: idx for idx, item_id in enumerate(sim_item_ids)}
            sim_index_to_item_id = {idx: item_id for idx, item_id in enumerate(sim_item_ids)}
            sim_item_id_index_map_df = pd.DataFrame({
                config.ITEM_ID_COL: sim_item_ids,
                'svd_index': range(sim_num_unique_items)
            }).astype({config.ITEM_ID_COL: str, 'svd_index': int})
        else:
            sim_item_id_to_index = {}
            sim_index_to_item_id = {}
            sim_item_id_index_map_df = pd.DataFrame(columns=[config.ITEM_ID_COL, 'svd_index'])

        print(f"Simulated item_id_index_map_df shape: {sim_item_id_index_map_df.shape}")
        # Need a dummy clean_item_meta_df for the row count check in assemble_hybrid_matrix
        # Its size should match the number of items used to generate X_items_reduced and item_id_index_map_df
        sim_clean_item_meta_df = pd.DataFrame({config.ITEM_ID_COL: sim_item_ids}) # Simple df with item_ids
        print(f"Simulated clean_item_meta_df shape (for validation): {sim_clean_item_meta_df.shape}")


        print("Simulated data preparation complete.")

        # --- Call the main function ---
        print("\nCalling assemble_hybrid_matrix...")
        # Define temporary paths for saving during standalone test
        temp_intermediate_dir = './temp_intermediate_for_hybrid_test'
        os.makedirs(temp_intermediate_dir, exist_ok=True)

        temp_batch_dir = os.path.join(temp_intermediate_dir, 'temp_batches')
        temp_x_train_hybrid = os.path.join(temp_intermediate_dir, 'test_X_train_hybrid.npz')
        temp_y_train = os.path.join(temp_intermediate_dir, 'test_y_train.csv')


        X_train_hybrid_assembled, y_train_assembled = assemble_hybrid_matrix(
            sim_train_pairs_df,
            sim_user_features_df,
            sim_user_numerical_scaler,
            sim_user_categorical_encoder,
            sim_user_categorical_svd,
            sim_user_num_cols_used, # Pass the simulated used lists
            sim_user_cat_cols_used,
            sim_X_items_reduced,
            sim_item_id_index_map_df,
            config.STACKING_BATCH_SIZE,
            temp_batch_dir,
            temp_x_train_hybrid,
            temp_y_train
        )

        print("\n--- Standalone Hybrid Matrix Assembly Complete ---")
        if X_train_hybrid_assembled is not None:
            print(f"Assembled X_train_hybrid shape: {X_train_hybrid_assembled.shape}")
        else:
            print("Assembled X_train_hybrid is None.")

        if y_train_assembled is not None:
             print(f"Assembled y_train shape: {y_train_assembled.shape}")
        else:
             print("Assembled y_train is None.")


        # --- Clean up temporary files ---
        print("\nCleaning up temporary test files...")
        try:
             if os.path.exists(temp_intermediate_dir):
                  shutil.rmtree(temp_intermediate_dir)
                  print(f"Removed temporary directory: {temp_intermediate_dir}")
        except Exception as e:
             print(f"Warning: Could not remove temporary directory {temp_intermediate_dir}: {e}")


    except ValueError as e:
         print(f"Script failed due to input data error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during script execution: {e}")