# item_processor.py

import pandas as pd
import numpy as np
import os
import time
import glob # Needed for checking files in standalone test cleanup
import shutil # Needed for standalone test cleanup
import pickle
import json
import scipy.sparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import traceback # <--- Added for detailed error reporting

# Import configuration from config.py
import config

def process_item_features(
    clean_item_meta_df: pd.DataFrame,
    item_svd_n_components: int,
    seed: int,
    x_items_reduced_npy_path: str,
    item_id_to_index_json_path: str,
    index_to_item_id_json_path: str,
    item_scaler_pkl_path: str,
    item_encoder_pkl_path: str,
    item_num_cols_json_path: str,
    item_cat_cols_json_path: str,
    item_binary_cols_json_path: str,
    item_original_num_cols_json_path: str,
    item_original_nom_cat_cols_json_path: str,
    item_details_cols_json_path: str,
) -> tuple[np.ndarray | None, dict | None, dict | None, pd.DataFrame | None, StandardScaler | None, OneHotEncoder | None, list, list, list, list, list, list]:
    """
    Processes item features from cleaned metadata.
    Loads saved data if available, otherwise generates and saves.

    Args:
        clean_item_meta_df (pd.DataFrame): DataFrame containing cleaned item metadata.
        item_svd_n_components (int): Number of components for item feature SVD.
        seed (int): Random seed for reproducibility.
        x_items_reduced_npy_path (str): Path to save/load the reduced item features numpy array.
        item_id_to_index_json_path (str): Path to save/load the item ID to index mapping JSON.
        index_to_item_id_json_path (str): Path to save/load the index to item ID mapping JSON.
        item_scaler_pkl_path (str): Path to save/load the fitted StandardScaler for item numerical features.
        item_encoder_pkl_path (str): Path to save/load the fitted OneHotEncoder for item categorical features.
        item_num_cols_json_path (str): Path to save/load the list of numerical columns used for scaling.
        item_cat_cols_json_path (str): Path to save/load the list of categorical columns used for encoding.
        item_binary_cols_json_path (str): Path to save/load the list of binary columns.
        item_original_num_cols_json_path (str): Path to save/load the list of original numerical columns.
        item_original_nom_cat_cols_json_path (str): Path to save/load the list of original nominal categorical columns.
        item_details_cols_json_path (str): Path to save/load the list of details columns.


    Returns:
        tuple: (X_items_reduced, item_id_to_index, index_to_item_id, item_id_index_map_df,
                item_scaler, item_encoder,
                item_numerical_cols_for_scaling, item_categorical_cols_for_encoding, item_binary_cols,
                item_original_numerical_cols, item_original_nominal_categorical_cols, item_details_cols)
                Returns None for outputs if generation/loading fails.

    Raises:
        ValueError: If input clean_item_meta_df is None or empty, or missing required columns.
        Exception: For other unexpected errors during processing or loading.
    """
    print("\n--- Step 4: Process Item Features ---")

    # --- Validate Inputs ---
    if clean_item_meta_df is None or clean_item_meta_df.empty:
        raise ValueError("Input clean_item_meta_df is None or empty.")
    if config.ITEM_ID_COL not in clean_item_meta_df.columns:
        raise ValueError(f"Input clean_item_meta_df is missing required column: {config.ITEM_ID_COL}")

    # Ensure item_id is string type in input for safety
    clean_item_meta_df[config.ITEM_ID_COL] = clean_item_meta_df[config.ITEM_ID_COL].astype(str)


    # --- Global variables to be populated (will be returned) ---
    X_items_reduced = None
    item_id_to_index = None
    index_to_item_id = None
    item_id_index_map_df = None
    item_scaler = None
    item_encoder = None
    item_numerical_cols_for_scaling = [] # Initialize as empty lists
    item_categorical_cols_for_encoding = []
    item_binary_cols = []
    item_original_numerical_cols = []
    item_original_nominal_categorical_cols = []
    item_details_cols = []

    prepare_item_features = True # Assume generation is needed unless loading succeeds

    # --- Check if saved item processed data exists to avoid regeneration ---
    # List ALL output files that *should* exist if a previous run completed successfully and saved outputs.
    # Scaler/Encoder are conditionally saved, so check their existence separately.
    required_item_output_files_for_loading = [
        x_items_reduced_npy_path,
        item_id_to_index_json_path,
        index_to_item_id_json_path,
        item_num_cols_json_path,
        item_cat_cols_json_path,
        item_binary_cols_json_path,
        item_original_num_cols_json_path,
        item_original_nom_cat_cols_json_path,
        item_details_cols_json_path
    ]

    all_core_files_exist = all(os.path.exists(f) for f in required_item_output_files_for_loading)
    scaler_file_exists = os.path.exists(item_scaler_pkl_path)
    encoder_file_exists = os.path.exists(item_encoder_pkl_path)


    if all_core_files_exist:
        print("\nDetected core saved item features and column lists. Attempting to load...")
        try:
            # Load column lists first, as they determine if transformers were needed
            with open(item_num_cols_json_path, 'r') as f: item_numerical_cols_for_scaling = json.load(f)
            with open(item_cat_cols_json_path, 'r') as f: item_categorical_cols_for_encoding = json.load(f)
            with open(item_binary_cols_json_path, 'r') as f: item_binary_cols = json.load(f)
            with open(item_original_num_cols_json_path, 'r') as f: item_original_numerical_cols = json.load(f)
            with open(item_original_nom_cat_cols_json_path, 'r') as f: item_original_nominal_categorical_cols = json.load(f)
            with open(item_details_cols_json_path, 'r') as f: item_details_cols = json.load(f)
            print("Loaded item column lists.")

            # Determine if scaler/encoder were expected based on loaded column lists
            is_scaler_expected = len(item_numerical_cols_for_scaling) > 0
            is_encoder_expected = len(item_categorical_cols_for_encoding) > 0

            # Load Transformers (pickle) - only if expected AND file exists
            if is_scaler_expected and scaler_file_exists:
                 with open(item_scaler_pkl_path, 'rb') as f: item_scaler = pickle.load(f)
                 print(f"Loaded {item_scaler_pkl_path}")
            elif is_scaler_expected and not scaler_file_exists:
                 print(f"Warning: {item_scaler_pkl_path} expected but not found. Scaler will be None.")
                 item_scaler = None # Ensure None if file missing but expected
            else:
                 print("Scaler not expected (no numerical columns identified).")
                 item_scaler = None # Ensure None if not expected


            if is_encoder_expected and encoder_file_exists:
                 with open(item_encoder_pkl_path, 'rb') as f: item_encoder = pickle.load(f)
                 print(f"Loaded {item_encoder_pkl_path}")
            elif is_encoder_expected and not encoder_file_exists:
                 print(f"Warning: {item_encoder_pkl_path} expected but not found. Encoder will be None.")
                 item_encoder = None # Ensure None if file missing but expected
            else:
                 print("Encoder not expected (no categorical columns identified).")
                 item_encoder = None # Ensure None if not expected


            # Load Reduced Matrix (numpy array)
            X_items_reduced = np.load(x_items_reduced_npy_path)
            print(f"Loaded X_items_reduced from {x_items_reduced_npy_path} (shape: {X_items_reduced.shape})")

            # Load Mappings (JSON)
            with open(item_id_to_index_json_path, 'r') as f:
                item_id_to_index = json.load(f)
            print(f"Loaded item_id_to_index from {item_id_to_index_json_path} ({len(item_id_to_index)} items)")

            with open(index_to_item_id_json_path, 'r') as f:
                index_to_item_id_str_keys = json.load(f)
                # Convert keys back to int as indices are numerical
                index_to_item_id = {int(k): v for k, v in index_to_item_id_str_keys.items()}
            print(f"Loaded index_to_item_id from {index_to_item_id_json_path} ({len(index_to_item_id)} indices)")

            # Recreate item_id_index_map_df from loaded data
            if item_id_to_index is not None and len(item_id_to_index) > 0:
                item_id_index_map_df = pd.DataFrame({
                    config.ITEM_ID_COL: list(item_id_to_index.keys()),
                    'svd_index': list(item_id_to_index.values())
                })
                item_id_index_map_df['svd_index'] = item_id_index_map_df['svd_index'].astype(int)
                print("Recreated item_id_index_map_df from loaded data.")
            else:
                print("Warning: item_id_to_index is empty after loading. Cannot recreate item_id_index_map_df.")
                item_id_index_map_df = pd.DataFrame(columns=[config.ITEM_ID_COL, 'svd_index']) # Create empty DF


            # --- Validation of loaded item data ---
            is_loaded_valid = True
            # Check key objects are not None/empty and have basic expected structure/attributes
            if X_items_reduced is None or not hasattr(X_items_reduced, 'shape'):
                 print("Validation Failed: X_items_reduced is None or missing shape.")
                 is_loaded_valid = False
            elif X_items_reduced.shape[0] != len(clean_item_meta_df): # Check number of rows matches input metadata
                 print(f"Validation Warning: Loaded X_items_reduced has {X_items_reduced.shape[0]} rows, expected {len(clean_item_meta_df)} (based on input metadata).")
                 # This discrepancy might be fatal depending on how strictly you need row counts to match.
                 # Let's treat it as fatal for robust pipelines.
                 print("FATAL Validation: Row count mismatch.")
                 is_loaded_valid = False

            # Check SVD dimension matches config if matrix is not empty
            # Allow 0 components if config is 0 or data doesn't support positive components
            n_components_expected = max(0, item_svd_n_components) # Treat <0 config as 0
            if is_loaded_valid and X_items_reduced.shape[0] > 0 and X_items_reduced.shape[1] != n_components_expected and X_items_reduced.shape[1] != 0:
                # This could happen if config changed or if the SVD component calculation had an edge case
                print(f"Validation Warning: Loaded X_items_reduced dimension {X_items_reduced.shape[1]} does not match configured/expected ({n_components_expected}) SVD components (and is not 0). This might indicate a configuration change since saving.")
                # Decide if this is fatal. For this script, mismatch is fatal to ensure consistency.
                print("FATAL Validation: Column count mismatch based on configured SVD components.")
                is_loaded_valid = False


            if item_id_to_index is None or not item_id_to_index:
                 print("Validation Failed: item_id_to_index is None or empty.")
                 is_loaded_valid = False
            # Also check if the number of items in mapping matches input metadata
            elif len(item_id_to_index) != len(clean_item_meta_df):
                 print(f"Validation Warning: Loaded item_id_to_index has {len(item_id_to_index)} items, expected {len(clean_item_meta_df)}.")
                 print("FATAL Validation: Item count mismatch in mapping.")
                 is_loaded_valid = False


            if index_to_item_id is None or not index_to_item_id:
                 print("Validation Failed: index_to_item_id is None or empty.")
                 is_loaded_valid = False
            # Check consistency between the two mappings
            elif len(item_id_to_index) != len(index_to_item_id):
                 print("Validation Failed: Item ID and index mapping sizes do not match.")
                 is_loaded_valid = False
            # Check item_id_index_map_df consistency
            if item_id_index_map_df is None or item_id_index_map_df.empty:
                 print("Validation Failed: item_id_index_map_df is None or empty.")
                 is_loaded_valid = False
            elif len(item_id_index_map_df) != len(clean_item_meta_df):
                 print(f"Validation Warning: item_id_index_map_df has {len(item_id_index_map_df)} rows, expected {len(clean_item_meta_df)}.")
                 print("FATAL Validation: Item count mismatch in map_df.")
                 is_loaded_valid = False


            # Check transformers validity *if* they were expected
            if is_scaler_expected and (item_scaler is None or not hasattr(item_scaler, 'mean_')):
                 print("Validation Failed: item_scaler was expected but is None or not fitted.")
                 is_loaded_valid = False # Force regeneration if scaler needed but invalid

            if is_encoder_expected and (item_encoder is None or not hasattr(item_encoder, 'categories_')):
                 print("Validation Failed: item_encoder was expected but is None or not fitted.")
                 is_loaded_valid = False # Force regeneration if encoder needed but invalid

            # Check column lists are not None (already implicitly done by file existence check) and match expected types
            if not (isinstance(item_numerical_cols_for_scaling, list) and isinstance(item_categorical_cols_for_encoding, list) and isinstance(item_binary_cols, list) and
                    isinstance(item_original_numerical_cols, list) and isinstance(item_original_nominal_categorical_cols, list) and isinstance(item_details_cols, list)):
                 print("Validation Failed: Loaded column lists are not all lists.")
                 is_loaded_valid = False

            if is_loaded_valid:
                print("Successfully loaded and validated all item data, mappings, and transformers.")
                prepare_item_features = False # Skip generation
            else:
                print("Warning: Saved item data failed validation. Proceeding with regeneration.")
                # Ensure variables are reset to None to avoid using invalid loaded data
                X_items_reduced = None; item_id_to_index = None; index_to_item_id = None; item_id_index_map_df = None
                item_scaler = None; item_encoder = None
                item_numerical_cols_for_scaling = []; item_categorical_cols_for_encoding = []
                item_binary_cols = []; item_original_numerical_cols = []; item_original_nominal_categorical_cols = []; item_details_cols = []

        except Exception as e:
            print(f"Error loading saved item files: {e}. Proceeding with regeneration.")
            traceback.print_exc() # Print traceback for loading errors
            # Ensure variables are reset to None
            X_items_reduced = None; item_id_to_index = None; index_to_item_id = None; item_id_index_map_df = None
            item_scaler = None; item_encoder = None
            item_numerical_cols_for_scaling = []; item_categorical_cols_for_encoding = []
            item_binary_cols = []; item_original_numerical_cols = []; item_original_nominal_categorical_cols = []; item_details_cols = []
            prepare_item_features = True # Force regeneration

    else:
        print("\nSaved core item features or transformers not found. Generating data...")
        prepare_item_features = True # Files don't exist, need to regenerate


    # --- Only proceed with item feature generation if prepare_item_features is True ---
    if prepare_item_features:
        print("\n--- Generating Item Features from Cleaned Metadata ---")

        # Work on a copy to avoid modifying the input DataFrame
        item_features_processed_df = clean_item_meta_df.copy()

        # --- Identify Feature Columns for Item Processing ---
        item_meta_cols_in_df = item_features_processed_df.columns.tolist()

        # Filter the lists from config based on columns present in the DataFrame
        # Ensure we use the return variable names for the lists that will be saved/returned
        item_original_numerical_cols = [col for col in config.ITEM_NUMERICAL_COLS if col in item_meta_cols_in_df]
        item_original_nominal_categorical_cols = [col for col in config.ITEM_NOMINAL_CATEGORICAL_COLS if col in item_meta_cols_in_df]
        item_binary_cols = [col for col in config.ITEM_BINARY_COLS if col in item_meta_cols_in_df]
        item_text_cols = [col for col in config.ITEM_TEXT_COLS if col in item_meta_cols_in_df] # Use a local name for text cols for vectorization
        item_details_cols = [col for col in item_meta_cols_in_df if col.startswith('details_')]


        # These are the columns that will be passed to the transformers for SVD
        item_numerical_cols_for_scaling = item_original_numerical_cols
        item_categorical_cols_for_encoding = item_original_nominal_categorical_cols + item_details_cols


        print(f"\nIdentified {len(item_original_numerical_cols)} numerical, {len(item_original_nominal_categorical_cols)} nominal categorical, {len(item_binary_cols)} binary, {len(item_text_cols)} text, and {len(item_details_cols)} details item feature columns.")


        # --- Item Feature Preprocessing (NaN Handling) ---
        print("\nHandling NaNs specifically in clean_item_meta_df before feature processing...")
        # Note: NaNs might have been introduced if items in train_df were not in the original metadata,
        # but here we process the clean_item_meta_df itself.
        # Ensure NaNs are handled before fitting transformers.

        # For Numerical Columns: Impute NaNs using median from this item features dataset
        for col in item_numerical_cols_for_scaling: # Use the list identified from the df
             if item_features_processed_df[col].isnull().sum() > 0:
                  # Calculate median only if there's non-NaN data, otherwise use 0 as fallback
                  median_val = item_features_processed_df[col].median() if not item_features_processed_df[col].isnull().all() else 0
                  item_features_processed_df[col] = item_features_processed_df[col].fillna(median_val)
                  item_features_processed_df[col] = item_features_processed_df[col].astype(float) # Ensure float type


        # For Binary Columns: Impute NaNs with 0, ensure int type
        for col in item_binary_cols: # Use the list identified from the df
            if item_features_processed_df[col].isnull().sum() > 0:
                item_features_processed_df[col] = item_features_processed_df[col].fillna(0)
            item_features_processed_df[col] = item_features_processed_df[col].astype(int)


        # For Categorical/Details Columns: Fill NaN with 'Missing' string, ensure string type
        for col in item_categorical_cols_for_encoding: # Use the list identified from the df
            if item_features_processed_df[col].isnull().sum() > 0:
                 item_features_processed_df[col] = item_features_processed_df[col].fillna('Missing')
            item_features_processed_df[col] = item_features_processed_df[col].astype(str)


        # For Text Columns ('title'): Fill NaN with empty string and ensure string type
        for col in item_text_cols: # Use the local list for text cols
            if item_features_processed_df[col].isnull().sum() > 0:
                 item_features_processed_df[col] = item_features_processed_df[col].fillna('')
            item_features_processed_df[col] = item_features_processed_df[col].astype(str)


        # Verify no NaNs remain in the processed feature columns used for transformations
        item_cols_for_processing = item_numerical_cols_for_scaling + item_binary_cols + item_categorical_cols_for_encoding + item_text_cols
        nan_count_after_item_handling = item_features_processed_df[item_cols_for_processing].isnull().sum().sum()
        if nan_count_after_item_handling == 0:
             print("NaN handling for item features complete. No NaNs remaining in target columns.")
        else:
             # Print columns with NaNs for debugging
             cols_with_nan = item_features_processed_df[item_cols_for_processing].isnull().sum()
             cols_with_nan = cols_with_nan[cols_with_nan > 0]
             print(f"Warning: {nan_count_after_item_handling} NaNs still found in item features after handling in columns:")
             print(cols_with_nan)


        # --- Scale Numerical Item Features ---
        print("\nScaling numerical item features...")
        # Initialize with 0 columns, shape matching the number of items in the input df
        scaled_numerical_features = scipy.sparse.csr_matrix((len(item_features_processed_df), 0))
        if item_numerical_cols_for_scaling: # Only process if there are columns to scale
             try:
                 item_scaler = StandardScaler() # Initialize scaler
                 # Ensure the data is float before scaling (done during NaN handling)
                 scaled_numerical_features = item_scaler.fit_transform(item_features_processed_df[item_numerical_cols_for_scaling])
                 scaled_numerical_features = scipy.sparse.csr_matrix(scaled_numerical_features) # Convert to sparse for hstack
                 print(f"Scaled numerical features shape: {scaled_numerical_features.shape}")
             except Exception as e:
                 print(f"Warning: Could not fit/transform Item StandardScaler: {e}. Skipping numerical scaling.")
                 traceback.print_exc()
                 item_scaler = None # Ensure None on error
                 scaled_numerical_features = scipy.sparse.csr_matrix((len(item_features_processed_df), 0)) # Ensure empty matrix on error
        else:
             print("No numerical columns identified for scaling. Skipping scaling.")
             item_scaler = None # Ensure None if not fitted


        # --- Encode Categorical Item Features ---
        print("\nEncoding categorical item features...")
        # Initialize with 0 columns
        encoded_categorical_features = scipy.sparse.csr_matrix((len(item_features_processed_df), 0))
        if item_categorical_cols_for_encoding: # Only process if there are columns to encode
            # Ensure data is string type before encoding (done during NaN handling)
            try:
               # Attempt to use sparse_output if available (scikit-learn >= 1.2)
               try:
                   item_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
                   print("Initialized OneHotEncoder for item features with sparse_output=True.")
               except TypeError: # Fallback to old sparse argument
                   item_encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)
                   print("Initialized OneHotEncoder for item features with sparse=True (deprecated).")

               encoded_categorical_features = item_encoder.fit_transform(item_features_processed_df[item_categorical_cols_for_encoding])
               print(f"Encoded categorical features shape: {encoded_categorical_features.shape}")

            except Exception as e:
                 print(f"Warning: Could not fit Item OneHotEncoder: {e}. Skipping encoding.")
                 traceback.print_exc()
                 item_encoder = None # Ensure None on error
                 encoded_categorical_features = scipy.sparse.csr_matrix((len(item_features_processed_df), 0)) # Ensure empty matrix on error
        else:
             print("No categorical columns identified for encoding. Skipping encoding.")
             item_encoder = None # Ensure None if not fitted


        # --- Vectorize Text Features ('title') ---
        print("\nVectorizing text item features ('title') using TF-IDF...")
        text_vectors = scipy.sparse.csr_matrix((len(item_features_processed_df), 0))
        # Assuming only one text column ('title') relevant from config.ITEM_TEXT_COLS
        # Filter config text cols to only those present in the dataframe
        item_text_cols_present = [col for col in config.ITEM_TEXT_COLS if col in item_features_processed_df.columns]

        if item_text_cols_present:
             text_col = item_text_cols_present[0] # Take the first text column identified
             print(f"Vectorizing text column: '{text_col}'")
             # Ensure data is string type (done during NaN handling)
             # Initialize and fit the vectorizer on the item data
             # Parameters match the strategy from previous cells if applicable
             try:
                 vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english') # Consider parameters from config
                 text_vectors = vectorizer.fit_transform(item_features_processed_df[text_col])
                 print(f"Text vectors shape: {text_vectors.shape}")
                 # Note: TF-IDF vectorizer is typically not saved or returned.
             except Exception as e:
                print(f"Warning: Could not fit TF-IDF vectorizer on column '{text_col}': {e}. Text features will be empty.")
                traceback.print_exc()
                text_vectors = scipy.sparse.csr_matrix((len(item_features_processed_df), 0)) # Fallback to empty
        else:
             print("No text columns identified or present for vectorization. Skipping text vectorization.")


        # --- Combine All Processed Item Features into a Single Sparse Matrix ---
        print("\nCombining all processed item features into a sparse matrix...")
        matrices_to_stack = []

        # Add matrices only if they have columns
        if scaled_numerical_features.shape[1] > 0:
             matrices_to_stack.append(scaled_numerical_features)
        # Add binary columns (converted to sparse matrix)
        if item_binary_cols:
             # Ensure binary columns are float/numerical before converting to sparse (done in NaN handling)
             matrices_to_stack.append(scipy.sparse.csr_matrix(item_features_processed_df[item_binary_cols].values))
        # Only add encoded categorical if encoder was fitted and produced features
        # Also check if the encoder has the categories_ attribute indicating it was fitted
        if encoded_categorical_features.shape[1] > 0 and item_encoder is not None and hasattr(item_encoder, 'categories_'):
             matrices_to_stack.append(encoded_categorical_features)
        # Add text vectors if vectorizer produced features
        if text_vectors.shape[1] > 0:
            matrices_to_stack.append(text_vectors)


        X_items = None # Initialize as None
        if matrices_to_stack:
            try:
                 X_items = scipy.sparse.hstack(matrices_to_stack)
                 print(f"Combined item feature matrix (sparse) shape: {X_items.shape}")

                 # Add check for NaNs/Infs in the final combined matrix before SVD
                 if hasattr(X_items, 'data') and X_items.data is not None:
                     if np.isnan(X_items.data).sum() > 0:
                         raise ValueError(f"NaNs found in the data array of the combined sparse matrix before SVD: {np.isnan(X_items.data).sum()}")
                     if np.isinf(X_items.data).sum() > 0:
                         raise ValueError(f"Infinities found in the data array of the combined sparse matrix before SVD: {np.isinf(X_items.data).sum()}")
                     print("Checked combined sparse matrix data: No NaNs or Infs found.")


            except Exception as e:
                print(f"FATAL ERROR: Could not stack item feature matrices or found invalid values: {e}. X_items cannot be created.")
                traceback.print_exc()
                X_items = None
                prepare_item_features = False # Cannot proceed if stacking fails
        else:
            # If no matrices were created (e.g., no feature columns in input df)
            print("Warning: No item feature matrices were prepared for stacking. X_items will be empty.")
            X_items = scipy.sparse.csr_matrix((len(item_features_processed_df), 0)) # Create an empty matrix with correct number of rows


        # --- Apply Dimensionality Reduction using TruncatedSVD ---
        X_items_reduced = None # Initialize as None
        svd_item = None # Initialize SVD object outside try block

        # Proceed only if X_items was created, has columns, and SVD components > 0
        n_components_actual = max(0, item_svd_n_components) # Treat <0 config as 0
        if prepare_item_features and X_items is not None and X_items.shape[1] > 0 and n_components_actual > 0:
            print(f"\nApplying TruncatedSVD to item features with {n_components_actual} components...")
            # Ensure n_components is not greater than the number of features or samples
            # TruncatedSVD n_components must be <= min(n_samples, n_features)
            n_components_fit = min(n_components_actual, X_items.shape[0], X_items.shape[1])

            if n_components_fit <= 0:
                 print("Warning: Item SVD components is 0 or less based on configured value or matrix shape. Skipping SVD.")
                 # If SVD is skipped but X_items exists, return a 0-dim matrix with the correct number of rows
                 X_items_reduced = np.zeros((X_items.shape[0], 0))
            else:
                 print(f"Reducing dimensionality to {n_components_fit} components.")
                 svd_item = TruncatedSVD(n_components=n_components_fit, random_state=seed)

                 try:
                     X_items_reduced = svd_item.fit_transform(X_items)

                     print(f"Reduced item feature matrix (dense) shape: {X_items_reduced.shape}")

                     # Check if fit was successful by looking for post-fit attributes
                     if hasattr(svd_item, 'n_components_') and hasattr(svd_item, 'explained_variance_ratio_') and svd_item.explained_variance_ratio_ is not None:
                          print(f"Explained variance ratio of the top {svd_item.n_components_} components: {svd_item.explained_variance_ratio_.sum():.4f}")
                     else:
                         # This warning indicates the SVD fit_transform might have finished without error
                         # but didn't set standard post-fit attributes, which is unexpected for TruncatedSVD.
                         print("Warning: SVD fit_transform completed, but standard post-fit attributes (like 'n_components_') were not found.")


                 except Exception as e:
                      # This block catches the actual error from svd_item.fit_transform()
                      print(f"FATAL ERROR: Could not fit or transform with TruncatedSVD for item features: {e}")
                      traceback.print_exc() # Print the full traceback of the *actual* error
                      print("X_items_reduced cannot be created due to the above error.")
                      X_items_reduced = None # Ensure None on error
                      prepare_item_features = False # Cannot proceed

        elif X_items is not None: # X_items exists but has 0 columns or SVD components requested <= 0
             print("Skipping Dimensionality Reduction for item features (0 input features or SVD components requested <= 0). X_items_reduced will be 0-dimensional.")
             # If SVD is skipped but X_items exists, return a 0-dim matrix with the correct number of rows
             X_items_reduced = np.zeros((X_items.shape[0], 0))


        else: # X_items was None or failed creation earlier
             print("Skipping Dimensionality Reduction for item features as X_items was not created.")
             X_items_reduced = None # Ensure None

        # Note: prepare_item_features flag is set to False if critical errors occurred.
        # The `svd_item` variable will either be None (if skipped or failed initialization)
        # or the SVD object (either fitted or failed during fit_transform).


        # --- Create item_id to index mapping and vice versa ---
        item_id_to_index = None; index_to_item_id = None; item_id_index_map_df = None # Initialize

        # Need item_features_processed_df (the source of IDs and indices) and X_items_reduced (implies successful prior steps)
        # Crucially, mapping is based on the *row index* of the input df, not the SVD output rows if filtering occurred.
        # Let's base the mapping on the *original* index from clean_item_meta_df
        # But map to the row index in the *processed* df used for SVD.
        # A safer approach is to map item_id -> index in the processed df (which is 0...N-1)
        # and then use that index to look up the corresponding row in X_items_reduced.
        # Let's stick to mapping item_id to the row index in the processed df used for SVD (0...N-1).
        # This requires X_items_reduced to have the same number of rows as item_features_processed_df.

        if prepare_item_features and item_features_processed_df is not None and config.ITEM_ID_COL in item_features_processed_df.columns and X_items_reduced is not None and X_items_reduced.shape[0] == len(item_features_processed_df):
             print("\nCreating item ID to index mappings...")
             try:
                  # Create dictionary mapping item_id to its index in the item_features_processed_df (which corresponds to rows in X_items_reduced)
                  # Use item_features_processed_df.index as the index, which should be 0, 1, 2...
                  item_id_to_index_dict = pd.Series(item_features_processed_df.index, index=item_features_processed_df[config.ITEM_ID_COL]).to_dict()
                  item_id_to_index = {str(k): v for k, v in item_id_to_index_dict.items()} # Ensure string keys for JSON

                  # Create dictionary mapping index to its item_id
                  index_to_item_id_dict = pd.Series(item_features_processed_df[config.ITEM_ID_COL], index=item_features_processed_df.index).to_dict()
                  index_to_item_id = {str(k): v for k, v in index_to_item_id_dict.items()} # Ensure string keys for JSON

                  print(f"Created item ID to index mappings for {len(item_id_to_index)} items.")

                  # Create item_id_index_map_df (item_id to its SVD row index)
                  if item_id_to_index is not None and len(item_id_to_index) > 0:
                       item_id_index_map_df = pd.DataFrame({
                          config.ITEM_ID_COL: list(item_id_to_index.keys()),
                          'svd_index': list(item_id_to_index.values())
                      })
                       item_id_index_map_df['svd_index'] = item_id_index_map_df['svd_index'].astype(int)
                       print("Created item_id_index_map_df.")
                  else:
                      print("Warning: item_id_to_index is empty, cannot create item_id_index_map_df.")
                      item_id_index_map_df = pd.DataFrame(columns=[config.ITEM_ID_COL, 'svd_index']) # Create empty DF

             except Exception as e:
                 print(f"FATAL ERROR creating item mappings: {e}. Skipping remaining steps.")
                 traceback.print_exc()
                 item_id_to_index = None
                 index_to_item_id = None
                 item_id_index_map_df = pd.DataFrame(columns=[config.ITEM_ID_COL, 'svd_index']) # Ensure empty DF on error
                 prepare_item_features = False # Cannot proceed if mappings fail
        else: # Mappings prerequisites not met (e.g., X_items_reduced None or row mismatch, or input df issue)
            if prepare_item_features: # Only print warning if generation was intended
                 print("Skipping item mapping creation as prerequisites were not met (X_items_reduced None/row mismatch or input df issue).")
            item_id_to_index = {} # Ensure empty dict if skipping
            index_to_item_id = {} # Ensure empty dict if skipping
            item_id_index_map_df = pd.DataFrame(columns=[config.ITEM_ID_COL, 'svd_index']) # Ensure empty DF


    # --- Save Processed Item Data and Transformers if Generation was Successful ---
    # Check the flag one last time before saving.
    # Saving should only happen if prepare_item_features is still True,
    # and the essential outputs were created (X_items_reduced, mappings, map_df).
    # The final validation block below does a good check for this.
    # Let's move the save logic *after* the final validation to only save if we are about to return valid data.

    # --- Final check on essential item features needed for the next phase ---
    # These are the items that MUST be returned successfully for the next step (hybrid_matrix_assembler) to work.
    # X_items_reduced must have rows matching the input df (if input df had rows).
    # item_id_to_index must be non-empty and match input df rows if input df had rows.
    # item_id_index_map_df must be non-empty and match input df rows if input df had rows.

    # Re-check the validity flag based on whether generation successfully produced required outputs
    is_item_processing_successful = prepare_item_features # Start with the generation flag state

    # Check X_items_reduced validity
    if is_item_processing_successful: # Only check if prior steps were OK
        if X_items_reduced is None or not hasattr(X_items_reduced, 'shape'):
             print("FATAL ERROR Check: X_items_reduced is None or invalid.")
             is_item_processing_successful = False
        elif len(clean_item_meta_df) > 0 and X_items_reduced.shape[0] != len(clean_item_meta_df):
            print(f"FATAL ERROR Check: X_items_reduced row count ({X_items_reduced.shape[0]}) does not match input metadata rows ({len(clean_item_meta_df)}).")
            is_item_processing_successful = False
        elif len(clean_item_meta_df) == 0 and X_items_reduced.shape[0] != 0:
             print(f"FATAL ERROR Check: X_items_reduced has {X_items_reduced.shape[0]} rows but input metadata was empty.")
             is_item_processing_successful = False
        # Check if the SVD dimension matches the requested, unless requested was 0
        n_components_actual = max(0, item_svd_n_components)
        if is_item_processing_successful and X_items_reduced is not None and X_items_reduced.shape[0] > 0 and X_items_reduced.shape[1] != n_components_actual and n_components_actual > 0:
             # This check is critical for ensuring consistency if SVD components were requested
             print(f"FATAL ERROR Check: X_items_reduced column count ({X_items_reduced.shape[1]}) does not match requested SVD components ({n_components_actual}).")
             is_item_processing_successful = False


    # Check item_id_to_index validity
    if is_item_processing_successful: # Only check if prior steps were OK
        if item_id_to_index is None or not item_id_to_index:
             # Only fail if input metadata had rows; if input was empty, an empty dict is expected/okay
             if len(clean_item_meta_df) > 0:
                  print("FATAL ERROR Check: item_id_to_index is None or empty, but input metadata had rows.")
                  is_item_processing_successful = False
             # No else needed, if input was empty and output is empty, it's ok (is_item_processing_successful remains True)

    # Check item_id_index_map_df validity
    if is_item_processing_successful: # Only check if prior steps were OK
        if item_id_index_map_df is None or item_id_index_map_df.empty:
             # Only fail if input metadata had rows; if input was empty, an empty df is expected/okay
             if len(clean_item_meta_df) > 0:
                  print("FATAL ERROR Check: item_id_index_map_df is None or empty, but input metadata had rows.")
                  is_item_processing_successful = False
             # No else needed
        # If map_df is not empty, check row count matches input metadata
        elif len(item_id_index_map_df) != len(clean_item_meta_df):
             print(f"FATAL ERROR Check: item_id_index_map_df row count ({len(item_id_index_map_df)}) does not match input metadata rows ({len(clean_item_meta_df)}).")
             is_item_processing_successful = False


    # Scaler and Encoder availability is checked in Step 5 based on column lists.
    # The column lists themselves should always be returned as lists, even if empty.
    if is_item_processing_successful:
        if not (isinstance(item_numerical_cols_for_scaling, list) and isinstance(item_categorical_cols_for_encoding, list) and isinstance(item_binary_cols, list) and
                isinstance(item_original_numerical_cols, list) and isinstance(item_original_nominal_categorical_cols, list) and isinstance(item_details_cols, list)):
             print("FATAL ERROR Check: Returned column lists are not all lists.")
             is_item_processing_successful = False


    # --- Save Processed Item Data and Transformers IF processing was successful ---
    # This block now runs only if is_item_processing_successful is True
    if is_item_processing_successful and prepare_item_features: # Only save if generation happened AND it was successful
        print("\nSaving processed item features, matrices, mappings, and transformers...")
        try:
            # Save Reduced Matrix (save even if 0 dim, loading needs to handle it)
            np.save(x_items_reduced_npy_path, X_items_reduced)
            print(f"Saved X_items_reduced to {x_items_reduced_npy_path}")

            # Save Mappings (save even if empty dict/df, loading needs to handle it)
            with open(item_id_to_index_json_path, 'w') as f:
                json.dump(item_id_to_index, f)
            print(f"Saved item_id_to_index to {item_id_to_index_json_path}")

            # Need to convert int keys back to string for JSON dump
            index_to_item_id_str_keys = {str(k): v for k, v in index_to_item_id.items()}
            with open(index_to_item_id_json_path, 'w') as f:
                 json.dump(index_to_item_id_str_keys, f)
            print(f"Saved index_to_item_id to {index_to_item_id_json_path}")

            # Save Transformers (only if fitted)
            if item_scaler is not None: # Check the actual variable, not just the list
                 with open(item_scaler_pkl_path, 'wb') as f:
                      pickle.dump(item_scaler, f)
                 print(f"Saved item_scaler to {item_scaler_pkl_path}")
            else: print("Skipping save: item_scaler is None.")

            if item_encoder is not None: # Check the actual variable, not just the list
                 with open(item_encoder_pkl_path, 'wb') as f:
                      pickle.dump(item_encoder, f)
                 print(f"Saved item_encoder to {item_encoder_pkl_path}")
            else: print("Skipping save: item_encoder is None.")

            # Save Column Lists (save even if empty lists)
            with open(item_num_cols_json_path, 'w') as f: json.dump(item_numerical_cols_for_scaling, f)
            print(f"Saved item_numerical_cols_for_scaling to {item_num_cols_json_path}")

            with open(item_cat_cols_json_path, 'w') as f: json.dump(item_categorical_cols_for_encoding, f)
            print(f"Saved item_categorical_cols_for_encoding to {item_cat_cols_json_path}")

            with open(item_binary_cols_json_path, 'w') as f: json.dump(item_binary_cols, f)
            print(f"Saved item_binary_cols to {item_binary_cols_json_path}")

            with open(item_original_num_cols_json_path, 'w') as f: json.dump(item_original_numerical_cols, f)
            print(f"Saved item_original_numerical_cols to {item_original_num_cols_json_path}")

            with open(item_original_nom_cat_cols_json_path, 'w') as f: json.dump(item_original_nominal_categorical_cols, f)
            print(f"Saved item_original_nominal_categorical_cols to {item_original_nom_cat_cols_json_path}")

            with open(item_details_cols_json_path, 'w') as f: json.dump(item_details_cols, f)
            print(f"Saved item_details_cols to {item_details_cols_json_path}")


            print("Item data, transformers, and lists saved.")

        except Exception as e:
            print(f"\nWarning: An error occurred during saving processed item data or transformers: {e}")
            traceback.print_exc() # Print traceback for saving errors
            # Saving errors are warnings here, as the function can still return the generated data in memory.
            # The caller (main.py) should check the returned objects regardless.

    elif prepare_item_features: # Generation was intended but failed the final check
         print("Skipping item feature saving as generation was not fully successful.")


    if is_item_processing_successful:
        print("\nEssential item data, mappings, and transformers are available.")
        print("Step 4 execution complete.")
        # Return all generated/loaded objects
        return (X_items_reduced, item_id_to_index, index_to_item_id, item_id_index_map_df,
                item_scaler, item_encoder,
                item_numerical_cols_for_scaling, item_categorical_cols_for_encoding, item_binary_cols,
                item_original_numerical_cols, item_original_nominal_categorical_cols, item_details_cols)

    else:
        print("\nStep 4 failed to produce essential outputs.")
        # Return None for critical outputs, and potentially empty lists/None for others
        # Ensure lists are returned as empty lists, not None
        # The column lists populated during the generation attempt might be partial,
        # but are still useful for debugging or downstream steps that handle missing features gracefully.
        return (None, None, None, pd.DataFrame(columns=[config.ITEM_ID_COL, 'svd_index']), # Ensure empty DF for map
                None, None,             # Transformers
                item_numerical_cols_for_scaling, item_categorical_cols_for_encoding, item_binary_cols,
                item_original_numerical_cols, item_original_nominal_categorical_cols, item_details_cols)


# Optional: Add a main block to test this script standalone
if __name__ == "__main__":
    print("Running item_processor.py as standalone script...")
    # To run this standalone, you need to provide a dummy or actual cleaned metadata dataframe.

    # --- Simulate Loading Required Input Data ---
    print("Simulating loading input data for standalone test...")
    sim_meta_df = None # Initialize as None
    try:
        # Load small subset of actual data for a more realistic test
        # Adjust path as needed for your test data location
        dummy_meta_path = '../data/dataset/clean_beauty_item_meta_with_details.csv' # Using actual config path for simulation

        if os.path.exists(dummy_meta_path):
            print(f"Attempting to load small actual data from {dummy_meta_path}")
            try:
                sim_meta_df = pd.read_csv(dummy_meta_path).head(200) # Load first 200 items
                sim_meta_df[config.ITEM_ID_COL] = sim_meta_df[config.ITEM_ID_COL].astype(str)
                print(f"Simulated loading small clean_item_meta_df ({len(sim_meta_df)} rows)")
                # Ensure expected columns are present for the test to pass fully
                # Note: This doesn't guarantee the *data* in these columns is clean or non-empty
                required_sim_cols = [config.ITEM_ID_COL] + config.ITEM_NUMERICAL_COLS + config.ITEM_BINARY_COLS + config.ITEM_NOMINAL_CATEGORICAL_COLS + config.ITEM_TEXT_COLS
                for col in required_sim_cols:
                     if col not in sim_meta_df.columns:
                          print(f"Warning: Required column '{col}' from config not found in simulated metadata slice.")
                          # Add a dummy column if missing to prevent errors downstream, but data will be empty
                          if col in config.ITEM_NUMERICAL_COLS: sim_meta_df[col] = np.nan
                          elif col in config.ITEM_BINARY_COLS: sim_meta_df[col] = np.nan
                          elif col in config.ITEM_NOMINAL_CATEGORICAL_COLS: sim_meta_df[col] = np.nan
                          elif col in config.ITEM_TEXT_COLS: sim_meta_df[col] = '' # Empty string for text
                          print(f"  Added dummy column '{col}' with placeholder/NaN values.")


                # Add dummy details cols if none exist and config lists them
                # If config doesn't list them explicitly, we rely on the 'startswith' logic.
                # Let's create some dummy ones if none exist in the loaded slice
                existing_details_cols = [col for col in sim_meta_df.columns if col.startswith('details_')]
                if not existing_details_cols: # If no 'details_' columns exist
                     print("Adding dummy details_ columns for simulation.")
                     sim_meta_df['details_dummy1'] = np.random.choice(['cat1', 'cat2', np.nan], len(sim_meta_df))
                     sim_meta_df['details_dummy2'] = np.random.choice(['valA', 'valB', np.nan], len(sim_meta_df))


            except Exception as e:
                 print(f"Error loading actual data for simulation: {e}. Falling back to dummy data.")
                 traceback.print_exc()
                 sim_meta_df = None # Reset to None to trigger dummy data creation


        if sim_meta_df is None:
            print("Creating simple dummy data.")
            # Create simple dummy dataframe if files aren't available or loading failed
            num_dummy_items = 50
            sim_meta_df = pd.DataFrame({
                 config.ITEM_ID_COL: [f'item{i:03d}' for i in range(num_dummy_items)],
                 # Include some columns from config
                 config.ITEM_NUMERICAL_COLS[0] if config.ITEM_NUMERICAL_COLS else 'dummy_num': np.random.rand(num_dummy_items) * 10,
                 config.ITEM_BINARY_COLS[0] if config.ITEM_BINARY_COLS else 'dummy_bin': np.random.randint(0, 2, num_dummy_items),
                 config.ITEM_NOMINAL_CATEGORICAL_COLS[0] if config.ITEM_NOMINAL_CATEGORICAL_COLS else 'dummy_cat': np.random.choice(['CatA', 'CatB', 'CatC'], num_dummy_items),
                 config.ITEM_TEXT_COLS[0] if config.ITEM_TEXT_COLS else 'dummy_text': [f'This is item {i} text' for i in range(num_dummy_items)],
                 # Dummy details cols
                 'details_brand': np.random.choice(['BrandX', 'BrandY', 'BrandZ', np.nan], num_dummy_items),
                 'details_color': np.random.choice(['Red', 'Blue', 'Green', np.nan], num_dummy_items),
                 'details_material': np.random.choice(['Cotton', 'Polyester', np.nan], num_dummy_items),
            }).astype({config.ITEM_ID_COL: str})

            # Ensure numeric types where expected (based on potential dummy columns)
            if config.ITEM_NUMERICAL_COLS: sim_meta_df[config.ITEM_NUMERICAL_COLS[0]] = sim_meta_df[config.ITEM_NUMERICAL_COLS[0]].astype(float)
            if config.ITEM_BINARY_COLS: sim_meta_df[config.ITEM_BINARY_COLS[0]] = sim_meta_df[config.ITEM_BINARY_COLS[0]].astype(int)


            # Introduce some NaNs for testing NaN handling in dummy data
            for col in sim_meta_df.columns:
                 if col != config.ITEM_ID_COL and sim_meta_df[col].dtype != 'object': # Avoid introducing NaN into string cols like ID or text initially
                     sim_meta_df.loc[sim_meta_df.sample(frac=0.15, random_state=config.SEED).index, col] = np.nan
                 elif sim_meta_df[col].dtype == 'object' and col != config.ITEM_ID_COL: # Introduce NaN into string/object cols
                      sim_meta_df.loc[sim_meta_df.sample(frac=0.15, random_state=config.SEED).index, col] = np.nan


            print(f"Created simple dummy metadata ({len(sim_meta_df)} rows)")


        print("Simulated loading complete.")

        # --- Call the main function ---
        print("\nCalling process_item_features...")
        # Define temporary paths for saving during standalone test
        temp_intermediate_dir = './temp_intermediate_for_item_proc_test'
        os.makedirs(temp_intermediate_dir, exist_ok=True)

        temp_x_items_reduced = os.path.join(temp_intermediate_dir, 'test_X_items_reduced.npy')
        temp_item_id_to_index = os.path.join(temp_intermediate_dir, 'test_item_id_to_index.json')
        temp_index_to_item_id = os.path.join(temp_intermediate_dir, 'test_index_to_item_id.json')
        temp_item_scaler = os.path.join(temp_intermediate_dir, 'test_item_scaler.pkl')
        temp_item_encoder = os.path.join(temp_intermediate_dir, 'test_item_encoder.pkl')
        temp_item_num_cols = os.path.join(temp_intermediate_dir, 'test_item_num_cols.json')
        temp_item_cat_cols = os.path.join(temp_intermediate_dir, 'test_item_cat_cols.json')
        temp_item_binary_cols = os.path.join(temp_intermediate_dir, 'test_item_binary_cols.json')
        temp_item_original_num_cols = os.path.join(temp_intermediate_dir, 'test_item_original_num_cols.json')
        temp_item_original_nom_cat_cols = os.path.join(temp_intermediate_dir, 'test_item_original_nom_cat_cols.json')
        temp_item_details_cols = os.path.join(temp_intermediate_dir, 'test_item_details_cols.json')


        # Configure SVD components for the standalone test - use a small number
        # Ensure this doesn't exceed the number of dummy items or features
        test_svd_components = min(config.ITEM_SVD_N_COMPONENTS, len(sim_meta_df) -1, 50) # Max 50 components for small test

        (X_items_reduced_gen, item_id_to_index_gen, index_to_item_id_gen, item_id_index_map_df_gen,
         item_scaler_gen, item_encoder_gen,
         item_numerical_cols_for_scaling_gen, item_categorical_cols_for_encoding_gen, item_binary_cols_gen,
         item_original_numerical_cols_gen, item_original_nominal_categorical_cols_gen, item_details_cols_gen) = process_item_features(
             sim_meta_df,
             test_svd_components, # Use test specific components
             config.SEED,
             temp_x_items_reduced,
             temp_item_id_to_index,
             temp_index_to_item_id,
             temp_item_scaler,
             temp_item_encoder,
             temp_item_num_cols,
             temp_item_cat_cols,
             temp_item_binary_cols,
             temp_item_original_num_cols,
             temp_item_original_nom_cat_cols,
             temp_item_details_cols
         )


        print("\n--- Standalone Item Feature Processing Complete ---")
        print(f"Called with {test_svd_components} SVD components.")

        if X_items_reduced_gen is not None:
            print(f"Final X_items_reduced shape: {X_items_reduced_gen.shape}")
        else:
            print("Final X_items_reduced is None.")

        if item_id_to_index_gen is not None:
             print(f"Item ID to index mapping size: {len(item_id_to_index_gen)}")
        else:
             print("Item ID to index mapping is None.")

        if item_id_index_map_df_gen is not None:
             print(f"Item ID index map DataFrame shape: {item_id_index_map_df_gen.shape}")
        else:
             print("Item ID index map DataFrame is None.")


        print(f"Item numerical cols for scaling: {item_numerical_cols_for_scaling_gen}")
        print(f"Item categorical cols for encoding: {item_categorical_cols_for_encoding_gen}")
        print(f"Item binary cols: {item_binary_cols_gen}")
        print(f"Item original numerical cols: {item_original_numerical_cols_gen}")
        print(f"Item original nominal categorical cols: {item_original_nominal_categorical_cols_gen}")
        print(f"Item details cols: {item_details_cols_gen}")


        if item_scaler_gen is not None:
             print(f"Item Scaler type: {type(item_scaler_gen)}")
        else:
             print("Item Scaler is None.")

        if item_encoder_gen is not None:
             print(f"Item Encoder type: {type(item_encoder_gen)}")
             if hasattr(item_encoder_gen, 'categories_'):
                 print(f"Item Encoder categories count: {sum(len(cats) for cats in item_encoder_gen.categories_)}")
             else:
                 print("Item Encoder missing 'categories_' attribute.")
        else:
             print("Item Encoder is None.")


        # --- Clean up temporary files ---
        print("\nCleaning up temporary test files...")
        try:
             if os.path.exists(temp_intermediate_dir):
                  # Use glob to find files saved by the script to ensure cleanup works
                  # file_patterns = [
                  #    temp_x_items_reduced, temp_item_id_to_index, temp_index_to_item_id,
                  #    temp_item_scaler, temp_item_encoder, temp_item_num_cols,
                  #    temp_item_cat_cols, temp_item_binary_cols, temp_item_original_num_cols,
                  #    temp_item_original_nom_cat_cols, temp_item_details_cols
                  # ]
                  # for pattern in file_patterns:
                  #      for f in glob.glob(pattern):
                  #           os.remove(f)
                  shutil.rmtree(temp_intermediate_dir) # Simpler to remove the dir
                  print(f"Removed temporary directory: {temp_intermediate_dir}")
        except Exception as e:
             print(f"Warning: Could not remove temporary directory {temp_intermediate_dir}: {e}")
             traceback.print_exc() # Print traceback for cleanup errors


    except FileNotFoundError as e:
         print(f"Script failed: {e}. Ensure dummy data paths or actual data paths are correct.")
         traceback.print_exc()
    except ValueError as e:
         print(f"Script failed due to input data error: {e}")
         traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during script execution: {e}")
        traceback.print_exc()