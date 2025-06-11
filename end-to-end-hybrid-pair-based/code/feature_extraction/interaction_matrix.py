# main.py

import pandas as pd
import numpy as np
import os
import time
import textwrap
import scipy.sparse as sp # Use sp alias consistently
import json
import pickle
# Keep transformer classes for type checking in validation steps
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import traceback # Import traceback for debugging

# Import configuration
import config

# Import custom modules (keep imports at top or grouped logically)
# Import all modules potentially used *before* the main execution flow starts
import data_loader
import pair_generator
import item_processor
import user_processor
import hybrid_matrix_assembler


print("--- Feature Extraction and Training Data Preparation Script ---")
start_pipeline_time = time.time()

# --- Global Variables to Hold Data/Results ---
pipeline_data = {}

# --- Step 0: Setup and Configuration Check ---
print("Step 0: Setup and Configuration Check...")
# ... (Configuration print statements - keep as is) ...

# Define paths for new artifacts created in this script
INTERACTION_MATRIX_PATH = os.path.join(config.INTERMEDIATE_DATA_PATH, 'interaction_matrix.npz')
USER_ID_TO_INDEX_JSON_PATH_MATRIX = os.path.join(config.INTERMEDIATE_DATA_PATH, 'user_id_to_index_matrix.json') # Differentiate from user features map

print(f"Input Train Data Path: {config.TRAIN_CSV_PATH}")
print(f"Input Metadata Path: {config.CLEANED_METADATA_CSV_PATH}")
print(f"Output Dataframes Path: {config.OUTPUT_DATAFRAMES_PATH}")
print(f"Intermediate Data Path: {config.INTERMEDIATE_DATA_PATH}")
print(f"Random Seed: {config.SEED}")
print(f"Negative Sample Ratio: {config.NEGATIVE_SAMPLE_RATIO}")
print(f"Item SVD Components: {config.ITEM_SVD_N_COMPONENTS}")
print(f"User Categorical SVD Components: {config.USER_CAT_SVD_N_COMPONENTS}")
print(f"Batch Size for Stacking: {config.STACKING_BATCH_SIZE}")
print("-" * 50)


# Ensure output directories exist
os.makedirs(config.OUTPUT_DATAFRAMES_PATH, exist_ok=True)
os.makedirs(config.INTERMEDIATE_DATA_PATH, exist_ok=True)
print("Ensured output and intermediate directories exist.")

pipeline_ok = True # Flag to track if pipeline should continue


# -----------------------------------------------------------------------------
# Step 1 & 2: Load Core Data and Initial Merge (Implemented by data_loader.py)
# -----------------------------------------------------------------------------
print("\n" + "="*50 + "\n")
print("Step 1 & 2: Load Core Data and Initial Merge...")
if pipeline_ok:
    try:
        pipeline_data['train_df'], pipeline_data['clean_item_meta_df'], pipeline_data['train_merged_df'] = data_loader.load_and_merge_data(
            config.TRAIN_CSV_PATH,
            config.CLEANED_METADATA_CSV_PATH
        )
        print("Step 1 & 2 successfully completed.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR in Step 1 & 2: Input file not found: {e}")
        pipeline_ok = False
    except ValueError as e:
        print(f"FATAL ERROR in Step 1 & 2: Missing required column: {e}")
        pipeline_ok = False
    except Exception as e:
        print(f"FATAL ERROR in Step 1 & 2: An unexpected error occurred: {e}")
        traceback.print_exc() # Add traceback
        pipeline_ok = False

    # Explicitly check if critical outputs were produced, even if no exception was raised
    if pipeline_ok:
         if pipeline_data.get('train_df') is None or pipeline_data['train_df'].empty:
              print("FATAL ERROR: Original train dataframe was not successfully created or is empty after Step 1 & 2.")
              pipeline_ok = False
         elif pipeline_data.get('clean_item_meta_df') is None or pipeline_data['clean_item_meta_df'].empty:
              print("FATAL ERROR: Clean item metadata dataframe was not successfully created or is empty after Step 1 & 2.")
              pipeline_ok = False
         # train_merged_df is less critical *at this stage* for downstream steps like pair_generator
         # but is needed for user features relying on item metadata. Let's check it but not make it strictly fatal yet.
         if pipeline_data.get('train_merged_df') is None or pipeline_data['train_merged_df'].empty:
              print("WARNING: Merged dataframe was not successfully created or is empty after Step 1 & 2. User features requiring item metadata may be limited.")


# -----------------------------------------------------------------------------
# Step 2.5: Filter Training Data to Keep Only Items Present in Cleaned Metadata
# -----------------------------------------------------------------------------
print("\n" + "="*50 + "\n")
print("Step 2.5: Filter Training Data by Cleaned Item Metadata...")
if pipeline_ok:
    try:
        # Get the set of item IDs present in the cleaned metadata
        valid_item_ids = set(pipeline_data['clean_item_meta_df'][config.ITEM_ID_COL].unique())
        initial_train_rows = len(pipeline_data['train_df'])
        initial_train_items = pipeline_data['train_df'][config.ITEM_ID_COL].nunique()
        initial_train_users = pipeline_data['train_df'][config.USER_ID_COL].nunique()

        print(f"Initial train_df shape: {pipeline_data['train_df'].shape}")
        print(f"Total unique items in clean_item_meta_df: {len(valid_item_ids)}")

        # Filter train_df to keep only interactions with valid item IDs
        pipeline_data['train_df'] = pipeline_data['train_df'][
            pipeline_data['train_df'][config.ITEM_ID_COL].isin(valid_item_ids)
        ].reset_index(drop=True)

        filtered_train_rows = len(pipeline_data['train_df'])
        filtered_train_items = pipeline_data['train_df'][config.ITEM_ID_COL].nunique()
        filtered_train_users = pipeline_data['train_df'][config.USER_ID_COL].nunique()


        print(f"Filtered train_df shape: {pipeline_data['train_df'].shape}")
        print(f"Interactions removed: {initial_train_rows - filtered_train_rows}")
        print(f"Unique items remaining in train_df: {filtered_train_items}")
        print(f"Unique items removed from train_df: {initial_train_items - filtered_train_items}")
        print(f"Unique users remaining in train_df: {filtered_train_users}")

        # Check if any data remains
        if pipeline_data['train_df'].empty:
             print("FATAL ERROR: train_df is empty after filtering by cleaned item metadata.")
             pipeline_ok = False
        else:
             print("Step 2.5 successfully completed.")
             # Re-create the merged_df based on the filtered train_df
             # This ensures the merged_df used for user features also reflects the filtered items
             print("Re-creating train_merged_df from filtered train_df...")
             if pipeline_data.get('clean_item_meta_df') is not None and not pipeline_data['clean_item_meta_df'].empty:
                  try:
                      # Assuming clean_item_meta_df has the columns needed for the merge
                      # Explicitly list columns to avoid issues if metadata has unexpected columns
                      # Get all columns from metadata except the ID column for merging
                      meta_cols_to_merge = [col for col in pipeline_data['clean_item_meta_df'].columns if col != config.ITEM_ID_COL]
                      merge_cols = [config.ITEM_ID_COL] + meta_cols_to_merge # Ensure ID column is first

                      pipeline_data['train_merged_df'] = pd.merge(
                           pipeline_data['train_df'],
                           pipeline_data['clean_item_meta_df'][merge_cols], # Select only necessary columns
                           on=config.ITEM_ID_COL,
                           how='left' # Use left merge to keep all interactions from filtered train_df
                      )
                      print(f"Re-created train_merged_df shape: {pipeline_data['train_merged_df'].shape}")
                      # Check for NaNs specifically in columns brought *from* clean_item_meta_df
                      nan_check_cols = [col for col in meta_cols_to_merge if col in pipeline_data['train_merged_df'].columns]
                      if pipeline_data['train_merged_df'][nan_check_cols].isnull().sum().sum() > 0:
                          print("WARNING: NaNs found in train_merged_df after re-creation in columns from item metadata. Check metadata processing.")

                      # Ensure dtypes are consistent after merge
                      pipeline_data['train_merged_df'][config.USER_ID_COL] = pipeline_data['train_merged_df'][config.USER_ID_COL].astype(str)
                      pipeline_data['train_merged_df'][config.ITEM_ID_COL] = pipeline_data['train_merged_df'][config.ITEM_ID_COL].astype(str)

                  except Exception as e:
                       print(f"WARNING: Could not re-create train_merged_df after filtering train_df: {e}")
                       traceback.print_exc()
                       pipeline_data['train_merged_df'] = None # Ensure it's None if re-creation fails

             else:
                  print("WARNING: clean_item_meta_df is not available, cannot re-create train_merged_df.")
                  pipeline_data['train_merged_df'] = None # Ensure it's None


    except Exception as e:
        print(f"FATAL ERROR in Step 2.5: An unexpected error occurred during filtering: {e}")
        traceback.print_exc() # Add traceback
        pipeline_ok = False


# -----------------------------------------------------------------------------
# Step 3 (Original Step 3): Generate Training Pairs and User Features (Implemented by pair_generator.py)
# Now renumbered to Step 4
# -----------------------------------------------------------------------------
print("\n" + "="*50 + "\n")
print("Step 4: Generate Training Pairs and User Features...") # Renumbered
if pipeline_ok:
    try:
        # Import pair_generator if not already imported (done at top now)
        # import pair_generator
        pipeline_data['train_pairs_df'], \
        pipeline_data['user_features_df'], \
        pipeline_data['user_feature_numerical_cols_final'], \
        pipeline_data['user_feature_categorical_cols_final'] = pair_generator.generate_pairs_and_user_features(
            pipeline_data['train_df'], # Pass the FILTERED train_df
            pipeline_data['clean_item_meta_df'], # Pass clean metadata
            pipeline_data['train_merged_df'], # Pass the re-created train_merged_df (could contain NaNs)
            config.NEGATIVE_SAMPLE_RATIO,
            config.SEED,
            config.TRAIN_PAIRS_CSV_PATH,
            config.USER_FEATURES_CSV_PATH
        )
        print("Step 4 successfully completed.") # Renumbered

    except ValueError as e:
        print(f"FATAL ERROR in Step 4: Input data error: {e}") # Renumbered
        pipeline_ok = False
    except Exception as e:
        print(f"FATAL ERROR in Step 4: An unexpected error occurred during generation: {e}") # Renumbered
        traceback.print_exc() # Add traceback
        pipeline_ok = False

    # Explicitly check if critical outputs were produced
    if pipeline_ok:
        if pipeline_data.get('train_pairs_df') is None or pipeline_data['train_pairs_df'].empty:
             print("FATAL ERROR: train_pairs_df was not successfully created or is empty after Step 4.") # Renumbered
             pipeline_ok = False
        if pipeline_data.get('user_features_df') is None or pipeline_data['user_features_df'].empty:
             print("FATAL ERROR: user_features_df was not successfully created or is empty after Step 4.") # Renumbered
             pipeline_ok = False
        if pipeline_data.get('user_feature_numerical_cols_final') is None or not isinstance(pipeline_data['user_feature_numerical_cols_final'], list):
             print("FATAL ERROR: user_feature_numerical_cols_final list was not successfully created or is not a list after Step 4.") # Renumbered
             pipeline_ok = False
        if pipeline_data.get('user_feature_categorical_cols_final') is None or not isinstance(pipeline_data['user_feature_categorical_cols_final'], list):
             print("FATAL ERROR: user_feature_categorical_cols_final list was not successfully created or is not a list after Step 4.") # Renumbered
             pipeline_ok = False

        # Check if the number of users in user_features_df matches the number of unique users in the *filtered* train_df
        if pipeline_ok and pipeline_data.get('train_df') is not None:
             num_users_train = pipeline_data['train_df'][config.USER_ID_COL].nunique()
             num_users_features = len(pipeline_data['user_features_df'])
             if num_users_train != num_users_features:
                  print(f"WARNING: Number of users in filtered train_df ({num_users_train}) does not match users in user_features_df ({num_users_features}). This could indicate users were dropped if they had no interactions remaining after item filtering.")


# -----------------------------------------------------------------------------
# Step 4 (Original Step 4): Process Item Features (Implemented by item_processor.py)
# Now renumbered to Step 5
# -----------------------------------------------------------------------------
print("\n" + "="*50 + "\n")
print("Step 5: Process Item Features...") # Renumbered
if pipeline_ok:
    try:
        # Import item_processor if not already imported (done at top now)
        # import item_processor
        (pipeline_data['X_items_reduced'],
         pipeline_data['item_id_to_index'], # This is created here!
         pipeline_data['index_to_item_id'],
         pipeline_data['item_id_index_map_df'],
         pipeline_data['item_scaler'],
         pipeline_data['item_encoder'],
         pipeline_data['item_numerical_cols_for_scaling'],
         pipeline_data['item_categorical_cols_for_encoding'],
         pipeline_data['item_binary_cols'],
         pipeline_data['item_original_numerical_cols'],
         pipeline_data['item_original_nominal_categorical_cols'],
         pipeline_data['item_details_cols']) = item_processor.process_item_features(
            pipeline_data['clean_item_meta_df'], # Item processing still uses the CLEANED metadata
            config.ITEM_SVD_N_COMPONENTS,
            config.SEED,
            config.X_ITEMS_REDUCED_NPY_PATH,
            config.ITEM_ID_TO_INDEX_JSON_PATH,
            config.INDEX_TO_ITEM_ID_JSON_PATH,
            config.ITEM_SCALER_PKL_PATH,
            config.ITEM_ENCODER_PKL_PATH,
            config.ITEM_NUM_COLS_JSON_PATH,
            config.ITEM_CAT_COLS_JSON_PATH,
            config.ITEM_BINARY_COLS_JSON_PATH,
            config.ITEM_ORIGINAL_NUM_COLS_JSON_PATH,
            config.ITEM_ORIGINAL_NOM_CAT_COLS_JSON_PATH,
            config.ITEM_DETAILS_COLS_JSON_PATH
        )
        print("Step 5 successfully completed.") # Renumbered

    except ValueError as e:
        print(f"FATAL ERROR in Step 5: Input data error: {e}") # Renumbered
        pipeline_ok = False
    except Exception as e:
        print(f"FATAL ERROR in Step 5: An unexpected error occurred during processing: {e}") # Renumbered
        traceback.print_exc() # Add traceback
        pipeline_ok = False

    # Explicitly check if essential outputs were produced
    if pipeline_ok:
        item_meta_has_rows = pipeline_data.get('clean_item_meta_df') is not None and not pipeline_data['clean_item_meta_df'].empty
        num_rows_expected = len(pipeline_data['clean_item_meta_df']) if item_meta_has_rows else 0

        if pipeline_data.get('X_items_reduced') is None or not hasattr(pipeline_data['X_items_reduced'], 'shape'):
             print("FATAL ERROR: X_items_reduced is None or invalid after Step 5.") # Renumbered
             pipeline_ok = False
        elif pipeline_data['X_items_reduced'].shape[0] != num_rows_expected:
             print(f"FATAL ERROR: X_items_reduced row count ({pipeline_data['X_items_reduced'].shape[0]}) does not match expected ({num_rows_expected}) after Step 5.") # Renumbered
             pipeline_ok = False

        if pipeline_data.get('item_id_to_index') is None or not isinstance(pipeline_data['item_id_to_index'], dict):
             print("FATAL ERROR: item_id_to_index is None or not a dict after Step 5.") # Renumbered
             pipeline_ok = False
        elif item_meta_has_rows and not pipeline_data['item_id_to_index']: # If input had rows, mapping must not be empty
             print("FATAL ERROR: item_id_to_index is empty despite input metadata having rows after Step 5.") # Renumbered
             pipeline_ok = False

        if pipeline_data.get('item_id_index_map_df') is None or not isinstance(pipeline_data['item_id_index_map_df'], pd.DataFrame):
             print("FATAL ERROR: item_id_index_map_df is None or not a DataFrame after Step 5.") # Renumbered
             pipeline_ok = False
        elif pipeline_data['item_id_index_map_df'].shape[0] != num_rows_expected:
             print(f"FATAL ERROR: item_id_index_map_df row count ({pipeline_data['item_id_index_map_df'].shape[0]}) does not match expected ({num_rows_expected}) after Step 5.") # Renumbered
             pipeline_ok = False

        # Column lists must be lists (essential for Step 8)
        col_lists_ok = (isinstance(pipeline_data.get('item_numerical_cols_for_scaling'), list) and
                        isinstance(pipeline_data.get('item_categorical_cols_for_encoding'), list) and
                        isinstance(pipeline_data.get('item_binary_cols'), list))
        if not col_lists_ok:
             print("FATAL ERROR: Essential item column lists were not successfully returned as lists after Step 5.") # Renumbered
             pipeline_ok = False

        if pipeline_ok:
             print("Step 5 validation passed.") # Renumbered
        else:
             print("Step 5 validation failed.") # Renumbered


# -----------------------------------------------------------------------------
# NEW Step: Create Sparse User-Item Interaction Matrix
# Inserted after Step 5 (Item Processing) - Now this is Step 6
# -----------------------------------------------------------------------------
print("\n" + "="*50 + "\n")
print("Step 6: Create Sparse User-Item Interaction Matrix...") # Renumbered
if pipeline_ok:
    try:
        # Require filtered train_df and item_id_to_index from previous steps
        if pipeline_data.get('train_df') is None or pipeline_data['train_df'].empty:
             raise ValueError("Filtered train_df is not available or empty.")
        # *** item_id_to_index is now available from Step 5 ***
        if pipeline_data.get('item_id_to_index') is None:
             raise ValueError("item_id_to_index map is not available from Step 5.") # Updated message


        train_df_filtered = pipeline_data['train_df']
        item_id_to_index = pipeline_data['item_id_to_index'] # This is now correctly populated

        # 1. Create User ID to Index Mapping (only for users present in the filtered train data)
        unique_users = train_df_filtered[config.USER_ID_COL].unique()
        user_id_to_index_matrix = {user_id: i for i, user_id in enumerate(unique_users)} # Use distinct name if needed elsewhere
        n_users = len(unique_users)
        # Use the total number of processable items *from the item processing step*
        # This ensures the matrix dimension matches the item feature matrix
        n_items = len(pipeline_data.get('item_id_to_index', {})) # Use item_id_to_index from Step 5


        print(f"Number of unique users in filtered train data for matrix: {n_users}")
        print(f"Number of processable items for matrix: {n_items}")

        # 2. Prepare data for COO matrix
        # Ensure all user IDs in train_df_filtered are in the user_id_to_index_matrix map
        # (This should be true since the map is built from unique users in the filtered df)
        user_indices = train_df_filtered[config.USER_ID_COL].map(user_id_to_index_matrix).values
        # Ensure all item IDs in train_df_filtered are in the item_id_to_index map
        # (This should be true due to the filtering in Step 2.5)
        item_indices = train_df_filtered[config.ITEM_ID_COL].map(item_id_to_index).values

        # Use 1 for implicit feedback (interaction occurred)
        data = np.ones(len(train_df_filtered), dtype=int)

        # 3. Create the sparse COO matrix
        interaction_matrix_coo = sp.coo_matrix( # Use sp alias
            (data, (user_indices, item_indices)),
            shape=(n_users, n_items) # Shape is (num_users, num_items)
        )

        # Convert to CSR for potential performance benefits depending on use case
        # CSR is generally better for matrix-vector multiplication and slicing
        interaction_matrix_csr = interaction_matrix_coo.tocsr()
        print(f"Created sparse interaction matrix (CSR format) with shape: {interaction_matrix_csr.shape}")
        print(f"Non-zero entries (interactions): {interaction_matrix_csr.nnz}")
        # Check if the number of non-zero entries matches the number of rows in filtered train_df
        if interaction_matrix_csr.nnz != len(train_df_filtered):
             print(f"WARNING: Non-zero entries in matrix ({interaction_matrix_csr.nnz}) do not match interactions in filtered train_df ({len(train_df_filtered)}). Check for duplicate user-item pairs in filtered train_df.")
        print(f"Sparsity: {1 - interaction_matrix_csr.nnz / (interaction_matrix_csr.shape[0] * interaction_matrix_csr.shape[1]):.4f}")

        # Store the matrix and mapping in the pipeline_data dictionary
        pipeline_data['interaction_matrix_csr'] = interaction_matrix_csr
        pipeline_data['user_id_to_index_matrix'] = user_id_to_index_matrix # Store the map used for the matrix
        # item_id_to_index is already in pipeline_data from Step 5


        # 4. Save the matrix and mapping
        # Use the defined paths from Step 0/config
        sp.save_npz(INTERACTION_MATRIX_PATH, interaction_matrix_csr) # Use sp.save_npz
        with open(USER_ID_TO_INDEX_JSON_PATH_MATRIX, 'w') as f:
            json.dump(user_id_to_index_matrix, f) # Save user map as JSON

        print(f"Interaction matrix saved to: {INTERACTION_MATRIX_PATH}")
        print(f"User ID to index map for matrix saved to: {USER_ID_TO_INDEX_JSON_PATH_MATRIX}")

        print("Step 6 successfully completed.") # Renumbered

    except ValueError as e:
        print(f"FATAL ERROR in Step 6: Data validation error: {e}") # Renumbered
        traceback.print_exc()
        pipeline_ok = False
    except Exception as e:
        print(f"FATAL ERROR in Step 6: An unexpected error occurred during matrix creation: {e}") # Renumbered
        traceback.print_exc() # Add traceback
        pipeline_ok = False

    # Validation (optional but good)
    if pipeline_ok:
        if pipeline_data.get('interaction_matrix_csr') is None or not isinstance(pipeline_data['interaction_matrix_csr'], sp.csr_matrix): # Use sp alias
             print("FATAL ERROR: Interaction matrix was not successfully created or is not CSR format after Step 6.") # Renumbered
             pipeline_ok = False
        if pipeline_data.get('user_id_to_index_matrix') is None or not isinstance(pipeline_data['user_id_to_index_matrix'], dict):
             print("FATAL ERROR: user_id_to_index_matrix map was not successfully created or is not a dict after Step 6.") # Renumbered
             pipeline_ok = False
        if pipeline_data.get('item_id_to_index') is None or not isinstance(pipeline_data['item_id_to_index'], dict):
             print("FATAL ERROR: item_id_to_index map was lost or is invalid after Step 6 (should be from Step 5).") # Check if map survived
             pipeline_ok = False


    if pipeline_ok:
         print("Step 6 validation passed.") # Renumbered
    else:
         print("Step 6 validation failed.") # Renumbered


# -----------------------------------------------------------------------------
# Step 5 (Original Step 5): Process User Features (Fit Transformers) (Implemented by user_processor.py)
# Now renumbered to Step 7
# -----------------------------------------------------------------------------
print("\n" + "="*50 + "\n")
print("Step 7: Process User Features (Fit Transformers)...") # Renumbered
if pipeline_ok:
    try:
        # Import user_processor if not already imported (done at top now)
        # import user_processor
        (pipeline_data['user_numerical_scaler'],
         pipeline_data['user_categorical_encoder'],
         pipeline_data['user_categorical_svd'],
         pipeline_data['user_num_cols_present_in_df'], # These are the lists *actually used* for transformation
         pipeline_data['user_cat_cols_present_in_df']) = user_processor.process_user_features(
            pipeline_data['user_features_df'], # Requires user_features_df from Step 4
            pipeline_data['user_feature_numerical_cols_final'], # Requires lists from Step 4
            pipeline_data['user_feature_categorical_cols_final'], # Requires lists from Step 4
            config.USER_CAT_SVD_N_COMPONENTS,
            config.SEED,
            config.USER_NUMERICAL_SCALER_PKL_PATH,
            config.USER_CATEGORICAL_ENCODER_PKL_PATH,
            config.USER_CATEGORICAL_SVD_PKL_PATH,
            config.USER_NUM_COLS_FOR_SCALING_JSON_PATH,
            config.USER_CAT_COLS_FOR_ENCODING_JSON_PATH
        )
        print("Step 7 successfully completed.") # Renumbered

    except ValueError as e:
        print(f"FATAL ERROR in Step 7: Input data error: {e}") # Renumbered
        pipeline_ok = False
    except Exception as e:
        print(f"FATAL ERROR in Step 7: An unexpected error occurred during processing: {e}") # Renumbered
        traceback.print_exc() # Add traceback
        pipeline_ok = False

    # Explicitly check if essential outputs were produced
    if pipeline_ok:
        # Column lists must be lists
        if pipeline_data.get('user_num_cols_present_in_df') is None or not isinstance(pipeline_data['user_num_cols_present_in_df'], list):
             print("FATAL ERROR: user_num_cols_present_in_df list was not successfully returned as a list after Step 7.") # Renumbered
             pipeline_ok = False
        if pipeline_data.get('user_cat_cols_present_in_df') is None or not isinstance(pipeline_data['user_cat_cols_present_in_df'], list):
             print("FATAL ERROR: user_cat_cols_present_in_df list was not successfully returned as a list after Step 7.") # Renumbered
             pipeline_ok = False

        # Check transformers validity *IF* their corresponding columns were present for fitting
        if pipeline_ok:
             is_scaler_needed = len(pipeline_data.get('user_num_cols_present_in_df', [])) > 0
             is_encoder_needed = len(pipeline_data.get('user_cat_cols_present_in_df', [])) > 0
             is_svd_needed = is_encoder_needed and config.USER_CAT_SVD_N_COMPONENTS > 0

             if is_scaler_needed and (pipeline_data.get('user_numerical_scaler') is None or not hasattr(pipeline_data['user_numerical_scaler'], 'mean_')):
                  print("FATAL ERROR: User numerical scaler is None or invalid despite numerical columns being present after Step 7.") # Renumbered
                  pipeline_ok = False

             if is_encoder_needed and (pipeline_data.get('user_categorical_encoder') is None or not hasattr(pipeline_data['user_categorical_encoder'], 'categories_')):
                  print("FATAL ERROR: User categorical encoder is None or invalid despite categorical columns being present after Step 7.") # Renumbered
                  pipeline_ok = False
             elif is_svd_needed and (pipeline_data.get('user_categorical_encoder') is not None and hasattr(pipeline_data['user_categorical_encoder'], 'categories_')):
                  if (pipeline_data.get('user_categorical_svd') is None or not (hasattr(pipeline_data['user_categorical_svd'], 'components_') and pipeline_data['user_categorical_svd'].n_components > 0)):
                       print("FATAL ERROR: User categorical SVD is None or invalid despite categorical columns and encoder being valid and SVD expected > 0 components after Step 7.") # Renumbered
                       pipeline_ok = False

    if pipeline_ok:
         print("Step 7 validation passed. Fitted user transformers and column lists are available.") # Renumbered
    else:
         print("Step 7 validation failed.") # Renumbered


# -----------------------------------------------------------------------------
# Step 6 (Original Step 6): Prepare Hybrid Training Matrix (Implemented by hybrid_matrix_assembler.py)
# Now renumbered to Step 8
# -----------------------------------------------------------------------------
print("\n" + "="*50 + "\n")
print("Step 8: Prepare Hybrid Training Matrix...") # Renumbered
if pipeline_ok:
    try:
        # Import hybrid_matrix_assembler just before its first use (or keep at top)
        # import hybrid_matrix_assembler # Keep import at top
        pipeline_data['X_train_hybrid'], pipeline_data['y_train'] = hybrid_matrix_assembler.assemble_hybrid_matrix(
            pipeline_data['train_pairs_df'], # Requires train_pairs_df from Step 4
            pipeline_data['user_features_df'], # Requires user_features_df from Step 4
            pipeline_data['user_numerical_scaler'], # Requires scaler from Step 7
            pipeline_data['user_categorical_encoder'], # Requires encoder from Step 7
            pipeline_data['user_categorical_svd'], # Requires SVD from Step 7 (or None)
            pipeline_data['user_num_cols_present_in_df'], # Requires list from Step 7
            pipeline_data['user_cat_cols_present_in_df'], # Requires list from Step 7
            pipeline_data['X_items_reduced'], # Requires item features from Step 5
            pipeline_data['item_id_index_map_df'], # Requires item map df from Step 5
            config.STACKING_BATCH_SIZE,
            config.BATCH_TEMP_DIR,
            config.X_TRAIN_HYBRID_PATH,
            config.Y_TRAIN_PATH
        )
        print("Step 8 successfully completed.") # Renumbered

    except ValueError as e:
        print(f"FATAL ERROR in Step 8: Input data error: {e}") # Renumbered
        traceback.print_exc()
        pipeline_ok = False
    except Exception as e:
        print(f"FATAL ERROR in Step 8: An unexpected error occurred during assembly: {e}") # Renumbered
        traceback.print_exc() # Add traceback
        pipeline_ok = False

    # Explicitly check if final outputs were produced and are valid
    if pipeline_ok:
        if pipeline_data.get('X_train_hybrid') is None or not hasattr(pipeline_data['X_train_hybrid'], 'shape'):
             print("FATAL ERROR: X_train_hybrid is None or invalid after Step 8.") # Renumbered
             pipeline_ok = False
        # Allow numpy array for y_train as generated by the assembler
        if pipeline_data.get('y_train') is None or not isinstance(pipeline_data['y_train'], (pd.Series, pd.DataFrame, np.ndarray)):
             print("FATAL ERROR: y_train is None or not a pandas Series/DataFrame/numpy array after Step 8.") # Renumbered
             pipeline_ok = False

        # Check shape consistency only if both X and y are valid objects
        if pipeline_ok and pipeline_data.get('X_train_hybrid') is not None and pipeline_data.get('y_train') is not None:
            if pipeline_data['X_train_hybrid'].shape[0] != len(pipeline_data['y_train']):
                 print(f"FATAL ERROR: X_train_hybrid row count ({pipeline_data['X_train_hybrid'].shape[0]}) does not match y_train length ({len(pipeline_data['y_train'])}) after Step 8.") # Renumbered
                 pipeline_ok = False

    if pipeline_ok:
         print("Step 8 validation passed. Final hybrid training data is ready.") # Renumbered
    else:
         print("Step 8 validation failed.") # Renumbered


# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------
print("\n" + "="*50 + "\n")
print("--- Script Execution Summary ---")

end_pipeline_time = time.time()
print(f"Total execution time: {end_pipeline_time - start_pipeline_time:.2f} seconds.")

if pipeline_ok:
    print("\nPipeline finished successfully.")
    # Report final outputs
    if pipeline_data.get('X_train_hybrid') is not None:
         print(f"Final X_train_hybrid shape: {pipeline_data['X_train_hybrid'].shape}")
         print(f"X_train_hybrid format: {type(pipeline_data['X_train_hybrid'])}")
    else:
         print("Final X_train_hybrid is not available.")

    if pipeline_data.get('y_train') is not None:
         print(f"Final y_train shape: {pipeline_data['y_train'].shape}")
         print(f"y_train type: {type(pipeline_data['y_train'])}")
    else:
         print("Final y_train is not available.")

    if pipeline_data.get('interaction_matrix_csr') is not None:
         print(f"Generated Interaction Matrix shape: {pipeline_data['interaction_matrix_csr'].shape}")
         print(f"Interaction Matrix format: {type(pipeline_data['interaction_matrix_csr'])}")
         print(f"Interaction Matrix saved to: {INTERACTION_MATRIX_PATH}")
         print(f"User index map for matrix saved to: {USER_ID_TO_INDEX_JSON_PATH_MATRIX}")
    else:
         print("Interaction Matrix was not generated.")


else:
    print("\nPipeline failed during execution. Check the logs above for the first FATAL ERROR.")