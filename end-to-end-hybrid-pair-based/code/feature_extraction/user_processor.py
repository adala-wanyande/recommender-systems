# user_processor.py

import pandas as pd
import numpy as np
import os
import time
import pickle
import json
import scipy.sparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import shutil

# Import configuration from config.py
import config

def process_user_features(
    user_features_df: pd.DataFrame,
    user_feature_numerical_cols_final: list,
    user_feature_categorical_cols_final: list,
    user_cat_svd_n_components: int,
    seed: int,
    user_numerical_scaler_pkl_path: str,
    user_categorical_encoder_pkl_path: str,
    user_categorical_svd_pkl_path: str,
    user_num_cols_for_scaling_json_path: str, # Path to save/load list of numerical cols used
    user_cat_cols_for_encoding_json_path: str # Path to save/load list of categorical cols used
) -> tuple[StandardScaler | None, OneHotEncoder | None, TruncatedSVD | None, list, list]:
    """
    Processes user features by fitting scalers, encoders, and SVD.
    Loads saved transformers and column lists if available, otherwise fits and saves.

    Args:
        user_features_df (pd.DataFrame): DataFrame containing raw user features.
        user_feature_numerical_cols_final (list): List of column names identified as numerical user features.
        user_feature_categorical_cols_final (list): List of column names identified as categorical user features.
        user_cat_svd_n_components (int): Number of components for user categorical feature SVD.
        seed (int): Random seed for reproducibility.
        user_numerical_scaler_pkl_path (str): Path to save/load the fitted StandardScaler.
        user_categorical_encoder_pkl_path (str): Path to save/load the fitted OneHotEncoder.
        user_categorical_svd_pkl_path (str): Path to save/load the fitted TruncatedSVD.
        user_num_cols_for_scaling_json_path (str): Path to save/load the list of numerical columns used for scaling.
        user_cat_cols_for_encoding_json_path (str): Path to save/load the list of categorical columns used for encoding.

    Returns:
        tuple: (user_numerical_scaler, user_categorical_encoder, user_categorical_svd,
                user_num_cols_present_in_df, user_cat_cols_present_in_df)
                Returns None for transformers if not fitted/loaded, and empty lists if no cols of that type found/used.

    Raises:
        ValueError: If input user_features_df is None or empty.
        Exception: For other errors during processing or loading.
    """
    print("\n--- Step 5: Process User Features (Fit Transformers) ---")

    # --- Validate Inputs ---
    if user_features_df is None or user_features_df.empty:
        raise ValueError("Input user_features_df is None or empty.")
    if user_feature_numerical_cols_final is None or not isinstance(user_feature_numerical_cols_final, list):
         raise ValueError("Input user_feature_numerical_cols_final must be a list (can be empty).")
    if user_feature_categorical_cols_final is None or not isinstance(user_feature_categorical_cols_final, list):
         raise ValueError("Input user_feature_categorical_cols_final must be a list (can be empty).")


    # --- Global variables to be populated (will be returned) ---
    user_numerical_scaler = None
    user_categorical_encoder = None
    user_categorical_svd = None
    user_num_cols_present_in_df = [] # Initialize as empty lists
    user_cat_cols_present_in_df = [] # Initialize as empty lists


    perform_user_transformer_fitting = True # Assume fitting is needed unless loading succeeds

    # --- Check for and Load Saved User Transformers ---
    print("\nChecking for and loading saved user feature transformers...")
    # Check for the existence of the required files
    required_user_transformer_files_for_loading = [
        user_numerical_scaler_pkl_path, # May not exist if no numerical cols
        user_categorical_encoder_pkl_path, # May not exist if no categorical cols
        user_categorical_svd_pkl_path, # May not exist if no categorical cols or SVD components <= 0
        user_num_cols_for_scaling_json_path, # Should always exist if processing ran
        user_cat_cols_for_encoding_json_path # Should always exist if processing ran
    ]

    # Check existence of the files that *should* always be saved (column lists)
    core_column_list_files_exist = os.path.exists(user_num_cols_for_scaling_json_path) and os.path.exists(user_cat_cols_for_encoding_json_path)

    if core_column_list_files_exist:
         print("Detected saved user feature column lists. Attempting to load transformers...")
         try:
              # Load column lists first
              with open(user_num_cols_for_scaling_json_path, 'r') as f: user_num_cols_present_in_df = json.load(f)
              with open(user_cat_cols_for_encoding_json_path, 'r') as f: user_cat_cols_present_in_df = json.load(f)
              print("Loaded user feature column lists used for transformation.")

              # Check column lists are actual lists
              if not (isinstance(user_num_cols_present_in_df, list) and isinstance(user_cat_cols_present_in_df, list)):
                  raise ValueError("Loaded user column lists are not lists.")

              # Determine if transformers were expected based on loaded column lists
              is_scaler_expected = len(user_num_cols_present_in_df) > 0
              is_encoder_expected = len(user_cat_cols_present_in_df) > 0
              # SVD is expected if encoder was expected and SVD components > 0
              is_svd_expected = is_encoder_expected and user_cat_svd_n_components > 0

              # Load Transformers (pickle) - only if expected AND file exists
              if is_scaler_expected and os.path.exists(user_numerical_scaler_pkl_path):
                   with open(user_numerical_scaler_pkl_path, 'rb') as f: user_numerical_scaler = pickle.load(f)
                   print(f"Loaded {user_numerical_scaler_pkl_path}")
              elif is_scaler_expected:
                   print(f"Warning: {user_numerical_scaler_pkl_path} expected but not found. Scaler will be None.")
                   user_numerical_scaler = None # Ensure None if file missing but expected
              else:
                   print("Scaler not expected (no numerical columns identified).")
                   user_numerical_scaler = None # Ensure None if not expected


              if is_encoder_expected and os.path.exists(user_categorical_encoder_pkl_path):
                   with open(user_categorical_encoder_pkl_path, 'rb') as f: user_categorical_encoder = pickle.load(f)
                   print(f"Loaded {user_categorical_encoder_pkl_path}")
              elif is_encoder_expected:
                   print(f"Warning: {user_categorical_encoder_pkl_path} expected but not found. Encoder will be None.")
                   user_categorical_encoder = None # Ensure None if file missing but expected
              else:
                   print("Encoder not expected (no categorical columns identified).")
                   user_categorical_encoder = None # Ensure None if not expected

              if is_svd_expected and os.path.exists(user_categorical_svd_pkl_path):
                   with open(user_categorical_svd_pkl_path, 'rb') as f: user_categorical_svd = pickle.load(f)
                   print(f"Loaded {user_categorical_svd_pkl_path}")
              elif is_svd_expected:
                   print(f"Warning: {user_categorical_svd_pkl_path} expected but not found. SVD will be None.")
                   user_categorical_svd = None # Ensure None if file missing but expected
              else:
                   print("SVD not expected (no categorical columns or SVD components <= 0).")
                   user_categorical_svd = None # Ensure None if not expected


              # Validate loaded transformers (check if fitted and correct type)
              is_user_num_scaler_valid = not is_scaler_expected or (user_numerical_scaler is not None and isinstance(user_numerical_scaler, StandardScaler) and hasattr(user_numerical_scaler, 'mean_'))
              is_user_cat_encoder_valid = not is_encoder_expected or (user_categorical_encoder is not None and isinstance(user_categorical_encoder, OneHotEncoder) and hasattr(user_categorical_encoder, 'categories_'))
              # SVD valid if expected AND loaded AND is correct type AND fitted AND has >0 components
              is_user_cat_svd_valid = not is_svd_expected or (user_categorical_svd is not None and isinstance(user_categorical_svd, TruncatedSVD) and hasattr(user_categorical_svd, 'components_') and user_categorical_svd.n_components > 0)


              # Also validate that the loaded column lists match the columns actually present in user_features_df
              # This check is important to ensure consistency between saved state and current input data
              cols_in_df_num = [col for col in user_feature_numerical_cols_final if col in user_features_df.columns]
              cols_in_df_cat = [col for col in user_feature_categorical_cols_final if col in user_features_df.columns]

              is_loaded_cols_num_valid = (user_num_cols_present_in_df == cols_in_df_num)
              is_loaded_cols_cat_valid = (user_cat_cols_present_in_df == cols_in_df_cat)

              # Overall validation check
              all_loaded_user_transformers_valid = (
                  is_user_num_scaler_valid and
                  is_user_cat_encoder_valid and
                  is_user_cat_svd_valid and
                  is_loaded_cols_num_valid and
                  is_loaded_cols_cat_valid
              )


              if all_loaded_user_transformers_valid:
                   print("Loaded and validated saved user feature transformers and column lists.")
                   perform_user_transformer_fitting = False # Data loaded successfully, skip regeneration
              else:
                   print("Warning: Saved user transformers or column lists found but failed validation. Proceeding with fitting.")
                   # Ensure return variables are reset to None/empty lists if validation fails
                   user_numerical_scaler = None; user_categorical_encoder = None; user_categorical_svd = None
                   user_num_cols_present_in_df = []; user_cat_cols_present_in_df = [] # Reset lists
                   perform_user_transformer_fitting = True # Force fitting


         except Exception as e:
              print(f"Error loading saved user transformers or column lists: {e}. Proceeding with fitting.")
              # Ensure return variables are reset to None/empty lists if loading fails
              user_numerical_scaler = None; user_categorical_encoder = None; user_categorical_svd = None
              user_num_cols_present_in_df = []; user_cat_cols_present_in_df = [] # Reset lists
              perform_user_transformer_fitting = True # Force fitting
    else:
        print("Saved user feature column lists not found. Proceeding with fitting.")
        perform_user_transformer_fitting = True # Files don't exist, need to regenerate


    # --- Fit User Scalers/Encoders/SVD if perform_user_transformer_fitting is True ---
    if perform_user_transformer_fitting:
        print("\nFitting user feature scalers/encoders/SVD on user_features_df...")

        # Use the input lists of user feature columns, filtering by presence in user_features_df
        user_num_cols_present_in_df = [col for col in user_feature_numerical_cols_final if col in user_features_df.columns]
        user_cat_cols_present_in_df = [col for col in user_feature_categorical_cols_final if col in user_features_df.columns]
        print(f"Identified {len(user_num_cols_present_in_df)} numerical and {len(user_cat_cols_present_in_df)} categorical user feature columns present in user_features_df to fit transformers on.")


        # --- 1. Fit Numerical User Scaler ---
        print("Fitting StandardScaler...")
        user_numerical_scaler = None # Initialize before fit
        if user_num_cols_present_in_df:
             user_features_df_num = user_features_df[user_num_cols_present_in_df].astype(float) # Ensure numerical copy
             # Impute NaNs in numerical columns before fitting, though pair_generator should handle this
             if user_features_df_num.isnull().sum().sum() > 0:
                  print("Warning: NaNs found in numerical columns during fitting preparation. Imputing with median.")
                  for col in user_num_cols_present_in_df:
                       if user_features_df_num[col].isnull().sum() > 0:
                            median_val = user_features_df_num[col].median() if not user_features_df_num[col].isnull().all() else 0
                            user_features_df_num[col] = user_features_df_num[col].fillna(median_val)

             user_numerical_scaler = StandardScaler()
             user_numerical_scaler.fit(user_features_df_num)
             print(f"Fitted StandardScaler on {len(user_num_cols_present_in_df)} columns.")
        else:
             print("No numerical columns present for scaling. Skipping scaling fitting.")
             # user_numerical_scaler remains None


        # --- 2. Fit Categorical User Encoder ---
        print("Fitting OneHotEncoder...")
        user_categorical_encoder = None # Initialize to None before try block
        if user_cat_cols_present_in_df:
            user_features_df_cat = user_features_df[user_cat_cols_present_in_df].astype(str) # Ensure string copy
            # Fill NaNs before encoding, though pair_generator should handle this
            if user_features_df_cat.isnull().sum().sum() > 0:
                 print("Warning: NaNs found in categorical columns during fitting preparation. Imputing with 'Missing'.")
                 user_features_df_cat = user_features_df_cat.fillna('Missing') # Ensure string type after fillna

            try:
                # Attempt to use sparse_output if available (scikit-learn >= 1.2)
                try:
                    user_categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
                    print("Initialized OneHotEncoder for user features with sparse_output=True.")
                except TypeError: # Fallback to old sparse argument
                    user_categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)
                    print("Initialized OneHotEncoder for user features with sparse=True (deprecated).")

                user_categorical_encoder.fit(user_features_df_cat)
                # Print output dimension robustly
                user_encoder_output_dim = getattr(user_categorical_encoder, 'n_features_out_', None)
                if user_encoder_output_dim is None: user_encoder_output_dim = getattr(user_categorical_encoder, 'n_features_', 'Unknown')
                if user_encoder_output_dim == 'Unknown' and hasattr(user_categorical_encoder, 'categories_'):
                     user_encoder_output_dim = sum(len(cats) for cats in user_categorical_encoder.categories_)

                print(f"Fitted OneHotEncoder on {len(user_cat_cols_present_in_df)} columns. Output dim: {user_encoder_output_dim}")

            except Exception as e:
               print(f"FATAL ERROR fitting User OneHotEncoder: {e}. Skipping encoding and SVD for user categorical features.")
               user_categorical_encoder = None # Ensure None on error
        else:
             print("No categorical columns present for encoding. Skipping encoding fitting.")
             # user_categorical_encoder remains None


        # --- 3. Fit User Categorical SVD ---
        user_categorical_svd = None # Initialize to None before try block
        # Only attempt fitting if the categorical encoder was successfully fitted AND produced features AND SVD components > 0
        is_encoder_fitted_and_has_output = (user_categorical_encoder is not None and
                                            hasattr(user_categorical_encoder, 'categories_') and
                                            sum(len(cats) for cats in user_categorical_encoder.categories_) > 0)
        n_components_actual = max(0, user_cat_svd_n_components) # Treat <0 config as 0

        if is_encoder_fitted_and_has_output and n_components_actual > 0:
             print(f"Fitting TruncatedSVD for user categorical features with {n_components_actual} components...")
             try:
                  # Use the same data used for encoder fitting (ensure string and fillna)
                  user_features_df_for_svd = user_features_df[user_cat_cols_present_in_df].astype(str).fillna('Missing')
                  user_cat_features_encoded_for_svd_fit = user_categorical_encoder.transform(user_features_df_for_svd) # Transform using fitted encoder

                  # Ensure encoded matrix is not empty before fitting SVD
                  if user_cat_features_encoded_for_svd_fit.shape[1] == 0:
                       print("Warning: User categorical encoded features have 0 columns after encoding. Skipping SVD fitting.")
                       user_categorical_svd = None
                  else:
                       # Ensure n_components is not greater than the number of encoded features or samples
                       n_components_fit = min(n_components_actual, user_cat_features_encoded_for_svd_fit.shape[0], user_cat_features_encoded_for_svd_fit.shape[1])

                       if n_components_fit <= 0:
                            print("Warning: User categorical SVD components is 0 or less based on encoded features. Skipping SVD fitting.")
                            user_categorical_svd = None
                       else:
                            user_categorical_svd = TruncatedSVD(n_components=n_components_fit, random_state=seed)
                            user_categorical_svd.fit(user_cat_features_encoded_for_svd_fit) # Fit on sparse matrix
                            print(f"Fitted User Categorical SVD with {user_categorical_svd.n_components} components. Explained variance: {user_categorical_svd.explained_variance_ratio_.sum():.4f}")

             except Exception as e:
                  print(f"FATAL ERROR fitting User Categorical SVD: {e}. Skipping user categorical SVD features.")
                  user_categorical_svd = None # Ensure None on error

        else:
             print("Skipping User Categorical SVD fitting (encoder not available/fitted, or fitted on 0 features, or SVD components <= 0).")
             # user_categorical_svd remains None


        # --- Validate Fitted User Transformers and Save if successful ---
        user_transformers_fitted_successfully = True # Assume success unless proven otherwise

        # Check if transformers are valid IF their corresponding columns were identified for fitting
        if len(user_num_cols_present_in_df) > 0 and (user_numerical_scaler is None or not hasattr(user_numerical_scaler, 'mean_')):
             print("FATAL ERROR: User numerical scaler failed validation after fitting attempt despite numerical columns being found.")
             user_transformers_fitted_successfully = False

        if len(user_cat_cols_present_in_df) > 0 and (user_categorical_encoder is None or not hasattr(user_categorical_encoder, 'categories_')):
             print("FATAL ERROR: User categorical encoder failed validation after fitting attempt despite categorical columns being found.")
             user_transformers_fitted_successfully = False
        # Check SVD *only* if categorical cols were present AND encoder was valid AND encoder produced features AND SVD was expected > 0 components
        elif len(user_cat_cols_present_in_df) > 0 and is_encoder_fitted_and_has_output and n_components_actual > 0:
             if user_categorical_svd is None or not (hasattr(user_categorical_svd, 'components_') and user_categorical_svd.n_components > 0):
                  print("FATAL ERROR: User categorical SVD failed validation after fitting attempt.")
                  user_transformers_fitted_successfully = False


        if user_transformers_fitted_successfully:
             print("\nUser feature transformers fitted and validated successfully.")
             # Save the fitted transformers and column lists
             print("Saving fitted user transformers and column lists...")
             try:
                  # Save Transformers (ensure they are not None)
                  if user_numerical_scaler is not None:
                       with open(user_numerical_scaler_pkl_path, 'wb') as f: pickle.dump(user_numerical_scaler, f)
                       print(f"  Saved {user_numerical_scaler_pkl_path}")
                  else: print(f"  Skipping save: {user_numerical_scaler_pkl_path} is None.") # Explicitly state when skipped

                  if user_categorical_encoder is not None:
                       with open(user_categorical_encoder_pkl_path, 'wb') as f: pickle.dump(user_categorical_encoder, f)
                       print(f"  Saved {user_categorical_encoder_pkl_path}")
                  else: print(f"  Skipping save: {user_categorical_encoder_pkl_path} is None.")

                  if user_categorical_svd is not None: # Only save SVD if it was successfully fitted (>0 components)
                       with open(user_categorical_svd_pkl_path, 'wb') as f: pickle.dump(user_categorical_svd, f)
                       print(f"  Saved {user_categorical_svd_pkl_path}")
                  else: print(f"  Skipping save: {user_categorical_svd_pkl_path} is None.")


                  # Save Column Lists (save even if empty lists)
                  with open(user_num_cols_for_scaling_json_path, 'w') as f: json.dump(user_num_cols_present_in_df, f)
                  print(f"  Saved {user_num_cols_for_scaling_json_path}")
                  with open(user_cat_cols_for_encoding_json_path, 'w') as f: json.dump(user_cat_cols_present_in_df, f)
                  print(f"  Saved {user_cat_cols_for_encoding_json_path}")

                  print("User transformers and column lists saved.")
                  # Set flag to False as generation was successful (even if save failed)
                  perform_user_transformer_fitting = False
             except Exception as e:
                  print(f"Warning: Error saving user transformers or column lists: {e}")
                  # Saving errors are warnings here, as the function can still return the fitted transformers in memory.
                  # The caller (main.py) should check the returned objects regardless.
                  # Keep perform_user_transformer_fitting = True here to indicate generation was attempted.
                  # But if save failed, it means the next run won't load, which is expected behavior.


    # Check if necessary outputs are available for the next step (hybrid matrix assembly)
    # These are the fitted transformers and the lists of columns they were fitted on.
    is_processing_successful = True # Assume success unless proven otherwise

    # Check column lists are lists (should be true if code reaches here, but belt and suspenders)
    if not (isinstance(user_num_cols_present_in_df, list) and isinstance(user_cat_cols_present_in_df, list)):
        print("FATAL ERROR: User column lists are not lists.")
        is_processing_successful = False

    # Check transformers validity *if* their corresponding columns were present for fitting
    # This check is more robust than the loading validation
    if is_processing_successful: # Only check if lists are valid
         if len(user_num_cols_present_in_df) > 0 and (user_numerical_scaler is None or not hasattr(user_numerical_scaler, 'mean_')):
              print("FATAL ERROR: User numerical scaler is None or invalid despite numerical columns being present.")
              is_processing_successful = False

         if len(user_cat_cols_present_in_df) > 0 and (user_categorical_encoder is None or not hasattr(user_categorical_encoder, 'categories_')):
              print("FATAL ERROR: User categorical encoder is None or invalid despite categorical columns being present.")
              is_processing_successful = False
         # Check SVD *only* if categorical cols were present AND encoder is valid AND SVD was expected > 0 components
         elif len(user_cat_cols_present_in_df) > 0 and (user_categorical_encoder is not None and hasattr(user_categorical_encoder, 'categories_')) and n_components_actual > 0:
              if user_categorical_svd is None or not (hasattr(user_categorical_svd, 'components_') and user_categorical_svd.n_components > 0):
                   print("FATAL ERROR: User categorical SVD is None or invalid despite categorical columns and encoder being valid and SVD expected > 0 components.")
                   is_processing_successful = False

    if is_processing_successful:
        print("\nUser feature transformers and column lists are available for next steps.")
        print("Step 5 execution complete.")
        # Return the fitted transformers and the *actual* lists of columns used for transformation
        return (user_numerical_scaler, user_categorical_encoder, user_categorical_svd,
                user_num_cols_present_in_df, user_cat_cols_present_in_df)
    else:
        print("\nStep 5 failed to produce essential transformer outputs.")
        # Return None for transformers that failed, and empty lists for columns that couldn't be processed
        # Ensure lists are returned as lists, even if empty due to failure
        return (None, None, None,
                user_num_cols_present_in_df if isinstance(user_num_cols_present_in_df, list) else [], # Return list even on failure
                user_cat_cols_present_in_df if isinstance(user_cat_cols_present_in_df, list) else []) # Return list even on failure


# Optional: Add a main block to test this script standalone
if __name__ == "__main__":
    print("Running user_processor.py as standalone script...")
    # To run this standalone, you need to provide a dummy or actual user features dataframe.

    # --- Simulate Loading Required Input Data ---
    print("Simulating loading input data for standalone test...")
    try:
        # Create dummy user features dataframe with numerical and categorical columns
        num_dummy_users = 100
        sim_user_features_df = pd.DataFrame({
             config.USER_ID_COL: [f'user{i:03d}' for i in range(num_dummy_users)],
             'user_interaction_count': np.random.randint(1, 50, num_dummy_users),
             'user_avg_price': np.random.rand(num_dummy_users) * 100 + 10, # Example numerical
             'user_most_frequent_store': np.random.choice(['StoreA', 'StoreB', 'StoreC', 'Unknown'], num_dummy_users, p=[0.3, 0.3, 0.3, 0.1]), # Example categorical
             'user_favorite_category': np.random.choice(['Cat1', 'Cat2', 'Cat3', 'Missing'], num_dummy_users, p=[0.4, 0.3, 0.2, 0.1]), # Example categorical
        }).astype({config.USER_ID_COL: str, 'user_most_frequent_store': str, 'user_favorite_category': str})
        sim_user_features_df['user_interaction_count'] = sim_user_features_df['user_interaction_count'].astype(int)
        sim_user_features_df['user_avg_price'] = sim_user_features_df['user_avg_price'].astype(float)

        # Introduce some NaNs for testing NaN handling
        for col in ['user_avg_price', 'user_favorite_category']:
            sim_user_features_df.loc[sim_user_features_df.sample(frac=0.1, random_state=config.SEED).index, col] = np.nan

        print(f"Created simple dummy user_features_df ({len(sim_user_features_df)} rows)")

        # Define the final user feature column lists as they would come from pair_generator
        sim_num_cols_final = ['user_interaction_count', 'user_avg_price']
        sim_cat_cols_final = ['user_most_frequent_store', 'user_favorite_category']

        print(f"Simulated numerical feature list: {sim_num_cols_final}")
        print(f"Simulated categorical feature list: {sim_cat_cols_final}")


        print("Simulated loading complete.")

        # --- Call the main function ---
        print("\nCalling process_user_features...")
        # Define temporary paths for saving during standalone test
        temp_intermediate_dir = './temp_intermediate_for_user_proc_test'
        os.makedirs(temp_intermediate_dir, exist_ok=True)

        temp_user_num_scaler = os.path.join(temp_intermediate_dir, 'test_user_numerical_scaler.pkl')
        temp_user_cat_encoder = os.path.join(temp_intermediate_dir, 'test_user_categorical_encoder.pkl')
        temp_user_cat_svd = os.path.join(temp_intermediate_dir, 'test_user_categorical_svd.pkl')
        temp_user_num_cols_path = os.path.join(temp_intermediate_dir, 'test_user_num_cols_for_scaling_final.json')
        temp_user_cat_cols_path = os.path.join(temp_intermediate_dir, 'test_user_cat_cols_for_encoding_final.json')


        (user_numerical_scaler_gen, user_categorical_encoder_gen, user_categorical_svd_gen,
         user_num_cols_used, user_cat_cols_used) = process_user_features(
             sim_user_features_df,
             sim_num_cols_final, # Pass the simulated final lists
             sim_cat_cols_final, # Pass the simulated final lists
             config.USER_CAT_SVD_N_COMPONENTS,
             config.SEED,
             temp_user_num_scaler,
             temp_user_cat_encoder,
             temp_user_cat_svd,
             temp_user_num_cols_path,
             temp_user_cat_cols_path
         )

        print("\n--- Standalone User Feature Processing Complete ---")
        if user_numerical_scaler_gen is not None:
            print(f"User Numerical Scaler type: {type(user_numerical_scaler_gen)}")
            print(f"User Numerical Scaler features_in_: {getattr(user_numerical_scaler_gen, 'n_features_in_', 'N/A')}")
        else:
            print("User Numerical Scaler is None.")

        if user_categorical_encoder_gen is not None:
            print(f"User Categorical Encoder type: {type(user_categorical_encoder_gen)}")
            print(f"User Categorical Encoder categories count: {sum(len(cats) for cats in user_categorical_encoder_gen.categories_)}")
        else:
            print("User Categorical Encoder is None.")

        if user_categorical_svd_gen is not None:
             print(f"User Categorical SVD type: {type(user_categorical_svd_gen)}")
             print(f"User Categorical SVD components: {user_categorical_svd_gen.n_components}")
        else:
             print("User Categorical SVD is None.")

        print(f"Numerical columns used for scaling: {user_num_cols_used}")
        print(f"Categorical columns used for encoding: {user_cat_cols_used}")


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