# pair_generator.py

import pandas as pd
import numpy as np
import os
import time
import textwrap # For pretty printing
import json # Needed if saving/loading lists
import shutil

# Import configuration from config.py
import config

def generate_pairs_and_user_features(
    train_df: pd.DataFrame,
    clean_item_meta_df: pd.DataFrame,
    train_merged_df: pd.DataFrame,
    negative_sample_ratio: int,
    seed: int,
    train_pairs_csv_path: str,
    user_features_csv_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    Generates user-item training pairs (positive and negative) and basic user features.
    Attempts to load saved data first.

    Args:
        train_df (pd.DataFrame): The original user-item interaction DataFrame.
        clean_item_meta_df (pd.DataFrame): DataFrame with cleaned item metadata.
                                           Used as the pool for negative sampling.
        train_merged_df (pd.DataFrame): Merged train_df and item_meta_df.
                                        Used for user features requiring item metadata.
        negative_sample_ratio (int): Number of negative samples per positive sample.
        seed (int): Random seed for reproducibility.
        train_pairs_csv_path (str): Path to save/load the training pairs CSV.
        user_features_csv_path (str): Path to save/load the user features CSV.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list, list]: A tuple containing
        (train_pairs_df, user_features_df, user_feature_numerical_cols_final, user_feature_categorical_cols_final).

    Raises:
        ValueError: If required input dataframes are None or empty, or missing columns.
        Exception: For other errors during generation or loading.
    """
    print("\n--- Step 3: Generate Training Pairs and User Features ---")

    # --- Validate Inputs ---
    if train_df is None or train_df.empty:
        raise ValueError("Input train_df is None or empty.")
    if clean_item_meta_df is None or clean_item_meta_df.empty:
        raise ValueError("Input clean_item_meta_df is None or empty.")
    # Corrected typo here: train_merged_merged_df -> train_merged_df
    if train_merged_df is None or train_merged_df.empty:
        # train_merged_df might be needed for certain user features, but pair generation doesn't strictly need it.
        # Let's allow it to be None/empty, but handle feature generation conditionally.
        print("Warning: Input train_merged_df is None or empty. User features requiring item metadata might be limited.")
        pass # Not a fatal error for pair generation itself

    required_train_cols = [config.USER_ID_COL, config.ITEM_ID_COL, config.TIMESTAMP_COL]
    if not all(col in train_df.columns for col in required_train_cols):
        missing = [col for col in required_train_cols if col not in train_df.columns]
        raise ValueError(f"Input train_df is missing required columns: {missing}")
    if config.ITEM_ID_COL not in clean_item_meta_df.columns:
         raise ValueError(f"Input clean_item_meta_df is missing required column: {config.ITEM_ID_COL}")
    if config.USER_ID_COL not in train_df.columns:
         raise ValueError(f"Input train_df is missing required column: {config.USER_ID_COL}")

    # Ensure ID columns are string type in inputs for safety
    train_df[config.USER_ID_COL] = train_df[config.USER_ID_COL].astype(str)
    train_df[config.ITEM_ID_COL] = train_df[config.ITEM_ID_COL].astype(str)
    clean_item_meta_df[config.ITEM_ID_COL] = clean_item_meta_df[config.ITEM_ID_COL].astype(str)
    if train_merged_df is not None and not train_merged_df.empty:
        train_merged_df[config.USER_ID_COL] = train_merged_df[config.USER_ID_COL].astype(str)
        train_merged_df[config.ITEM_ID_COL] = train_merged_df[config.ITEM_ID_COL].astype(str)


    # --- Global variables to be populated ---
    train_pairs_df = None
    user_features_df = None
    user_feature_numerical_cols_final = []
    user_feature_categorical_cols_final = []


    # --- Check if saved train_pairs_df exists to potentially skip generation ---
    regenerate_pairs = True # Assume regeneration is needed unless loading succeeds
    print(f"\nChecking for saved train_pairs_df at {train_pairs_csv_path}")
    if os.path.exists(train_pairs_csv_path):
        try:
            # Attempt to load
            loaded_train_pairs_df = pd.read_csv(train_pairs_csv_path)
            # Perform basic validation
            required_pair_cols = [config.USER_ID_COL, config.ITEM_ID_COL, config.INTERACTION_COL]
            if all(col in loaded_train_pairs_df.columns for col in required_pair_cols) and not loaded_train_pairs_df.empty:
                 print("Saved train_pairs_df found and appears valid.")
                 train_pairs_df = loaded_train_pairs_df
                 # Ensure dtypes after loading
                 train_pairs_df[config.USER_ID_COL] = train_pairs_df[config.USER_ID_COL].astype(str)
                 train_pairs_df[config.ITEM_ID_COL] = train_pairs_df[config.ITEM_ID_COL].astype(str)
                 # Ensure interaction is integer
                 train_pairs_df[config.INTERACTION_COL] = train_pairs_df[config.INTERACTION_COL].astype(int)
                 print(f"Loaded train_pairs_df shape: {train_pairs_df.shape}")
                 regenerate_pairs = False # Skip generation
            else:
                 missing = [col for col in required_pair_cols if col not in loaded_train_pairs_df.columns]
                 print(f"Saved train_pairs_df found but is missing critical columns ({missing}) or is empty. Regenerating.")
                 train_pairs_df = None # Force regeneration

        except Exception as e:
            print(f"Error loading {train_pairs_csv_path}: {e}. Regenerating train_pairs_df.")
            train_pairs_df = None # Force regeneration

    if regenerate_pairs: # Regenerate if loading failed or file didn't exist/was invalid
        print("Generating new training pairs...")
        try:
            # Positive samples (from train_df)
            positive_pairs = train_df[[config.USER_ID_COL, config.ITEM_ID_COL]].copy()
            positive_pairs[config.INTERACTION_COL] = 1
            print(f"Generated {len(positive_pairs)} positive pairs.")

            # Get unique items from the cleaned item metadata to sample negatives from
            item_pool_for_neg_sampling = clean_item_meta_df[config.ITEM_ID_COL].unique()
            print(f"Using {len(item_pool_for_neg_sampling)} unique item IDs from metadata for negative sampling pool.")
            if len(item_pool_for_neg_sampling) == 0:
                 raise ValueError("Item pool for negative sampling is empty. Cannot generate negative samples.")

            # Create a set of existing (user, item) interactions for efficient lookup
            user_item_set = set(zip(train_df[config.USER_ID_COL], train_df[config.ITEM_ID_COL]))

            # Negative sampling - Faster per-user approach
            negative_pairs_list = []
            all_users = train_df[config.USER_ID_COL].unique()
            num_users = len(all_users) # Get total number of users
            num_negatives_per_user_base = negative_sample_ratio # The target ratio

            print(f"Generating negative samples (Ratio: {negative_sample_ratio} negative per positive)...")
            start_time_neg_sampling = time.time()

            rng = np.random.RandomState(seed) # Use provided seed for reproducibility

            # Iterate through users
            for i, user in enumerate(all_users): # Use enumerate to get the index
                if (i + 1) % 1000 == 0 or i == 0: # Print progress every 1000 users, and at the start
                    print(f"  Processing user {i + 1}/{num_users}...")

                positive_items_for_user = set([item for u, item in user_item_set if u == user])
                # Items in the pool not interacted with by this user
                items_to_sample_from = [item for item in item_pool_for_neg_sampling if item not in positive_items_for_user]

                num_interactions_for_user = len(positive_items_for_user)
                # Number of negative samples to aim for for this user
                num_negatives_this_user = num_interactions_for_user * num_negatives_per_user_base

                # Ensure we don't try to sample more items than available non-interacted items
                num_samples_actual = min(int(num_negatives_this_user), len(items_to_sample_from))

                if num_samples_actual > 0:
                     # Sample negative items
                     sampled_items = rng.choice(items_to_sample_from, size=num_samples_actual, replace=False)

                     # Append to the list of negative pairs
                     for item in sampled_items:
                          negative_pairs_list.append({config.USER_ID_COL: user, config.ITEM_ID_COL: item, config.INTERACTION_COL: 0})
            negative_pairs = pd.DataFrame(negative_pairs_list)
            print(f"Generated {len(negative_pairs)} negative pairs in {time.time() - start_time_neg_sampling:.2f} seconds.")

            # Combine positive and negative pairs
            train_pairs_df = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
            print(f"Combined train_pairs_df shape (pos + neg): {train_pairs_df.shape}")

            # Shuffle the combined dataset (important for training batches)
            train_pairs_df = train_pairs_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            print("Shuffled training pairs.")

            # Save the generated pairs
            try:
                train_pairs_df.to_csv(train_pairs_csv_path, index=False)
                print(f"Saved generated train_pairs_df to {train_pairs_csv_path}")
            except Exception as e:
                print(f"Warning: Could not save train_pairs_df to {train_pairs_csv_path}: {e}")


        except Exception as e:
            raise Exception(f"An error occurred during training pair generation: {e}") from e

    if train_pairs_df is None or train_pairs_df.empty:
         raise Exception("train_pairs_df could not be loaded or generated.")


    # --- Check if saved user_features_df exists to potentially skip generation ---
    regenerate_user_features = True # Assume regeneration is needed unless loading succeeds
    print(f"\nChecking for saved user_features_df at {user_features_csv_path}")

    # Also check for saved column lists which are generated/saved alongside user features
    user_num_cols_path = config.USER_NUM_COLS_FOR_SCALING_JSON_PATH # Using the _final path from config
    user_cat_cols_path = config.USER_CAT_COLS_FOR_ENCODING_JSON_PATH # Using the _final path from config

    if os.path.exists(user_features_csv_path) and os.path.exists(user_num_cols_path) and os.path.exists(user_cat_cols_path):
         try:
              # Attempt to load user features
              loaded_user_features_df = pd.read_csv(user_features_csv_path)
              # Basic validation: check for user_id and at least one other column
              if config.USER_ID_COL in loaded_user_features_df.columns and len(loaded_user_features_df.columns) > 1 and not loaded_user_features_df.empty:
                  print("Saved user_features_df found and appears valid.")
                  user_features_df = loaded_user_features_df
                  # Ensure user_id dtype
                  user_features_df[config.USER_ID_COL] = user_features_df[config.USER_ID_COL].astype(str)

                  # Load column lists
                  with open(user_num_cols_path, 'r') as f: user_feature_numerical_cols_final = json.load(f)
                  with open(user_cat_cols_path, 'r') as f: user_feature_categorical_cols_final = json.load(f)
                  print("Loaded user feature column lists.")

                  # Check for NaNs in numerical columns after loading and impute using median of loaded data
                  user_num_cols_loaded_in_df = [col for col in user_feature_numerical_cols_final if col in user_features_df.columns]
                  if user_features_df[user_num_cols_loaded_in_df].isnull().sum().sum() > 0:
                      print("Warning: NaNs found in numerical user features after loading. Imputing with median.")
                      for col in user_num_cols_loaded_in_df:
                           if user_features_df[col].isnull().sum() > 0:
                                median_val = user_features_df[col].median()
                                user_features_df[col] = user_features_df[col].fillna(median_val)

                  # Ensure categorical columns are string type and fill NaNs with 'Missing' for consistency
                  user_cat_cols_loaded_in_df = [col for col in user_feature_categorical_cols_final if col in user_features_df.columns]
                  for col in user_cat_cols_loaded_in_df:
                       if user_features_df[col].isnull().sum() > 0:
                            user_features_df[col] = user_features_df[col].fillna('Missing')
                       user_features_df[col] = user_features_df[col].astype(str)

                  print(f"Loaded user_features_df shape: {user_features_df.shape}")
                  print(f"Loaded numerical user features ({len(user_feature_numerical_cols_final)}): {user_feature_numerical_cols_final}")
                  print(f"Loaded categorical user features ({len(user_feature_categorical_cols_final)}): {user_feature_categorical_cols_final}")
                  regenerate_user_features = False # Skip generation

              else:
                   missing = [config.USER_ID_COL] # Minimum required column
                   print(f"Saved user_features_df found but is missing critical columns ({missing}) or is empty. Regenerating.")
                   user_features_df = None # Force regeneration
                   user_feature_numerical_cols_final = [] # Reset lists
                   user_feature_categorical_cols_final = []

         except Exception as e:
              print(f"Error loading saved user features or column lists: {e}. Regenerating.")
              user_features_df = None # Force regeneration
              user_feature_numerical_cols_final = [] # Reset lists
              user_feature_categorical_cols_final = []

    if regenerate_user_features: # Regenerate if loading failed or files didn't exist
         print("Generating new user features...")
         try:
             # Calculate simple aggregate features from the *original* train_df
             # This avoids using information leaked from the merged metadata or negative samples.
             user_features_df = train_df.groupby(config.USER_ID_COL).agg(
                 user_interaction_count=(config.ITEM_ID_COL, 'count'),
                 # Add more numerical features based on train_df if available/relevant, e.g.:
                 # user_avg_timestamp = (config.TIMESTAMP_COL, 'mean'),
                 # user_unique_item_count = (config.ITEM_ID_COL, lambda x: x.nunique()),
             ).reset_index()

             # Identify the numerical columns just created
             # Corrected the dtype check here
             user_feature_numerical_cols_final = [col for col in user_features_df.columns if col != config.USER_ID_COL and np.issubdtype(user_features_df[col].dtype, np.number)]
             user_feature_categorical_cols_final = [] # Initialize categorical list


             # Calculate user_most_frequent_store using train_merged_df IF it's available and has the 'store' column
             if train_merged_df is not None and not train_merged_df.empty and 'store' in train_merged_df.columns:
                  print("Generating 'user_most_frequent_store' feature from merged data...")
                  user_most_frequent_store = train_merged_df.groupby(config.USER_ID_COL)['store'].agg(
                      user_most_frequent_store = (lambda x: x.mode()[0] if not x.mode().empty else 'Unknown') # Handles cases with no store or multiple modes
                  ).reset_index()
                  # Merge this new categorical feature
                  user_features_df = pd.merge(user_features_df, user_most_frequent_store, on=config.USER_ID_COL, how='left')
                  # Handle potential NaNs in the new column (e.g., user had no interactions with valid stores)
                  user_features_df['user_most_frequent_store'] = user_features_df['user_most_frequent_store'].fillna('Unknown').astype(str) # Treat NaN as 'Unknown' category
                  user_feature_categorical_cols_final.append('user_most_frequent_store') # Add to the list

             # Ensure user_id is string after merges
             user_features_df[config.USER_ID_COL] = user_features_df[config.USER_ID_COL].astype(str)

             print(f"Generated user_features_df shape: {user_features_df.shape}")
             print(f"Identified numerical user features ({len(user_feature_numerical_cols_final)}): {user_feature_numerical_cols_final}")
             print(f"Identified categorical user features ({len(user_feature_categorical_cols_final)}): {user_feature_categorical_cols_final}")


             # Save the generated features and column lists
             try:
                 user_features_df.to_csv(user_features_csv_path, index=False)
                 print(f"Saved generated user_features_df to {user_features_csv_path}")
                 with open(user_num_cols_path, 'w') as f: json.dump(user_feature_numerical_cols_final, f)
                 print(f"Saved user_feature_numerical_cols_final to {user_num_cols_path}")
                 with open(user_cat_cols_path, 'w') as f: json.dump(user_feature_categorical_cols_final, f)
                 print(f"Saved user_feature_categorical_cols_final to {user_cat_cols_path}")

             except Exception as e:
                print(f"Warning: Could not save user features or column lists: {e}")


         except Exception as e:
              raise Exception(f"An error occurred during user feature generation: {e}") from e

    if user_features_df is None or user_features_df.empty:
         raise Exception("user_features_df could not be loaded or generated.")
    if user_feature_numerical_cols_final is None or user_feature_categorical_cols_final is None:
         raise Exception("User feature column lists were not successfully loaded or generated.")


    print("Step 3 execution complete.")

    return train_pairs_df, user_features_df, user_feature_numerical_cols_final, user_feature_categorical_cols_final

# Optional: Add a main block to test this script standalone
if __name__ == "__main__":
    print("Running pair_generator.py as standalone script...")
    # To run this standalone, you need to provide dummy dataframes
    # or load actual small slices/dummy versions of the required inputs.
    # Loading small actual files is better for testing the function logic.

    # --- Simulate Loading Required Input Data ---
    print("Simulating loading input data for standalone test...")
    try:
        # Create dummy inputs or load small actual files
        # Adjust paths as needed for your test data location
        dummy_train_path = '../data/train.csv' # Using actual config path for simulation
        dummy_meta_path = '../data/dataset/clean_beauty_item_meta_with_details.csv' # Using actual config path for simulation

        if os.path.exists(dummy_train_path) and os.path.exists(dummy_meta_path):
            # Load small subset of actual data for a more realistic test
            sim_train_df = pd.read_csv(dummy_train_path).head(1000)
            sim_train_df[config.USER_ID_COL] = sim_train_df[config.USER_ID_COL].astype(str)
            sim_train_df[config.ITEM_ID_COL] = sim_train_df[config.ITEM_ID_COL].astype(str)
            print(f"Simulated loading small train_df ({len(sim_train_df)} rows)")

            sim_meta_df = pd.read_csv(dummy_meta_path)
            sim_meta_df[config.ITEM_ID_COL] = sim_meta_df[config.ITEM_ID_COL].astype(str)
            print(f"Simulated loading clean_item_meta_df ({len(sim_meta_df)} rows)")

            # Simulate merged df with 'store' column for user features
            sim_merged_df = pd.merge(sim_train_df, sim_meta_df[['item_id', 'store']], on='item_id', how='left')
            sim_merged_df['store'] = sim_merged_df['store'].fillna('Unknown').astype(str) # Simple fill for test
            print(f"Simulated merged_df ({len(sim_merged_df)} rows)")

        else:
            print("Warning: Actual data files not found for simulation. Creating simple dummy data.")
            # Create simple dummy dataframes if files aren't available
            sim_train_df = pd.DataFrame({
                config.USER_ID_COL: ['u1', 'u1', 'u2', 'u3', 'u3', 'u3', 'u4'],
                config.ITEM_ID_COL: ['i1', 'i2', 'i1', 'i3', 'i4', 'i5', 'i6'],
                config.TIMESTAMP_COL: [1, 2, 3, 4, 5, 6, 7],
                config.INTERACTION_COL: [1, 1, 1, 1, 1, 1, 1] # Assume all positives in raw train
            }).astype(str)
            sim_train_df[config.TIMESTAMP_COL] = sim_train_df[config.TIMESTAMP_COL].astype(int) # Timestamp can be int/float

            sim_meta_df = pd.DataFrame({
                 config.ITEM_ID_COL: ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10'],
                 'store': ['StoreA', 'StoreB', 'StoreA', 'StoreC', 'StoreA', 'StoreB', 'StoreC', 'StoreA', 'StoreB', 'StoreC'],
                 'price_category_mapped': [1,2,3,1,2,3,1,2,3,1], # Example item feature
                 'title': [f'Item {i}' for i in range(1, 11)] # Example item feature
            }).astype(str)
            sim_meta_df['price_category_mapped'] = sim_meta_df['price_category_mapped'].astype(int)

            sim_merged_df = pd.merge(sim_train_df, sim_meta_df[['item_id', 'store']], on='item_id', how='left')
            sim_merged_df['store'] = sim_merged_df['store'].fillna('Unknown').astype(str)


        print("Simulated loading complete.")

        # --- Call the main function ---
        print("\nCalling generate_pairs_and_user_features...")
        # Define temporary paths for saving during standalone test
        temp_intermediate_dir = './temp_intermediate_for_pair_gen_test'
        os.makedirs(temp_intermediate_dir, exist_ok=True)
        temp_pairs_csv = os.path.join(temp_intermediate_dir, 'test_train_pairs.csv')
        temp_user_features_csv = os.path.join(temp_intermediate_dir, 'test_user_features.csv')
        temp_user_num_cols_json = os.path.join(temp_intermediate_dir, 'test_user_num_cols.json')
        temp_user_cat_cols_json = os.path.join(temp_intermediate_dir, 'test_user_cat_cols.json')

        # Override config paths temporarily for the test call
        original_user_num_cols_path = config.USER_NUM_COLS_FOR_SCALING_JSON_PATH
        original_user_cat_cols_path = config.USER_CAT_COLS_FOR_ENCODING_JSON_PATH
        config.USER_NUM_COLS_FOR_SCALING_JSON_PATH = temp_user_num_cols_json
        config.USER_CAT_COLS_FOR_ENCODING_JSON_PATH = temp_user_cat_cols_json


        train_pairs_df_generated, user_features_df_generated, num_cols, cat_cols = generate_pairs_and_user_features(
            sim_train_df,
            sim_meta_df,
            sim_merged_df, # Pass the simulated merged df
            config.NEGATIVE_SAMPLE_RATIO,
            config.SEED,
            temp_pairs_csv,
            temp_user_features_csv
        )

        # Restore original config paths
        config.USER_NUM_COLS_FOR_SCALING_JSON_PATH = original_user_num_cols_path
        config.USER_CAT_COLS_FOR_ENCODING_JSON_PATH = original_user_cat_cols_path


        print("\n--- Standalone Pair Generation and User Features Complete ---")
        if train_pairs_df_generated is not None:
            print(f"Final train_pairs_df shape: {train_pairs_df_generated.shape}")
            print(f"Positive pairs: {len(train_pairs_df_generated[train_pairs_df_generated[config.INTERACTION_COL] == 1])}")
            print(f"Negative pairs: {len(train_pairs_df_generated[train_pairs_df_generated[config.INTERACTION_COL] == 0])}")
            print("\nFirst 5 rows of train_pairs_df:")
            print(textwrap.indent(train_pairs_df_generated.head().__str__(), '  '))

        if user_features_df_generated is not None:
             print(f"\nFinal user_features_df shape: {user_features_df_generated.shape}")
             print(f"User feature numerical columns: {num_cols}")
             print(f"User feature categorical columns: {cat_cols}")
             print("\nFirst 5 rows of user_features_df:")
             print(textwrap.indent(user_features_df_generated.head().__str__(), '  '))


        # --- Clean up temporary files ---
        print("\nCleaning up temporary test files...")
        try:
             if os.path.exists(temp_intermediate_dir):
                  shutil.rmtree(temp_intermediate_dir)
                  print(f"Removed temporary directory: {temp_intermediate_dir}")
        except Exception as e:
             print(f"Warning: Could not remove temporary directory {temp_intermediate_dir}: {e}")


    except FileNotFoundError as e:
         print(f"Script failed: {e}. Ensure dummy data paths or actual data paths are correct.")
    except ValueError as e:
         print(f"Script failed due to input data error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during script execution: {e}")