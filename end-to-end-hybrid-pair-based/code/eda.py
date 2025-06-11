# eda.py

import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import statistics
import collections
import re
import string
import ast
import textwrap

# --- Configuration ---
# Data path relative to the script location
data_path = "../data/"
train_csv_path = os.path.join(data_path, "train.csv")
test_csv_path = os.path.join(data_path, "test.csv")
item_meta_csv_path = os.path.join(data_path, "item_meta.csv")
submission_csv_path = os.path.join(data_path, "sample_submission.csv")

# Visuals path relative to the script location
visuals_base_path = "../visuals/"
eda_visuals_path = os.path.join(visuals_base_path, "eda")

# Ensure the visuals directory exists
os.makedirs(eda_visuals_path, exist_ok=True)
print(f"Ensured visuals directory exists: {eda_visuals_path}")


# Set seeds for reproducibility (optional for this analysis)
seed = 42
random.seed(seed)
np.random.seed(seed)

# --- Helper Function for safe literal eval ---
# Used for parsing strings that represent lists/dicts (like categories, images, details)
def safe_literal_eval(s):
    try:
        # Ensure it's a string before parsing
        if isinstance(s, str):
            # Handle common cases of empty list string or empty dict string
            s_stripped = s.strip()
            if s_stripped == '[]':
                return []
            if s_stripped == '{}':
                 return {}
            # Attempt parsing for non-empty lists/dicts
            return ast.literal_eval(s)
        return None # Return None for non-string or NaN
    except (ValueError, SyntaxError):
        return None # Return None if parsing fails

# --- Helper Function for generating n-grams ---
def generate_ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


print("--- Starting Exploratory Data Analysis ---")

# --- Analysis 1: User History Length Distribution ---
print("\n--- Analysis 1: User History Length Distribution ---")

train_df = None
test_df = None
interactions_df = pd.DataFrame() # Initialize combined df

# Load Interaction Data (Train and Test)
print(f"Loading interaction data from {train_csv_path} and {test_csv_path}...")
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    print(f"Loaded train data: {train_df.shape[0]} interactions")
else:
    print(f"Warning: Train data not found at {train_csv_path}")

if os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded test data: {test_df.shape[0]} interactions")
else:
    print(f"Warning: Test data not found at {test_csv_path}")

# Combine train and test data for full history if at least one loaded
if train_df is not None and not train_df.empty and test_df is not None and not test_df.empty:
     interactions_df = pd.concat([train_df, test_df], ignore_index=True)
elif train_df is not None and not train_df.empty:
     interactions_df = train_df.copy()
elif test_df is not None and not test_df.empty:
     interactions_df = test_df.copy()
else:
     print("Error: No interaction data (neither train nor test) loaded. Cannot analyze history.")
     interactions_df = pd.DataFrame()

users_for_analysis_orig = []
if not interactions_df.empty:
    required_interaction_cols = ['user_id', 'item_id', 'timestamp']
    if not all(col in interactions_df.columns for col in required_interaction_cols):
         print("Error: Required interaction columns ('user_id', 'item_id', 'timestamp') not found in combined data.")
         interactions_df = pd.DataFrame()
    else:
        interactions_df = interactions_df[required_interaction_cols].dropna(subset=['user_id']).copy()
        print(f"Combined interaction data shape after selecting columns and removing NaN in user_id: {interactions_df.shape}")

    # Load Users for Analysis (from submission file)
    print(f"\nLoading users for analysis from {submission_csv_path}...")
    submission_df = pd.DataFrame()
    if os.path.exists(submission_csv_path):
        try:
            submission_df = pd.read_csv(submission_csv_path)
            if 'user_id' in submission_df.columns:
                users_for_analysis_orig = submission_df['user_id'].unique().tolist()
                print(f"Loaded {len(users_for_analysis_orig)} unique user IDs for analysis from submission file.")
            else:
                 print(f"Error: 'user_id' column not found in {submission_csv_path}.")
        except Exception as e:
            print(f"Error loading {submission_csv_path}: {e}")
    else:
        print(f"Error: Submission file not found at {submission_csv_path}. Cannot determine users for analysis.")


# Calculate Interaction Counts for Users in Submission File
all_submission_users_counts = pd.Series(dtype=int)
if users_for_analysis_orig and not interactions_df.empty:
    print("\nCalculating interaction counts for users in the submission file...")
    interactions_for_analysis_users = interactions_df[interactions_df['user_id'].isin(users_for_analysis_orig)].copy()

    if not interactions_for_analysis_users.empty:
         user_interaction_counts = interactions_for_analysis_users.groupby('user_id').size()
         print(f"Calculated interaction counts for {len(user_interaction_counts)} users from submission file who had interactions.")
    else:
         print("No interaction data found for users in the submission file.")
         user_interaction_counts = pd.Series(dtype=int)

    users_with_interactions_set = set(user_interaction_counts.index)
    users_from_submission_set = set(users_for_analysis_orig)
    users_with_zero_history_set = users_from_submission_set - users_with_interactions_set
    print(f"Number of users from submission file with ZERO interactions found: {len(users_with_zero_history_set)}")

    zero_history_series = pd.Series(0, index=list(users_with_zero_history_set))
    all_submission_users_counts = pd.concat([user_interaction_counts, zero_history_series])
    print(f"Total users from submission file included in count analysis: {len(all_submission_users_counts)}")

elif users_for_analysis_orig: # and interactions_df is empty
     print("Warning: No interaction data available to count history length for submission users. Assuming 0 history for all.")
     all_submission_users_counts = pd.Series(0, index=users_for_analysis_orig)
     print(f"Assigned history count 0 to all {len(all_submission_users_counts)} users from submission file.")
else:
     print("No users loaded from submission file for history analysis.")


# Analyze and Visualize History Length Distribution
if not all_submission_users_counts.empty:
    print(f"\nAnalyzing history lengths for {len(all_submission_users_counts)} users from submission file.")
    history_length_distribution = all_submission_users_counts.value_counts().sort_index()

    print("\nHistory Length Distribution (Count of users per history length):")
    print(history_length_distribution)

    print("\nSummary Statistics for User History Lengths:")
    print(all_submission_users_counts.describe())

    print("\nGenerating history length distribution graph...")
    plt.figure(figsize=(15, 7))
    x_labels = history_length_distribution.index.astype(str)
    x_positions = np.arange(len(x_labels))
    plt.bar(x_positions, history_length_distribution.values, align='center')
    plt.xlabel("Number of Past Interactions (Total History Length in Train + Test)")
    plt.ylabel("Number of Users")
    plt.title("Distribution of Interaction History Lengths for Users in Submission File")
    plt.xticks(x_positions, x_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save the plot instead of showing
    save_path = os.path.join(eda_visuals_path, "user_history_length_distribution.png")
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory
else:
    print("Skipping history length analysis due to missing data.")


# --- Analysis 2: Store Distribution in Test Set ---
print("\n--- Analysis 2: Store Distribution in Test Set ---")

test_df = None
item_meta_df = None

# Load Test Interaction Data
print(f"Loading test data from {test_csv_path}...")
if os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded test data: {test_df.shape[0]} interactions")
else:
    print(f"Error: Test data not found at {test_csv_path}. Cannot analyze store distribution in test set.")

# Load Item Metadata for store information
print(f"Loading item metadata for store analysis from {item_meta_csv_path}...")
item_id_to_store = {}
if os.path.exists(item_meta_csv_path):
    try:
        item_meta_df = pd.read_csv(item_meta_csv_path, low_memory=False)
        print(f"Loaded item metadata: {item_meta_df.shape[0]} items")
        required_meta_cols = ['item_id', 'store']
        if not all(col in item_meta_df.columns for col in required_meta_cols):
            print("Error: Required columns ('item_id', 'store') not found in item metadata for store analysis.")
        else:
             item_meta_df_relevant = item_meta_df[required_meta_cols].dropna().copy()
             print(f"Metadata shape after selecting columns and removing NaN in 'store': {item_meta_df_relevant.shape}")
             item_id_to_store = item_meta_df_relevant.set_index('item_id')['store'].to_dict()
             print(f"Created store mapping for {len(item_id_to_store)} items.")
    except Exception as e:
        print(f"Error loading {item_meta_csv_path}: {e}")
else:
    print(f"Error: Item metadata file not found at {item_meta_csv_path}. Cannot get store information.")


# Merge Test Data with Store Information & Calculate Distribution
store_distribution_test_set = pd.Series(dtype=int)
if test_df is not None and not test_df.empty and item_id_to_store:
    required_test_cols = ['user_id', 'item_id'] # Only these needed for merge/grouping
    if not all(col in test_df.columns for col in required_test_cols):
         print("Error: Required columns ('user_id', 'item_id') not found in test data for store analysis.")
         test_df_relevant = pd.DataFrame()
    else:
        test_df_relevant = test_df[required_test_cols].dropna(subset=['item_id']).copy()

    if not test_df_relevant.empty:
        print("\nMerging test data with store information...")
        test_df_with_stores = test_df_relevant.copy()
        test_df_with_stores['store'] = test_df_with_stores['item_id'].map(item_id_to_store)
        test_df_with_stores = test_df_with_stores.dropna(subset=['store']).copy() # Drop interactions where item had no store
        print(f"Test interactions with store information shape: {test_df_with_stores.shape}")

        if not test_df_with_stores.empty:
            print("\nCalculating store distribution in the test set...")
            store_distribution_test_set = test_df_with_stores['store'].value_counts()
            print(f"Calculated distribution for {len(store_distribution_test_set)} unique stores in the test set.")
        else:
             print("No test interactions with store information available.")
    else:
         print("Test data is empty or missing required columns.")
elif test_df is None or test_df.empty:
    print("Skipping test set store distribution analysis: Test data not loaded or empty.")
elif not item_id_to_store:
     print("Skipping test set store distribution analysis: Store metadata not loaded or mapping is empty.")

# Analyze and Visualize Store Distribution
if not store_distribution_test_set.empty:
    print(f"\nTotal unique stores found in test set interactions with metadata: {len(store_distribution_test_set)}")
    print("\nDistribution of interactions per store (Top 20):")
    print(store_distribution_test_set.head(20))

    print("\nSummary Statistics for Interactions per Store:")
    print(store_distribution_test_set.describe())

    print("\nGenerating store distribution graph (Top 20 stores)...")
    top_n_stores_to_plot = 20
    store_distribution_top_n = store_distribution_test_set.head(top_n_stores_to_plot)
    plt.figure(figsize=(15, 7))
    x_labels = store_distribution_top_n.index.tolist()
    x_positions = np.arange(len(x_labels))
    plt.bar(x_positions, store_distribution_top_n.values, align='center')
    plt.xlabel(f"Store Name (Top {top_n_stores_to_plot} by Interaction Count)")
    plt.ylabel("Number of Interactions in Test Set")
    plt.title(f"Distribution of Interactions across Top {top_n_stores_to_plot} Stores in Test Set")
    plt.xticks(x_positions, x_labels, rotation=90, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save the plot instead of showing
    save_path = os.path.join(eda_visuals_path, "test_set_top20_store_distribution.png")
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path)
    plt.close() # Close the figure
else:
    print("Skipping store distribution analysis in test set due to missing data.")


# --- Analysis 3: Unique Store Count per User in Train Set (for Submission Users) ---
print("\n--- Analysis 3: Unique Store Count per User (Train Set, Submission Users) ---")

users_for_analysis_orig = []
train_df = None
item_meta_df = None

# Load Users for Analysis (from submission file)
print(f"Loading users for analysis from {submission_csv_path}...")
if os.path.exists(submission_csv_path):
    try:
        submission_df = pd.read_csv(submission_csv_path)
        if 'user_id' in submission_df.columns:
            users_for_analysis_orig = submission_df['user_id'].unique().tolist()
            print(f"Loaded {len(users_for_analysis_orig)} unique user IDs for analysis from submission file.")
        else:
             print(f"Error: 'user_id' column not found in {submission_csv_path}.")
             users_for_analysis_orig = []
    except Exception as e:
        print(f"Error loading {submission_csv_path}: {e}")
        users_for_analysis_orig = []
else:
    print(f"Error: Submission file not found at {submission_csv_path}. Cannot determine users for analysis.")


# Load Train Interaction Data and Filter for Submission Users
print(f"\nLoading train data from {train_csv_path} and filtering for submission users...")
train_df_filtered = pd.DataFrame()
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    print(f"Loaded train data: {train_df.shape[0]} interactions")
    required_train_cols = ['user_id', 'item_id']
    if not all(col in train_df.columns for col in required_train_cols):
         print("Error: Required columns ('user_id', 'item_id') not found in train data.")
    else:
        train_df = train_df[required_train_cols].dropna(subset=required_train_cols).copy()
        print(f"Train data shape after removing NaN: {train_df.shape}")

        if users_for_analysis_orig and not train_df.empty:
             train_df_filtered = train_df[train_df['user_id'].isin(users_for_analysis_orig)].copy()
             print(f"Filtered train interactions shape for submission users: {train_df_filtered.shape}")
        elif users_for_analysis_orig:
             print("Warning: No train interaction data available or is empty to filter for submission users.")
        # If users_for_analysis_orig is empty, train_df_filtered remains empty.
else:
    print(f"Error: Train data not found at {train_csv_path}. Cannot filter for submission users.")


# Load Item Metadata for store information
print(f"\nLoading item metadata for store analysis from {item_meta_csv_path}...")
item_id_to_store = {}
if os.path.exists(item_meta_csv_path):
    try:
        item_meta_df = pd.read_csv(item_meta_csv_path, low_memory=False)
        print(f"Loaded item metadata: {item_meta_df.shape[0]} items")
        required_meta_cols = ['item_id', 'store']
        if not all(col in item_meta_df.columns for col in required_meta_cols):
            print("Error: Required columns ('item_id', 'store') not found in item metadata for store analysis.")
        else:
             item_meta_df_relevant = item_meta_df[required_meta_cols].dropna().copy()
             print(f"Metadata shape after selecting columns and removing NaN in 'store': {item_meta_df_relevant.shape}")
             item_id_to_store = item_meta_df_relevant.set_index('item_id')['store'].to_dict()
             print(f"Created store mapping for {len(item_id_to_store)} items.")
    except Exception as e:
        print(f"Error loading {item_meta_csv_path}: {e}")
else:
    print(f"Error: Item metadata file not found at {item_meta_csv_path}. Cannot get store information.")


# Merge Filtered Train Data with Store Information & Calculate Unique Store Counts
all_submission_users_unique_store_counts = pd.Series(dtype=int)
unique_store_count_distribution = pd.Series(dtype=int)

if not train_df_filtered.empty and item_id_to_store:
    print("\nMerging filtered train data with store information...")
    train_df_with_stores = train_df_filtered.copy()
    train_df_with_stores['store'] = train_df_with_stores['item_id'].map(item_id_to_store)
    train_df_with_stores = train_df_with_stores.dropna(subset=['store']).copy()
    print(f"Filtered train interactions with store information shape: {train_df_with_stores.shape}")

    if not train_df_with_stores.empty:
        print("\nCalculating unique store counts per user in the train set...")
        user_unique_store_counts_with_interactions = train_df_with_stores.groupby('user_id')['store'].nunique()
        print(f"Calculated unique store counts for {len(user_unique_store_counts_with_interactions)} users with interactions.")

        users_with_interactions_set = set(user_unique_store_counts_with_interactions.index)
        users_from_submission_set = set(users_for_analysis_orig)
        users_with_zero_unique_stores_set = users_from_submission_set - users_with_interactions_set
        print(f"Number of users from submission file with ZERO interactions in filtered train data: {len(users_with_zero_unique_stores_set)}")

        zero_unique_stores_series = pd.Series(0, index=list(users_with_zero_unique_stores_set))
        all_submission_users_unique_store_counts = pd.concat([user_unique_store_counts_with_interactions, zero_unique_stores_series])
        print(f"Total users from submission file included in unique store count analysis: {len(all_submission_users_unique_store_counts)}")

        unique_store_count_distribution = all_submission_users_unique_store_counts.value_counts().sort_index()

        print("\nUnique Store Count Distribution (Number of users per unique store count):")
        print(unique_store_count_distribution)

        print("\nSummary Statistics for Unique Stores per User:")
        print(all_submission_users_unique_store_counts.describe())

    else:
        print("No filtered train interactions with store information available for unique store count analysis.")

elif not users_for_analysis_orig:
     print("Skipping unique store count analysis: No users loaded from submission file.")
elif train_df_filtered.empty:
     print("Skipping unique store count analysis: Filtered train data is empty or missing.")
elif not item_id_to_store:
     print("Skipping unique store count analysis: Store metadata mapping is empty.")


# Visualize Unique Store Count Distribution
if not unique_store_count_distribution.empty:
    print("\nGenerating unique store count distribution graph...")
    plt.figure(figsize=(15, 7))
    x_labels = unique_store_count_distribution.index.astype(str)
    x_positions = np.arange(len(x_labels))
    plt.bar(x_positions, unique_store_count_distribution.values, align='center')
    plt.xlabel("Number of Unique Stores Interacted With (in Train Set)")
    plt.ylabel("Number of Users")
    plt.title("Distribution of Unique Stores Interacted With by Users in Submission File (in Train Set)")
    plt.xticks(x_positions, x_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save the plot instead of showing
    save_path = os.path.join(eda_visuals_path, "user_unique_store_count_distribution.png")
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path)
    plt.close() # Close the figure
else:
    print("Skipping unique store count distribution visualization due to missing data.")


# --- Analysis 4: Item Metadata Analysis (Focused on 'All Beauty') ---
print("\n--- Analysis 4: Item Metadata Analysis ('All Beauty') ---")

item_meta_df = None
# Load Item Metadata
print(f"Loading item metadata from {item_meta_csv_path}...")
if os.path.exists(item_meta_csv_path):
    try:
        item_meta_df = pd.read_csv(item_meta_csv_path, low_memory=False)
        print(f"Loaded item metadata: {item_meta_df.shape[0]} items")
    except Exception as e:
        print(f"Error loading {item_meta_csv_path}: {e}")
else:
    print(f"Error: Item metadata file not found at {item_meta_csv_path}. Cannot perform metadata analysis.")

if item_meta_df is None or item_meta_df.empty:
    print("\nNo item metadata loaded. Cannot perform analysis.")
else:
    print("\nInitial DataFrame Info:")
    item_meta_df.info()

    print("\nInitial DataFrame Head:")
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
        print(item_meta_df.head())

    print("\nInitial DataFrame Describe (numerical columns):")
    print(item_meta_df.describe())

    print("\nMissing Value Analysis (Full Dataset):")
    missing_counts = item_meta_df.isnull().sum()
    missing_percentage = (missing_counts / len(item_meta_df)) * 100
    missing_info = pd.DataFrame({'Missing Count': missing_counts, 'Missing %': missing_percentage})
    print(missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing %', ascending=False))

    print("\nUnique Value Counts per Column (Full Dataset):")
    for col in item_meta_df.columns:
        nunique = item_meta_df[col].nunique()
        print(f"  '{col}': {nunique} unique values")


    # Filter for 'All Beauty' category
    beauty_items_df = item_meta_df[item_meta_df['main_category'] == 'All Beauty'].copy()
    print(f"\n--- Focused Analysis on 'All Beauty' Category ({len(beauty_items_df)} items) ---")

    if beauty_items_df.empty:
        print("No items found in the 'All Beauty' category. Skipping detailed 'All Beauty' analysis.")
    else:
        print("\nMissing Value Analysis (All Beauty):")
        missing_counts_beauty = beauty_items_df.isnull().sum()
        missing_percentage_beauty = (missing_counts_beauty / len(beauty_items_df)) * 100
        missing_info_beauty = pd.DataFrame({'Missing Count': missing_counts_beauty, 'Missing %': missing_percentage_beauty})
        print(missing_info_beauty[missing_info_beauty['Missing Count'] > 0].sort_values(by='Missing %', ascending=False))

        print("\nUnique Value Counts per Column (All Beauty):")
        for col in beauty_items_df.columns:
            nunique = beauty_items_df[col].nunique()
            print(f"  '{col}': {nunique} unique values")


        # --- Analyze specific columns in 'All Beauty' ---

        # parent_asin
        print("\n--- Analyzing 'parent_asin' (All Beauty) ---")
        unique_parent_asins_beauty = beauty_items_df['parent_asin'].nunique()
        print(f"Number of unique parent_ASINs: {unique_parent_asins_beauty}")
        items_with_parent_beauty = beauty_items_df['parent_asin'].notna().sum()
        print(f"Number of items with a non-null parent_ASIN: {items_with_parent_beauty}")
        print(f"Number of items with a null parent_ASIN: {len(beauty_items_df) - items_with_parent_beauty}")

        if items_with_parent_beauty > 0:
             parent_counts = beauty_items_df.groupby('parent_asin').size().sort_values(ascending=False)
             print("\nDistribution of items per parent_ASIN (Top 10):")
             print(parent_counts.head(10))
             print("\nSummary statistics for number of child items per parent_ASIN:")
             print(parent_counts.describe())

             print("\nGenerating histogram for items per parent_ASIN...")
             plt.figure(figsize=(12, 6))
             parent_counts.hist(bins=50, log=True)
             plt.title('Distribution of Number of Child Items per Parent_ASIN (All Beauty)')
             plt.xlabel('Number of Child Items')
             plt.ylabel('Number of Parent_ASINs (log scale)')
             plt.grid(axis='y', alpha=0.75)
             plt.tight_layout()
             # Save the plot instead of showing
             save_path = os.path.join(eda_visuals_path, "beauty_items_per_parent_asin_histogram.png")
             print(f"Saving plot to {save_path}")
             plt.savefig(save_path)
             plt.close() # Close the figure


             parents_with_multiple_children = parent_counts[parent_counts > 1].index
             if not parents_with_multiple_children.empty:
                 example_parent_asin = parents_with_multiple_children[0]
                 print(f"\nExample: Items grouped under parent_ASIN '{example_parent_asin}':")
                 example_items = beauty_items_df[beauty_items_df['parent_asin'] == example_parent_asin]
                 print(example_items[['item_id', 'title', 'price', 'details', 'parent_asin']])
             else:
                  print("\nNo parent_ASINs found with multiple child items in 'All Beauty'.")


        # average_rating, rating_number, store, timestamp (basic info)
        print("\n--- Analyzing 'average_rating', 'rating_number', 'store', 'timestamp' (All Beauty) ---")
        basic_cols_to_analyze = ['average_rating', 'rating_number', 'store', 'timestamp']
        for col in basic_cols_to_analyze:
            if col in beauty_items_df.columns:
                print(f"\n  Analysis for column: '{col}'")
                dtype = beauty_items_df[col].dtype
                non_null_count = beauty_items_df[col].notnull().sum()
                nunique = beauty_items_df[col].nunique()
                missing_count = beauty_items_df[col].isnull().sum()

                print(f"    Dtype: {dtype}")
                print(f"    Non-null count: {non_null_count} (Missing: {missing_count})")
                print(f"    Unique values: {nunique}")

                if dtype in ['int64', 'float64']:
                    if non_null_count > 0:
                         print("    Describe:")
                         print(beauty_items_df[col].dropna().describe())
                    else:
                         print("    No non-null values for describe.")
                elif dtype == 'object':
                    if non_null_count > 0:
                         print("    Sample values (Top 10 Value Counts):")
                         print(beauty_items_df[col].value_counts().head(10))
                    else:
                         print("    No non-null values for value counts.")
                # Special handling for timestamp if it's an object before conversion
                if col == 'timestamp' and dtype == 'object' and non_null_count > 0:
                     try:
                          timestamps_dt = pd.to_datetime(beauty_items_df['timestamp'].dropna(), unit='ms')
                          print(f"    First timestamp: {timestamps_dt.min()}")
                          print(f"    Last timestamp: {timestamps_dt.max()}")
                          print(f"    Range: {timestamps_dt.max() - timestamps_dt.min()}")
                     except Exception as e:
                          print(f"    Could not convert timestamp to datetime: {e}")


        # categories
        print("\n--- Analyzing 'categories' (All Beauty) ---")
        if 'categories' in beauty_items_df.columns:
            null_categories_count = beauty_items_df['categories'].isnull().sum()
            print(f"Number of items with null 'categories': {null_categories_count}")
            non_null_categories = beauty_items_df['categories'].dropna()
            print(f"Number of items with non-null 'categories': {len(non_null_categories)}")

            if len(non_null_categories) > 0:
                print("\nSample non-null 'categories' values:")
                sample_categories = non_null_categories.sample(min(5, len(non_null_categories))).tolist()
                for i, cat in enumerate(sample_categories):
                     print(f"Sample {i+1}: {cat} (Type: {type(cat)})")

                # Attempt parsing
                parsed_categories = non_null_categories.apply(safe_literal_eval)
                # Filter out None results and results that are not lists or are empty lists
                valid_category_lists = parsed_categories[parsed_categories.apply(lambda x: isinstance(x, list) and len(x) > 0)]

                print(f"\nSuccessfully parsed 'categories' into non-empty lists for {len(valid_category_lists)} items.")

                if len(valid_category_lists) > 0:
                    # Flatten the list of lists to count individual category elements
                    all_category_elements = [item for sublist in valid_category_lists for item in sublist]

                    if all_category_elements:
                        category_counts = pd.Series(all_category_elements).value_counts()
                        print("\nTop 10 most frequent category elements:")
                        print(category_counts.head(10))
                        print("\nNumber of unique category elements found after parsing:", len(category_counts))
                    else:
                        print("\nNo category elements found after parsing (all parsed as empty lists?).")
                else:
                    print("\nNo items with valid, non-empty parsed category lists.")
            else:
                 print("No non-null 'categories' values found.")
        else:
             print("'categories' column not found in item metadata.")


        # images
        print("\n--- Analyzing 'images' (All Beauty) ---")
        if 'images' in beauty_items_df.columns:
            null_images_count = beauty_items_df['images'].isnull().sum()
            print(f"Number of items with null 'images': {null_images_count}")
            non_null_images = beauty_items_df['images'].dropna()
            print(f"Number of items with non-null 'images': {len(non_null_images)}")

            if len(non_null_images) > 0:
                print("\nSample non-null 'images' values:")
                sample_images = non_null_images.sample(min(5, len(non_null_images))).tolist()
                for i, img_str in enumerate(sample_images):
                     print(f"Sample {i+1}: {img_str[:100]}... (truncated) (Type: {type(img_str)})")

                # Attempt parsing
                parsed_images = non_null_images.apply(safe_literal_eval)
                # Filter out None results and results that are not lists or are empty lists
                valid_image_lists = parsed_images[parsed_images.apply(lambda x: isinstance(x, list) and len(x) > 0)]
                print(f"\nSuccessfully parsed 'images' into non-empty lists for {len(valid_image_lists)} items.")

                if len(valid_image_lists) > 0:
                    image_counts_per_item = valid_image_lists.apply(len)
                    print("\nSummary statistics for number of images per item (for items with at least one parsed image):")
                    print(image_counts_per_item.describe())

                    print("\nGenerating histogram for number of images per item...")
                    plt.figure(figsize=(12, 6))
                    # Use bins starting from 1
                    bins = np.arange(image_counts_per_item.min(), image_counts_per_item.max() + 2) - 0.5
                    plt.hist(image_counts_per_item, bins=bins, rwidth=0.8)
                    plt.xticks(np.arange(image_counts_per_item.min(), image_counts_per_item.max() + 1))
                    plt.title('Distribution of Number of Images per Item (All Beauty, non-empty image lists)')
                    plt.xlabel('Number of Images')
                    plt.ylabel('Number of Items')
                    plt.grid(axis='y', alpha=0.75)
                    plt.tight_layout()
                    # Save the plot instead of showing
                    save_path = os.path.join(eda_visuals_path, "beauty_image_count_histogram.png")
                    print(f"Saving plot to {save_path}")
                    plt.savefig(save_path)
                    plt.close() # Close the figure


                    # Show an example of parsed image data
                    example_item_index = valid_image_lists.index[0]
                    example_item_id = beauty_items_df.loc[example_item_index, 'item_id']
                    example_original_string = beauty_items_df.loc[example_item_index, 'images']
                    example_parsed_list = valid_image_lists.loc[example_item_index]

                    print(f"\nExample of parsed 'images' data for item_id {example_item_id}:")
                    print(f"Original string: {example_original_string[:200]}... (truncated)")
                    print(f"Parsed data: {example_parsed_list}")
                    print(f"Number of images for this item: {len(example_parsed_list)}")

                else:
                    print("\nNo items with valid, non-empty image lists found after parsing.")
            else:
                 print("No non-null 'images' values found.")
        else:
             print("'images' column not found in item metadata.")


        # price
        print("\n--- Analyzing 'price' (All Beauty) ---")
        if 'price' in beauty_items_df.columns:
            null_prices_count = beauty_items_df['price'].isnull().sum()
            print(f"Number of items with null 'price': {null_prices_count}")
            valid_prices = beauty_items_df['price'].dropna()
            print(f"Number of items with non-null 'price': {len(valid_prices)}")

            if len(valid_prices) > 0:
                print("\nSummary statistics for 'price' (non-null values):")
                print(valid_prices.describe())

                print("\nGenerating price distribution visualizations...")

                # Histogram with log scale on X
                positive_prices = valid_prices[valid_prices > 0]
                if len(positive_prices) > 0:
                     plt.figure(figsize=(14, 7))
                     # Use log scale on Y axis to better see counts of rarer prices
                     plt.hist(positive_prices, bins=100, log=True)
                     plt.xscale('log')
                     plt.title('Histogram of Prices (All Beauty, Positive Prices, Log X-axis)')
                     plt.xlabel('Price (Log Scale)')
                     plt.ylabel('Number of Items (Log Scale)')
                     plt.grid(axis='both', alpha=0.75)
                     plt.tight_layout()
                     # Save the plot
                     save_path = os.path.join(eda_visuals_path, "beauty_price_logx_histogram.png")
                     print(f"Saving plot to {save_path}")
                     plt.savefig(save_path)
                     plt.close() # Close the figure
                else:
                    print("No positive prices found to plot on log scale.")

                # Box plot
                plt.figure(figsize=(10, 6))
                plt.boxplot(valid_prices, vert=False)
                plt.title('Box Plot of Prices (All Beauty)')
                plt.xlabel('Price')
                # Adjust xlim to focus on main distribution, ignoring extreme outliers visually
                if valid_prices.quantile(0.99) > 0:
                     plt.xlim(0, valid_prices.quantile(0.99) * 1.1)
                plt.grid(axis='x', alpha=0.75)
                plt.tight_layout()
                # Save the plot
                save_path = os.path.join(eda_visuals_path, "beauty_price_boxplot.png")
                print(f"Saving plot to {save_path}")
                plt.savefig(save_path)
                plt.close() # Close the figure

            else:
                 print("No valid price data available for visualization.")
        else:
             print("'price' column not found in item metadata.")

        # title
        print("\n--- Analyzing 'title' (All Beauty) ---")
        if 'title' in beauty_items_df.columns:
            null_titles_count = beauty_items_df['title'].isnull().sum()
            print(f"Number of items with null 'title': {null_titles_count}")
            valid_titles = beauty_items_df['title'].dropna()
            print(f"Number of items with non-null 'title': {len(valid_titles)}")

            if len(valid_titles) > 0:
                unique_titles_count = valid_titles.nunique()
                print(f"\nNumber of unique titles: {unique_titles_count}")
                print(f"Percentage of unique titles: {(unique_titles_count / len(valid_titles) * 100):.2f}%")

                print("\nSample 'title' values:")
                sample_titles = valid_titles.sample(min(5, len(valid_titles))).tolist()
                for i, title in enumerate(sample_titles):
                    print(f"Sample {i+1}: {title}")

                # Title length analysis
                title_lengths = valid_titles.str.len()
                print("\nSummary statistics for title length:")
                print(title_lengths.describe())

                print("\nGenerating title length distribution histogram...")
                plt.figure(figsize=(12, 6))
                # Plot up to roughly the 99th percentile to handle potential outliers
                max_len_to_plot = title_lengths.quantile(0.99) if len(title_lengths) > 100 else title_lengths.max() if len(title_lengths)>0 else 0
                if max_len_to_plot > 0:
                    plt.hist(title_lengths, bins=50, range=(0, max_len_to_plot * 1.1))
                    plt.title('Distribution of Title Lengths (All Beauty)')
                    plt.xlabel('Title Length (Number of Characters)')
                    plt.ylabel('Number of Items')
                    plt.grid(axis='y', alpha=0.75)
                    plt.tight_layout()
                    # Save the plot
                    save_path = os.path.join(eda_visuals_path, "beauty_title_length_histogram.png")
                    print(f"Saving plot to {save_path}")
                    plt.savefig(save_path)
                    plt.close() # Close the figure
                else:
                     print("Not enough data to plot title lengths.")


                # Text analysis (word/ngram frequency)
                print("\nPerforming text analysis on titles...")
                all_titles_text = ' '.join(valid_titles.tolist()).lower()
                translator = str.maketrans('', '', string.punctuation)
                cleaned_text = all_titles_text.translate(translator)
                words = cleaned_text.split()

                print(f"Total number of words in all titles: {len(words)}")
                if len(words) > 0:
                    print(f"Number of unique words in all titles: {len(set(words))}")
                    word_counts = collections.Counter(words)
                    print("\nTop 20 most frequent words in titles:")
                    print(word_counts.most_common(20))

                    if len(words) >= 2:
                        bigrams = generate_ngrams(words, 2)
                        bigram_counts = collections.Counter(bigrams)
                        print("\nTop 20 most frequent Bigrams in titles:")
                        print(bigram_counts.most_common(20))

                    if len(words) >= 3:
                         trigrams = generate_ngrams(words, 3)
                         trigram_counts = collections.Counter(trigrams)
                         print("\nTop 20 most frequent Trigrams in titles:")
                         print(trigram_counts.most_common(20))
                else:
                    print("No words found in titles after cleaning.")

            else:
                 print("No valid title data available for analysis.")
        else:
             print("'title' column not found in item metadata.")


        # description
        print("\n--- Analyzing 'description' (All Beauty) ---")
        if 'description' in beauty_items_df.columns:
            null_description_count = beauty_items_df['description'].isnull().sum()
            print(f"Number of items with null 'description': {null_description_count}")
            non_null_description = beauty_items_df['description'].dropna()
            print(f"Number of items with non-null 'description': {len(non_null_description)}")

            if len(non_null_description) > 0:
                print("\nSample non-null 'description' values:")
                sample_description = non_null_description.sample(min(5, len(non_null_description))).tolist()
                for i, desc in enumerate(sample_description):
                     print(f"Sample {i+1}: {desc[:200]}... (truncated) (Type: {type(desc)})")

                # Check for substantive descriptions (not null and not the string '[]')
                description_strings = beauty_items_df['description'].astype(str).str.strip()
                substantive_descriptions_mask = (beauty_items_df['description'].notna()) & \
                                                (description_strings != '[]') & \
                                                (~description_strings.str.lower().isin(['nan', 'none'])) # Handle potential 'nan' strings
                substantive_descriptions = beauty_items_df.loc[substantive_descriptions_mask, 'description']

                print(f"\nNumber of items with a SUBSTANTIVE description: {len(substantive_descriptions)}")

                # Analyze substantive descriptions
                if len(substantive_descriptions) > 0:
                    description_lengths = substantive_descriptions.str.len()
                    print("\nSummary statistics for description length (Substantive Descriptions):")
                    print(description_lengths.describe())

                    print("\nGenerating description length distribution histogram (Substantive Descriptions)...")
                    plt.figure(figsize=(14, 7))
                    max_len_to_plot = description_lengths.quantile(0.99) if len(description_lengths) > 100 else description_lengths.max() if len(description_lengths) > 0 else 0
                    if max_len_to_plot > 0:
                        plt.hist(description_lengths, bins=100, range=(0, max_len_to_plot * 1.1))
                        plt.title('Distribution of Description Lengths (All Beauty, Substantive Descriptions)')
                        plt.xlabel('Description Length (Number of Characters)')
                        plt.ylabel('Number of Items')
                        plt.grid(axis='y', alpha=0.75)
                        plt.tight_layout()
                        # Save the plot
                        save_path = os.path.join(eda_visuals_path, "beauty_description_length_histogram.png")
                        print(f"Saving plot to {save_path}")
                        plt.savefig(save_path)
                        plt.close() # Close the figure
                    else:
                         print("Not enough substantive descriptions to plot lengths.")


                    print("\nPerforming text analysis on substantive descriptions...")
                    all_descriptions_text = ' '.join(substantive_descriptions.tolist()).lower()
                    translator = str.maketrans('', '', string.punctuation)
                    cleaned_text = all_descriptions_text.translate(translator)
                    words = cleaned_text.split()

                    print(f"Total number of words in substantive descriptions: {len(words)}")
                    if len(words) > 0:
                         print(f"Number of unique words in substantive descriptions: {len(set(words))}")
                         word_counts = collections.Counter(words)
                         print("\nTop 20 most frequent words in substantive descriptions:")
                         print(word_counts.most_common(20))
                    else:
                         print("No words found in substantive descriptions after cleaning.")

                else:
                    print("No 'All Beauty' items with SUBSTANTIVE descriptions found for further analysis.")
            else:
                 print("No non-null 'description' values found.")
        else:
             print("'description' column not found in item metadata.")


        # details
        print("\n--- Analyzing 'details' (All Beauty) ---")
        if 'details' in beauty_items_df.columns:
            null_details_count = beauty_items_df['details'].isnull().sum()
            print(f"Number of items with null 'details': {null_details_count}")
            non_null_details = beauty_items_df['details'].dropna()
            print(f"Number of items with non-null 'details': {len(non_null_details)}")

            if len(non_null_details) > 0:
                print("\nSample non-null 'details' values:")
                sample_details = non_null_details.sample(min(5, len(non_null_details))).tolist()
                for i, detail_str in enumerate(sample_details):
                     print(f"Sample {i+1}: {detail_str[:200]}... (truncated) (Type: {type(detail_str)})")

                # Attempt parsing into dictionaries
                parsed_details = non_null_details.apply(safe_literal_eval)
                 # Filter out None results and results that are not dictionaries or are empty dictionaries
                valid_parsed_details = parsed_details[parsed_details.apply(lambda x: isinstance(x, dict) and len(x) > 0)]

                print(f"\nSuccessfully parsed 'details' into non-empty dictionaries for {len(valid_parsed_details)} items.")

                if len(valid_parsed_details) > 0:
                    all_keys = []
                    common_key_value_types = collections.defaultdict(collections.Counter)

                    for detail_dict in valid_parsed_details:
                        if isinstance(detail_dict, dict): # Double check type after filtering
                            all_keys.extend(detail_dict.keys())
                            for key, value in detail_dict.items():
                                common_key_value_types[key][type(value).__name__] += 1

                    key_counts = collections.Counter(all_keys)
                    print(f"\nTotal number of detail entries found across all parsed dicts: {len(all_keys)}")
                    print(f"Number of unique detail keys found: {len(key_counts)}")

                    print("\nTop 20 most frequent detail keys and their counts:")
                    for key, count in key_counts.most_common(20):
                         print(f"  '{key}': {count}")

                    # Sample values for the most common keys
                    print("\nSample values for the Top 10 most frequent detail keys:")
                    top_10_keys = [key for key, count in key_counts.most_common(10)]

                    for key in top_10_keys:
                         print(f"\n--- Sample values for key: '{key}' ---")
                         # Find up to 3 items among those successfully parsed that have this key and are in the original beauty_items_df index
                         items_with_this_key_index = valid_parsed_details[valid_parsed_details.apply(lambda d: key in d)].index.intersection(beauty_items_df.index).tolist()

                         if items_with_this_key_index:
                              sample_indices = random.sample(items_with_this_key_index, min(3, len(items_with_this_key_index)))
                              sample_parsed_details = parsed_details.loc[sample_indices]

                              for index in sample_indices:
                                   item_id = beauty_items_df.loc[index, 'item_id']
                                   detail_dict = sample_parsed_details.loc[index]
                                   value = detail_dict.get(key)
                                   # Wrap long values for better printing
                                   wrapped_value = textwrap.fill(str(value), width=80, subsequent_indent="    ")
                                   print(f"  Item ID {item_id} - Value: '{wrapped_value}' (Type: {type(value).__name__})")
                         else:
                              print("  (No sample items found for this key)")

                    # Analyze Value Types for Top Keys
                    print("\nValue type distribution for Top 10 most frequent detail keys:")
                    for key in top_10_keys:
                         if key in common_key_value_types:
                             print(f"  '{key}': {common_key_value_types[key].most_common()}")
                         else:
                              print(f"  '{key}': (No type data collected)") # Should not happen for top keys if logic is right


                else:
                    print("\nNo 'All Beauty' items successfully parsed into non-empty dictionaries from 'details'.")
            else:
                 print("No non-null 'details' values found.")
        else:
             print("'details' column not found in item metadata.")


print("\n--- Exploratory Data Analysis Complete ---")