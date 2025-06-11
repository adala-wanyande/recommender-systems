# config.py

import os

# --- Configuration ---
# Define paths using relative paths
# Assume the script is run from the project root directory,
# or adjust relative paths accordingly.

# Input Data Paths
# train.csv is in ../data/
TRAIN_CSV_PATH = os.path.join('../', 'data', 'train.csv')
# clean_beauty_item_meta_with_details.csv is in ../data/dataset/
CLEANED_METADATA_CSV_PATH = os.path.join('../', 'data', 'dataset', 'clean_beauty_item_meta_with_details.csv')

# Kaggle Submission Files
SAMPLE_SUBMISSION_CSV_PATH = os.path.join('../', 'data', 'sample_submission.csv') # Path to sample submission file
SUBMISSION_OUTPUT_DIR = os.path.join('..', 'submissions') # Directory to save generated submission files

# Output Dataframes Path (Final X_train, y_train)
# Output dataframes should go into ../data/dataframes/
OUTPUT_DATAFRAMES_PATH = os.path.join('..', 'data', 'dataframes')

# Intermediate Data Path (for generated features, transformers, temp batches)
# Let's create a dedicated folder for intermediate processing artifacts.
INTERMEDIATE_DATA_PATH = os.path.join('..', 'data', 'intermediate')

# Define paths for saving/loading intermediate artifacts within INTERMEDIATE_DATA_PATH
USER_FEATURES_CSV_PATH = os.path.join(INTERMEDIATE_DATA_PATH, 'user_features.csv')
TRAIN_PAIRS_CSV_PATH = os.path.join(INTERMEDIATE_DATA_PATH, 'train_pairs.csv')

# Item feature processing artifacts (from Cell 3 logic)
X_ITEMS_REDUCED_NPY_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "X_items_reduced.npz") # Changed to npz if it's sparse
ITEM_ID_TO_INDEX_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "item_id_to_index.json")
INDEX_TO_ITEM_ID_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "index_to_item_id.json")
ITEM_SCALER_PKL_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "item_scaler.pkl")
ITEM_ENCODER_PKL_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "item_encoder.pkl")
ITEM_NUM_COLS_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "item_numerical_cols_for_scaling.json")
ITEM_CAT_COLS_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "item_categorical_cols_for_encoding.json")
ITEM_BINARY_COLS_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "item_binary_cols.json")
ITEM_ORIGINAL_NUM_COLS_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "item_numerical_cols.json")
ITEM_ORIGINAL_NOM_CAT_COLS_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "item_nominal_categorical_cols.json")
ITEM_DETAILS_COLS_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "item_details_cols.json")

# User feature processing artifacts (from Cell 5 logic)
USER_NUMERICAL_SCALER_PKL_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "user_numerical_scaler.pkl")
USER_CATEGORICAL_ENCODER_PKL_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "user_categorical_encoder.pkl")
USER_CATEGORICAL_SVD_PKL_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "user_categorical_svd.pkl")
USER_NUM_COLS_FOR_SCALING_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "user_numerical_cols_for_scaling_final.json") # Using _final suffix for clarity
USER_CAT_COLS_FOR_ENCODING_JSON_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "user_categorical_cols_for_encoding_final.json") # Using _final suffix for clarity


# Final output paths (X_train, y_train) - Based on OUTPUT_DATAFRAMES_PATH
X_TRAIN_HYBRID_PATH = os.path.join(OUTPUT_DATAFRAMES_PATH, "X_train_hybrid.npz") # Using .npz for sparse matrix
Y_TRAIN_PATH = os.path.join(OUTPUT_DATAFRAMES_PATH, "y_train.csv") # Using .npy as it's a numpy array

# Temporary directory for batch processing - within INTERMEDIATE_DATA_PATH
BATCH_TEMP_DIR = os.path.join(INTERMEDIATE_DATA_PATH, "temp_batches_train_hybrid") # Specific name for hybrid batches

# Directory where trained models are saved
TRAINED_MODELS_PATH = os.path.join('..', 'trained_models') # Assuming models are saved here

# Model configuration constants
SEED = 42 # For reproducibility
NEGATIVE_SAMPLE_RATIO = 1 # Number of negative samples per positive sample (1:1 ratio)
ITEM_SVD_N_COMPONENTS = 20 # Number of components for item feature SVD (from Cell 3)
USER_CAT_SVD_N_COMPONENTS = 50 # Number of components for user categorical feature SVD (from Cell 5)
STACKING_BATCH_SIZE = 50000 # Batch size for processing and saving training pairs (from Cell 5)
TRAIN_VALIDATION_SPLIT_SIZE = 0.2 # Size of the validation set (e.g., 0.2 for 20%)


# --- Model Hyperparameters ---
# Define hyperparameters for each model type
MODEL_CONFIGS = {
    'LogisticRegression': {
        'solver': 'liblinear',
        'C': 1.0,
        'random_state': SEED,
        'n_jobs': -1,
    },
    'RandomForestClassifier': {
        'n_estimators': 500, # Increased estimators slightly from 100
        'max_depth': 10, # No max depth by default
        'random_state': SEED,
        'n_jobs': -1,
    },
    'LGBMClassifier': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 500, # More estimators, rely on early stopping
        'learning_rate': 0.05,
        'num_leaves': 31,
        'seed': SEED,
        'n_jobs': -1,
        'colsample_bytree': 0.8, # Feature fractionalization
        'subsample': 0.8, # Data fractionalization
        'reg_alpha': 0.1, # L1 regularization
        'reg_lambda': 0.1, # L2 regularization
        'early_stopping_rounds': 20, # Early stopping patience
    },
     'XGBClassifier': {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 500, # More estimators, rely on early stopping
        'learning_rate': 0.05,
        'max_depth': 6,
        'seed': SEED,
        'n_jobs': -1,
        'colsample_bytree': 0.8, # Feature fractionalization
        'subsample': 0.8, # Data fractionalization
        'reg_alpha': 0.1, # L1 regularization
        'reg_lambda': 0.1, # L2 regularization
        'use_label_encoder': False, # Suppress warning
        'early_stopping_rounds': 20, # Early stopping patience
    },
    'CatBoostClassifier': {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 500, # Equivalent to n_estimators
        'learning_rate': 0.05,
        'depth': 6,
        'random_seed': SEED,
        'verbose': 100, # Set to > 0 to see progress, 0 for silent
        'l2_leaf_reg': 3, # L2 regularization
        'early_stopping_rounds': 20, # Early stopping patience
    },
    'MLPClassifier': {
        'hidden_layer_sizes': (128, 64), # Example: two layers, 128 then 64 neurons
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001, # L2 penalty
        'max_iter': 500, # Maximum number of epochs
        'random_state': SEED,
        'learning_rate': 'adaptive',
        'early_stopping': True, # Use validation set for early stopping
        'validation_fraction': 0.1, # Fraction of training data for validation
        'n_iter_no_change': 20, # Stop if validation score doesn't improve for this many iterations
        'verbose': True, # Show training progress
    },
    'SVC': {
        'C': 1.0, # Regularization parameter
        'kernel': 'rbf', # Kernel type
        'gamma': 'scale', # Kernel coefficient
        'probability': True, # Enable probability estimates (required for predict_proba)
        'random_state': SEED,
        'verbose': True, # Show progress (can be very slow)
        # WARNING: SVC is NOT RECOMMENDED for large datasets due to scalability issues (O(n^2) or O(n^3))
        # Use LinearSVC or rethink approach for large data. Keeping for completeness but with caution.
    }
}


# --- Column Definitions ---
# Define expected column names for clarity and consistency
USER_ID_COL = 'user_id'
ITEM_ID_COL = 'item_id'
TIMESTAMP_COL = 'timestamp'
INTERACTION_COL = 'interaction'

# Item feature column types - used for processing steps
# These should align with the columns *expected* after metadata cleaning
ITEM_TEXT_COLS = ['title'] # Assuming 'title' is the primary text column
ITEM_NOMINAL_CATEGORICAL_COLS = ['store']
ITEM_NUMERICAL_COLS = ['number_of_images', 'weighted_rating', 'price_category_mapped']
ITEM_BINARY_COLS = ['has_video', 'has_substantive_features', 'has_substantive_description']
# Note: details_ cols are identified dynamically based on prefix in the metadata loading phase
# ITEM_DETAILS_COLS_PREFIX = 'details_' # Could define prefix here if needed

    # 'RandomForestClassifier': { Original random forest parameters
    #     'n_estimators': 200, # Increased estimators slightly from 100
    #     'max_depth': None, # No max depth by default
    #     'random_state': SEED,
    #     'n_jobs': -1,
    # },