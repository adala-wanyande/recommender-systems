# config.py

import os

# --- Configuration ---
# Define paths using relative paths
# Assume the script is run from the project root directory,
# or adjust relative paths accordingly.

# Input Data Paths
# train.csv is in ../data/
TRAIN_CSV_PATH = os.path.join('../', 'data', 'train.csv')
TEST_CSV_PATH = '../data/test.csv' # Add this line
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
X_ITEMS_REDUCED_NPY_PATH = os.path.join(INTERMEDIATE_DATA_PATH, "X_items_reduced.npy") # Changed to npz if it's sparse
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

# Specific model filenames (ensure these match your saved files)
RF_MODEL_FILENAME = 'random_forest_model.joblib'
LGBM_MODEL_FILENAME = 'lightgbm_model.joblib'


# Model configuration constants
SEED = 42 # For reproducibility
NEGATIVE_SAMPLE_RATIO = 1 # Number of negative samples per positive sample (1:1 ratio)
ITEM_SVD_N_COMPONENTS = 20 # Number of components for item feature SVD (from Cell 3)
USER_CAT_SVD_N_COMPONENTS = 50 # Number of components for user categorical feature SVD (from Cell 5)
STACKING_BATCH_SIZE = 50000 # Batch size for processing and saving training pairs (from Cell 5)
TRAIN_VALIDATION_SPLIT_SIZE = 0.2 # Size of the validation set (e.g., 0.2 for 20%)


# --- Model Hyperparameters ---
# Define hyperparameters for each model type
# Keep existing model configs for training scripts
MODEL_CONFIGS = {
    'LogisticRegression': {
        'solver': 'liblinear',
        'C': 1.0,
        'random_state': SEED,
        'n_jobs': -1,
    },
    'RandomForestClassifier': {
        'n_estimators': 200,
        'max_depth': None,
        'random_state': SEED,
        'n_jobs': -1,
    },
    'LGBMClassifier': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'seed': SEED,
        'n_jobs': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'early_stopping_rounds': 20,
    },
     'XGBClassifier': {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'seed': SEED,
        'n_jobs': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'use_label_encoder': False,
        'early_stopping_rounds': 20,
    },
    'CatBoostClassifier': {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'random_seed': SEED,
        'verbose': 100,
        'l2_leaf_reg': 3,
        'early_stopping_rounds': 20,
    },
    'MLPClassifier': {
        'hidden_layer_sizes': (128, 64),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'max_iter': 500,
        'random_state': SEED,
        'learning_rate': 'adaptive',
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20,
        'verbose': True,
    },
    'SVC': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': SEED,
        'verbose': True,
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
ITEM_TEXT_COLS = ['title']
ITEM_NOMINAL_CATEGORICAL_COLS = ['store']
ITEM_NUMERICAL_COLS = ['number_of_images', 'weighted_rating', 'price_category_mapped']
ITEM_BINARY_COLS = ['has_video', 'has_substantive_features', 'has_substantive_description']


# Recommendation specific configurations for blending script
RECOMMENDATION_COUNT = 10 # Total number of items to recommend per user
NUM_POPULAR_ITEMS_TO_BLEND = 5 # Target number of popular items to include
NUM_RANDOM_ITEMS_TO_BLEND = 0 # Target number of random items to include
# Note: The actual number of popular/random items included may be less than the target
# if not enough unique, unseen items are available in these categories.
# The remaining slots up to RECOMMENDATION_COUNT will be filled by model recommendations.
# The blend script will also have internal targets for model recommendations (e.g., 3 from RF, 3 from LGBM).