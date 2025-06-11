# Hybrid Recommendation System for E-commerce (Beauty Category)

This project implements a hybrid recommendation system focused on user-item interactions and item metadata, specifically targeting the 'All Beauty' category. The pipeline involves data cleaning, feature extraction using techniques like SVD, training gradient boosting models (LightGBM, CatBoost), and a simple logistic regression and random forest as baselines, and finally blending model predictions with popularity and randomness for generating recommendations.

This README provides a step-by-step guide to replicate the results obtained from this codebase.

## Project Structure

The expected structure of the project directory is as follows:
├── data/
│ ├── raw/ # Optional: Placeholder for original downloaded files
│ ├── item_meta.csv # <--- REQUIRED INPUT: Raw item metadata
│ ├── train.csv # <--- REQUIRED INPUT: Training interaction data (user_id, item_id, timestamp, interaction)
│ ├── test.csv # <--- REQUIRED INPUT: Test interaction data (user_id, item_id, timestamp) - Used for user identification for prediction
│ ├── sample_submission.csv # <--- REQUIRED INPUT: Defines users for prediction and submission format
│ ├── dataset/ # Created by construct_beauty_dataset.py
│ │ └── clean_beauty_item_meta_with_details.csv # Cleaned item metadata
│ └── intermediate/ # Created by feature_extraction/main.py
│ ├── X_train_hybrid.npz # Sparse training feature matrix
│ ├── y_train.csv # Training labels
│ ├── item_id_to_index.json # Mapping item IDs to matrix indices
│ ├── index_to_item_id.json # Mapping matrix indices back to item IDs
│ ├── item_numerical_cols_for_scaling.json # List of numerical item columns used for scaling
│ ├── item_categorical_cols_for_encoding.json # List of categorical item columns used for encoding
│ ├── item_binary_cols.json # List of binary item columns
│ ├── item_original_numerical_cols.json # Original numerical item columns
│ ├── item_original_nominal_categorical_cols.json # Original nominal categorical item columns
│ ├── item_details_cols.json # List of detail keys extracted as columns
│ ├── item_scaler.pkl # Fitted StandardScaler for item numerical features
│ ├── item_encoder.pkl # Fitted OneHotEncoder for item categorical features
│ ├── user_features.csv # Extracted user features
│ ├── user_num_cols_for_scaling.json # List of numerical user columns used for scaling
│ ├── user_cat_cols_for_encoding.json # List of categorical user columns used for encoding
│ ├── user_numerical_scaler.pkl # Fitted StandardScaler for user numerical features
│ ├── user_categorical_encoder.pkl # Fitted OneHotEncoder for user categorical features
│ └── user_categorical_svd.pkl # Fitted TruncatedSVD for user categorical features (if components > 0)
│ └── X_items_reduced.npz # Processed item features (sparse, recommended)
├── visuals/ # Created by eda.py
│ └── eda/
│ └── *.png # Various plots from EDA
├── models/ # Created by training scripts
│ ├── catboost_model.joblib # Trained CatBoost model
│ ├── lightgbm_model.joblib # Trained LightGBM model
│ ├── logistic_regression_model.joblib # Trained Logistic Regression model
│ └── random_forest_model.joblib # Trained Random Forest model
├── submissions/ # Created by recommendation scripts
│ ├── *.csv # Individual model submissions (optional)
│ └── random_forest_model_lightgbm_model_popular_random_blended_submission.csv # FINAL SUBMISSION
├── code/ # <--- All Python code is contained here
│ ├── feature_extraction/
│ │ ├── main.py # Orchestrates feature extraction pipeline
│ │ ├── data_loader.py # Loads and merges data
│ │ ├── pair_generator.py # Generates user-item pairs and user features
│ │ ├── item_processor.py # Processes item features
│ │ ├── user_processor.py # Processes user features
│ │ └── hybrid_matrix_assembler.py # Assembles hybrid feature matrix
│ ├── train/
│ │ ├── train_catboost.py
│ │ ├── train_lightgbm.py
│ │ ├── train_logistic_regression.py
│ │ └── train_random_forest.py
│ ├── recommend/
│ │ ├── generate_multiple_submissions.py # Generates submissions for each individual trained model
│ │ └── blended_recommendations.py # Generates the final multi-model blended submission
│ ├── config.py # IMPORTANT: Project configuration file
│ ├── construct_beauty_dataset.py # First script: Cleans item metadata
│ └── eda.py # Optional: Performs EDA
└── README.md # This file

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.7+**: The code is written in Python.
2.  **Required Libraries**: Install the necessary Python packages. It's recommended to use a virtual environment. Create a `requirements.txt` file (if not already present) with the following content:

    ```
    pandas
    numpy
    scipy
    scikit-learn
    catboost
    lightgbm
    matplotlib
    joblib
    ```
    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note on CatBoost/LightGBM:* Depending on your OS and setup, these libraries might require additional system dependencies. Refer to their official installation guides if you encounter issues (e.g., `catboost` documentation for GPU support, `lightgbm` documentation for system requirements).

## Data

Obtain the following raw data files and place them directly in the `data/` directory:

*   `item_meta.csv`
*   `train.csv`
*   `test.csv`
*   `sample_submission.csv`

Make sure these files are in CSV format and match the expected column names (`user_id`, `item_id`, `timestamp`, `interaction`, `main_category`, `images`, `videos`, `price`, `features`, `description`, `details`, etc. as used in the scripts). The column names `user_id`, `item_id`, `timestamp`, and `interaction` are defined in `config.py`.

## Configuration (`config.py`)

The `config.py` file contains essential configuration for paths, hyperparameters, and model settings. **It is crucial to review and potentially modify this file** before running the scripts to match your environment and desired experimental setup.

Key parameters in `config.py` to check:

*   **Paths:** Ensure `DATA_PATH`, `TRAIN_CSV_PATH`, `TEST_CSV_PATH`, `ITEM_META_CSV_PATH`, `SAMPLE_SUBMISSION_CSV_PATH`, `CLEANED_METADATA_CSV_FILENAME`, `OUTPUT_DATAFRAMES_PATH`, `INTERMEDIATE_DATA_PATH`, `TRAINED_MODELS_PATH`, `SUBMISSION_OUTPUT_DIR`, `VISUALS_BASE_PATH` are correctly set up relative to your project root and match where you place/expect files.
*   **Column Names:** Verify `USER_ID_COL`, `ITEM_ID_COL`, `TIMESTAMP_COL`, `INTERACTION_COL` match your input data.
*   **Feature Extraction Parameters:** Adjust `NEGATIVE_SAMPLE_RATIO`, `ITEM_SVD_N_COMPONENTS`, `USER_CAT_SVD_N_COMPONENTS`, `STACKING_BATCH_SIZE`, `BATCH_TEMP_DIR`. The batch size is critical for memory management.
*   **Model Training Parameters:** Modify `TRAIN_VALIDATION_SPLIT_SIZE` and the `MODEL_CONFIGS` dictionary to tune hyperparameters for CatBoost, LightGBM, Logistic Regression, and Random Forest.
*   **Recommendation Parameters:** Set `RECOMMENDATION_COUNT` (number of items per user) and blending counts (`NUM_POPULAR_ITEMS_TO_BLEND`, `NUM_RANDOM_ITEMS_TO_BLEND`). Note that the blending logic in `blended_recommendations.py` also uses internal counts (`NUM_RF_ITEMS_TO_BLEND`, `NUM_LGBM_ITEMS_TO_BLEND`) which you might also want to adjust within that script or move to config.
*   **Model Filenames:** Ensure `RF_MODEL_FILENAME` and `LGBM_MODEL_FILENAME` match the filenames used when saving the respective models in `train/`.

## Replication Steps

Follow these steps sequentially to run the pipeline and replicate the blended recommendation results:

### Step 1: Clean and Preprocess Item Metadata

This script loads the raw `item_meta.csv`, filters for the 'All Beauty' category, extracts and transforms specific features (like images count, has video, price category, substantive features/description, weighted rating), extracts details keys, and saves the resulting cleaned dataframe.

*   **Review `construct_beauty_dataset.py`:** Confirm `INPUT_CSV_FILENAME` matches your raw metadata file name.
*   **Run the script:**
    ```bash
    python code/construct_beauty_dataset.py
    ```
*   **Expected Output:** `data/dataset/clean_beauty_item_meta_with_details.csv`

### Step 2: Perform Exploratory Data Analysis (Optional)

This script analyzes the interaction data and the cleaned item metadata, printing summary statistics and generating visualizations. It's helpful for understanding the dataset but not strictly required to run the rest of the pipeline.

*   **Run the script:**
    ```bash
    python code/eda.py
    ```
*   **Expected Output:** Various statistics printed to the console and image files saved in the `visuals/eda/` directory.

### Step 3: Feature Extraction and Training Data Preparation Pipeline

This is the main feature engineering and data preparation step. It loads the cleaned metadata and interaction data, generates positive and negative user-item pairs, computes user features based on historical interactions, processes item features, fits necessary transformers (scalers, encoders, SVD), and finally assembles the hybrid feature matrix (`X_train_hybrid`) and target vector (`y_train`) for training.

*   **Review `config.py`:** Ensure paths and parameters related to feature extraction and intermediate outputs are correct. Pay close attention to `STACKING_BATCH_SIZE`.
*   **Run the script:**
    ```bash
    python code/feature_extraction/main.py
    ```
*   **Expected Output:**
    *   `data/intermediate/X_train_hybrid.npz` (sparse matrix)
    *   `data/intermediate/y_train.csv`
    *   `data/intermediate/X_items_reduced.npz` (processed item features, hopefully sparse for memory efficiency)
    *   `data/intermediate/user_features.csv` (processed user features)
    *   Multiple `.pkl` files (transformers) and `.json` files (mappings, column lists) in `data/intermediate/`. These are essential for prediction.

### Step 4: Train Models

Train the individual models using the prepared `X_train_hybrid` and `y_train` data. The final blended submission script (`blended_recommendations.py`) uses `random_forest_model.joblib` and `lightgbm_model.joblib` by default. You should run at least these two training scripts.

*   **Review `config.py`:** Adjust `MODEL_CONFIGS` as needed for hyperparameters. Ensure `TRAIN_VALIDATION_SPLIT_SIZE` is set for evaluation during training.
*   **Run the desired training scripts:**
    ```bash
    python code/train/train_random_forests.py
    python code/train/train_lightgbm.py

    # Optional (if you want to train/use these models):
    # python code/train/train_logistic_regression.py
    # python code/train/train_catboost.py
    ```
*   **Expected Output:** Trained model files (e.g., `random_forest_model.joblib`, `lightgbm_model.joblib`) saved in the `models/` directory. Ensure the filenames match the `RF_MODEL_FILENAME` and `LGBM_MODEL_FILENAME` settings in `config.py`.

### Step 5: Generate Individual Model Submissions (Optional)

This script generates a submission file for each trained model found in the `models/` directory, blending its predictions with popular items based on `config.py`. This step is useful for evaluating individual model performance but is not strictly necessary for the final blended submission.

*   **Review `config.py`:** Check `RECOMMENDATION_COUNT` and `NUM_POPULAR_ITEMS_TO_BLEND`.
*   **Run the script:**
    ```bash
    python code/recommend/generate_multiple_submissions.py
    ```
*   **Expected Output:** Individual submission files (e.g., `random_forest_model_submission.csv`, `lightgbm_model_submission.csv`) created in the `submissions/` directory.

### Step 6: Generate Multi-Model Blended Submission

This script loads specific trained models (Random Forest and LightGBM by default, based on `config.py` filenames), the trained transformers, the processed item features, and item popularity information. It generates predictions for all relevant user-item pairs for the users in the `sample_submission.csv`, blends the scores from the two models with popular and random items, ranks the results, and generates the final submission file in the required format.

*   **Review `config.py`:** Ensure `RF_MODEL_FILENAME`, `LGBM_MODEL_FILENAME`, `RECOMMENDATION_COUNT`, `NUM_POPULAR_ITEMS_TO_BLEND`, and `NUM_RANDOM_ITEMS_TO_BLEND` are set. Also, check the hardcoded blending counts for RF and LGBM within `recommend/blended_recommendations.py` (`NUM_RF_ITEMS_TO_BLEND`, `NUM_LGBM_ITEMS_TO_BLEND`) if you want to modify the weights.
*   **Run the script:**
    ```bash
    python code/recommend/blended_recommendations.py
    ```
*   **Expected Output:** The final submission file, named `random_forest_model_lightgbm_model_popular_random_blended_submission.csv` (the specific name is constructed based on the loaded model filenames), saved in the `submissions/` directory. This file should contain `user_id` and `item_id` columns, where `item_id` is a comma-separated string of recommended item IDs.

## Results

The main deliverable is the file `submissions/random_forest_model_lightgbm_model_popular_random_blended_submission.csv`, which contains the final set of 10 recommended items for each user listed in the `sample_submission.csv`.

Intermediate outputs like cleaned data, processed features, trained transformers, individual model submissions, and EDA plots are also generated and saved in their respective directories as described above.

## Customization and Further Experimentation

*   **Configuration:** Modify `config.py` to experiment with different hyperparameters for feature extraction (SVD components, negative sampling) and models.
*   **Features:** Modify the logic in `feature_extraction/item_processor.py` and `feature_extraction/user_processor.py` to include different features or transformation methods. Remember to update the saving/loading of column lists and transformers accordingly.
*   **Models:** Train and evaluate other models by creating new training scripts in the `train/` directory.
*   **Blending Strategy:** The blending logic in `recommend/blended_recommendations.py` is a simple weighted blend (by count). You can modify this script to implement different blending strategies (e.g., weighted average of scores, ranking aggregation) or include more models. Consider making the individual model blending counts (`NUM_RF_ITEMS_TO_BLEND`, `NUM_LGBM_ITEMS_TO_BLEND`) configurable in `config.py`.
*   **Memory Optimization:** If encountering `MemoryError`, focus on reducing `STACKING_BATCH_SIZE`, decreasing SVD components, and ensuring sparse matrices (`.npz`) are used where possible (e.g., for `X_train_hybrid` and `X_items_reduced`).

## Troubleshooting Common Issues

*   **`FileNotFoundError`:** This usually means a script couldn't find an input file or a file generated by a previous step.
    *   Ensure the raw data files are in `data/`.
    *   Verify that the output files from the *previous* step in the pipeline were successfully created in their designated directories (`data/dataset/`, `data/intermediate/`, `models/`).
    *   Check the file paths and filenames in `config.py` and `construct_beauty_dataset.py` for typos or mismatches.
*   **`MemoryError`:** Feature extraction and prediction can be memory-intensive.
    *   **Crucially, ensure `X_items_reduced` is saved as a sparse `.npz` file.** The current `item_processor.py` uses `np.save` which creates a dense `.npy`. Modify `item_processor.py` to use `scipy.sparse.save_npz` and update the loading in prediction scripts (`generate_multiple_submissions.py`, `blended_recommendations.py`) to `scipy.sparse.load_npz` for `X_items_reduced_path_npz`.
    *   Reduce the `STACKING_BATCH_SIZE` in `config.py`. This is the most direct control over memory usage during batch processing.
    *   Reduce `ITEM_SVD_N_COMPONENTS` and `USER_CAT_SVD_N_COMPONENTS` if SVD is enabled and memory is an issue.
    *   Ensure intermediate batch arrays are explicitly deleted (`del array_name`) and `gc.collect()` is called periodically in batch processing loops (this is already present in the provided code, but worth reviewing).
*   **`ValueError` or Data Issues During Processing/Training:**
    *   Check the console output from the scripts. Errors often indicate issues with column names, data types, or unexpected data formats (e.g., non-numeric values in expected numeric columns, strings that fail `ast.literal_eval`).
    *   Inspect the intermediate `.csv` and `.json` files (e.g., `clean_beauty_item_meta_with_details.csv`, `user_features.csv`) to see if they look as expected.
    *   The `eda.py` script can help identify data quality issues early on.
*   **Transformer/Model Loading Errors:** Ensure the `.pkl` and `.joblib` files in `data/intermediate/` and `models/` correspond to the transformers and models saved by the previous steps and that their filenames match what the prediction scripts are trying to load (`config.py`).
*   **`y_train.csv` Loading Errors in Training Scripts:** The training scripts contain specific robust logic for loading `y_train.csv` generated by `feature_extraction/main.py`. If this still fails, inspect the `y_train.csv` file to understand its exact structure (e.g., does it have an index column, is the label in the first column, is there a header?).

Remember to execute the scripts in the specified order to ensure dependencies are met. Monitor the console output for progress updates, warnings, and error messages.