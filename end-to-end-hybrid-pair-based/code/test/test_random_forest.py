# test_random_forest.py

import pandas as pd
import numpy as np
import os
import time
import joblib
import scipy.sparse as sp
import traceback # Import traceback for debugging

import config

print("--- Testing Random Forest Model ---")
start_time = time.time()

# --- Configuration and Paths ---
# Path to the prepared test features (output of prepare_test_data.py)
X_test_hybrid_path = config.X_TEST_HYBRID_PATH
# Path to the trained model
model_path = os.path.join(config.TRAINED_MODELS_PATH, 'random_forest_model.joblib')
# Path to the original raw test data (needed to add predictions back for context)
raw_test_data_path = config.TEST_CSV_PATH
# Output path for saving the original test data with predictions
output_predictions_path = os.path.join(config.OUTPUT_DATAFRAMES_PATH, 'test_predictions_rf.csv') # Define a path for test results

# Ensure output directory for predictions exists
os.makedirs(config.OUTPUT_DATAFRAMES_PATH, exist_ok=True)


print(f"Loading test features from: {X_test_hybrid_path}")
print(f"Loading model from: {model_path}")
print(f"Loading raw test data from: {raw_test_data_path}")
print(f"Saving predictions to: {output_predictions_path}")
print("-" * 50)

# --- Load Data and Model ---
try:
    # Load the prepared test feature matrix
    X_test_hybrid = sp.load_npz(X_test_hybrid_path)
    print(f"X_test_hybrid loaded successfully. Shape: {X_test_hybrid.shape}")

    # Load the original raw test data to merge predictions later
    test_df_original = pd.read_csv(raw_test_data_path)
    print(f"Raw test data loaded successfully. Shape: {test_df_original.shape}")

    # Basic check to ensure row counts match (they should if prepare_test_data processed all rows)
    if X_test_hybrid.shape[0] != test_df_original.shape[0]:
        raise ValueError(f"Row count mismatch between X_test_hybrid ({X_test_hybrid.shape[0]}) and raw test data ({test_df_original.shape[0]}).")

    # Load the trained Random Forest model
    model = joblib.load(model_path)
    print("Model loaded successfully.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Required file not found: {e}")
    exit()
except ValueError as e:
    print(f"FATAL ERROR: Data shape mismatch: {e}")
    traceback.print_exc()
    exit()
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during loading: {e}")
    traceback.print_exc()
    exit()

# --- Predict on Test Data ---
print("\nGenerating predictions on test data...")
try:
    if X_test_hybrid.shape[0] > 0:
        # Predict probabilities for the positive class (class 1)
        # model.predict_proba returns an array where each row is [prob_class_0, prob_class_1]
        test_predictions_proba = model.predict_proba(X_test_hybrid)[:, 1]
        print(f"Predictions generated. Number of predictions: {len(test_predictions_proba)}")
    else:
        print("No test data to predict on. X_test_hybrid is empty.")
        test_predictions_proba = np.array([])

except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during prediction: {e}")
    traceback.print_exc()
    exit()

# --- Calculate Simple Hit Rate ---
# Define a probability threshold for considering a prediction a "hit"
# This threshold might need tuning based on your model's output distribution and goals.
prediction_threshold = 0.5 # Example threshold

print(f"\nCalculating simple hit rate using threshold: {prediction_threshold}")
try:
    if len(test_predictions_proba) > 0:
        # The 'true positives' in this context are simply all the instances in test.csv,
        # as test.csv is assumed to contain only positive interactions.
        # A 'hit' is counted if the model predicts a high probability for this positive interaction.
        total_positive_interactions = len(test_predictions_proba)
        hits = np.sum(test_predictions_proba >= prediction_threshold)

        if total_positive_interactions > 0:
            hit_rate = hits / total_positive_interactions
            print(f"Number of 'hits' (predicted prob >= {prediction_threshold}): {hits}")
            print(f"Total test instances (assumed positive interactions): {total_positive_interactions}")
            print(f"Simple Hit Rate: {hit_rate:.4f}")
        else:
            print("No predictions to calculate hit rate.")
            hit_rate = 0

    else:
        print("No predictions available for hit rate calculation.")
        hit_rate = 0

except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during hit rate calculation: {e}")
    traceback.print_exc()
    exit()


# --- Save Predictions ---
print("\nSaving test data with predictions...")
try:
    if not test_df_original.empty and len(test_predictions_proba) == len(test_df_original):
        # Add the predicted probabilities and the binary hit indicator to the original test DataFrame
        test_df_original['predicted_proba'] = test_predictions_proba
        test_df_original['predicted_hit'] = (test_predictions_proba >= prediction_threshold).astype(int)

        # Save the DataFrame to a new CSV file
        test_df_original.to_csv(output_predictions_path, index=False)
        print(f"Test data with predictions saved successfully to {output_predictions_path}")
    elif test_df_original.empty:
        print("Skipping save: Original test data is empty.")
    else:
        print(f"Skipping save: Prediction count ({len(test_predictions_proba)}) does not match original test data row count ({len(test_df_original)}).")

except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during saving predictions: {e}")
    traceback.print_exc()
    exit()


print("\n--- Random Forest Testing Script Finished ---")
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds.")