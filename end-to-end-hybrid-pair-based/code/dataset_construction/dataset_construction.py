# construct_beauty_dataset.py

import pandas as pd
import numpy as np
import ast  # For safely evaluating string representations of lists/dicts
import string  # For punctuation removal
import collections  # For word counting (needed for substantive check)
import re  # For general regex
import os  # Import os for path joining and existence checks

# --- Configuration ---
# Define the relative path to your data directory (where the input file is located)
DATA_PATH = '../data/'
# Define the name of your input CSV file containing raw item metadata
INPUT_CSV_FILENAME = 'item_meta.csv' # <-- *** CHANGE THIS TO YOUR ACTUAL FILE NAME ***
# Define the full relative path and filename for the output cleaned CSV file
# This now includes the 'dataset' subdirectory
OUTPUT_FULL_PATH_AND_FILENAME = 'dataset/clean_beauty_item_meta_with_details.csv'
# Number of top detail keys to extract
TOP_N_DETAILS_KEYS = 20

# --- File Paths ---
input_csv_path = os.path.join(DATA_PATH, INPUT_CSV_FILENAME)
# The full output path is constructed here, including the subdirectory
output_csv_path = os.path.join(DATA_PATH, OUTPUT_FULL_PATH_AND_FILENAME)

print(f"Attempting to construct dataset from: {input_csv_path}")
print(f"Output will be saved to: {output_csv_path}")

# --- Check if input file exists ---
if not os.path.exists(input_csv_path):
    print(f"\nError: Input file not found at {input_csv_path}")
    print("Please ensure the raw item metadata CSV is in the './data' folder (relative to your script's parent directory)")
    print(f"and the INPUT_CSV_FILENAME variable is set correctly (currently '{INPUT_CSV_FILENAME}').")
else:
    try:
        # --- Load Initial Data ---
        print("\nLoading raw item metadata...")
        # Assuming the initial data is in a CSV format that pandas can read directly
        # Use low_memory=False to avoid dtype warnings with mixed data types if any
        item_meta_df = pd.read_csv(input_csv_path, low_memory=False)
        print(f"Successfully loaded dataset with shape: {item_meta_df.shape}")
        print("Original columns:", item_meta_df.columns.tolist())

        # --- 1. Filter for 'All Beauty' ---
        if 'main_category' in item_meta_df.columns:
             beauty_items_df = item_meta_df[item_meta_df['main_category'] == 'All Beauty'].copy()
             print(f"\nFiltered for 'All Beauty', starting with {len(beauty_items_df)} items.")
        else:
             print("\nWarning: 'main_category' column not found. Skipping filtering.")
             beauty_items_df = item_meta_df.copy() # Work with the full dataset if category missing

        # --- 2. Extract Number of Images ---
        print("\nExtracting number of images...")
        # Safe parser for list-like strings or non-strings
        def safe_literal_eval_list_or_empty(s):
             try:
                if isinstance(s, str):
                     s_stripped = s.strip()
                     if s_stripped == '[]' or s_stripped.lower() == 'nan':
                         return []
                     # Attempt to parse non-empty strings that look like lists
                     if s_stripped.startswith('[') and s_stripped.endswith(']'):
                         return ast.literal_eval(s)
                # Handle NaN or other non-string/non-list-like formats
                return None
             except (ValueError, SyntaxError, TypeError): # Added TypeError
                return None # Return None if parsing fails for non-[] strings

        if 'images' in beauty_items_df.columns:
             # Handle potential NaN or non-string inputs by converting to string first
             parsed_images = beauty_items_df['images'].astype(str).apply(safe_literal_eval_list_or_empty)
             beauty_items_df['number_of_images'] = parsed_images.apply(lambda x: len(x) if isinstance(x, list) else 0)
             print(f"Created 'number_of_images' column. Max images: {beauty_items_df['number_of_images'].max()}")
             # Drop original images column if it exists and was processed
             beauty_items_df = beauty_items_df.drop(columns=['images'])
             print("Dropped original 'images' column.")
        else:
             print("Warning: 'images' column not found. Cannot extract image count.")
             beauty_items_df['number_of_images'] = 0 # Add column with default 0

        # --- 3. Transform 'videos' to a binary signal 'has_video' ---
        print("\nTransforming 'videos' column...")
        if 'videos' in beauty_items_df.columns:
            # Handle potential NaN or non-string inputs by converting to string first
            parsed_videos = beauty_items_df['videos'].astype(str).apply(safe_literal_eval_list_or_empty) # Re-use image parser
            beauty_items_df['has_video'] = parsed_videos.apply(lambda x: 1 if isinstance(x, list) and len(x) > 0 else 0)
            print("Created 'has_video' signal.")
            # Drop original videos column
            beauty_items_df = beauty_items_df.drop(columns=['videos'])
            print("Dropped original 'videos' column.")
        else:
             print("Warning: 'videos' column not found. Cannot create 'has_video'.")
             beauty_items_df['has_video'] = 0 # Add column with default 0


        # --- 4. Transform 'price' to numerical 'price_category_mapped' ---
        print("\nTransforming 'price' column to numerical 'price_category_mapped'...")
        if 'price' in beauty_items_df.columns:
            # Ensure price column is numeric, coercing errors to NaN
            beauty_items_df['price'] = pd.to_numeric(beauty_items_df['price'], errors='coerce')

            price_bins = [0, 10, 25, 50, 100, 500, float('inf')]
            price_labels = ['Under $10', '$10 - $25', '$25 - $50', '$50 - $100', '$100 - $500', 'Over $500']
            price_missing_label = 'Price Missing'

            # Create intermediate price_category column (string) for pd.cut
            beauty_items_df['price_category_temp'] = price_missing_label
            price_not_null_mask = beauty_items_df['price'].notna()

            # Use try-except for pd.cut in case of issues
            try:
                 # pd.cut will only work on non-null numeric data filtered by the mask
                 beauty_items_df.loc[price_not_null_mask, 'price_category_temp'] = pd.cut(
                     beauty_items_df.loc[price_not_null_mask, 'price'],
                     bins=price_bins,
                     labels=price_labels,
                     right=True,
                     include_lowest=True,
                     duplicates='drop'
                 )
            except Exception as e:
                 print(f"Warning: Could not apply pd.cut to 'price' column. Error: {e}")
                 # If pd.cut fails, the temporary column will remain 'Price Missing' or NaN where original price was NaN


            # Define and apply numerical mapping
            price_ordered_labels = price_labels # Use the bin labels for order
            price_mapping = {label: i for i, label in enumerate(price_ordered_labels)}
            price_mapping[price_missing_label] = len(price_ordered_labels) # Map missing to last code (6)

            # Ensure the temp column is string before mapping, map NaN from pd.cut issues too
            beauty_items_df['price_category_mapped'] = beauty_items_df['price_category_temp'].astype(str).map(price_mapping).fillna(price_mapping[price_missing_label]).astype(int) # Ensure integer type, fill any remaining NaNs with missing code

            # Drop the intermediate string column and the original price column
            beauty_items_df = beauty_items_df.drop(columns=['price_category_temp', 'price'])
            print("Created numerical 'price_category_mapped' and dropped original 'price'.")
            print(f"Price Category Mapping: {price_mapping}")

        else:
            print("Warning: 'price' column not found. Cannot create 'price_category_mapped'.")
            # Add the column with the code for missing if price column is absent
            price_mapping = {label: i for i, label in enumerate(['Under $10', '$10 - $25', '$25 - $50', '$50 - $100', '$100 - $500', 'Over $500'])}
            price_mapping['Price Missing'] = len(price_mapping)
            beauty_items_df['price_category_mapped'] = price_mapping['Price Missing']
            print(f"Added 'price_category_mapped' with default value {price_mapping['Price Missing']}.")


        # --- 5. Transform 'features' to binary 'has_substantive_features' ---
        print("\nTransforming 'features' column to 'has_substantive_features'...")
        if 'features' in beauty_items_df.columns:
            translator = str.maketrans('', '', string.punctuation) # Define translator here

            # Safe parser for list-like strings (corrected from original for robustness)
            def safe_literal_eval_list(s):
                 try:
                    # Handles parsing lists, [] string, 'nan' string, other non-list strings
                    if isinstance(s, str):
                         s_stripped = s.strip()
                         if s_stripped == '[]' or s_stripped.lower() == 'nan':
                             return [] # Treat empty list string or 'nan' string as empty list
                         # Use re.match to be more strict about starting with '[' and ending with ']'
                         if re.match(r'\[.*\]$', s_stripped):
                              return ast.literal_eval(s)
                    return None # For other non-string or non-list-like strings
                 except (ValueError, SyntaxError, TypeError): # Add TypeError for better handling
                     return None # Parsing fails

            # Function to check for substantive text in a list of strings
            def contains_substantive_text(feature_list):
                 # Handle None results from parsing failure explicitly
                 if feature_list is None:
                     return False
                 if not isinstance(feature_list, list): # Should be lists or None now
                     return False
                 for feature_str in feature_list:
                     # Ensure list item is a string before processing
                     if isinstance(feature_str, str):
                         # Clean and check for words
                         cleaned_str = feature_str.lower().translate(translator).strip()
                         words = cleaned_str.split()
                         if len(words) > 0: # Check if there are any words
                             return True
                 return False # No substantive text found in any item

            # Apply parsing and substantive check - convert to string first for safety
            parsed_features_all = beauty_items_df['features'].astype(str).apply(safe_literal_eval_list)
            has_substantive_features_mask = parsed_features_all.apply(contains_substantive_text)
            beauty_items_df['has_substantive_features'] = has_substantive_features_mask.astype(int)

            # Drop original features column
            beauty_items_df = beauty_items_df.drop(columns=['features'])
            print("Created 'has_substantive_features' column and dropped original 'features'.")
        else:
            print("Warning: 'features' column not found. Cannot create 'has_substantive_features'.")
            beauty_items_df['has_substantive_features'] = 0 # Add column with default 0


        # --- 6. Calculate Weighted Rating (numerical 'weighted_rating') ---
        print("\nCalculating numerical 'weighted_rating'...")
        if 'average_rating' in beauty_items_df.columns and 'rating_number' in beauty_items_df.columns:
            # Ensure rating columns are numeric, coercing errors to NaN
            beauty_items_df['average_rating'] = pd.to_numeric(beauty_items_df['average_rating'], errors='coerce')
            beauty_items_df['rating_number'] = pd.to_numeric(beauty_items_df['rating_number'], errors='coerce').fillna(0).astype(int) # Fill NaNs with 0 for count

            # Identify items with both average_rating (non-NaN) and rating_number (non-NaN, which are now non-NaN int after fillna)
            rated_items_for_calc = beauty_items_df.dropna(subset=['average_rating']).copy() # Only need average_rating to be non-null for the calc formula input

            if len(rated_items_for_calc) > 0:
                # Calculate C (overall average rating) and m (median rating count) using the items that had *original* non-null ratings
                # This ensures C and m are representative of items with rating info.
                C_source_df = beauty_items_df.dropna(subset=['average_rating', 'rating_number']).copy()
                if len(C_source_df) > 0:
                    C = C_source_df['average_rating'].mean()
                    m = C_source_df['rating_number'].median()
                    m = max(1, m) # Ensure m is at least 1
                else:
                    C = 0 # Default C if no items have both ratings
                    m = 1 # Default m if no items have both ratings
                print(f"Calculated C (overall average rating): {C:.4f}")
                print(f"Calculated m (median rating number): {m}")


                def weighted_rating_calc(row, C, m):
                    R = row['average_rating']
                    N = row['rating_number']
                    # N is already handled for NaNs by fillna(0)
                    # R is handled by the dropna(subset=['average_rating']) mask applied to rated_items_for_calc
                    # Avoid division by zero (N+m should be > 0 with m>=1)
                    if (N + m) == 0: # Should not happen with m>=1
                         return np.nan
                    return ((R * N) + (C * m)) / (N + m)

                # Calculate Weighted Rating *only* for the items with valid average_rating
                calculated_weighted_ratings = rated_items_for_calc.apply(weighted_rating_calc, axis=1, C=C, m=m)

                # Initialize weighted_rating column with NaN in the main DataFrame
                beauty_items_df['weighted_rating'] = np.nan

                # Assign the calculated values back using the index
                beauty_items_df.loc[calculated_weighted_ratings.index, 'weighted_rating'] = calculated_weighted_ratings

                print(f"Calculated numerical 'weighted_rating' for {len(calculated_weighted_ratings)} items.")
                # Drop original rating columns (rating_number was filled, average_rating had NaNs)
                beauty_items_df = beauty_items_df.drop(columns=['average_rating', 'rating_number'])
                print("Dropped original 'average_rating' and 'rating_number' columns.")

            else:
                print("Not enough items with average_rating to calculate Weighted Rating.")
                beauty_items_df['weighted_rating'] = np.nan # Ensure column exists even if empty
                # Still need to drop the original columns if they exist
                cols_to_drop_ratings = [col for col in ['average_rating', 'rating_number'] if col in beauty_items_df.columns]
                if cols_to_drop_ratings:
                     beauty_items_df = beauty_items_df.drop(columns=cols_to_drop_ratings)
                     print(f"Dropped original rating columns: {cols_to_drop_ratings}")

        else:
            print("Warning: 'average_rating' or 'rating_number' column not found. Cannot calculate 'weighted_rating'.")
            beauty_items_df['weighted_rating'] = np.nan # Add column as NaN


        # --- 7. Transform 'description' to binary 'has_substantive_description' ---
        print("\nTransforming 'description' column to 'has_substantive_description'...")
        if 'description' in beauty_items_df.columns:
            translator = str.maketrans('', '', string.punctuation) # Ensure translator is defined

            def is_substantive_description(desc_str):
                if not isinstance(desc_str, str):
                     return False
                cleaned_str = desc_str.lower().translate(translator).strip()
                # Check if the cleaned string is empty or contains no words
                if len(cleaned_str) == 0:
                    return False
                words = cleaned_str.split()
                if len(words) == 0:
                    return False
                # Also explicitly check for the '[]' string representation which might indicate empty
                if desc_str.strip() == '[]':
                     return False
                return True

            # Ensure description is treated as string before apply to handle NaNs etc.
            beauty_items_df['has_substantive_description'] = beauty_items_df['description'].astype(str).apply(is_substantive_description).astype(int)

            # Drop original description column
            beauty_items_df = beauty_items_df.drop(columns=['description'])
            print("Created 'has_substantive_description' column and dropped original 'description'.")
        else:
            print("Warning: 'description' column not found. Cannot create 'has_substantive_description'.")
            beauty_items_df['has_substantive_description'] = 0 # Add column with default 0


        # --- 8. Extract Features from Top Details Keys ---
        print(f"\nAnalyzing and extracting features from 'details' column (Top {TOP_N_DETAILS_KEYS} keys)...")
        if 'details' in beauty_items_df.columns:

            # Function to safely parse dictionary strings
            def safe_literal_eval_dict(s):
                if not isinstance(s, str):
                     return None
                try:
                    # Be strict about starting with '{' and ending with '}'
                    if re.match(r'{.*}$', s.strip()):
                         # Use ast.literal_eval to safely evaluate the string
                         return ast.literal_eval(s)
                    return None # Return None for strings not looking like dicts
                except (ValueError, SyntaxError, TypeError): # Added TypeError
                    return None # Return None if evaluation fails


            # Apply the parsing function - convert to string first for safety
            parsed_details = beauty_items_df['details'].astype(str).apply(safe_literal_eval_dict)

            # Filter for successfully parsed dictionaries to count keys
            valid_parsed_details = parsed_details[parsed_details.apply(lambda x: isinstance(x, dict))]
            print(f"Successfully parsed 'details' into dictionaries for {len(valid_parsed_details)} items.")

            # Identify Top N Keys
            all_keys = []
            for detail_dict in valid_parsed_details:
                 if isinstance(detail_dict, dict) and detail_dict: # Ensure it's a dict and not empty
                     # Exclude keys that are just empty strings or whitespace
                     valid_keys = [key for key in detail_dict.keys() if isinstance(key, str) and key.strip()]
                     all_keys.extend(valid_keys)


            key_counts = collections.Counter(all_keys)
            num_keys_to_extract = min(TOP_N_DETAILS_KEYS, len(key_counts))
            top_n_keys = [key for key, count in key_counts.most_common(num_keys_to_extract)]

            if num_keys_to_extract > 0:
                print(f"Identified top {num_keys_to_extract} most frequent detail keys:")
                for key, count in key_counts.most_common(num_keys_to_extract):
                    print(f"  '{key}': {count}")

                # Extract Values for Top N Keys and Create New Columns
                print(f"\nExtracting values for top {num_keys_to_extract} keys...")

                # Create new columns directly on the DataFrame
                for key in top_n_keys:
                     # Create column name, replacing spaces/special chars for safer column names
                     safe_key_name = re.sub(r'[^a-zA-Z0-9_]+', '_', key).strip('_').lower()
                     if not safe_key_name: # Handle cases where key was just special chars
                         safe_key_name = 'unknown_detail_key'
                     new_col_name = f'details_{safe_key_name}'

                     # Ensure column names are unique if original keys map to the same safe name
                     original_new_col_name = new_col_name
                     counter = 1
                     while new_col_name in beauty_items_df.columns:
                         new_col_name = f'{original_new_col_name}_{counter}'
                         counter += 1


                     # Apply the extraction for this key
                     # Use parsed_details series which aligns with beauty_items_df index
                     beauty_items_df[new_col_name] = parsed_details.apply(
                         lambda x: x.get(key, None) if isinstance(x, dict) and key in x else None
                     )

                print(f"Created {num_keys_to_extract} new columns for detail keys.")

            else:
                 print("No substantive detail keys found or parsed successfully.")


            # Drop the original 'details' column now that it's processed
            beauty_items_df = beauty_items_df.drop(columns=['details'])
            print("Dropped original 'details' column.")

        else:
            print("Warning: 'details' column not found. Cannot extract detail keys.")
            # No new details columns will be created

        # --- Drop other specified columns that were not transformed ---
        # Assuming these should be dropped if they weren't processed above
        # Also drop 'item_id' if it exists, as we assume the index is sufficient or it's not needed as a feature
        columns_to_drop_unprocessed = ['bought_together', 'parent_asin', 'categories', 'main_category']
        existing_cols_to_drop_unprocessed = [col for col in columns_to_drop_unprocessed if col in beauty_items_df.columns]
        if existing_cols_to_drop_unprocessed:
             beauty_items_df = beauty_items_df.drop(columns=existing_cols_to_drop_unprocessed)
             print(f"\nDropped unprocessed columns: {existing_cols_to_drop_unprocessed}")
        else:
             print("\nNo additional columns found to drop.")


        # --- Final Clean DataFrame ---
        clean_beauty_meta_df = beauty_items_df.copy()

        print(f"\nFinal clean dataset shape: {clean_beauty_meta_df.shape}")
        print("\nFinal clean dataset columns:")
        print(clean_beauty_meta_df.columns.tolist())
        print("\nFinal clean dataset info:")
        clean_beauty_meta_df.info()

        print("\nFirst 5 rows of the final clean dataset:")
        print(clean_beauty_meta_df.head())

        # Optional: Check value counts for some new columns
        print("\nValue counts for selected new columns:")
        cols_to_check_value_counts = [
            'number_of_images', 'has_video', 'price_category_mapped',
            'has_substantive_features', 'has_substantive_description',
            'weighted_rating' # Check non-null count/distribution for weighted rating
        ]
        for col in cols_to_check_value_counts:
            if col in clean_beauty_meta_df.columns:
                 print(f"\n--- {col} ---")
                 if col == 'weighted_rating':
                     # For numeric like weighted_rating, describe is more useful than value_counts
                     print(clean_beauty_meta_df[col].describe())
                 else:
                     # Use dropna=False to see counts of missing values if any
                     print(clean_beauty_meta_df[col].value_counts(dropna=False).head()) # Use head for potentially many values

        # --- 9. Save the clean DataFrame to CSV ---
        try:
            # Get the directory path for the output file
            output_dir = os.path.dirname(output_csv_path)

            # Create the output directory if it doesn't exist (including any necessary parent directories)
            # exist_ok=True prevents an error if the directory already exists
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nEnsured output directory exists: {output_dir}")

            # Save with index=False
            clean_beauty_meta_df.to_csv(output_csv_path, index=False)
            print(f"Clean dataset saved successfully to {output_csv_path}")
        except Exception as e:
            print(f"\nError saving clean dataset to CSV: {e}")


    except FileNotFoundError:
         # This specific error is handled by the initial check, but kept as defensive
         print(f"Error: Input file not found at {input_csv_path}")
    except pd.errors.EmptyDataError:
         print(f"Error: The file at {input_csv_path} is empty.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc() # Print the full traceback for debugging

print("\nDataset construction script finished.")