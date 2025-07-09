import os
import pandas as pd
import numpy as np
import re
from tensorflow.keras.utils import to_categorical
import time
import chardet # chardet is less relevant for .xlsx files, but kept if mixed file types exist or for future csv use.

# --- Configuration ---
# IMPORTANT: Adjust this path to where your 'final_csv_files' folder is located,
# or where your LDF, IDF, NF, DLF, RDF, SCF folders containing data files are.
data_root_dir = '.'  # Keep '.' if your fault type folders are in the same directory as this script.
                      # Set to a specific path if your data structure is different.

# Added 'SCF' to the list of fault types
fault_types = ['LDF', 'IDF', 'NF', 'DLF', 'RDF', 'SCF']
expected_frequency_points = 4999 # Changed from 5002 to 4999 as requested

# --- Data Storage Initialization ---
all_fr_data = []  # To store the processed frequency response data
all_labels = []   # To store the corresponding fault type labels
# Create a mapping from fault type name to an integer label for one-hot encoding
label_to_int = {label: i for i, label in enumerate(fault_types)}

print("--- Starting Data Merging and Preprocessing Script ---")
print(f"Data root directory: '{os.path.abspath(data_root_dir)}'")
print(f"Fault types to be processed: {fault_types}")
print(f"Expected frequency points per case: {expected_frequency_points}\n")

total_start_time = time.time()

# Loop through each defined fault type
for i, fault_type in enumerate(fault_types):
    # Construct path to the current fault type's data directory
    fault_type_dir = os.path.join(data_root_dir, fault_type)
    
    if not os.path.isdir(fault_type_dir):
        print(f"ERROR: Directory not found: '{fault_type_dir}'. Please ensure '{fault_type}' folder exists in '{data_root_dir}'. Skipping this fault type.\n")
        continue

    print(f"--- Processing fault type: '{fault_type}' ({i + 1}/{len(fault_types)}) ---")
    fault_type_start_time = time.time()

    # --- MODIFICATION HERE: Check for .xlsx files instead of .csv ---
    all_files_in_dir = os.listdir(fault_type_dir)
    print(f"  DEBUG: All files found in '{fault_type_dir}': {all_files_in_dir}") # Debugging print
    
    # Filter for both .csv and .xlsx files, preferring .xlsx for 'SCF' if that's the primary type there
    data_files = []
    if fault_type == 'SCF': # Special handling for SCF based on user's clarification
        data_files = [f for f in all_files_in_dir if f.lower().endswith('.xlsx')]
    else: # For other fault types, continue looking for .csv files
        data_files = [f for f in all_files_in_dir if f.lower().endswith('.csv')]

    data_files.sort() # Ensure consistent order across runs

    if not data_files:
        print(f"  WARNING: No data files found (.csv or .xlsx) directly in '{fault_type_dir}'. Skipping this fault type.\n")
        continue

    print(f"  Found {len(data_files)} data files in '{fault_type_dir}'.")
    
    processed_count_in_folder = 0

    # Loop through each data file for the current fault type
    for j, filename in enumerate(data_files):
        filepath = os.path.join(fault_type_dir, filename)
        
        if (j + 1) % 10 == 0 or (j + 1) == len(data_files):
            print(f"  Processing file {j + 1}/{len(data_files)}: '{filename}'")
        
        df = None

        try:
            # --- MODIFICATION HERE: Use pd.read_excel for .xlsx files, pd.read_csv for .csv ---
            if filename.lower().endswith('.xlsx'):
                df = pd.read_excel(filepath, engine='openpyxl')
            elif filename.lower().endswith('.csv'):
                # For CSVs, keep robust encoding detection
                with open(filepath, 'rb') as f:
                    raw_data_sample = f.read(10000)
                detection = chardet.detect(raw_data_sample)
                detected_encoding = detection['encoding']
                confidence = detection['confidence']

                encodings_to_try = []
                if detected_encoding and confidence > 0.8:
                    encodings_to_try.append(detected_encoding)
                encodings_to_try.extend(['latin-1', 'cp1252', 'utf-8'])
                encodings_to_try = list(dict.fromkeys(encodings_to_try))

                read_success = False
                for enc in encodings_to_try:
                    try:
                        df = pd.read_csv(filepath, encoding=enc, sep=None, engine='python')
                        read_success = True
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e_read:
                        continue
                if not read_success or df is None:
                    raise Exception(f"Failed to load dataframe after trying all encoding options.")
            else:
                print(f"  WARNING: Skipping unsupported file type: '{filename}'.")
                continue

            # --- Step 1: Robustly find and parse the Magnitude/Phase column ---
            data_column_name = None
            
            if len(df.columns) == 2:
                for col in df.columns:
                    if col.lower() != 'freq.':
                        data_column_name = col
                        break
            
            if data_column_name is None:
                if 'V(e1)/I(V1)' in df.columns:
                    data_column_name = 'V(e1)/I(V1)'
                elif 'V(e1)/I(v1)' in df.columns:
                    data_column_name = 'V(e1)/I(v1)'

            # --- Handle potential direct Magnitude/Phase columns for robustness ---
            magnitude_col_found = False
            phase_col_found = False
            
            # Check for common direct magnitude/phase column names (case-insensitive)
            # This is a new addition to handle potential different excel formats
            for col in df.columns:
                if 'magnitude' in col.lower() and 'db' in col.lower():
                    magnitude_db = pd.to_numeric(df[col], errors='coerce').values
                    magnitude_col_found = True
                if 'phase' in col.lower() and 'deg' in col.lower():
                    phase_deg = pd.to_numeric(df[col], errors='coerce').values
                    phase_col_found = True
            
            if magnitude_col_found and phase_col_found:
                # If separate columns are found, use them directly
                parsed_df = pd.DataFrame({'magnitude_db': magnitude_db, 'phase_deg': phase_deg})
                parsed_df.dropna(inplace=True) # Drop NaNs introduced by coerce
                print(f"  Note: Using direct Magnitude (dB) and Phase (deg) columns from '{filename}'.")
            elif data_column_name is not None:
                # Fallback to parsing combined string column if separate columns not found
                regex_pattern = r'\(?([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*dB[,\s]*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*°?\)?'
                
                parsed_data = df[data_column_name].astype(str).apply(
                    lambda x: re.findall(regex_pattern, x.replace('–', '-'))[0] if re.findall(regex_pattern, x.replace('–', '-')) else (np.nan, np.nan)
                )
                
                parsed_df = pd.DataFrame(parsed_data.tolist(), columns=['magnitude_db', 'phase_deg'])
                
                parsed_df['magnitude_db'] = pd.to_numeric(parsed_df['magnitude_db'])
                parsed_df['phase_deg'] = pd.to_numeric(parsed_df['phase_deg'])

                initial_rows_after_parse = len(parsed_df)
                parsed_df.dropna(inplace=True)
                if len(parsed_df) < initial_rows_after_parse:
                    print(f"  Note: Dropped {initial_rows_after_parse - len(parsed_df)} rows with invalid magnitude/phase data in '{filename}'.")
            else:
                print(f"  ERROR: Could not find suitable data columns (e.g., 'V(e1)/I(V1)', 'V(e1)/I(v1)', or direct 'Magnitude (dB)'/'Phase (deg)') in '{filename}'. Available columns: {df.columns.tolist()}. Skipping this case.")
                continue

            # Ensure 'Freq.' column is numeric and filter NaNs - This is critical for alignment and consistency
            freq_col_name = None
            for col in df.columns:
                if col.lower() == 'freq.' or 'frequency' in col.lower(): # More robust frequency column detection
                    freq_col_name = col
                    break
            if freq_col_name is None:
                print(f"  ERROR: 'Freq.' or 'Frequency' column not found in '{filename}'. Skipping this case.")
                continue

            frequencies = pd.to_numeric(df[freq_col_name].values, errors='coerce')
            frequencies = frequencies[parsed_df.index] # Align frequencies with the (possibly filtered) parsed_df


            # --- Step 2: Handle Data Length Consistency ---
            if len(parsed_df) == 0 or len(frequencies) == 0:
                print(f"  WARNING: '{filename}' resulted in zero valid data points after initial parsing and filtering. Skipping this file.")
                continue

            if len(parsed_df) != expected_frequency_points:
                if len(parsed_df) > expected_frequency_points:
                    parsed_df = parsed_df.iloc[:expected_frequency_points]
                    frequencies = frequencies[:expected_frequency_points] # Truncate frequencies as well
                    print(f"  Truncated '{filename}' from {len(df)} (initial) to {expected_frequency_points} rows.") # Note: len(df) might be different due to parsing/dropping
                else:
                    print(f"  WARNING: '{filename}' has {len(parsed_df)} rows, expected {expected_frequency_points}. Skipping this case due to insufficient data.")
                    continue 
            
            magnitude_db = parsed_df['magnitude_db'].values
            phase_deg = parsed_df['phase_deg'].values

            # --- Step 3: Apply Preprocessing Transforms ---
            magnitude_log = np.log10(magnitude_db + 1e-9)
            phase_normalized = (phase_deg + 180) / 360 

            # --- Step 4: Combine features for this case ---
            case_data = np.stack((magnitude_log, phase_normalized), axis=-1)
            
            all_fr_data.append(case_data)
            one_hot_label = to_categorical(label_to_int[fault_type], num_classes=len(fault_types))
            all_labels.append(one_hot_label)
            processed_count_in_folder += 1

            print(f"    Processed 1 sample for '{fault_type}'. Current total samples: {len(all_fr_data)}")

        except Exception as e:
            print(f"  ERROR: Failed to process '{filepath}': {e}. Skipping this case.")
            continue
    
    fault_type_end_time = time.time()
    elapsed_time_fault_type = fault_type_end_time - fault_type_start_time
    print(f"--- Finished processing '{fault_type}'. Successfully processed {processed_count_in_folder} files in {elapsed_time_fault_type:.2f} seconds. ---\n")

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

print(f"--- Finished loading all data. Total cases loaded: {len(all_fr_data)} ---")
print(f"Total data loading and preprocessing time: {total_elapsed_time:.2f} seconds.\n")

# --- Step 5: Convert Lists to Final NumPy Arrays ---
print("Converting processed data to NumPy arrays and applying one-hot encoding...")
if len(all_fr_data) == 0:
    print("No data was loaded. Cannot create NumPy arrays. Please check your data paths and file contents.")
    exit()

X = np.array(all_fr_data)
y_one_hot = np.array(all_labels)

print(f"Shape of X (features array): {X.shape}")
print(f"Shape of y (one-hot encoded labels array): {y_one_hot.shape}")
if X.shape[0] > 0:
    print(f"Shape of a single case's data (e.g., X[0]): {X[0].shape}")

# --- Step 6: Save the Processed Data ---
output_dir = 'processed_fr_dataset'
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, 'X_fr_data.npy'), X)
np.save(os.path.join(output_dir, 'y_fr_labels.npy'), y_one_hot)
np.save(os.path.join(output_dir, 'fault_class_names.npy'), np.array(fault_types, dtype=object))

print(f"\n--- Processed data saved successfully to '{output_dir}' directory ---")
print(f"- X_fr_data.npy (contains your FR data)")
print(f"- y_fr_labels.npy (contains your one-hot encoded labels)")
print(f"- fault_class_names.npy (contains the list of fault type names)")

print("\n--- Script Finished ---")