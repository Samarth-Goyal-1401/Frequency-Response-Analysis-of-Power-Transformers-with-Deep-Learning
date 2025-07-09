import numpy as np
import pandas as pd
import os
import re
import chardet

# --- Configuration (Minimal User Input) ---
output_dir = 'processed_fr_dataset'
data_root_dir = '.' # Same as in databaseCreation.py
fault_types = ['LDF', 'IDF', 'NF'] # IMPORTANT: Make sure this matches your databaseCreation.py

# expected_frequency_points will be determined automatically from X_fr_data.npy

# --- Helper Functions ---

def find_sample_csv(root_dir, fault_dirs):
    """
    Finds the first .csv file in the specified fault directories.
    """
    for fault_type in fault_dirs:
        fault_type_dir = os.path.join(root_dir, fault_type)
        if os.path.isdir(fault_type_dir):
            csv_files = [f for f in os.listdir(fault_type_dir) if f.endswith('.csv')]
            if csv_files:
                return os.path.join(fault_type_dir, csv_files[0])
    return None

def load_frequencies(csv_path, expected_points):
    """
    Loads and preprocesses frequencies from a sample CSV file.
    Applies encoding detection and truncation logic.
    """
    if csv_path is None:
        print("ERROR: No sample CSV path provided to load frequencies.")
        return None

    try:
        with open(csv_path, 'rb') as f:
            raw_data = f.read()
        
        detection = chardet.detect(raw_data)
        detected_encoding = detection['encoding']
        
        encodings_to_try = []
        if detected_encoding and detection['confidence'] > 0.8:
            encodings_to_try.append(detected_encoding)
        encodings_to_try.extend(['latin-1', 'cp1252', 'utf-8'])
        encodings_to_try = list(dict.fromkeys(encodings_to_try))

        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                continue # Just continue quietly if a read fails with a specific encoding

        if df is None:
            raise Exception(f"Failed to load sample CSV at {csv_path} with any known encoding.")
        
        if 'Freq.' not in df.columns:
            freq_col_found = False
            for col_name in ['Freq.', 'Frequency', 'freq', 'frequency']:
                if col_name in df.columns:
                    df.rename(columns={col_name: 'Freq.'}, inplace=True)
                    freq_col_found = True
                    break
            if not freq_col_found:
                raise ValueError(f"Column 'Freq.' (or variations) not found in sample CSV at {csv_path}. Please check file format.")
            
        frequencies = pd.to_numeric(df['Freq.'].values, errors='coerce')
        frequencies = frequencies[~np.isnan(frequencies)]

        if len(frequencies) > expected_points:
            frequencies = frequencies[:expected_points]
        elif len(frequencies) < expected_points:
            print(f"  Warning: Sample CSV has {len(frequencies)} frequencies, but expected {expected_points}. Using available frequencies.")
            
        return frequencies

    except Exception as e:
        print(f"ERROR: Could not load frequencies from '{csv_path}': {e}")
        return None

def display_single_sample_details(sample_idx, X_data, y_labels, class_names, frequencies_data, display_type="Sample"):
    """
    Helper function to display details for a single sample.
    """
    print(f"\n--- {display_type} (Dataset Index: {sample_idx}) ---")
    
    one_hot_label = y_labels[sample_idx]
    label_index = np.argmax(one_hot_label)
    decoded_label = class_names[label_index]

    print(f"Fault Type: {decoded_label} (One-Hot: {one_hot_label})")

    sample_features = X_data[sample_idx]
    
    # Ensure frequencies match the length of processed features
    display_frequencies_for_sample = frequencies_data[:sample_features.shape[0]]

    display_df = pd.DataFrame({
        'Frequency': display_frequencies_for_sample,
        'Log Magnitude (dB)': sample_features[:, 0],
        'Normalized Phase (0-1)': sample_features[:, 1]
    })

    if len(display_df) > 10:
        print(display_df.head(5).to_string())
        print("...")
        print(display_df.tail(5).to_string())
    else:
        print(display_df.to_string())
    print("-" * 40)


# --- Main Script Execution ---

# --- Load the .npy files ---
X_loaded = None
y_loaded = None
class_names_loaded = None
expected_frequency_points = 0 # Will be updated after loading X_loaded

x_filepath = os.path.join(output_dir, 'X_fr_data.npy')
y_filepath = os.path.join(output_dir, 'y_fr_labels.npy')
class_names_filepath = os.path.join(output_dir, 'fault_class_names.npy')

print("--- Loading processed data (.npy files) ---")
try:
    X_loaded = np.load(x_filepath)
    print(f"Successfully loaded {x_filepath}")
    print(f"Shape of loaded X: {X_loaded.shape}")
    print(f"Data type of loaded X: {X_loaded.dtype}")
    expected_frequency_points = X_loaded.shape[1] 
    print(f"Automatically set expected_frequency_points based on X_fr_data.npy: {expected_frequency_points}\n")

except FileNotFoundError:
    print(f"ERROR: '{x_filepath}' not found. Please run databaseCreation.py first to generate these files.")
    exit()
except Exception as e:
    print(f"An error occurred while loading X_fr_data.npy: {e}")
    exit()

try:
    y_loaded = np.load(y_filepath)
    print(f"Successfully loaded {y_filepath}")
    print(f"Shape of loaded y: {y_loaded.shape}")
    print(f"Data type of loaded y: {y_loaded.dtype}")
except FileNotFoundError:
    print(f"ERROR: '{y_filepath}' not found. Please run databaseCreation.py first.")
except Exception as e:
    print(f"An error occurred while loading y_fr_labels.npy: {e}")

try:
    class_names_loaded = np.load(class_names_filepath, allow_pickle=True)
    print(f"Successfully loaded {class_names_filepath}")
    print(f"Contents of class names: {class_names_loaded}")
except FileNotFoundError:
    print(f"ERROR: '{class_names_filepath}' not found. Please run databaseCreation.py first.")
except Exception as e:
    print(f"An error occurred while loading fault_class_names.npy: {e}")

print("\n" + "=" * 50 + "\n") # Separator

# --- Display loaded data ---
if X_loaded is not None and y_loaded is not None and class_names_loaded is not None and X_loaded.shape[0] > 0:
    
    # --- Dynamically find a sample CSV for frequencies ---
    sample_csv_path = find_sample_csv(data_root_dir, fault_types)
    frequencies = None
    if sample_csv_path:
        print(f"Automatically selected sample CSV for frequencies: '{sample_csv_path}'")
        frequencies = load_frequencies(sample_csv_path, expected_frequency_points)
    else:
        print(f"WARNING: No CSV files found in {fault_types} within {data_root_dir}. Cannot load frequencies.")
    
    if frequencies is None or len(frequencies) == 0:
        print("Could not load frequencies. Displaying X values with dummy frequencies.")
        frequencies = np.arange(1, expected_frequency_points + 1) # Dummy frequencies

    total_samples = X_loaded.shape[0]

    # --- Display First 5 Samples ---
    print("\n" + "#" * 15 + " Displaying First 5 Samples " + "#" * 15)
    num_first_samples = min(5, total_samples)
    for i in range(num_first_samples):
        display_single_sample_details(i, X_loaded, y_loaded, class_names_loaded, frequencies, display_type=f"First Sample {i+1}")

    # --- Display Last 5 Samples ---
    if total_samples > 5:
        print("\n" + "#" * 15 + " Displaying Last 5 Samples " + "#" * 15)
        num_last_samples = min(5, total_samples)
        for i in range(total_samples - num_last_samples, total_samples):
            display_single_sample_details(i, X_loaded, y_loaded, class_names_loaded, frequencies, display_type=f"Last Sample {i - (total_samples - num_last_samples) + 1}")
    else:
        print("\nLess than 5 total samples, skipping 'Last 5 Samples' display.")

    # --- Display 5 Random Samples ---
    if total_samples > 5: # Ensure there are enough samples to pick 5 unique random ones
        print("\n" + "#" * 15 + " Displaying 5 Random Samples " + "#" * 15)
        num_random_samples = 5
        random_indices = np.random.choice(total_samples, num_random_samples, replace=False)
        for i, sample_idx in enumerate(random_indices):
            display_single_sample_details(sample_idx, X_loaded, y_loaded, class_names_loaded, frequencies, display_type=f"Random Sample {i+1}")
    elif total_samples > 0:
        print("\nLess than 5 total samples, displaying all available samples as 'Random'.")
        for i in range(total_samples):
            display_single_sample_details(i, X_loaded, y_loaded, class_names_loaded, frequencies, display_type=f"Random Sample {i+1}")
    else:
        print("\nNo samples to display.")

else:
    print("Cannot display samples: Data (X, y, or class names) not loaded correctly or dataset is empty.")

print("\n--- Script Finished ---")