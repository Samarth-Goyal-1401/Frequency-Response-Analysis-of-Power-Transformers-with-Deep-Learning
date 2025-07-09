import os
import pandas as pd
import chardet
import shutil
import re 

def raw_data_to_standard_csv_converter():
    """
    Converts raw data files to standardized .txt files, then to .csv files.
    Specifically handles challenging 'NF' file formats.
    """
    print("--- Step 1: Prepare raw data (ensure .txt extension) ---")
    input_folder = input("Enter the path to the input folder containing your raw data files (e.g., C:\\Users\\YourUser\\Data): ")
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    intermediate_txt_folder = os.path.join(input_folder, "standardized_txt_files")
    os.makedirs(intermediate_txt_folder, exist_ok=True)
    print(f"Intermediate .txt files will be placed in: '{intermediate_txt_folder}'")

    txt_standardized_count = 0
    print("\nStandardizing file extensions to .txt...")
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        if os.path.isfile(filepath) and filepath != intermediate_txt_folder: # Skip directories and self-copy
            base_name, current_ext = os.path.splitext(filename)
            new_filename = base_name + '.txt'
            new_filepath = os.path.join(intermediate_txt_folder, new_filename)
            
            try:
                shutil.copy2(filepath, new_filepath)
                txt_standardized_count += 1
            except Exception as e:
                print(f"  ERROR: Could not copy '{filename}' to .txt: {e}")

    print(f"\nFinished Step 1. {txt_standardized_count} files prepared in '{intermediate_txt_folder}'.")
    print("\n" + "=" * 50 + "\n")

    print("--- Step 2: Convert .txt files to .csv ---")
    final_csv_output_folder = os.path.join(input_folder, "final_csv_files")
    os.makedirs(final_csv_output_folder, exist_ok=True)
    print(f"Final .csv files will be saved in: '{final_csv_output_folder}'")

    delimiter_input = input(r"Enter the primary delimiter used IN MOST OF YOUR ORIGINAL TEXT DATA (e.g., , for comma, \t for tab, ' ' for space/auto-detect): ")
    
    general_delimiter = None
    if delimiter_input == r'\t':
        general_delimiter = '\t'
    elif delimiter_input == ' ' or delimiter_input == r'\s':
        general_delimiter = None
    else:
        general_delimiter = delimiter_input

    csv_converted_count = 0
    skipped_conversion_count = 0

    txt_files_to_convert = [f for f in os.listdir(intermediate_txt_folder) if f.endswith('.txt')]

    if not txt_files_to_convert:
        print(f"No .txt files found in '{intermediate_txt_folder}' to convert to CSV.")
        return

    for filename in txt_files_to_convert:
        input_filepath = os.path.join(intermediate_txt_folder, filename)
        output_filename = os.path.splitext(filename)[0] + '.csv'
        output_filepath = os.path.join(final_csv_output_folder, output_filename)

        print(f"Converting: '{filename}' to '{output_filename}'")

        is_nf_file = 'NF' in filename.upper()

        df = None
        read_success = False
        
        encodings_to_try = []
        try:
            with open(input_filepath, 'rb') as f:
                raw_data_sample = f.read(10000)
            detection = chardet.detect(raw_data_sample)
            if detection['encoding'] and detection['confidence'] > 0.8:
                encodings_to_try.append(detection['encoding'])
            encodings_to_try.extend(['latin-1', 'cp1252', 'utf-8'])
            encodings_to_try = list(dict.fromkeys(encodings_to_try))

        except Exception as e:
            print(f"  ERROR: Could not detect encoding for '{filename}': {e}. Skipping conversion.")
            skipped_conversion_count += 1
            continue

        # --- NF File Specific Parsing (Revised) ---
        if is_nf_file:
            print(f"  Applying special NF parsing for '{filename}'...")
            for enc in encodings_to_try:
                try:
                    # Step 1: Read the header line directly as a string to avoid pandas' initial interpretation
                    with open(input_filepath, 'r', encoding=enc) as f:
                        header_line = f.readline().strip() # Read first line and remove leading/trailing whitespace
                        
                    print(f"  --- DEBUG: NF Raw header line (from direct read): '{header_line}'")
                        
                    # Split header by any sequence of whitespace (spaces or tabs)
                    header_cols = re.split(r'\s+|\t+', header_line)

                    if len(header_cols) != 2:
                        raise ValueError(f"NF Header format unexpected after direct read: '{header_line}'. Expected two column names.")

                    # Step 2: Read the rest of the file (skipping the header) assuming tab-separated data
                    # Use header=None because we're providing column names manually
                    # Use engine='python' for robust tab splitting
                    df = pd.read_csv(input_filepath, sep='\t', encoding=enc, header=None, skiprows=1, engine='python')
                    
                    if df.empty:
                        raise ValueError("File is empty or contains no data rows after skipping header.")
                    
                    # Step 3: Handle the case where the data might be in one column or two
                    if df.shape[1] == 1:
                        # If pandas read the whole data line into one column (e.g., "Freq\t(Mag,Phase)"), split it
                        data_series_to_split = df.iloc[:, 0].astype(str)
                        split_data = data_series_to_split.str.split('\t', n=1, expand=True) # Split only at the first tab
                        if split_data.shape[1] != 2:
                            raise ValueError("NF Data rows did not split into two columns by tab delimiter (case: single column read).")
                        df = pd.DataFrame({header_cols[0]: split_data[0], header_cols[1]: split_data[1]})
                    elif df.shape[1] == 2:
                        # If pandas already split it into two columns correctly, just assign names
                        df.columns = header_cols
                    else:
                        raise ValueError(f"NF Data rows parsed into {df.shape[1]} columns, expected 1 or 2. Content: {df.head().to_string()}")
                    
                    read_success = True
                    break # Break out of encoding loop if NF parsing was successful
                except Exception as e_nf_parse:
                    print(f"  NF Specific Parsing Error for '{filename}' with encoding '{enc}': {e_nf_parse}. Trying next encoding for NF.")
                    continue # Try next encoding for NF specific parsing

        # --- General CSV Parsing (for non-NF files or if NF specific parsing failed) ---
        if not read_success: 
            print(f"  Attempting general CSV parsing for '{filename}' (not NF or NF parsing failed)...")
            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(input_filepath, sep=general_delimiter, encoding=enc, header='infer', engine='python') # Added engine='python' for robustness
                    read_success = True
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e_read:
                    print(f"  Warning: Could not read '{filename}' with encoding '{enc}' and delimiter '{general_delimiter}': {e_read}. Trying next.")
                    continue

        # --- Final check and save ---
        if not read_success or df is None:
            print(f"  ERROR: Failed to read or parse '{filename}' with any tried method. Skipping conversion.")
            skipped_conversion_count += 1
            continue
            
        try:
            # Save as CSV (comma-separated by default)
            df.to_csv(output_filepath, index=False)
            print(f"  Saved as: '{output_filepath}'")
            csv_converted_count += 1
        except Exception as e_save:
            print(f"  ERROR: Failed to save '{output_filename}' as CSV: {e_save}. Skipping.")
            skipped_conversion_count += 1
            continue

    print("\n--- Conversion Summary ---")
    print(f"Total files prepared (standardized to .txt): {txt_standardized_count}")
    print(f"Successfully converted to .csv: {csv_converted_count} files")
    print(f"Skipped during .csv conversion: {skipped_conversion_count} files")
    print(f"Standardized .txt files are in: '{intermediate_txt_folder}'")
    print(f"Final .csv files are in: '{final_csv_output_folder}'")
    print("--- Script Finished ---")

if __name__ == "__main__":
    raw_data_to_standard_csv_converter()