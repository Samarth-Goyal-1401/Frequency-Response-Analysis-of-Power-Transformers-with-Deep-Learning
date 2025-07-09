import os
import numpy as np
import matplotlib.pyplot as plt

print("--- Starting Data Visualization Script ---")

# --- Configuration ---
processed_data_dir = 'processed_fr_dataset' # Directory where databaseCreation.py saved the .npy files
output_image_dir = 'data_visualizations'    # Directory to save the generated image plots
num_samples_to_plot = 5                     # Number of samples to plot from each fault type

# Create output directory if it doesn't exist
os.makedirs(output_image_dir, exist_ok=True)
print(f"Image plots will be saved in: '{output_image_dir}'")

# --- 1. Load the Processed Data ---
print(f"Loading data from '{processed_data_dir}'...")
try:
    X = np.load(os.path.join(processed_data_dir, 'X_fr_data.npy'))
    y_one_hot = np.load(os.path.join(processed_data_dir, 'y_fr_labels.npy'))
    fault_class_names = np.load(os.path.join(processed_data_dir, 'fault_class_names.npy'), allow_pickle=True)
    print("Data loaded successfully.")
    print(f"Features (X) shape: {X.shape}")
    print(f"Labels (y_one_hot) shape: {y_one_hot.shape}")
    print(f"Fault classes: {fault_class_names.tolist()}")
except FileNotFoundError:
    print(f"ERROR: Processed data files not found in '{processed_data_dir}'.")
    print("Please ensure 'databaseCreation.py' has been run successfully and saved the .npy files there.")
    exit()
except Exception as e:
    print(f"ERROR: An error occurred while loading data: {e}")
    exit()

if X.shape[0] == 0:
    print("ERROR: Loaded dataset is empty. No graphs to plot.")
    exit()

# Convert one-hot labels back to integer labels for easier indexing
y_labels = np.argmax(y_one_hot, axis=1)

# --- 2. Generate and Save Plots ---
print("\nGenerating and saving plots...")
plotted_count = 0
for class_idx, class_name in enumerate(fault_class_names):
    # Find indices for the current fault type
    indices = np.where(y_labels == class_idx)[0]
    
    if len(indices) == 0:
        print(f"  No samples found for class '{class_name}'.")
        continue

    print(f"  Plotting up to {num_samples_to_plot} samples for class '{class_name}'...")
    
    # Plot a subset of samples from this class
    for i in range(min(num_samples_to_plot, len(indices))):
        sample_index = indices[i]
        
        # Extract features for the current sample
        # X[sample_index, :, 0] is magnitude_log
        # X[sample_index, :, 1] is phase_normalized
        magnitude_log = X[sample_index, :, 0]
        phase_normalized = X[sample_index, :, 1]
        
        # Create a "frequency" axis (since actual freq values aren't in X)
        frequency_points = np.arange(X.shape[1]) # Just use index as x-axis

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot Magnitude
        plt.subplot(2, 1, 1) # 2 rows, 1 column, 1st plot
        plt.plot(frequency_points, magnitude_log, color='blue')
        plt.title(f'Sample {sample_index} ({class_name}) - Magnitude (Log10)')
        plt.ylabel('Log10 Magnitude')
        plt.grid(True)

        # Plot Phase
        plt.subplot(2, 1, 2) # 2 rows, 1 column, 2nd plot
        plt.plot(frequency_points, phase_normalized, color='red')
        plt.xlabel('Frequency Point Index (0 to 5001)')
        plt.ylabel('Normalized Phase (0-1)')
        plt.grid(True)

        plt.tight_layout() # Adjust layout to prevent overlapping titles/labels

        # Save the plot as a PNG image
        plot_filename = f'{class_name}_sample_{sample_index}.png'
        plot_filepath = os.path.join(output_image_dir, plot_filename)
        plt.savefig(plot_filepath)
        plt.close() # Close the plot to free memory

        plotted_count += 1

print(f"\nFinished plotting. {plotted_count} graphs saved as PNG images in '{output_image_dir}'.")
print("--- Script Finished ---")




