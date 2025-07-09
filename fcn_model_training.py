import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight # <--- NEW IMPORT
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

print("--- Starting FCN Model Training Script ---")

# --- Configuration ---
processed_data_dir = 'processed_fr_dataset' # Directory where your .npy files are saved
model_output_dir = 'fcn_model_output'      # Directory to save trained model and plots

# Create output directories if they don't exist
os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

# Training parameters
BATCH_SIZE = 16
EPOCHS = 10 # Increased epochs because EarlyStopping will manage it
VALIDATION_SPLIT = 0.2 # 20% of data for validation

# --- 1. Load the Processed Dataset ---
print(f"\nLoading data from '{processed_data_dir}'...")
try:
    X = np.load(os.path.join(processed_data_dir, 'X_fr_data.npy'))
    y = np.load(os.path.join(processed_data_dir, 'y_fr_labels.npy'))
    fault_class_names = np.load(os.path.join(processed_data_dir, 'fault_class_names.npy'), allow_pickle=True)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Data files not found in '{processed_data_dir}'. Please run databaseCreation.py first.")
    exit()

# Verify data shapes
print(f"Features (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")
print(f"Fault Class Names: {fault_class_names.tolist()}")

if X.shape[0] != y.shape[0]:
    print("ERROR: Number of samples in X and y do not match. Please check your database creation script.")
    exit()

num_samples, sequence_length, num_features = X.shape
num_classes = y.shape[1]

# --- 2. Split Data into Training, Validation, and Test Sets ---
print("\nSplitting data into training, validation, and test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=VALIDATION_SPLIT + 0.1, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Validation set shape: X_val={X_val.shape}, y_val={y_val.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# --- NEW: Calculate Class Weights ---
# Convert one-hot encoded y_train back to class labels for class_weight calculation
y_train_labels = np.argmax(y_train, axis=1)

print("\nCalculating class weights for imbalanced dataset...")
# 'balanced' mode automatically computes weights inversely proportional to class frequencies
# i.e., smaller classes get higher weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
# Convert to a dictionary as expected by Keras
class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}

print(f"Computed Class Weights: {class_weights_dict}")

# Optional: Print training set class distribution to understand imbalance
print("\nTraining set class distribution (for reference):")
unique_train, counts_train = np.unique(y_train_labels, return_counts=True)
for i, count in zip(unique_train, counts_train):
    print(f"  Class {i} ({fault_class_names[i]}): {count} samples")


# --- 3. Define the FCN Model Architecture ---
print("\nDefining the Fully Convolutional Network (FCN) model...")

def build_fcn_model(input_shape, num_classes):
    input_layer = keras.Input(shape=input_shape)

    # First Convolutional Block with Dropout
    conv1 = layers.Conv1D(filters=512, kernel_size=8, padding='same')(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv1 = layers.Dropout(0.4)(conv1) # Added Dropout

    # Second Convolutional Block with Dropout
    conv2 = layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Dropout(0.3)(conv2) # Added Dropout

    # Third Convolutional Block with Dropout
    conv3 = layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)
    conv3 = layers.Dropout(0.2)(conv3) # Added Dropout

    # Global Average Pooling for classification
    gap_layer = layers.GlobalAveragePooling1D()(conv3)

    # Output Layer
    output_layer = layers.Dense(num_classes, activation='softmax')(gap_layer)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

model = build_fcn_model(input_shape=(sequence_length, num_features), num_classes=num_classes)
model.summary()

# --- 4. Compile the Model ---
print("\nCompiling the model...")

# Step 2: Create an instance of the Adam optimizer with a specified learning rate
# Common starting points for learning rate: 0.001 (default), 0.0005, 0.0001
# Let's try 0.0005 as an example, as a slightly lower rate can sometimes help with stability and generalization
custom_optimizer = Adam(learning_rate=0.0005)

# Step 3: Pass the custom optimizer instance to model.compile()
model.compile(optimizer=custom_optimizer, # Use the custom optimizer here
              loss='categorical_crossentropy',
              metrics=['accuracy']
            )





# --- 5. Train the Model ---
print("\nStarting model training...")

# Define EarlyStopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15, # Increased patience slightly as training might be more volatile with class weights
    restore_best_weights=True,
    verbose=1
)

history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping], # Use early stopping
                    class_weight=class_weights_dict, # <--- NEW: Apply class weights
                    verbose=1)

print("\nModel training finished.")

# --- 6. Evaluate the Model on Test Set and Generate Final Plots ---
print("\nEvaluating model on the test set and generating final plots...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Generate classification report
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=fault_class_names))

# Generate and save Confusion Matrix (Heatmap)
print("\nGenerating Confusion Matrix heatmap...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=fault_class_names, yticklabels=fault_class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(model_output_dir, 'confusion_matrix.png'))
plt.show()

# Plot and save Training History (Accuracy and Loss)
print("\nGenerating Training History plots (Accuracy and Loss)...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(model_output_dir, 'training_history.png'))
plt.show()

# --- 7. Save the Trained Model ---
model_save_path = os.path.join(model_output_dir, 'fcn_transformer_fault_model.keras')
model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

print("\n--- FCN Model Training Script Finished ---")