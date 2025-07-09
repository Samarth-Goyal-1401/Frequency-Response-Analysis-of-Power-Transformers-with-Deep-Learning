Transformer Fault Diagnosis using SFRA and Deep Learning
This project develops a deep learning framework for automated power transformer fault diagnosis using Sweep Frequency Response Analysis (SFRA) data. It classifies various fault types (e.g., LDF, IDF, NF, DLF, RDF, SCF) by leveraging advanced neural network architectures.

🚀 Features
Automated Data Preprocessing: Transforms raw SFRA (CSV/XLSX) into a standardized numerical dataset.

Feature Engineering: Extracts and preprocesses magnitude (log-transformed) and phase (normalized) components.

Dataset Creation: Generates a unified NumPy dataset from hundreds of SFRA samples.

Deep Learning Model Training: Implements and trains a Fully Convolutional Network (FCN) for sequential data.

Robust Evaluation: Provides detailed model performance metrics (accuracy, classification reports, confusion matrices).

Model Persistence: Saves trained models for future use.

🛠️ Technologies & Algorithms Used
Programming Language: Python

Core Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Chardet, Openpyxl.

Deep Learning Frameworks & Algorithms: TensorFlow/Keras, Fully Convolutional Network (FCN). Actively exploring ResNet and Xception for image-based SFRA data.

📊 Dataset
Processes SFRA data for Line-to-Ground Fault (LDF), Inter-Disk Fault (IDF), No Fault (NF), Deformed Low Voltage Winding Fault (DLF), Radial Displacement Fault (RDF), and Short Circuit Fault (SCF). The dataset consists of hundreds of SFRA samples, each with 4999 frequency points and 2 features (magnitude, phase).

📂 Project Structure
.
├── databaseCreation.py
├── fcn_model_training.py
├── processed_fr_dataset/
│   ├── X_fr_data.npy
│   ├── y_fr_labels.npy
│   └── fault_class_names.npy
└── fcn_model_output/
    ├── confusion_matrix.png
    ├── training_history.png
    └── fcn_transformer_fault_model.keras
├── LDF/
├── IDF/
├── NF/
├── DLF/
├── RDF/
└── SCF/

🚀 Setup and Installation
Clone the repository:

git clone <repository_url>
cd <repository_name>

Create a virtual environment (recommended):

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install dependencies:

pip install numpy pandas tensorflow scikit-learn matplotlib seaborn chardet openpyxl

🏃 Usage
Place your raw SFRA data: Organize raw SFRA files (CSV/XLSX) into subdirectories named by fault type (e.g., LDF/, SCF/).

Create the processed dataset: Run databaseCreation.py. Follow prompts for data_root_dir (e.g., .).

python databaseCreation.py

Train the FCN model: Run fcn_model_training.py. Training progress will display in the terminal. Plots and reports will be saved in fcn_model_output/.

python fcn_model_training.py

📈 Results (Placeholder)
Add your key performance metrics here after training, e.g.:

Test Accuracy: XX.XX%

Include screenshots of confusion_matrix.png and training_history.png here.

💡 Future Work
Hyperparameter Tuning (Keras Tuner with Hyperband).

Full ResNet & Xception Integration (with image data transformation).

Advanced Regularization & Cross-Validation.

Model Deployment.
