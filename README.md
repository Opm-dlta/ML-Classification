# ML-Classification
A full end‑to‑end machine learning pipeline for classifying cow behaviour using accelerometer sensor data. This project applies time‑aware data splitting, feature preprocessing, model training, hyperparameter tuning, and dimensionality education techniques to build reliable behavior prediction models for Japanese Black Beef cattle.

📚 Table of Contents
Project Overview

Features

Dataset

Getting Started

Prerequisites

Installation

Configuration

Usage Examples

Technical Details

Contributing

Testing

Roadmap

Known Issues

License

Credits

📌 Project Overview
This project builds a complete machine learning workflow to classify cow behaviour from accelerometer sensor data collected across multiple animals and sessions.
It focuses on time‑series integrity, class imbalance, and model comparison, producing reproducible evaluation results and exportable summary files.

⭐ Features
Full ML pipeline: loading → cleaning → preprocessing → training → evaluation

Time‑aware session‑based splitting (prevents leakage)

Median imputation + standardization

Class imbalance handling using class_weight="balanced"

Multiple models: MLP, SVM, Random Forest

Hyperparameter tuning (MLP alpha sensitivity)

Dimensionality reduction: PCA, LDA

Feature selection: SelectKBest, RFE, RF importance

Correlation analysis

Visualizations (Seaborn + Matplotlib)

Exported CSV summaries for reporting

📊 Dataset
The dataset contains:

Accelerometer sensor readings

Multiple animals

Multiple time‑ordered sessions

Behaviour labels (multi‑class, imbalanced)

Required columns:

Code
session, animal, behaviour
🛠️ Getting Started
✔️ Prerequisites
Install the following:

Python 3.10+

pip 23+

Recommended: virtual environment (venv or conda)

Python libraries used:

Code
numpy
pandas
matplotlib
seaborn
scikit-learn
✔️ Installation
1. Clone the repository
bash
git clone https://github.com/yourusername/cow-behaviour-classification.git
cd cow-behaviour-classification
2. Create a virtual environment
bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
3. Install dependencies
bash
pip install -r requirements.txt
✔️ Configuration
1. Set your dataset path
Open the main script and update:

python
DATA_PATH = r"C:\path\to\your\cow_accelerometer_data.xlsx"
2. Ensure your dataset contains:
session

animal

behaviour

Sensor feature columns

✔️ Usage Examples
Run the full pipeline:
bash
python cow_behavior_classification.py
Output files generated:
Summary.csv

Sensitivity.csv

random_forest_experiments.csv

Example: training only the Random Forest model
python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight="balanced_subsample")
model.fit(X_train_proc, y_train)
pred = model.predict(X_test_proc)
🧠 Technical Details
✔️ Time‑Aware Splitting
Sessions are sorted chronologically and split:

70% training

15% validation

15% testing

This prevents future data leaking into the past.

✔️ Preprocessing
Median imputation

StandardScaler normalization

Class weights for imbalance

✔️ Models
MLPClassifier (with early stopping)

SVC (RBF kernel)

RandomForestClassifier

✔️ Feature Reduction
PCA (95% variance)

LDA (supervised)

SelectKBest

RFE

RF importance ranking

🤝 Contributing
We welcome contributions!

How to contribute:
Fork the repository

Create a feature branch

Commit your changes

Open a pull request

Issue guidelines:
Use descriptive titles

Include steps to reproduce

Attach logs or screenshots

🧪 Testing
Run tests (if you add a test suite):

bash
pytest tests/
Or basic script validation:

bash
python cow_behavior_classification.py --dry-run
🗺️ Roadmap
Planned Features
Add LSTM/CNN deep learning models

Add sliding‑window segmentation

Add SHAP explainability

Add Streamlit dashboard

Add Docker container support

🐞 Known Issues
Some behaviour classes may have extremely low sample counts

Sensor noise may affect PCA variance

RFE may be slow on large datasets

No built‑in cross‑validation yet
