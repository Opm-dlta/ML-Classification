# Assignment for the Japanese Black Beef Cow Behavior Classification Dataset
# Replace DATA_PATH below with the actual CSV file location on your machine or Google Drive.
# Upgraded and adjusted code for Assigment 1 machine learning 
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

warnings.filterwarnings("ignore")

# 1 Load the dataset
def load_dataset(path):
    lower_path = path.lower()
    if lower_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


DATA_PATH =r"C:\Users\Lunga\Desktop\study new\machine learning\cow_accelerometer_data.xlsx"
data = load_dataset(DATA_PATH)

print("Dataset loaded from:", DATA_PATH)
print("Shape:", data.shape)
print("\nFirst 10 rows:")
first_rows = data.head(10).copy()
first_rows.index = np.arange(1, len(first_rows) + 1)
print(first_rows)

# 2 Basic dataset inspection
required_columns = {"session", "animal"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
    raise ValueError(f"The dataset is missing required columns: {sorted(missing_columns)}")

if "behaviour" in data.columns:
    target_column = "behaviour"
elif "behavior" in data.columns:
    target_column = "behavior"
else:
    raise ValueError("The dataset does not contain a target column named 'behaviour' or 'behavior'.")

if target_column != "behaviour":
    data = data.rename(columns={target_column: "behaviour"})

print("\nMissing values per column:")
print(data.isnull().sum())

# Simple imputation is applied later, but this is useful to know whether the dataset is clean.
if data.isnull().sum().sum() > 0:
    data = data.dropna()

print("\nDataset size after cleaning:", data.shape)
print("Number of classes:", data["behaviour"].nunique())
print("Class distribution:\n", data["behaviour"].value_counts().sort_index())
print("\nBehaviour frequency by animal:\n", data.groupby("animal")["behaviour"].value_counts().unstack(fill_value=0))

behaviour_counts = data["behaviour"].value_counts().sort_index()
animal_behaviour_counts = data.groupby(["animal", "behaviour"]).size().unstack(fill_value=0)

print("\nAssignment-ready summary notes:")
print(f"- Dataset size after cleaning: {data.shape[0]} rows and {data.shape[1]} columns")
print(f"- Number of classes: {data['behaviour'].nunique()}")
print(f"- Most frequent behaviour: {behaviour_counts.idxmax()} ({behaviour_counts.max()} samples)")
print(f"- Least frequent behaviour: {behaviour_counts.idxmin()} ({behaviour_counts.min()} samples)")
print("- The global distribution is useful for reporting imbalance, while the animal-wise table helps check whether the same pattern is consistent across animals.")
print("- A session-wise split was used because the data is time-ordered sensor data and random splitting would risk information leakage.")

# 3 prepare feature matrix and target vector
# For the classification task, I separate the features from the target variable.
# The features are all columns except for "session", "animal", and "behaviour".
# The target variable is the "behaviour" column, which I convert to string type for classification.

feature_columns = [c for c in data.columns if c not in ["session", "animal", "behaviour", "__row_order"]]
X = data[feature_columns]
y = data["behaviour"].astype(str)

print("\nFeature matrix and target vector prepared.")
print("Number of features:", X.shape[1])
print("Target labels:", sorted(y.unique()))

# 4 Time-aware split (session-wise, respecting order)
# This dataset is sequence-based, so a random split would leak information across sessions.
# A safer approach is to split by session in chronological order and keep each session intact.
# If you have a real timestamp column, use that instead of "session" for the ordering.
data = data.copy()
data["__row_order"] = np.arange(len(data))
data = data.sort_values(["session", "__row_order"], kind="mergesort").reset_index(drop=True)

X = data[feature_columns]
y = data["behaviour"].astype(str)

sessions = sorted(data["session"].unique())
train_end = int(len(sessions) * 0.70)
val_end = int(len(sessions) * 0.85)

train_sessions = sessions[:train_end]
val_sessions = sessions[train_end:val_end]
test_sessions = sessions[val_end:]

train_mask = data["session"].isin(train_sessions)
val_mask = data["session"].isin(val_sessions)
test_mask = data["session"].isin(test_sessions)

X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()
X_val = X.loc[val_mask].copy()
y_val = y.loc[val_mask].copy()
X_test = X.loc[test_mask].copy()
y_test = y.loc[test_mask].copy()

print("\nSplit sizes:")
print("Train:", X_train.shape[0])
print("Validation:", X_val.shape[0])
print("Test:", X_test.shape[0])

# 5 Preprocessing and normalization decision
# For this dataset, normalization is useful because the features are continuous sensor-derived values.
# It is especially important for MLP and SVM. Tree-based models do not strictly require it, but
# applying the same preprocessing keeps the comparison fair.

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
X_val_proc = scaler.transform(imputer.transform(X_val))
X_test_proc = scaler.transform(imputer.transform(X_test))

# 6 Class imbalance handling
# Compute balanced weights for the training set so models that support sample_weight can use them.
class_labels = np.unique(y_train)
class_weights = dict(
    zip(class_labels, compute_class_weight(class_weight="balanced", classes=class_labels, y=y_train))
)
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
# I will use class_weight='balanced' for the classifiers to reduce bias towards the majority classes.
# This is a practical and commonly used strategy for imbalanced data.
# Add this later when I get the data 


# 7 Utility functions
# Compute evaluation metric for the given model 
def evaluate_model(y_true, y_pred, model_name):
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics

# Print out the table of data that will be input 
def print_metrics_table(rows, title):
    table = pd.DataFrame(rows)
    table = table.set_index("model")
    print(f"\n{title}\n")
    print(table.round(4).to_string())
    return table

def fit_model(model, X, y, sample_weight=None):
    if sample_weight is not None:
        try:
            return model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            return model.fit(X, y)
    return model.fit(X, y)


def fit_and_report(model, X_train_, y_train_, X_val_, y_val_, X_test_, y_test_, name, train_sample_weight=None):
    fit_model(model, X_train_, y_train_, sample_weight=train_sample_weight)
    val_pred = model.predict(X_val_)
    val_metrics = evaluate_model(y_val_, val_pred, name + "_val")

    # Refit on train + validation for a final test evaluation.
    combined_X = np.vstack([X_train_, X_val_])
    combined_y = np.concatenate([y_train_, y_val_])
    combined_sample_weights = compute_sample_weight(class_weight="balanced", y=combined_y)
    fit_model(model, combined_X, combined_y, sample_weight=combined_sample_weights)
    test_pred = model.predict(X_test_)
    test_metrics = evaluate_model(y_test_, test_pred, name + "_test")
    return val_metrics, test_metrics, val_pred, test_pred


# 8 Baseline classifiers: MLP, SVM, Random Forest
models = {}
models["MLP"] = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    alpha=0.0001,
    random_state=42,
    early_stopping=True,
)
models["SVM"] = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)
models["RandomForest"] = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

results = []
for name, model in models.items():
    val_metrics, test_metrics, _, _ = fit_and_report(
        model, X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test, name
    )
    results.append(val_metrics)
    results.append(test_metrics)

summary_table = print_metrics_table(results, "Validation and test results for the three classifiers")
summary_table.to_csv("Summary.csv")

# Print detailed reports for each model on  test .
for name, model in models.items():
    # Refit once more for a clean report on the test set.
    model.fit(np.vstack([X_train_proc, X_val_proc]), np.concatenate([y_train, y_val]))
    test_pred = model.predict(X_test_proc)
    print(f"\n===== {name} test report =====")
    print(classification_report(y_test, test_pred, zero_division=0))

# 9) MLP alpha tuning
alpha_values = [1e-4, 1e-3, 1e-2, 1e-1]
alpha_rows = []
for alpha in alpha_values:
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        alpha=alpha,
        random_state=42,
        early_stopping=True,
    )
    _val_metrics, _test_metrics, _, _ = fit_and_report(
        mlp, X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test, f"MLP_alpha_{alpha}"
    )
    alpha_rows.append({"alpha": alpha, "val_f1_macro": _val_metrics["f1_macro"], "test_f1_macro": _test_metrics["f1_macro"]})

alpha_table = pd.DataFrame(alpha_rows)
alpha_table.to_csv("Sensitivity.csv", index=False)
print("\nSensitivity:\n")
print(alpha_table.round(4).to_string(index=False))

# 10 Correlation analysis
# Analyse 

correlation_matrix = pd.DataFrame(X_train_proc, columns=feature_columns).corr()
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
correlated_pairs = [
    (column, row, round(correlation_matrix.loc[row, column], 4))
    for column in upper.columns
    for row in upper.index
    if abs(upper.loc[row, column]) > 0.95 and row != column
]
correlated_pairs = sorted(correlated_pairs, key=lambda x: abs(x[2]), reverse=True)
print("\nHighly correlated feature pairs (|corr| > 0.95):")
for item in correlated_pairs[:20]:
    print(item)


# 11 PCA, FLD and Random Forest experiment
# For the reduction methods, I fit on the training data only and transform train/validation/test.
# PCA is unsupervised, while FLD (LDA) is supervised and uses the class labels (this is for my note).
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_proc)
X_val_pca = pca.transform(X_val_proc)
X_test_pca = pca.transform(X_test_proc)

# FLD / LDA
lda = LinearDiscriminantAnalysis(n_components=min(3, len(np.unique(y_train)) - 1))
X_train_lda = lda.fit_transform(X_train_proc, y_train)
X_val_lda = lda.transform(X_val_proc)
X_test_lda = lda.transform(X_test_proc)

# Feature selection: SelectKBest and RFE
selector_kbest = SelectKBest(score_func=f_classif, k=20)
X_train_kbest = selector_kbest.fit_transform(X_train_proc, y_train)
X_val_kbest = selector_kbest.transform(X_val_proc)
X_test_kbest = selector_kbest.transform(X_test_proc)

rfe_estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=20)
X_train_rfe = rfe.fit_transform(X_train_proc, y_train)
X_val_rfe = rfe.transform(X_val_proc)
X_test_rfe = rfe.transform(X_test_proc)

# Random Forest feature-importance based selection
rf_importance = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1,
)
rf_importance.fit(X_train_proc, y_train)
importances = rf_importance.feature_importances_
selected_idx = np.argsort(importances)[::-1][:20]
X_train_rf_importance = X_train_proc[:, selected_idx]
X_val_rf_importance = X_val_proc[:, selected_idx]
X_test_rf_importance = X_test_proc[:, selected_idx]

# Evaluate RF(Random Forest) on all variants
rf_variants = {
    "Baseline_RF": (X_train_proc, X_val_proc, X_test_proc),
    "RF_with_PCA": (X_train_pca, X_val_pca, X_test_pca),
    "RF_with_LDA": (X_train_lda, X_val_lda, X_test_lda),
    "RF_with_SelectKBest": (X_train_kbest, X_val_kbest, X_test_kbest),
    "RF_with_RFE": (X_train_rfe, X_val_rfe, X_test_rfe),
    "RF_with_importance_selection": (X_train_rf_importance, X_val_rf_importance, X_test_rf_importance),
}

rf_rows = []
for name, (X_tr, X_va, X_te) in rf_variants.items():
    rf_model = RandomForestClassifier( #Class the animal behavior using Random Forest Classifier
        n_estimators=200,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_tr, y_train)
    val_pred = rf_model.predict(X_va)
    test_pred = rf_model.predict(X_te)

    val_metrics = evaluate_model(y_val, val_pred, name + "_val")
    test_metrics = evaluate_model(y_test, test_pred, name + "_test")
    rf_rows.append({
        "method": name,
        "val_f1_macro": val_metrics["f1_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
    })

rf_table = pd.DataFrame(rf_rows)
rf_table.to_csv("random_forest_experiments.csv", index=False)
print("\nRandom Forest experiments with dimensionality reduction and feature selection:\n")
print(rf_table.round(4).to_string(index=False))
print("\nSaved result files:")
print("- classification_summary.csv")
print("- mlp_alpha_sensitivity.csv")
print("- random_forest_experiments.csv")

# 12 Optional visualizations (remove if needed)
# A few simple plots to support the report.
plt.figure(figsize=(8, 4))
sns.barplot(data=alpha_table, x="alpha", y="test_f1_macro")
plt.title("MLP alpha vs. macro F1")
plt.xlabel("Alpha")
plt.ylabel("Macro F1")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 13 Discussion notes for the report

print("\nReport :")
print("- The temporal split uses sessions as the time axis, which avoids leakage between train/validation/test.")
print("- Standardization is applied because the features are continuous sensor values and MLP/SVM benefit from it.")
print("- Class imbalance is handled with class_weight='balanced' for the classifiers.")
print("- Macro, micro, and weighted metrics are reported because the classes are imbalanced and the assignment asks for multiple views of performance.")
print("- For significance testing in a multi-class setting, use paired bootstrap or permutation tests rather than a simple paired t-test.")
