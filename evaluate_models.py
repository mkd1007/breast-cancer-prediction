# src/evaluate_models.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

# Load the processed data
data_path = "data/processed/wdbc_processed.csv"
data = pd.read_csv(data_path)

# Split the data into features and labels
X = data.drop(columns=["Diagnosis"])
y = data["Diagnosis"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
model_dir = "models"
model_names = ["SVM", "RandomForest", "KNN", "LogisticRegression"]
models = {name: joblib.load(os.path.join(model_dir, f"{name}.joblib")) for name in model_names}

# Evaluate models
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred_proba)
    }

# Print results
for name, metrics in results.items():
    print(f"Model: {name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

print("All models evaluated.")
