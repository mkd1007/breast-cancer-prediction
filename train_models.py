# src/train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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

# Define models
models = {
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=10000, random_state=42)
}

# Train and save models
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(model_dir, f"{name}.joblib"))
    print(f"{name} model trained and saved.")

print("All models trained and saved.")
