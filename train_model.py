"""
Diabetes Detection — Model Training
--------------------------------------
Trains Logistic Regression and Random Forest classifiers on patient
health data to predict diabetes risk, evaluates both, and saves the
best-performing model (plus encoders/scaler) for the Streamlit app.

Dataset: data/diabetes.csv
Columns: Patient_ID, Age, Gender, BMI, Blood_Pressure, Glucose_Level,
         Insulin_Level, Physical_Activity, Smoking, Family_History,
         High_Blood_Pressure, Diabetes

Usage:
    python train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "data/diabetes.csv"
TARGET_COL = "Diabetes"
ID_COL = "Patient_ID"
RANDOM_STATE = 42

CATEGORICAL_COLS = [
    "Gender",
    "Physical_Activity",
    "Smoking",
    "Family_History",
    "High_Blood_Pressure",
]
NUMERIC_COLS = ["Age", "BMI", "Blood_Pressure", "Glucose_Level", "Insulin_Level"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Class balance:\n{df[TARGET_COL].value_counts()}")
    return df


def clean_and_encode(df: pd.DataFrame):
    """Drop ID column, encode categoricals + target, return df + encoders."""
    df = df.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=ID_COL)

    df = df.dropna()

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le.classes_.tolist()  # e.g. ['Female', 'Male'] -> 0, 1

    target_le = LabelEncoder()
    df[TARGET_COL] = target_le.fit_transform(df[TARGET_COL])  # No=0, Yes=1
    encoders[TARGET_COL] = target_le.classes_.tolist()

    return df, encoders


def train_and_evaluate(df: pd.DataFrame):
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        }
        fitted[name] = model

        print(f"\n=== {name} ===")
        for k, v in results[name].items():
            print(f"{k.capitalize():10}: {v}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, zero_division=0))

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest performing model (by F1-Score): {best_name}")

    return fitted[best_name], best_name, scaler, results, feature_cols


def main():
    df = load_data(DATA_PATH)
    df, encoders = clean_and_encode(df)
    best_model, best_name, scaler, results, feature_cols = train_and_evaluate(df)

    # Save everything the Streamlit app needs
    joblib.dump(best_model, "model/diabetes_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    metadata = {
        "best_model": best_name,
        "feature_cols": feature_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "encoders": encoders,
        "uses_scaled_input": best_name == "Logistic Regression",
        "results": results,
    }
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model, scaler, and metadata to ./model/")


if __name__ == "__main__":
    main()
