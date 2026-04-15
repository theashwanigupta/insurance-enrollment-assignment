"""Train and persist insurance enrollment prediction model.

This script reproduces the notebook's ML flow in a reusable form:
- loads data
- applies feature engineering (age_group, salary_band)
- trains Logistic Regression and Random Forest pipelines
- selects best model by ROC-AUC then F1
- saves fitted pipeline + metadata in model directory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "enrolled"

NUMERIC_FEATURES = ["tenure_years"]
CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "employment_type",
    "region",
    "has_dependents",
    "age_group",
    "salary_band",
]
IDENTIFIER_FEATURES = ["employee_id"]


def engineer_features(df: pd.DataFrame, salary_bins: list[float] | None = None) -> pd.DataFrame:
    """Apply notebook-consistent feature engineering."""
    out = df.copy()
    out["age_group"] = pd.cut(
        out["age"],
        bins=[20, 30, 40, 50, 60, 70],
        labels=["20s", "30s", "40s", "50s", "60+"],
        include_lowest=True,
    )
    if salary_bins is None:
        _, bins = pd.qcut(out["salary"], q=4, retbins=True, duplicates="drop")
        salary_bins = bins.tolist()

    labels = ["Low", "Mid", "High", "Very High"][: max(1, len(salary_bins) - 1)]
    out["salary_band"] = pd.cut(
        out["salary"], bins=salary_bins, labels=labels, include_lowest=True
    )
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
            ("id", "drop", IDENTIFIER_FEATURES),
        ]
    )


def train_and_select(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    results: dict[str, dict[str, float]] = {}
    pipelines: dict[str, Pipeline] = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
        pipelines[name] = pipeline

    # Notebook-style priority: ROC-AUC first, then F1
    best_name = max(results.keys(), key=lambda n: (results[n]["roc_auc"], results[n]["f1"]))
    return best_name, pipelines[best_name], results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train insurance enrollment model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--model-dir", default="Assignment/model", help="Directory to save model artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(data_path)
    _, salary_bins = pd.qcut(df_raw["salary"], q=4, retbins=True, duplicates="drop")
    salary_bins = salary_bins.tolist()
    df = engineer_features(df_raw, salary_bins=salary_bins)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    best_name, best_pipeline, results = train_and_select(X_train, y_train, X_test, y_test)

    model_path = model_dir / "enrollment_model.joblib"
    metadata_path = model_dir / "metadata.json"

    payload = {
        "model": best_pipeline,
        "model_name": best_name,
        "feature_engineering": {
            "age_bins": [20, 30, 40, 50, 60, 70],
            "age_labels": ["20s", "30s", "40s", "50s", "60+"],
            "salary_band_labels": ["Low", "Mid", "High", "Very High"][: max(1, len(salary_bins) - 1)],
            "salary_bins": salary_bins,
        },
    }
    joblib.dump(payload, model_path)

    metadata = {
        "selected_model": best_name,
        "metrics": results,
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Training completed. Best model: {best_name}")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
