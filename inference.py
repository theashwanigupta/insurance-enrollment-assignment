"""Run inference for insurance enrollment prediction model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for enrollment model")
    parser.add_argument("--model-dir", default="Assignment/model", help="Directory containing model artifact")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", help="Single JSON record as string")
    group.add_argument("--record-file", help="Path to JSON file containing one record or list of records")
    return parser.parse_args()


def load_records(args: argparse.Namespace) -> pd.DataFrame:
    if args.record:
        obj = json.loads(args.record)
    else:
        obj = json.loads(Path(args.record_file).read_text(encoding="utf-8"))

    if isinstance(obj, dict):
        obj = [obj]
    return pd.DataFrame(obj)


def engineer_features(df: pd.DataFrame, feature_engineering: dict) -> pd.DataFrame:
    out = df.copy()
    age_bins = feature_engineering.get("age_bins", [20, 30, 40, 50, 60, 70])
    age_labels = feature_engineering.get("age_labels", ["20s", "30s", "40s", "50s", "60+"])
    salary_labels = feature_engineering.get("salary_band_labels", ["Low", "Mid", "High", "Very High"])
    salary_bins = feature_engineering.get("salary_bins")

    out["age_group"] = pd.cut(out["age"], bins=age_bins, labels=age_labels, include_lowest=True)
    if salary_bins:
        labels = salary_labels[: max(1, len(salary_bins) - 1)]
        out["salary_band"] = pd.cut(out["salary"], bins=salary_bins, labels=labels, include_lowest=True)
    else:
        # Fallback for older artifacts without persisted bins.
        quantiles = [0.25, 0.5, 0.75]
        edges = [float("-inf")] + [float(out["salary"].quantile(q)) for q in quantiles] + [float("inf")]
        out["salary_band"] = pd.cut(out["salary"], bins=edges, labels=salary_labels[:4], include_lowest=True)
    return out


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_dir) / "enrollment_model.joblib"

    payload = joblib.load(model_path)
    model = payload["model"]
    feature_engineering = payload.get("feature_engineering", {})

    records = load_records(args)
    engineered = engineer_features(records, feature_engineering)

    predictions = model.predict(engineered)
    probabilities = model.predict_proba(engineered)[:, 1]

    output = []
    for i in range(len(engineered)):
        output.append(
            {
                "record_index": i,
                "predicted_enrolled": int(predictions[i]),
                "enrollment_probability": float(probabilities[i]),
            }
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
