#!/usr/bin/env python3
"""Predict student placement status from tabular features.

Usage:
    python predict.py --input some_students.csv --output predictions.csv
"""

import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd

# Expected feature columns for the pipeline
EXPECTED_FEATURES = [
    "department",
    "city_tier",
    "cgpa",
    "backlogs",
    "internship_months",
    "coding_score",
    "projects",
    "workshops",
    "status_code",
]

# Required columns in the input CSV (including identifier)
REQUIRED_COLUMNS = ["student_id"] + EXPECTED_FEATURES


def load_model(model_path: Path):
    """Load pre-trained model artifact without refitting."""
    if not model_path.exists():
        sys.stderr.write(f"Error: Model artifact not found at {model_path}.\n")
        sys.stderr.write(
            "Please train and serialize the model first (see analysis.ipynb).\n"
        )
        sys.exit(1)
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        sys.stderr.write(f"Error loading model artifact from {model_path}: {e}\n")
        sys.exit(1)


def predict(input_path: Path, output_path: Path, model_path: Path) -> None:
    """Run inference on input CSV and write binary predictions to output CSV."""
    if not input_path.exists():
        sys.stderr.write(f"Error: Input file does not exist: {input_path}\n")
        sys.exit(1)

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        # Gracefully handle completely empty files
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["student_id", "placed"]).to_csv(output_path, index=False)
        print(f"Empty input file processed. Wrote empty schema to {output_path}")
        return
    except Exception as e:
        sys.stderr.write(f"Error reading input CSV {input_path}: {e}\n")
        sys.exit(1)

    # Validate that required columns are present
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        sys.stderr.write(
            f"Error: Missing required column(s) in input CSV: {missing_cols}\n"
        )
        sys.stderr.write(f"Expected required columns: {REQUIRED_COLUMNS}\n")
        sys.exit(1)

    # Handle empty dataset (columns present, 0 rows)
    if len(df) == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(columns=["student_id", "placed"])
        out_df.to_csv(output_path, index=False)
        print(f"Empty dataset (0 rows). Wrote empty schema to {output_path}")
        return

    # Load fitted pipeline
    model = load_model(model_path)

    # Select only the feature columns expected by the model pipeline
    X = df[EXPECTED_FEATURES]

    # Predict binary outcomes (0 or 1)
    try:
        preds = model.predict(X)
    except Exception as e:
        sys.stderr.write(f"Error generating predictions through model pipeline: {e}\n")
        sys.exit(1)

    # Ensure integer 0 or 1
    placed_binary = [int(p) for p in preds]

    # Construct output DataFrame preserving input row order and student_id
    out_df = pd.DataFrame({
        "student_id": df["student_id"],
        "placed": placed_binary,
    })

    # Validate schema
    if list(out_df.columns) != ["student_id", "placed"]:
        sys.stderr.write("Internal Error: Output columns do not match expected schema.\n")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(out_df)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict student placement status using pre-trained model."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to input CSV containing student records.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to output CSV to write predictions.",
    )
    args = parser.parse_args()

    # Resolve model artifact relative to predict.py directory
    model_path = Path(__file__).resolve().parent / "model.joblib"
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    predict(input_path, output_path, model_path)


if __name__ == "__main__":
    main()
