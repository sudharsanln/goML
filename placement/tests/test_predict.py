import os
import subprocess
import sys
import tempfile
from pathlib import Path
import joblib
import pandas as pd
import pytest

PLACEMENT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PLACEMENT_DIR / "model.joblib"
PREDICT_SCRIPT = PLACEMENT_DIR / "predict.py"
SAMPLE_INPUT = PLACEMENT_DIR / "data" / "train.csv"


def test_1_model_artifact_loading():
    """Verify that model.joblib exists and loads cleanly."""
    assert MODEL_PATH.exists(), f"Model artifact missing at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded artifact missing predict method"
    assert hasattr(model, "predict_proba"), "Loaded artifact missing predict_proba method"


def test_2_normal_input_prediction():
    """Test standard execution of predict.py."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "preds.csv"
        # Take first 10 rows of train data
        df = pd.read_csv(SAMPLE_INPUT).head(10)
        in_csv = Path(tmpdir) / "input.csv"
        df.to_csv(in_csv, index=False)

        cmd = [sys.executable, str(PREDICT_SCRIPT), "--input", str(in_csv), "--output", str(out_csv)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"predict.py failed: {res.stderr}"
        assert out_csv.exists()
        preds_df = pd.read_csv(out_csv)
        assert len(preds_df) == 10


def test_3_input_with_missing_feature_values():
    """Verify inference succeeds when required features contain null values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "preds.csv"
        df = pd.read_csv(SAMPLE_INPUT).head(10).copy()
        # Inject nulls
        df.loc[0, "cgpa"] = None
        df.loc[1, "coding_score"] = None
        df.loc[2, "status_code"] = None
        df.loc[3, "backlogs"] = None
        in_csv = Path(tmpdir) / "missing_input.csv"
        df.to_csv(in_csv, index=False)

        cmd = [sys.executable, str(PREDICT_SCRIPT), "--input", str(in_csv), "--output", str(out_csv)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"predict.py failed on missing features: {res.stderr}"
        preds_df = pd.read_csv(out_csv)
        assert len(preds_df) == 10
        assert not preds_df["placed"].isnull().any()


def test_4_input_with_extra_irrelevant_column():
    """Verify predict.py ignores additional irrelevant columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "preds.csv"
        df = pd.read_csv(SAMPLE_INPUT).head(5).copy()
        df["favorite_hobby"] = ["Reading", "Chess", "Gaming", "Music", "Cricket"]
        df["random_score"] = [99, 88, 77, 66, 55]
        in_csv = Path(tmpdir) / "extra_cols.csv"
        df.to_csv(in_csv, index=False)

        cmd = [sys.executable, str(PREDICT_SCRIPT), "--input", str(in_csv), "--output", str(out_csv)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"predict.py failed on extra columns: {res.stderr}"
        preds_df = pd.read_csv(out_csv)
        assert list(preds_df.columns) == ["student_id", "placed"]


def test_5_missing_required_feature_column():
    """Verify predict.py exits with non-zero status when a required feature is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "preds.csv"
        df = pd.read_csv(SAMPLE_INPUT).head(5).copy()
        # Drop a required column
        df = df.drop(columns=["department"])
        in_csv = Path(tmpdir) / "invalid_input.csv"
        df.to_csv(in_csv, index=False)

        cmd = [sys.executable, str(PREDICT_SCRIPT), "--input", str(in_csv), "--output", str(out_csv)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "Missing required column" in res.stderr


def test_6_correct_output_column_names_and_order():
    """Verify output columns are exactly 'student_id,placed' in that order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "preds.csv"
        df = pd.read_csv(SAMPLE_INPUT).head(5)
        in_csv = Path(tmpdir) / "input.csv"
        df.to_csv(in_csv, index=False)

        cmd = [sys.executable, str(PREDICT_SCRIPT), "--input", str(in_csv), "--output", str(out_csv)]
        subprocess.run(cmd, check=True)
        with open(out_csv, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
        assert header_line == "student_id,placed"


def test_7_binary_prediction_values():
    """Verify predictions are strictly integers 0 or 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "preds.csv"
        df = pd.read_csv(SAMPLE_INPUT).head(50)
        in_csv = Path(tmpdir) / "input.csv"
        df.to_csv(in_csv, index=False)

        cmd = [sys.executable, str(PREDICT_SCRIPT), "--input", str(in_csv), "--output", str(out_csv)]
        subprocess.run(cmd, check=True)
        preds_df = pd.read_csv(out_csv)
        assert set(preds_df["placed"]).issubset({0, 1})
        assert pd.api.types.is_integer_dtype(preds_df["placed"])


def test_8_preservation_of_row_order_and_student_ids():
    """Verify input student_id sequence is preserved exactly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "preds.csv"
        df = pd.read_csv(SAMPLE_INPUT).sample(frac=1.0, random_state=123).head(20)
        expected_ids = list(df["student_id"])
        in_csv = Path(tmpdir) / "shuffled.csv"
        df.to_csv(in_csv, index=False)

        cmd = [sys.executable, str(PREDICT_SCRIPT), "--input", str(in_csv), "--output", str(out_csv)]
        subprocess.run(cmd, check=True)
        preds_df = pd.read_csv(out_csv)
        assert list(preds_df["student_id"]) == expected_ids


def test_9_invocation_from_different_working_directory():
    """Verify predict.py works when invoked from any arbitrary CWD."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "preds.csv"
        df = pd.read_csv(SAMPLE_INPUT).head(5)
        in_csv = Path(tmpdir) / "input.csv"
        df.to_csv(in_csv, index=False)

        # Run from tmpdir as current working directory
        cmd = [sys.executable, str(PREDICT_SCRIPT.resolve()), "--input", str(in_csv), "--output", str(out_csv)]
        res = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
        assert res.returncode == 0, f"Failed when invoked from external cwd: {res.stderr}"
        assert out_csv.exists()


def test_10_proof_that_predict_does_not_train_model():
    """Verify model artifact timestamp remains completely unchanged during inference."""
    mtime_before = MODEL_PATH.stat().st_mtime_ns
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "preds.csv"
        df = pd.read_csv(SAMPLE_INPUT).head(10)
        in_csv = Path(tmpdir) / "input.csv"
        df.to_csv(in_csv, index=False)

        cmd = [sys.executable, str(PREDICT_SCRIPT), "--input", str(in_csv), "--output", str(out_csv)]
        subprocess.run(cmd, check=True)

    mtime_after = MODEL_PATH.stat().st_mtime_ns
    assert mtime_before == mtime_after, "Model artifact was modified during prediction!"
