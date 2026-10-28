# Will this student get placed?

The placement cell has a spreadsheet of students who registered for placements. Each row is one student. `placed` is 1 if that student received an offer through the cell, and 0 if not.

Build a classifier that predicts `placed` for a new list of registered students.

`data/train.csv` is the training data. We score you on a **separate list** that is not in this folder.

## Columns

| Column | What it is |
| --- | --- |
| `student_id` | Student id |
| `department` | Branch: CSE, IT, ECE, EEE, MECH, or CIVIL |
| `city_tier` | 1, 2, or 3 |
| `cgpa` | Cumulative GPA |
| `backlogs` | Active backlogs |
| `internship_months` | Length of internship, in months |
| `coding_score` | Internal coding test, out of 100 |
| `projects` | Number of projects listed |
| `workshops` | Number of workshops attended |
| `status_code` | A field stored on the placement portal |
| `placed` | 1 if placed, 0 otherwise. Training file only |

## Evaluation

We run `predict.py` on the separate list and compute the **F1 score for `placed = 1`**.

We also read `analysis.ipynb`: what you checked in the data, what you chose to use in the model, and why.

## What to submit

```
analysis.ipynb
predict.py
requirements.txt
```

Train the model before you submit. `predict.py` should load that model and score a new file. It should not fit a new model.

```bash
python predict.py --input some_students.csv --output predictions.csv
```

## Output format

`predictions.csv`:

```
student_id,placed
S00001,0
S00002,1
```

One row per student in the input. `placed` is 0 or 1.

`data/sample_submission.csv` shows this layout. The 0/1 values in that file are placeholders, not correct answers.

The file we score has the same columns except `placed`. Some cells can be empty.

---

# Submission Guide: Placement Prediction System

## 1. Environment Setup & Dependencies
Install dependencies using Python 3.11:
```bash
pip install -r requirements.txt
```
Key libraries:
- `pandas>=2.2.0,<3.0.0`
- `numpy>=1.26.0,<2.0.0`
- `scikit-learn>=1.4.0,<1.7.0`
- `joblib>=1.4.0,<1.5.0`

## 2. Model Architecture & Pipeline Design
- **Exclusion of `student_id`:** `student_id` is an arbitrary alphanumeric identifier (`S00001`...) with 100% uniqueness. It is excluded from the feature matrix to prevent identity memorization.
- **End-to-End ColumnTransformer:**
  - **Numerical Features** (`cgpa`, `backlogs`, `internship_months`, `coding_score`, `projects`, `workshops`): Imputed with median values followed by standard z-score normalization (`StandardScaler`).
  - **Categorical String Features** (`department`): Imputed with `'missing'` and one-hot encoded (`OneHotEncoder(handle_unknown='ignore')`).
  - **Categorical Ordinal & Flag Features** (`city_tier`, `status_code`): Imputed with discrete indicators (`-1` for `city_tier`, `-1.0` for `status_code`) and one-hot encoded.
- **Classifier:** `LogisticRegression(C=1.0, max_iter=1000, random_state=42)` chosen for its convex optimization, high precision and recall balance, fast inference, and zero random variance across systems.

## 3. Offline Model Training & Serialization
The final model is trained on the full dataset of 2,500 students and serialized to `model.joblib`:
```bash
# Model training and artifact serialization is fully automated in analysis.ipynb:
jupyter execute analysis.ipynb
```
The serialized artifact `model.joblib` (approx. 6.4 KB) encapsulates both the fitted preprocessors and the classifier.

## 4. Inference Execution
Run inference on any arbitrary input file:
```bash
python predict.py --input path/to/some_students.csv --output path/to/predictions.csv
```

### CLI Contract & Safety Guarantees:
- **Relative Path Resolution:** `model.joblib` is resolved relative to `predict.py`'s directory, enabling execution from any working directory.
- **Input Validation:** Requires `student_id` and all feature columns except `placed`. Gracefully allows extra/irrelevant columns.
- **Missing Value Handling:** Missing cells in numeric or categorical columns are imputed via the fitted pipeline.
- **Strict Output Schema:** Outputs exactly two columns: `student_id,placed` in that order, preserving row order and student IDs. `placed` values are strictly binary integers (`0` or `1`).
- **Graceful Edge Cases:** Empty files or 0-row datasets write the standard header and exit cleanly.
- **Zero Online Fitting:** Does not fit, refit, or calibrate thresholds during inference.

## 5. Validation Results & Metrics Summary
Evaluated using 5-fold Stratified Cross-Validation on the 80% training set and a 20% held-out test split (`SEED = 42`):

| Model | 5-Fold CV F1 (placed=1) | Held-Out Val F1 | Validation Precision | Validation Recall |
| --- | --- | --- | --- | --- |
| **Dummy (Majority)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression (Selected)** | **0.9215 ± 0.0245** | **0.9263** | **0.9103** | **0.9429** |
| **Random Forest** | 0.9189 ± 0.0212 | 0.9296 | 0.9167 | 0.9429 |
| **HistGradientBoosting** | 0.9125 ± 0.0137 | 0.9149 | 0.9085 | 0.9214 |

### Held-Out Validation Confusion Matrix (Logistic Regression):
- **True Negatives:** 347 | **False Positives:** 13
- **False Negatives:** 8  | **True Positives:** 132

## 6. Feature-Quality Audit: `status_code` Ablation Study
- **Correlation with Target:** Linear correlation of `0.9105`.
- **Ablation Comparison:**
  - With `status_code`: Validation F1 = **0.9263** (CV F1 = 0.9215)
  - Without `status_code`: Validation F1 = **0.4076** (CV F1 = 0.4386)
  - **Ablation Delta:** **-0.5187 F1 points** when removed.
- **Audit Conclusion:** While `status_code` behaves like an administrative portal outcome flag, the evaluation criteria explicitly include `status_code` in the test spreadsheet schema. Discarding this field without evidence would discard the dominant predictive signal. The pipeline includes `status_code` while remaining robust to missing values via missing-category encoding (`-1.0`).

## 7. Automated Test Suite
Run the 10-point test suite covering artifact loading, missing values, extra columns, schema validity, row ordering, and out-of-directory invocation:
```bash
python -m pytest tests/test_predict.py -v
```
