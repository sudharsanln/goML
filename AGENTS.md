# Project Constraints and Engineering Standards

## 1. Overview
This repository contains a two-part technical take-home assignment:
- **Part A (`placement/`)**: Tabular binary classification predicting campus placement (`placed` $\in \{0, 1\}$).
- **Part B (`class_messages/`)**: Local LLM-driven structured task extraction from raw class communications.

---

## 2. Part A: Placement Prediction (`placement/`)
### Critical Constraints
- **Target**: `placed` (positive class = 1).
- **Primary Optimization Metric**: F1 score for `placed = 1` (positive class). Precision, recall, and confusion matrix are required secondary metrics; accuracy/ROC-AUC are supplementary only.
- **Model Training**: Model training, validation, feature ablation, and threshold selection MUST occur exclusively during offline development (in `analysis.ipynb`).
- **Persistence**: Save the complete end-to-end preprocessing + classifier pipeline to `placement/model.joblib`.
- **Inference Script (`predict.py`)**:
  - Command: `python predict.py --input <path> --output <path>`
  - Path Resolution: The model artifact path must resolve relative to `predict.py` itself, allowing execution from any arbitrary working directory.
  - Required Features: Validates that all expected features are present; accepts extra/irrelevant columns gracefully; processes missing values via the fitted preprocessing pipeline.
  - Strict Output Contract: Produces a CSV with exactly two columns (`student_id,placed`), preserving the input row order and identifiers. `placed` values must be strictly binary integers (`0` or `1`).
  - No Online Fitting: Must NEVER train, refit, or calibrate thresholds during inference.
  - Error Handling: Exits with non-zero status and an actionable error message on missing columns, missing model artifact, or unreadable inputs. Gracefully handles empty valid input files.
- **Feature-Quality & Leakage Audit**:
  - `student_id`: Pure identifier, must be excluded from predictive features.
  - `status_code`: Portal field investigated for target leakage / unstable proxy behavior. An explicit ablation study (with vs. without) must be documented in the notebook.

---

## 3. Part B: Local Class-Message Task Extraction (`class_messages/`)
### Critical Constraints
- **Zero Remote APIs**: Strictly NO calls to OpenAI, Anthropic, Gemini, Groq, or any hosted/remote LLM services. All inference must be performed via a local runtime (Ollama) running on `localhost:11434`.
- **Local Model**: Use a compact instruction-following model (`qwen2.5:1.5b`). Setup instructions and commands must be explicitly documented.
- **Extraction Script (`extract.py`)**:
  - Command: `python extract.py --inbox <dir> --output <json_path>`
  - Output Contract: Single JSON object `{"tasks": [...]}` with keys `message_id`, `task`, `owner` (string or `null`), and `due` (string or `null`).
  - Strict Fidelity: Do not invent tasks, owners, or deadlines. Relative deadlines are preserved verbatim; dates are not inferred or converted to calendar dates.
  - Prompt Injection Defense: Untrusted message content must be strictly separated from system instructions. Embedded commands attempting to override extraction instructions, leak data, or generate synthetic tasks (e.g. `m04.txt`) must be safely resisted.
  - Fault Tolerance & Atomic I/O: If an individual message fails or produces invalid JSON, the error is logged and remaining messages continue. Output JSON is written atomically to prevent corrupt or truncated files. Non-zero exit code if any message fails.
  - Determinism: Configured with `temperature = 0` (or deterministic sampling). A two-run diff (`diff -u run1.json run2.json`) is tested and recorded.
  - Token Accounting: Accurately measure and report prompt and completion tokens using Ollama API metadata or a documented tokenizer.

---

## 4. Testing & Verification Expectations
- Automated test suites for both components covering:
  - Part A: Model loading, end-to-end scoring, missing features, extra features, invalid schemas, row order preservation, and out-of-directory execution.
  - Part B: Semantic matching on `example/expected.json`, regression assertions for practice messages (`m01`–`m06`), injection rejection on `m04`, output schema validation, and missing model handling.
- Never modify or overwrite original input data (`train.csv`, `sample_submission.csv`, `messages/*.txt`, `example/*`).

---

## 5. Documentation Standards
- `placement/analysis.ipynb`: Fully executable Jupyter notebook with verified markdown rationale, reproducible splits (fixed random seeds), leakage analysis, ablation study, and metric logs.
- `placement/README.md`: Original assignment text preserved, followed by a dedicated Submission Guide (setup, reproduction, inference, and known limitations).
- `class_messages/NOTES.md`: Concise, factual responses addressing all required points (Model, Tokens, Prompt, Deterministic Output, Hallucination/Rejection, Validation, Errors, Guardrails).
- `class_messages/README.md`: Submission guide with setup commands, model requirements, runtime commands, and reproduction steps.
- `requirements.txt`: Lightweight, pinned dependencies separated by module.
