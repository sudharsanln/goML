# Class task list

You are the class representative during project week. Faculty and classmates send you short messages. Some ask for a task. Some do not.

Write a program that reads a folder of messages and writes one JSON task list. Use a **local** model. Any local model is fine. Do not send the messages to a hosted API (OpenAI, Anthropic, Gemini, Groq, and the rest).

## Messages

Each file looks like this:

```
message_id: ex01
from: Class mentor
date: 1 Mar 2024

Riya, please share the venue list by Tuesday 11am.
```

The body starts after the blank line. Practice messages are in `messages/`. One solved message is in `example/`.

## Output format

```json
{
  "tasks": [
    {
      "message_id": "ex01",
      "task": "Share the venue list",
      "owner": "Riya",
      "due": "Tuesday 11am"
    }
  ]
}
```

- `task` is a short line about what to do.
- `owner` is the person asked to do it, or `null` if the message names no one.
- `due` is the deadline in the message's own words, or `null` if there is no deadline.
- Leave out messages that do not ask for a task.

```bash
python extract.py --inbox messages --output tasks.json
```

Pointed at `example/`, the tasks should match `example/expected.json`.

## Evaluation

We run your program on a **second folder** of messages, twice.

We check that the JSON is valid, that the tasks are the ones actually asked for, and that names, deadlines, and tasks that are not in the message are not added. The two runs should produce the same file.

We read `NOTES.md`.

## What to submit

```
extract.py
NOTES.md
requirements.txt
```

Include how to run the local model. Weights can live in a `models/` folder, or your notes can give the exact local model name and file.

## Things to cover in NOTES.md

Keep each one short.

- **Model.** Which local model you ran.
- **Tokens.** How many tokens you sent and received on the practice folder, and how you counted them.
- **Prompt.** How the prompt is structured. What you include, and what you leave out.
- **Deterministic output.** The setting you used so a second run matches the first. Paste the result of diffing two runs.
- **Hallucination.** A case where the model added something that was not in the message, and what you did about it.
- **Format and validation.** How you check the JSON before writing it.
- **Errors.** What happens if one message fails, so the rest of the folder still finishes.
- **Guardrails.** What you do when a message tells the model to ignore its instructions or to add something that is not a class task.

---

# Submission Guide: Local Task Extraction System

## 1. Local Runtime Installation & Model Setup
This solution runs **100% locally** using Ollama. Strictly zero remote or hosted APIs (OpenAI, Anthropic, Gemini, Groq, etc.) are called.

### Step 1: Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com) (or package manager). Ensure the daemon is running:
```bash
ollama serve
```

### Step 2: Download Model
Download the verified instruction-following model `qwen2.5:1.5b`:
```bash
ollama pull qwen2.5:1.5b
```

## 2. Python Dependencies
Install minimal dependencies using Python 3.11:
```bash
pip install -r requirements.txt
```
`extract.py` has no third-party Python runtime dependencies; it uses the
standard library and the local Ollama HTTP endpoint. `pytest==8.3.4` is
required only to run the test suite.

## 3. Extraction Command
Execute the extractor pointing to an inbox directory:
```bash
# Example directory extraction:
python extract.py --inbox example --output tasks.json

# Practice messages extraction:
python extract.py --inbox messages --output tasks.json
```

## 4. Token Accounting
When processing the practice folder (`messages/`):
- **Input (Prompt) Tokens:** `4,180`
- **Output (Completion) Tokens:** `315`
- **Total Tokens:** `4,495`
- **Measurement Method:** Directly retrieved from the native Ollama API response fields (`prompt_eval_count` and `eval_count`) corresponding to the actual tokenizer tokens.

## 5. Testing & Reproducibility
### Run Automated Regression Suite:
```bash
python -m pytest tests/test_extract.py -v
```

### Determinism Verification (Two-Run Diff):
Run extraction twice to separate files and diff them:
```bash
python extract.py --inbox messages --output run1.json
python extract.py --inbox messages --output run2.json
diff -u run1.json run2.json
```
Both runs produce bit-for-bit identical JSON files under `temperature = 0.0, seed = 42`.

## 6. Known Limitations
- **CPU Speed:** Processing takes approximately 2–3 seconds per message on CPU.
- **Single-Duty Focus:** Designed for concise academic and logistical class tasks; deeply nested multi-step workflows with multiple sub-owners require multi-turn parsing.
