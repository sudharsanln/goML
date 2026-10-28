import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

CLASS_MESSAGES_DIR = Path(__file__).resolve().parent.parent
EXTRACT_SCRIPT = CLASS_MESSAGES_DIR / "extract.py"
MESSAGES_DIR = CLASS_MESSAGES_DIR / "messages"
EXAMPLE_DIR = CLASS_MESSAGES_DIR / "example"
EXPECTED_JSON = EXAMPLE_DIR / "expected.json"


@pytest.fixture(scope="session")
def practice_tasks():
    """Run extract.py once on practice messages and yield parsed JSON tasks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "tasks.json"
        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--inbox",
            str(MESSAGES_DIR),
            "--output",
            str(out_json),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"extract.py failed: {res.stderr}"
        assert out_json.exists(), "Output JSON file was not written"
        data = json.loads(out_json.read_text(encoding="utf-8"))
        tasks_by_id = {t["message_id"]: t for t in data.get("tasks", [])}
        return tasks_by_id


def test_1_example_semantic_match():
    """Verify that pointed at example/, output matches example/expected.json semantically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "ex_out.json"
        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--inbox",
            str(EXAMPLE_DIR),
            "--output",
            str(out_json),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Extraction on example failed: {res.stderr}"
        assert out_json.exists()

        actual = json.loads(out_json.read_text(encoding="utf-8"))
        expected = json.loads(EXPECTED_JSON.read_text(encoding="utf-8"))

        assert len(actual.get("tasks", [])) == len(expected.get("tasks", []))
        act_task = actual["tasks"][0]
        exp_task = expected["tasks"][0]

        assert act_task["message_id"] == exp_task["message_id"]
        assert act_task["owner"] == exp_task["owner"]
        assert act_task["due"] == exp_task["due"]
        assert "venue" in act_task["task"].lower()


def test_2_m01_extracts_neha_and_deadline(practice_tasks):
    """m01 extracts Neha's task and its stated deadline."""
    assert "m01" in practice_tasks, "Task for m01 was not extracted"
    t = practice_tasks["m01"]
    assert t["owner"] == "Neha"
    assert "abstract" in t["task"].lower()
    assert "friday 5pm" in t["due"].lower()


def test_3_m02_is_excluded(practice_tasks):
    """m02 (social tea invitation) must be excluded."""
    assert "m02" not in practice_tasks, f"m02 should be excluded, got: {practice_tasks.get('m02')}"


def test_4_m03_extracts_arjun_and_hall_booking(practice_tasks):
    """m03 extracts seminar-hall booking with Arjun as owner and preserves deadline."""
    assert "m03" in practice_tasks, "Task for m03 was not extracted"
    t = practice_tasks["m03"]
    assert t["owner"] == "Arjun"
    assert "hall" in t["task"].lower() or "book" in t["task"].lower()
    assert "18 march" in t["due"].lower()


def test_5_m04_extracts_legitimate_attendance_task(practice_tasks):
    """m04 extracts attendance-sheet task with Meera and 6pm deadline."""
    assert "m04" in practice_tasks, "Task for m04 was not extracted"
    t = practice_tasks["m04"]
    assert t["owner"] == "Meera"
    assert "attendance" in t["task"].lower()
    assert "6pm" in t["due"].lower()


def test_6_m04_excludes_injected_answer_key(practice_tasks):
    """Verify prompt-injection attempt to email answer key is completely excluded."""
    for mid, t in practice_tasks.items():
        assert "answer key" not in t["task"].lower(), f"Injection leaked in {mid}: {t}"
        assert "grader@" not in t["task"].lower()


def test_7_m05_extracts_printing_task_with_owner_null(practice_tasks):
    """m05 extracts poster printing task with owner null and preserves deadline."""
    assert "m05" in practice_tasks, "Task for m05 was not extracted"
    t = practice_tasks["m05"]
    assert t["owner"] is None
    assert "print" in t["task"].lower() or "poster" in t["task"].lower()
    assert "before the review" in t["due"].lower()


def test_8_m06_is_excluded(practice_tasks):
    """m06 (newsletter announcement, no action needed) must be excluded."""
    assert "m06" not in practice_tasks, f"m06 should be excluded, got: {practice_tasks.get('m06')}"


def test_9_missing_deadlines_become_null():
    """Verify messages without deadlines produce due = null."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inbox = Path(tmpdir) / "inbox"
        inbox.mkdir()
        msg_file = inbox / "msg_no_due.txt"
        msg_file.write_text(
            "message_id: nodue01\nfrom: Prof. Rao\ndate: 8 Mar 2024\n\nVikram, please update the software repository.",
            encoding="utf-8",
        )
        out_json = Path(tmpdir) / "out.json"

        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--inbox",
            str(inbox),
            "--output",
            str(out_json),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["due"] is None


def test_10_malformed_json_rejected_safely():
    """Verify validate_and_guard safely rejects malformed or non-list tasks."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from extract import validate_and_guard

    with pytest.raises(ValueError):
        validate_and_guard("not a list", "test01", "body", "sender")

    with pytest.raises(ValueError):
        validate_and_guard([{"invalid_key": "val"}], "test01", "body", "sender")


def test_11_message_failure_does_not_halt_processing():
    """Verify that an unparseable/corrupted message does not halt remaining messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inbox = Path(tmpdir) / "inbox"
        inbox.mkdir()
        # Good message
        (inbox / "good.txt").write_text(
            "message_id: good01\nfrom: Prof. X\ndate: 1 Jan 2024\n\nSam, please submit the project report by Monday 10am.",
            encoding="utf-8",
        )
        # Empty/corrupt file
        (inbox / "corrupt.txt").write_text("", encoding="utf-8")

        out_json = Path(tmpdir) / "out.json"
        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--inbox",
            str(inbox),
            "--output",
            str(out_json),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        # Output JSON should still exist and contain good task
        assert out_json.exists()
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert any(t["message_id"] == "good01" for t in data.get("tasks", []))


def test_12_missing_local_model_actionable_error():
    """Verify that specifying a non-installed model produces an actionable error message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "out.json"
        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--inbox",
            str(EXAMPLE_DIR),
            "--output",
            str(out_json),
            "--model",
            "nonexistent_model:99b",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "not installed in Ollama" in res.stderr
        assert "ollama pull" in res.stderr


def test_13_no_hosted_model_api_contacted():
    """Audit extract.py source code to prove strictly no hosted APIs are referenced."""
    code = EXTRACT_SCRIPT.read_text(encoding="utf-8").lower()
    for forbidden in ["api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com", "api.groq.com"]:
        assert forbidden not in code, f"Forbidden remote API reference found: {forbidden}"


def test_14_determinism_two_runs_identical():
    """Verify two runs on practice messages produce 100% identical JSON outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "run1.json"
        out2 = Path(tmpdir) / "run2.json"

        cmd1 = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--inbox",
            str(EXAMPLE_DIR),
            "--output",
            str(out1),
        ]
        cmd2 = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--inbox",
            str(EXAMPLE_DIR),
            "--output",
            str(out2),
        ]

        subprocess.run(cmd1, check=True)
        subprocess.run(cmd2, check=True)

        assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_15_validation_logic_emits_complete_output_schema():
    """Verify emitted tasks have exactly the required keys and value types."""
    sys.path.insert(0, str(CLASS_MESSAGES_DIR))
    from extract import validate_and_guard

    tasks = validate_and_guard(
        [{"task": "Update the software repository", "owner": None, "due": None}],
        "schema01",
        "Could someone update the software repository?",
        "Prof. Rao",
    )

    assert len(tasks) == 1
    task = tasks[0]
    assert set(task) == {"message_id", "task", "owner", "due"}
    assert isinstance(task["message_id"], str)
    assert isinstance(task["task"], str) and task["task"]
    assert task["owner"] is None or isinstance(task["owner"], str)
    assert task["due"] is None or isinstance(task["due"], str)
