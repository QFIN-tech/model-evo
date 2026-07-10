from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run.py"


def _run(payload: dict, run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), "--input", "-", "--output-dir", str(run_dir)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )


def test_experiment_id_required_and_validated(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    completed = _run({"action": "train", "payload": {"experiment_id": "bad"}}, run_dir)

    assert completed.returncode == 0
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "needs_input"
    assert stdout["issues"][0]["code"] == "INVALID_EXPERIMENT_ID"
    assert not (run_dir / "new-models").exists()


def test_experiment_dir_is_not_overwritten(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    experiment_dir = run_dir / "new-models" / "exp-001-s_learner_baseline"
    experiment_dir.mkdir(parents=True)
    sentinel = experiment_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    completed = _run(
        {"action": "train", "payload": {"experiment_id": "exp-001-s_learner_baseline"}},
        run_dir,
    )

    assert completed.returncode == 0
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "needs_input"
    assert stdout["issues"][0]["code"] == "EXPERIMENT_ALREADY_EXISTS"
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_experiment_layout_records_failed_train_without_dependencies(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    completed = _run(
        {"action": "train", "payload": {"experiment_id": "exp-001-s_learner_baseline"}},
        run_dir,
    )

    assert completed.returncode == 0
    stdout = json.loads(completed.stdout)
    outputs = stdout["outputs"]
    assert outputs["experiment_id"] == "exp-001-s_learner_baseline"
    assert outputs["experiment_dir"] == "new-models/exp-001-s_learner_baseline"
    assert outputs["result_path"] == "new-models/exp-001-s_learner_baseline/results/0001_train.result.json"

    experiment_dir = run_dir / outputs["experiment_dir"]
    assert (experiment_dir / "inputs" / "0001_train.request.json").is_file()
    assert (experiment_dir / "results" / "0001_train.result.json").is_file()
    manifest = json.loads((experiment_dir / "_experiment_manifest.json").read_text(encoding="utf-8"))
    log_lines = (experiment_dir / "_experiment_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert manifest["outputs"]["result_path"] == outputs["result_path"]
    assert json.loads(log_lines[0])["result_path"] == outputs["result_path"]
