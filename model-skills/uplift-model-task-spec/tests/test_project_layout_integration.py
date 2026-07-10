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


def test_draft_task_config_uses_existing_run_dir_and_relative_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "data.csv").write_text("uid,treat,y\n1,1,1\n2,0,0\n", encoding="utf-8")

    completed = _run(
        {
            "action": "draft_task_config",
            "payload": {
                "data_ref": "data.csv",
                "output_dir": str(run_dir),
                "outcome_column": "y",
                "outcome_type": "binary",
                "treatment_column": "treat",
                "treatment_value": 1,
                "control_value": 0,
                "unit_id_column": "uid",
            },
        },
        run_dir,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "needs_confirmation"
    outputs = stdout["outputs"]
    assert outputs["flow_dir"].startswith("uplift-model-task-spec/")
    assert outputs["result_path"].endswith("/results/0001_draft_task_config.result.json")
    assert not Path(outputs["result_path"]).is_absolute()

    flow_dir = run_dir / outputs["flow_dir"]
    assert (flow_dir / "inputs" / "0001_draft_task_config.request.json").is_file()
    result = json.loads((flow_dir / "results" / "0001_draft_task_config.result.json").read_text(encoding="utf-8"))
    manifest = json.loads((flow_dir / "_flow_manifest.json").read_text(encoding="utf-8"))
    log_lines = (flow_dir / "_flow_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    log_event = json.loads(log_lines[0])
    assert result["outputs"]["task_config_path"].startswith("uplift-model-task-spec/")
    assert manifest["log_path"].startswith("uplift-model-task-spec/")
    assert manifest["outputs"]["result_path"] == outputs["result_path"]
    assert log_event["request_path"].endswith("/inputs/0001_draft_task_config.request.json")
    assert log_event["result_path"] == outputs["result_path"]
    assert len(log_lines) == 1


def test_missing_run_dir_is_not_created(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing"
    completed = _run({"action": "draft_task_config", "payload": {}}, run_dir)

    assert completed.returncode == 0
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "needs_input"
    assert stdout["issues"][0]["code"] == "RUN_DIR_NOT_FOUND"
    assert not run_dir.exists()
