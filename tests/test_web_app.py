import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web_app import Analyzer, build_job_config, load_steps, normalize_scoring_thresholds, read_scores, select_decode_strategy


def test_load_steps_reads_operation_case_buttons():
    steps = load_steps(PROJECT_ROOT / "configs" / "steps.operation_case.yaml")

    assert len(steps) == 13
    assert steps[0]["id"] == "step_01_power_cable_connection"
    assert steps[0]["name"] == "接电连接"
    assert steps[-1]["name"] == "贴角度标签"


def test_build_job_config_uses_uploaded_video_and_job_outputs(tmp_path):
    base_config = {
        "model": {"path": "./models/Qwen3-VL-Embedding-2B"},
        "video": {
            "path": "./data/operation_case/input.mp4",
            "windows_dir": "./data/operation_case_1s/windows",
            "window_seconds": 1,
            "stride_seconds": 1,
            "min_window_seconds": 1,
            "skip_existing": True,
        },
        "embedding": {"batch_size": 1, "fps": 2.0, "max_frames": 4},
        "scoring": {"completion_threshold": 0.60, "missing_threshold": 0.45},
        "outputs": {},
    }

    config = build_job_config(
        base_config=base_config,
        upload_path=tmp_path / "input.mp4",
        windows_dir=tmp_path / "windows",
        output_dir=tmp_path / "outputs",
    )

    assert config["video"]["path"] == str(tmp_path / "input.mp4")
    assert config["video"]["windows_dir"] == str(tmp_path / "windows")
    assert config["video"]["skip_existing"] is False
    assert config["scoring"]["completion_threshold"] == 0.44
    assert config["scoring"]["missing_threshold"] == 0.40
    assert config["scoring"]["ignore_margin_for_completion"] is True
    assert config["scoring"]["score_median_context_seconds"] == 1.0
    assert config["outputs"]["scores_csv"] == str(tmp_path / "outputs" / "scores.csv")
    assert config["outputs"]["report_md"] == str(tmp_path / "outputs" / "report.md")


def test_build_job_config_accepts_per_run_thresholds(tmp_path):
    config = build_job_config(
        base_config={"video": {}, "scoring": {}, "outputs": {}},
        upload_path=tmp_path / "input.mp4",
        windows_dir=tmp_path / "windows",
        output_dir=tmp_path / "outputs",
        completion_threshold=0.52,
        missing_threshold=0.48,
    )

    assert config["scoring"]["completion_threshold"] == 0.52
    assert config["scoring"]["missing_threshold"] == 0.48


def test_normalize_scoring_thresholds_validates_review_band():
    assert normalize_scoring_thresholds("0.44", "0.40") == (0.44, 0.40)
    assert normalize_scoring_thresholds(None, None) == (0.44, 0.40)

    try:
        normalize_scoring_thresholds("0.30", "0.40")
    except ValueError as error:
        assert "复核下限必须小于亮灯阈值" in str(error)
    else:
        raise AssertionError("expected invalid threshold order to fail")


def test_read_scores_maps_rows_to_button_states(tmp_path):
    scores_path = tmp_path / "scores.csv"
    with scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step_id",
                "step_name",
                "best_window_id",
                "best_time_range",
                "candidate_window_id",
                "candidate_time_range",
                "best_score",
                "margin",
                "status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step_id": "step_01",
                "step_name": "步骤一",
                "best_window_id": "window_000001",
                "best_time_range": "00:00.000-00:01.000",
                "candidate_window_id": "window_000001",
                "candidate_time_range": "00:00.000-00:01.000",
                "best_score": "0.81234",
                "margin": "0.21",
                "status": "completed",
                "reason": "分数和 margin 均达到阈值",
            }
        )

    rows = read_scores(scores_path)

    assert rows == [
        {
            "step_id": "step_01",
            "step_name": "步骤一",
            "status": "completed",
            "is_active": True,
            "best_time_range": "00:00.000-00:01.000",
            "best_score": 0.81234,
            "margin": 0.21,
            "reason": "分数和 margin 均达到阈值",
        }
    ]
    json.dumps(rows, ensure_ascii=False)


def test_select_decode_strategy_falls_back_when_windows_are_insufficient():
    config = {"scoring": {"decode_strategy": "sequence", "sequence_min_gap_windows": 1}}

    selected = select_decode_strategy(config, window_count=4, step_count=13)

    assert selected["scoring"]["decode_strategy"] == "independent"
    assert config["scoring"]["decode_strategy"] == "sequence"


def test_select_decode_strategy_keeps_sequence_when_windows_are_sufficient():
    config = {"scoring": {"decode_strategy": "sequence", "sequence_min_gap_windows": 1}}

    selected = select_decode_strategy(config, window_count=20, step_count=13)

    assert selected["scoring"]["decode_strategy"] == "sequence"


def test_analyzer_reload_steps_config_for_button_labels(tmp_path):
    steps_path = tmp_path / "steps.yaml"
    steps_path.write_text(
        "steps:\n"
        "  - id: step_05_assemble_gasket\n"
        "    name: 装配密封垫\n"
        "    descriptions:\n"
        "      - 旧描述\n",
        encoding="utf-8",
    )
    base_config_path = tmp_path / "run.yaml"
    base_config_path.write_text("outputs: {}\n", encoding="utf-8")
    analyzer = Analyzer(
        project_root=tmp_path,
        base_config_path=base_config_path,
        steps_config_path=steps_path,
        job_root=tmp_path / "jobs",
        upload_root=tmp_path / "uploads",
        python_executable=sys.executable,
    )

    steps_path.write_text(
        "steps:\n"
        "  - id: step_05_assemble_gasket\n"
        "    name: 装配黑色密封圈\n"
        "    descriptions:\n"
        "      - 工人取黑色橡胶密封垫。\n",
        encoding="utf-8",
    )

    states = analyzer._initial_step_states()

    assert states[0]["step_name"] == "装配黑色密封圈"


def test_prepare_step_embeddings_regenerates_when_step_metadata_is_stale(tmp_path):
    steps_path = tmp_path / "steps.yaml"
    steps_path.write_text(
        "steps:\n"
        "  - id: step_05_assemble_gasket\n"
        "    name: 装配黑色密封圈\n"
        "    descriptions:\n"
        "      - 工人取黑色橡胶密封垫。\n",
        encoding="utf-8",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    np.save(source_dir / "step_embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    (source_dir / "step_metadata.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "id": "step_05_assemble_gasket",
                        "name": "装配密封垫",
                        "descriptions": ["旧描述"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    base_config_path = tmp_path / "run.yaml"
    base_config_path.write_text(
        "outputs:\n"
        f"  step_embeddings: {source_dir / 'step_embeddings.npy'}\n"
        f"  step_metadata: {source_dir / 'step_metadata.json'}\n",
        encoding="utf-8",
    )
    analyzer = Analyzer(
        project_root=tmp_path,
        base_config_path=base_config_path,
        steps_config_path=steps_path,
        job_root=tmp_path / "jobs",
        upload_root=tmp_path / "uploads",
        python_executable=sys.executable,
    )
    commands = []

    def record_command(job_id, command):
        commands.append(command)

    analyzer._run_command = record_command
    output_dir = tmp_path / "job"
    analyzer.store.write_status("job_1", {"job_id": "job_1", "logs": []})
    config = {
        "outputs": {
            "step_embeddings": str(output_dir / "step_embeddings.npy"),
            "step_metadata": str(output_dir / "step_metadata.json"),
        }
    }

    analyzer._prepare_step_embeddings("job_1", config, output_dir / "run.yaml")

    assert commands
    assert commands[0][1] == "scripts/embed_steps.py"
    assert not (output_dir / "step_metadata.json").exists()
