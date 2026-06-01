import numpy as np

from scripts.score_steps import (
    compute_similarity_matrix,
    summarize_steps,
    summarize_steps_sequence,
    write_report_md,
)


def test_compute_similarity_matrix_returns_cosine_scores():
    step_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    window_embeddings = np.array([[1.0, 0.0], [1.0, 1.0]])

    matrix = compute_similarity_matrix(step_embeddings, window_embeddings)

    np.testing.assert_allclose(matrix[0], np.array([1.0, 1 / np.sqrt(2)]))
    np.testing.assert_allclose(matrix[1], np.array([0.0, 1 / np.sqrt(2)]))


def test_summarize_steps_marks_completed_missing_and_uncertain():
    matrix = np.array(
        [
            [0.91, 0.10, 0.05],
            [0.20, 0.44, 0.30],
            [0.11, 0.50, 0.54],
        ]
    )
    steps = [
        {"id": "step_01", "name": "步骤一"},
        {"id": "step_02", "name": "步骤二"},
        {"id": "step_03", "name": "步骤三"},
    ]
    windows = [
        {"window_id": "window_000001", "start": 0.0, "end": 8.0, "time_range": "00:00.000-00:08.000"},
        {"window_id": "window_000002", "start": 8.0, "end": 16.0, "time_range": "00:08.000-00:16.000"},
        {"window_id": "window_000003", "start": 16.0, "end": 24.0, "time_range": "00:16.000-00:24.000"},
    ]

    rows = summarize_steps(
        matrix,
        steps,
        windows,
        completion_threshold=0.60,
        missing_threshold=0.45,
        margin_threshold=0.05,
        enforce_order=True,
    )

    assert rows[0]["status"] == "completed"
    assert rows[1]["status"] == "missing"
    assert rows[2]["status"] == "uncertain"
    assert rows[0]["best_window_id"] == "window_000001"
    assert rows[2]["best_time_range"] == "00:16.000-00:24.000"
    assert rows[0]["runner_up_step_name"] == "步骤二"
    assert rows[0]["runner_up_score"] == 0.2
    assert rows[2]["runner_up_step_name"] == "步骤二"


def test_write_report_md_includes_runner_up_step_name(tmp_path):
    rows = [
        {
            "step_id": "step_01",
            "step_name": "步骤一",
            "best_window_id": "window_000001",
            "best_time_range": "00:00.000-00:01.000",
            "candidate_window_id": "window_000001",
            "candidate_time_range": "00:00.000-00:01.000",
            "best_score": 0.91,
            "margin": 0.21,
            "runner_up_step_id": "step_02",
            "runner_up_step_name": "步骤二",
            "runner_up_score": 0.7,
            "status": "completed",
            "reason": "分数和 margin 均达到阈值",
        }
    ]

    report_path = tmp_path / "report.md"
    write_report_md(report_path, rows)

    report = report_path.read_text(encoding="utf-8")
    assert "同一窗口第二名步骤" in report
    assert "| 步骤一 | completed | 00:00.000-00:01.000 | 0.910000 | 0.210000 | 步骤二 |" in report


def test_summarize_steps_excludes_completed_previous_steps_from_later_margin():
    matrix = np.array(
        [
            [0.80],
            [0.70],
        ]
    )
    steps = [
        {"id": "step_01", "name": "步骤一"},
        {"id": "step_02", "name": "步骤二"},
    ]
    windows = [
        {"window_id": "window_000001", "start": 0.0, "end": 1.0, "time_range": "00:00.000-00:01.000"},
    ]

    rows = summarize_steps(
        matrix,
        steps,
        windows,
        completion_threshold=0.60,
        missing_threshold=0.45,
        margin_threshold=0.05,
        enforce_order=True,
    )

    assert rows[0]["status"] == "completed"
    assert rows[1]["status"] == "completed"
    assert rows[1]["margin"] == 0.70
    assert rows[1]["runner_up_step_name"] == "--"


def test_summarize_steps_keeps_uncompleted_previous_steps_in_later_margin():
    matrix = np.array(
        [
            [0.59],
            [0.70],
        ]
    )
    steps = [
        {"id": "step_01", "name": "步骤一"},
        {"id": "step_02", "name": "步骤二"},
    ]
    windows = [
        {"window_id": "window_000001", "start": 0.0, "end": 1.0, "time_range": "00:00.000-00:01.000"},
    ]

    rows = summarize_steps(
        matrix,
        steps,
        windows,
        completion_threshold=0.60,
        missing_threshold=0.45,
        margin_threshold=0.05,
        enforce_order=True,
    )

    assert rows[0]["status"] == "uncertain"
    assert rows[1]["status"] == "completed"
    assert rows[1]["margin"] == 0.11
    assert rows[1]["runner_up_step_name"] == "步骤一"


def test_summarize_steps_skips_equivalent_steps_when_calculating_margin():
    matrix = np.array(
        [
            [0.90],
            [0.80],
            [0.78],
        ]
    )
    steps = [
        {"id": "step_03_scan_into_system", "name": "扫入系统"},
        {"id": "step_12_traceability_scan", "name": "追溯扫码"},
        {"id": "step_04_pre_loosen_bolts", "name": "预松螺栓"},
    ]
    windows = [
        {"window_id": "window_000001", "start": 0.0, "end": 1.0, "time_range": "00:00.000-00:01.000"},
    ]

    rows = summarize_steps(
        matrix,
        steps,
        windows,
        completion_threshold=0.60,
        missing_threshold=0.45,
        margin_threshold=0.05,
        enforce_order=True,
        equivalent_step_groups=[["step_03_scan_into_system", "step_12_traceability_scan"]],
    )

    assert rows[0]["status"] == "completed"
    assert rows[0]["margin"] == 0.12
    assert rows[0]["runner_up_step_name"] == "预松螺栓"


def test_summarize_steps_sequence_skips_equivalent_steps_when_calculating_margin():
    matrix = np.array(
        [
            [0.90, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.78, 0.70, 0.85],
        ]
    )
    steps = [
        {"id": "step_03_scan_into_system", "name": "扫入系统"},
        {"id": "step_12_traceability_scan", "name": "追溯扫码"},
        {"id": "step_04_pre_loosen_bolts", "name": "预松螺栓"},
    ]
    windows = [
        {
            "window_id": f"window_{index + 1:06d}",
            "start": float(index),
            "end": float(index + 1),
            "time_range": f"00:0{index}.000-00:0{index + 1}.000",
        }
        for index in range(matrix.shape[1])
    ]

    rows = summarize_steps_sequence(
        matrix,
        steps,
        windows,
        completion_threshold=0.60,
        missing_threshold=0.45,
        margin_threshold=0.05,
        equivalent_step_groups=[["step_03_scan_into_system", "step_12_traceability_scan"]],
    )

    assert rows[0]["status"] == "completed"
    assert rows[0]["margin"] == 0.12
    assert rows[0]["runner_up_step_name"] == "预松螺栓"


def test_summarize_steps_flags_order_conflict_as_uncertain():
    matrix = np.array(
        [
            [0.10, 0.90],
            [0.88, 0.20],
        ]
    )
    steps = [
        {"id": "step_01", "name": "步骤一"},
        {"id": "step_02", "name": "步骤二"},
    ]
    windows = [
        {"window_id": "window_000001", "start": 0.0, "end": 8.0, "time_range": "00:00.000-00:08.000"},
        {"window_id": "window_000002", "start": 8.0, "end": 16.0, "time_range": "00:08.000-00:16.000"},
    ]

    rows = summarize_steps(
        matrix,
        steps,
        windows,
        completion_threshold=0.60,
        missing_threshold=0.45,
        margin_threshold=0.05,
        enforce_order=True,
    )

    assert rows[0]["status"] == "completed"
    assert rows[1]["status"] == "uncertain"
    assert "顺序冲突" in rows[1]["reason"]


def test_summarize_steps_can_ignore_small_margin_for_completion():
    matrix = np.array(
        [
            [0.70],
            [0.68],
        ]
    )
    steps = [
        {"id": "step_01", "name": "步骤一"},
        {"id": "step_02", "name": "步骤二"},
    ]
    windows = [
        {"window_id": "window_000001", "start": 0.0, "end": 1.0, "time_range": "00:00.000-00:01.000"},
    ]

    rows = summarize_steps(
        matrix,
        steps,
        windows,
        completion_threshold=0.60,
        missing_threshold=0.45,
        margin_threshold=0.05,
        enforce_order=True,
        ignore_margin_for_completion=True,
    )

    assert rows[0]["status"] == "completed"
    assert "忽略 margin" in rows[0]["reason"]


def test_summarize_steps_uses_forty_to_forty_four_as_uncertain_band():
    matrix = np.array(
        [
            [0.44],
            [0.40],
            [0.399],
        ]
    )
    steps = [
        {"id": "step_01", "name": "步骤一"},
        {"id": "step_02", "name": "步骤二"},
        {"id": "step_03", "name": "步骤三"},
    ]
    windows = [
        {"window_id": "window_000001", "start": 0.0, "end": 1.0, "time_range": "00:00.000-00:01.000"},
    ]

    rows = summarize_steps(
        matrix,
        steps,
        windows,
        completion_threshold=0.44,
        missing_threshold=0.40,
        margin_threshold=0.05,
        enforce_order=True,
        ignore_margin_for_completion=True,
    )

    assert rows[0]["status"] == "completed"
    assert rows[1]["status"] == "uncertain"
    assert rows[2]["status"] == "missing"


def test_summarize_steps_uses_neighboring_one_second_median_for_thresholds():
    matrix = np.array([[0.20, 0.50, 0.20]])
    steps = [{"id": "step_01", "name": "步骤一"}]
    windows = [
        {"window_id": "window_000001", "start": 0.0, "end": 1.0, "time_range": "00:00.000-00:01.000"},
        {"window_id": "window_000002", "start": 1.0, "end": 2.0, "time_range": "00:01.000-00:02.000"},
        {"window_id": "window_000003", "start": 2.0, "end": 3.0, "time_range": "00:02.000-00:03.000"},
    ]

    rows = summarize_steps(
        matrix,
        steps,
        windows,
        completion_threshold=0.44,
        missing_threshold=0.40,
        margin_threshold=0.05,
        enforce_order=True,
        ignore_margin_for_completion=True,
        score_median_context_seconds=1.0,
    )

    assert rows[0]["candidate_window_id"] == "window_000002"
    assert rows[0]["best_score"] == 0.20
    assert rows[0]["status"] == "missing"
    assert "前后 1 秒中位数" in rows[0]["reason"]


def test_summarize_steps_sequence_assigns_repeated_actions_to_ordered_windows():
    matrix = np.array(
        [
            [0.92, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            [0.10, 0.70, 0.10, 0.10, 0.10, 0.95, 0.10],
            [0.10, 0.10, 0.90, 0.10, 0.10, 0.10, 0.10],
            [0.10, 0.70, 0.10, 0.10, 0.10, 0.95, 0.10],
        ]
    )
    steps = [
        {"id": "step_01", "name": "接电连接"},
        {"id": "step_02", "name": "第一次扫码"},
        {"id": "step_03", "name": "中间动作"},
        {"id": "step_04", "name": "第二次扫码"},
    ]
    windows = [
        {
            "window_id": f"window_{index + 1:06d}",
            "start": float(index),
            "end": float(index + 1),
            "time_range": f"00:0{index}.000-00:0{index + 1}.000",
        }
        for index in range(matrix.shape[1])
    ]

    rows = summarize_steps_sequence(
        matrix,
        steps,
        windows,
        completion_threshold=0.60,
        missing_threshold=0.45,
        margin_threshold=0.05,
    )

    assert rows[1]["step_name"] == "第一次扫码"
    assert rows[1]["candidate_window_id"] == "window_000002"
    assert rows[1]["best_time_range"] == "00:01.000-00:02.000"
    assert rows[3]["step_name"] == "第二次扫码"
    assert rows[3]["candidate_window_id"] == "window_000006"
    assert rows[3]["best_time_range"] == "00:05.000-00:06.000"
    assert "时序解码" in rows[1]["reason"]
