import numpy as np

from tests import realtime_1s_inference_benchmark as benchmark


def test_classify_embedding_returns_top_step_score_and_margin():
    step_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"id": "step_01", "name": "步骤一"},
        {"id": "step_02", "name": "步骤二"},
        {"id": "step_03", "name": "步骤三"},
    ]
    window_embedding = np.array([0.0, 1.0], dtype=np.float32)

    result = benchmark.classify_embedding(window_embedding, step_embeddings, metadata)

    assert result["predicted_step_id"] == "step_02"
    assert result["predicted_step_name"] == "步骤二"
    assert result["predicted_index"] == 1
    assert result["score"] == 1.0
    assert round(result["margin"], 6) == round(1.0 - (1 / np.sqrt(2)), 6)


def test_summarize_latencies_reports_percentiles_and_realtime_factor():
    summary = benchmark.summarize_latencies([0.1, 0.2, 0.4, 0.8], window_seconds=1.0)

    assert summary["num_windows"] == 4
    assert summary["mean_latency_seconds"] == 0.375
    assert summary["p50_latency_seconds"] == 0.3
    assert summary["p90_latency_seconds"] == 0.68
    assert summary["windows_per_second"] == round(4 / 1.5, 6)
    assert summary["video_seconds_per_wall_second"] == round(4 / 1.5, 6)


def test_parse_batch_sizes_accepts_comma_separated_positive_sizes():
    assert hasattr(benchmark, "parse_batch_sizes")
    assert benchmark.parse_batch_sizes("1,2,4,8") == [1, 2, 4, 8]


def test_parse_backend_accepts_native_and_vllm_only():
    assert hasattr(benchmark, "parse_backend")
    assert benchmark.parse_backend("native") == "native"
    assert benchmark.parse_backend("vllm") == "vllm"


def test_batch_windows_groups_windows_by_requested_batch_size():
    windows = [{"window_id": f"window_{index}"} for index in range(5)]

    assert hasattr(benchmark, "batch_windows")
    groups = benchmark.batch_windows(windows, batch_size=2)

    assert [[window["window_id"] for window in group] for group in groups] == [
        ["window_0", "window_1"],
        ["window_2", "window_3"],
        ["window_4"],
    ]


def test_summarize_batch_records_reports_batch_and_amortized_latency():
    records = [
        {"batch_size": 2, "actual_batch_size": 2, "batch_latency_seconds": 0.2},
        {"batch_size": 2, "actual_batch_size": 2, "batch_latency_seconds": 0.4},
        {"batch_size": 2, "actual_batch_size": 1, "batch_latency_seconds": 0.3},
    ]

    assert hasattr(benchmark, "summarize_batch_records")
    summary = benchmark.summarize_batch_records(records, configured_batch_size=2, window_seconds=1.0)

    assert summary["batch_size"] == 2
    assert summary["num_windows"] == 5
    assert summary["num_batches"] == 3
    assert summary["total_inference_seconds"] == 0.9
    assert summary["mean_batch_latency_seconds"] == 0.3
    assert summary["mean_seconds_per_window"] == 0.18
    assert summary["windows_per_second"] == round(5 / 0.9, 6)
    assert summary["video_seconds_per_wall_second"] == round(5 / 0.9, 6)


def test_build_qwen_vllm_text_messages_keeps_instruction_in_system_role():
    assert hasattr(benchmark, "build_qwen_vllm_text_messages")

    messages = benchmark.build_qwen_vllm_text_messages("工步描述", "检索相关视频片段")

    assert messages == [
        {"role": "system", "content": "检索相关视频片段"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "工步描述"},
            ],
        },
        {"role": "assistant", "content": ""},
    ]


def test_build_qwen_vllm_video_messages_includes_sampling_settings():
    assert hasattr(benchmark, "build_qwen_vllm_video_messages")

    messages = benchmark.build_qwen_vllm_video_messages(
        "/tmp/window.mp4",
        "检索相关视频片段",
        fps=1.0,
        max_frames=2,
    )

    assert messages[0] == {"role": "system", "content": "检索相关视频片段"}
    assert messages[1]["content"][0] == {
        "type": "video",
        "video": "/tmp/window.mp4",
        "fps": 1.0,
        "max_frames": 2,
    }
    assert messages[1]["content"][1] == {"type": "text", "text": ""}
    assert messages[2] == {"role": "assistant", "content": ""}


def test_extract_vllm_embedding_reads_pooling_output():
    assert hasattr(benchmark, "extract_vllm_embedding")

    class FakeOutputs:
        embedding = [0.1, 0.2]

    class FakeOutput:
        outputs = FakeOutputs()

    embedding = benchmark.extract_vllm_embedding(FakeOutput())

    np.testing.assert_allclose(embedding, np.array([0.1, 0.2], dtype=np.float32))


def test_parse_gpu_sample_reads_nvidia_smi_metrics():
    assert hasattr(benchmark, "parse_gpu_sample")

    sample = benchmark.parse_gpu_sample("0, NVIDIA RTX 4090, 24576, 12000, 48, 320.50, 450.00, 62")

    assert sample == {
        "gpu_index": 0,
        "gpu_name": "NVIDIA RTX 4090",
        "memory_total_mib": 24576,
        "memory_used_mib": 12000,
        "gpu_utilization_percent": 48.0,
        "power_draw_watts": 320.5,
        "power_limit_watts": 450.0,
        "temperature_celsius": 62.0,
    }


def test_filter_gpu_samples_respects_numeric_cuda_visible_devices():
    samples = [
        {"gpu_index": 0, "gpu_name": "GPU-0"},
        {"gpu_index": 1, "gpu_name": "GPU-1"},
        {"gpu_index": 2, "gpu_name": "GPU-2"},
    ]

    assert hasattr(benchmark, "filter_gpu_samples_for_visible_devices")
    visible = benchmark.filter_gpu_samples_for_visible_devices(samples, "1,2")

    assert [sample["gpu_index"] for sample in visible] == [1, 2]


def test_summarize_resource_samples_reports_memory_power_and_stability():
    samples = [
        {
            "elapsed_seconds": 0.0,
            "gpus": [
                {
                    "gpu_index": 0,
                    "gpu_name": "GPU-A",
                    "memory_total_mib": 1000,
                    "memory_used_mib": 200,
                    "gpu_utilization_percent": 40.0,
                    "power_draw_watts": 100.0,
                    "power_limit_watts": 200.0,
                    "temperature_celsius": 60.0,
                }
            ],
            "process_max_rss_mib": 512.0,
            "process_cpu_seconds": 1.0,
        },
        {
            "elapsed_seconds": 1.0,
            "gpus": [
                {
                    "gpu_index": 0,
                    "gpu_name": "GPU-A",
                    "memory_total_mib": 1000,
                    "memory_used_mib": 500,
                    "gpu_utilization_percent": 80.0,
                    "power_draw_watts": 140.0,
                    "power_limit_watts": 200.0,
                    "temperature_celsius": 70.0,
                }
            ],
            "process_max_rss_mib": 640.0,
            "process_cpu_seconds": 2.0,
        },
    ]

    assert hasattr(benchmark, "summarize_resource_samples")
    summary = benchmark.summarize_resource_samples(samples, duration_seconds=1.25)

    assert summary["sample_count"] == 2
    assert summary["duration_seconds"] == 1.25
    assert summary["process_peak_rss_mib"] == 640.0
    assert summary["process_cpu_avg_percent"] == 80.0
    assert summary["gpus"][0]["memory_peak_mib"] == 500
    assert summary["gpus"][0]["memory_peak_percent"] == 50.0
    assert summary["gpus"][0]["gpu_utilization_avg_percent"] == 60.0
    assert summary["gpus"][0]["power_avg_watts"] == 120.0
    assert summary["gpus"][0]["power_stddev_watts"] == 20.0
    assert summary["gpus"][0]["power_coefficient_of_variation_percent"] == 16.666667
    assert summary["gpus"][0]["power_stability"] == "stable"


def test_summarize_accuracy_metrics_uses_ground_truth_when_present():
    rows = [
        {"window_id": "window_1", "predicted_step_id": "step_1", "score": 0.9, "margin": 0.3},
        {"window_id": "window_2", "predicted_step_id": "step_1", "score": 0.7, "margin": 0.1},
    ]
    windows = [
        {"window_id": "window_1", "expected_step_id": "step_1"},
        {"window_id": "window_2", "expected_step_id": "step_2"},
    ]

    assert hasattr(benchmark, "summarize_accuracy_metrics")
    summary = benchmark.summarize_accuracy_metrics(rows, windows)

    assert summary["has_ground_truth"] is True
    assert summary["accuracy"] == 0.5
    assert summary["correct_predictions"] == 1
    assert summary["evaluated_windows"] == 2
    assert summary["mean_score"] == 0.8
    assert summary["mean_margin"] == 0.2


def test_summarize_accuracy_metrics_reports_na_without_ground_truth():
    rows = [
        {"window_id": "window_1", "predicted_step_id": "step_1", "score": 0.9, "margin": 0.3},
    ]
    windows = [{"window_id": "window_1"}]

    summary = benchmark.summarize_accuracy_metrics(rows, windows)

    assert summary["has_ground_truth"] is False
    assert summary["accuracy"] is None
    assert summary["evaluated_windows"] == 0
    assert summary["mean_score"] == 0.9
    assert summary["mean_margin"] == 0.3


def test_build_terminal_report_includes_speed_accuracy_resource_and_power_sections():
    summary = {
        "best_throughput_batch_size": 4,
        "best_windows_per_second": 12.5,
        "lowest_mean_batch_latency_size": 1,
        "lowest_mean_batch_latency_seconds": 0.12,
        "runs": [
            {
                "batch_size": 4,
                "windows_per_second": 12.5,
                "video_seconds_per_wall_second": 12.5,
                "mean_seconds_per_window": 0.08,
                "p95_seconds_per_window": 0.1,
            }
        ],
    }
    resource_summary = {
        "duration_seconds": 3.0,
        "sample_count": 6,
        "process_peak_rss_mib": 1024.0,
        "process_cpu_avg_percent": 95.0,
        "cpu_count": 16,
        "cuda_visible_devices": "0",
        "gpus": [
            {
                "gpu_index": 0,
                "gpu_name": "GPU-A",
                "memory_peak_mib": 500,
                "memory_total_mib": 1000,
                "memory_peak_percent": 50.0,
                "memory_avg_mib": 400.0,
                "gpu_utilization_avg_percent": 60.0,
                "gpu_utilization_peak_percent": 80.0,
                "temperature_avg_celsius": 65.0,
                "temperature_peak_celsius": 70.0,
                "power_avg_watts": 120.0,
                "power_min_watts": 100.0,
                "power_peak_watts": 140.0,
                "power_stddev_watts": 20.0,
                "power_coefficient_of_variation_percent": 16.666667,
                "power_stability": "stable",
            }
        ],
    }
    accuracy_summary = {
        "has_ground_truth": True,
        "accuracy": 0.5,
        "correct_predictions": 1,
        "evaluated_windows": 2,
        "mean_score": 0.8,
        "min_score": 0.7,
        "mean_margin": 0.2,
        "min_margin": 0.1,
    }

    assert hasattr(benchmark, "build_terminal_report")
    report = benchmark.build_terminal_report(summary, resource_summary, accuracy_summary)

    assert "速度指标" in report
    assert "精度指标" in report
    assert "资源占用" in report
    assert "功耗稳定性" in report
    assert "GPU 0 GPU-A" in report
    assert "accuracy: 50.00%" in report


def test_build_terminal_report_prints_resource_power_and_accuracy_per_batch():
    summary = {
        "best_throughput_batch_size": 2,
        "best_windows_per_second": 20.0,
        "lowest_mean_batch_latency_size": 1,
        "lowest_mean_batch_latency_seconds": 0.1,
        "runs": [
            {
                "batch_size": 1,
                "windows_per_second": 10.0,
                "video_seconds_per_wall_second": 10.0,
                "mean_seconds_per_window": 0.1,
                "p95_seconds_per_window": 0.11,
                "accuracy": {
                    "has_ground_truth": True,
                    "accuracy": 0.5,
                    "correct_predictions": 1,
                    "evaluated_windows": 2,
                    "mean_score": 0.8,
                    "min_score": 0.7,
                    "mean_margin": 0.2,
                    "min_margin": 0.1,
                },
                "resources": {
                    "duration_seconds": 1.0,
                    "sample_count": 2,
                    "process_peak_rss_mib": 1000.0,
                    "process_cpu_avg_percent": 100.0,
                    "cpu_count": 16,
                    "cuda_visible_devices": "1",
                    "gpus": [
                        {
                            "gpu_index": 1,
                            "gpu_name": "GPU-A",
                            "memory_peak_mib": 100,
                            "memory_total_mib": 1000,
                            "memory_peak_percent": 10.0,
                            "memory_avg_mib": 90.0,
                            "gpu_utilization_avg_percent": 50.0,
                            "gpu_utilization_peak_percent": 60.0,
                            "temperature_avg_celsius": 40.0,
                            "temperature_peak_celsius": 45.0,
                            "power_avg_watts": 100.0,
                            "power_min_watts": 90.0,
                            "power_peak_watts": 110.0,
                            "power_stddev_watts": 5.0,
                            "power_coefficient_of_variation_percent": 5.0,
                            "power_stability": "stable",
                        }
                    ],
                },
            },
            {
                "batch_size": 2,
                "windows_per_second": 20.0,
                "video_seconds_per_wall_second": 20.0,
                "mean_seconds_per_window": 0.05,
                "p95_seconds_per_window": 0.06,
                "accuracy": {
                    "has_ground_truth": True,
                    "accuracy": 0.75,
                    "correct_predictions": 3,
                    "evaluated_windows": 4,
                    "mean_score": 0.9,
                    "min_score": 0.8,
                    "mean_margin": 0.3,
                    "min_margin": 0.2,
                },
                "resources": {
                    "duration_seconds": 2.0,
                    "sample_count": 4,
                    "process_peak_rss_mib": 2000.0,
                    "process_cpu_avg_percent": 200.0,
                    "cpu_count": 16,
                    "cuda_visible_devices": "1",
                    "gpus": [
                        {
                            "gpu_index": 1,
                            "gpu_name": "GPU-A",
                            "memory_peak_mib": 200,
                            "memory_total_mib": 1000,
                            "memory_peak_percent": 20.0,
                            "memory_avg_mib": 180.0,
                            "gpu_utilization_avg_percent": 70.0,
                            "gpu_utilization_peak_percent": 80.0,
                            "temperature_avg_celsius": 46.0,
                            "temperature_peak_celsius": 50.0,
                            "power_avg_watts": 140.0,
                            "power_min_watts": 120.0,
                            "power_peak_watts": 160.0,
                            "power_stddev_watts": 10.0,
                            "power_coefficient_of_variation_percent": 7.142857,
                            "power_stability": "stable",
                        }
                    ],
                },
            },
        ],
    }

    report = benchmark.build_terminal_report(summary, {}, {})

    assert "精度指标 batch=1" in report
    assert "精度指标 batch=2" in report
    assert "accuracy: 50.00%" in report
    assert "accuracy: 75.00%" in report
    assert "资源占用 batch=1" in report
    assert "资源占用 batch=2" in report
    assert "显存峰值 100 MiB" in report
    assert "显存峰值 200 MiB" in report
    assert "功耗稳定性 batch=1" in report
    assert "功耗稳定性 batch=2" in report
    assert "avg 100.000 W" in report
    assert "avg 140.000 W" in report
