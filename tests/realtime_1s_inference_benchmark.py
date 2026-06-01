from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import chunked, ensure_dir, l2_normalize, load_yaml, mean_normalized, read_json, resolve_model_name_or_path, resolve_project_path
from scripts.embed_steps import load_embedder, tensor_to_numpy
from scripts.embed_windows import read_windows_manifest


def classify_embedding(
    window_embedding: np.ndarray,
    step_embeddings: np.ndarray,
    step_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    step_vectors = l2_normalize(step_embeddings)
    window_vector = l2_normalize(np.asarray(window_embedding, dtype=np.float32).reshape(1, -1))[0]
    scores = step_vectors @ window_vector
    predicted_index = int(np.argmax(scores))
    best_score = float(scores[predicted_index])
    if len(scores) == 1:
        margin = best_score
        second_score = None
        second_index = None
    else:
        other_indices = [index for index in range(len(scores)) if index != predicted_index]
        second_index = max(other_indices, key=lambda index: float(scores[index]))
        second_score = float(scores[second_index])
        margin = best_score - second_score
    step = step_metadata[predicted_index]
    return {
        "predicted_index": predicted_index,
        "predicted_step_id": step["id"],
        "predicted_step_name": step["name"],
        "score": round(best_score, 6),
        "margin": round(margin, 6),
        "second_index": second_index,
        "second_score": round(second_score, 6) if second_score is not None else None,
    }


def summarize_latencies(latencies: list[float], window_seconds: float) -> dict[str, Any]:
    if not latencies:
        raise ValueError("latencies 不能为空")
    values = np.asarray(latencies, dtype=np.float64)
    total_seconds = float(values.sum())
    percentiles = np.percentile(values, [50, 90, 95, 99])
    windows_per_second = len(values) / total_seconds if total_seconds > 0 else 0.0
    return {
        "num_windows": int(len(values)),
        "window_seconds": float(window_seconds),
        "total_inference_seconds": round(total_seconds, 6),
        "mean_latency_seconds": round(float(values.mean()), 6),
        "p50_latency_seconds": round(float(percentiles[0]), 6),
        "p90_latency_seconds": round(float(percentiles[1]), 6),
        "p95_latency_seconds": round(float(percentiles[2]), 6),
        "p99_latency_seconds": round(float(percentiles[3]), 6),
        "min_latency_seconds": round(float(values.min()), 6),
        "max_latency_seconds": round(float(values.max()), 6),
        "windows_per_second": round(float(windows_per_second), 6),
        "video_seconds_per_wall_second": round(float(windows_per_second * window_seconds), 6),
    }


def parse_batch_sizes(value: str) -> list[int]:
    batch_sizes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        batch_size = int(item)
        if batch_size <= 0:
            raise ValueError("batch size 必须大于 0")
        batch_sizes.append(batch_size)
    if not batch_sizes:
        raise ValueError("至少需要一个 batch size")
    return batch_sizes


def parse_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend not in {"native", "vllm"}:
        raise ValueError("backend 只能是 native 或 vllm")
    return backend


def batch_windows(windows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    return [windows[index : index + batch_size] for index in range(0, len(windows), batch_size)]


def build_qwen_vllm_text_messages(text: str, instruction: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ],
        },
        {"role": "assistant", "content": ""},
    ]


def build_qwen_vllm_video_messages(
    video_path: str,
    instruction: str,
    fps: float,
    max_frames: int,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "max_frames": max_frames,
                },
                {"type": "text", "text": ""},
            ],
        },
        {"role": "assistant", "content": ""},
    ]


def extract_vllm_embedding(output: Any) -> np.ndarray:
    outputs = output.outputs
    if hasattr(outputs, "embedding"):
        embedding = outputs.embedding
    elif hasattr(outputs, "data"):
        embedding = outputs.data
    else:
        raise TypeError("无法从 vLLM pooling 输出中提取 embedding")
    return np.asarray(embedding, dtype=np.float32)


def summarize_batch_records(
    records: list[dict[str, Any]],
    configured_batch_size: int,
    window_seconds: float,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records 不能为空")
    batch_latencies = np.asarray([float(record["batch_latency_seconds"]) for record in records], dtype=np.float64)
    actual_batch_sizes = np.asarray([int(record["actual_batch_size"]) for record in records], dtype=np.int64)
    total_windows = int(actual_batch_sizes.sum())
    total_seconds = float(batch_latencies.sum())
    batch_percentiles = np.percentile(batch_latencies, [50, 90, 95, 99])

    amortized_samples: list[float] = []
    for latency, actual_batch_size in zip(batch_latencies, actual_batch_sizes, strict=True):
        amortized_samples.extend([float(latency / actual_batch_size)] * int(actual_batch_size))
    amortized_values = np.asarray(amortized_samples, dtype=np.float64)
    amortized_percentiles = np.percentile(amortized_values, [50, 90, 95, 99])
    windows_per_second = total_windows / total_seconds if total_seconds > 0 else 0.0
    return {
        "batch_size": int(configured_batch_size),
        "num_windows": total_windows,
        "num_batches": int(len(records)),
        "window_seconds": float(window_seconds),
        "total_inference_seconds": round(total_seconds, 6),
        "mean_batch_latency_seconds": round(float(batch_latencies.mean()), 6),
        "p50_batch_latency_seconds": round(float(batch_percentiles[0]), 6),
        "p90_batch_latency_seconds": round(float(batch_percentiles[1]), 6),
        "p95_batch_latency_seconds": round(float(batch_percentiles[2]), 6),
        "p99_batch_latency_seconds": round(float(batch_percentiles[3]), 6),
        "min_batch_latency_seconds": round(float(batch_latencies.min()), 6),
        "max_batch_latency_seconds": round(float(batch_latencies.max()), 6),
        "mean_seconds_per_window": round(float(total_seconds / total_windows), 6),
        "p50_seconds_per_window": round(float(amortized_percentiles[0]), 6),
        "p90_seconds_per_window": round(float(amortized_percentiles[1]), 6),
        "p95_seconds_per_window": round(float(amortized_percentiles[2]), 6),
        "p99_seconds_per_window": round(float(amortized_percentiles[3]), 6),
        "min_seconds_per_window": round(float(amortized_values.min()), 6),
        "max_seconds_per_window": round(float(amortized_values.max()), 6),
        "windows_per_second": round(float(windows_per_second), 6),
        "video_seconds_per_wall_second": round(float(windows_per_second * window_seconds), 6),
    }


GPU_QUERY_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "power.draw",
    "power.limit",
    "temperature.gpu",
]
GROUND_TRUTH_FIELDS = ("expected_step_id", "ground_truth_step_id", "label_step_id", "target_step_id", "step_id")


def round_metric(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _parse_optional_float(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or normalized.upper() in {"N/A", "[N/A]"}:
        return None
    return float(normalized)


def _parse_optional_int(value: str) -> int | None:
    parsed = _parse_optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def parse_gpu_sample(line: str) -> dict[str, Any]:
    row = next(csv.reader([line], skipinitialspace=True))
    if len(row) != len(GPU_QUERY_FIELDS):
        raise ValueError(f"nvidia-smi 输出字段数量不匹配: {line}")
    return {
        "gpu_index": int(row[0].strip()),
        "gpu_name": row[1].strip(),
        "memory_total_mib": _parse_optional_int(row[2]),
        "memory_used_mib": _parse_optional_int(row[3]),
        "gpu_utilization_percent": _parse_optional_float(row[4]),
        "power_draw_watts": _parse_optional_float(row[5]),
        "power_limit_watts": _parse_optional_float(row[6]),
        "temperature_celsius": _parse_optional_float(row[7]),
    }


def _visible_gpu_indices(cuda_visible_devices: str | None) -> set[int] | None:
    if not cuda_visible_devices:
        return None
    values = [value.strip() for value in cuda_visible_devices.split(",") if value.strip()]
    if not values:
        return None
    indices: set[int] = set()
    for value in values:
        if not value.isdigit():
            return None
        indices.add(int(value))
    return indices


def filter_gpu_samples_for_visible_devices(
    samples: list[dict[str, Any]],
    cuda_visible_devices: str | None = None,
) -> list[dict[str, Any]]:
    visible_indices = _visible_gpu_indices(
        os.environ.get("CUDA_VISIBLE_DEVICES") if cuda_visible_devices is None else cuda_visible_devices
    )
    if visible_indices is None:
        return samples
    return [sample for sample in samples if int(sample.get("gpu_index", -1)) in visible_indices]


def query_gpu_samples() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    samples: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            samples.append(parse_gpu_sample(line))
        except ValueError:
            continue
    return filter_gpu_samples_for_visible_devices(samples)


def current_process_max_rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        bytes_used = float(usage.ru_maxrss)
    else:
        bytes_used = float(usage.ru_maxrss) * 1024.0
    return round(bytes_used / (1024.0 * 1024.0), 6)


def current_process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def _min(values: list[float]) -> float | None:
    return min(values) if values else None


def _stddev(values: list[float]) -> float | None:
    if not values:
        return None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return variance ** 0.5


def classify_power_stability(coefficient_of_variation_percent: float | None) -> str:
    if coefficient_of_variation_percent is None:
        return "unknown"
    if coefficient_of_variation_percent <= 20.0:
        return "stable"
    if coefficient_of_variation_percent <= 40.0:
        return "moderate"
    return "unstable"


def _numeric_gpu_values(samples: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = sample.get(key)
        if value is not None:
            values.append(float(value))
    return values


def summarize_resource_samples(samples: list[dict[str, Any]], duration_seconds: float) -> dict[str, Any]:
    process_rss_values = [
        float(sample["process_max_rss_mib"])
        for sample in samples
        if sample.get("process_max_rss_mib") is not None
    ]
    process_cpu_values = [
        float(sample["process_cpu_seconds"])
        for sample in samples
        if sample.get("process_cpu_seconds") is not None
    ]
    process_cpu_avg_percent = None
    if duration_seconds > 0 and len(process_cpu_values) >= 2:
        process_cpu_avg_percent = max(0.0, (process_cpu_values[-1] - process_cpu_values[0]) / duration_seconds * 100.0)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        for gpu in sample.get("gpus", []):
            grouped.setdefault(int(gpu["gpu_index"]), []).append(gpu)

    gpu_summaries: list[dict[str, Any]] = []
    for gpu_index, gpu_samples in sorted(grouped.items()):
        memory_used_values = _numeric_gpu_values(gpu_samples, "memory_used_mib")
        total_values = _numeric_gpu_values(gpu_samples, "memory_total_mib")
        util_values = _numeric_gpu_values(gpu_samples, "gpu_utilization_percent")
        power_values = _numeric_gpu_values(gpu_samples, "power_draw_watts")
        power_limit_values = _numeric_gpu_values(gpu_samples, "power_limit_watts")
        temp_values = _numeric_gpu_values(gpu_samples, "temperature_celsius")

        memory_total = int(_max(total_values) or 0)
        memory_peak = int(_max(memory_used_values) or 0)
        power_avg = _mean(power_values)
        power_stddev = _stddev(power_values)
        power_cv = (power_stddev / power_avg * 100.0) if power_avg and power_stddev is not None else None
        gpu_summaries.append(
            {
                "gpu_index": gpu_index,
                "gpu_name": str(gpu_samples[0].get("gpu_name", "")),
                "memory_total_mib": memory_total,
                "memory_avg_mib": round_metric(_mean(memory_used_values)),
                "memory_peak_mib": memory_peak,
                "memory_peak_percent": round_metric(memory_peak / memory_total * 100.0) if memory_total else None,
                "gpu_utilization_avg_percent": round_metric(_mean(util_values)),
                "gpu_utilization_peak_percent": round_metric(_max(util_values)),
                "power_avg_watts": round_metric(power_avg),
                "power_min_watts": round_metric(_min(power_values)),
                "power_peak_watts": round_metric(_max(power_values)),
                "power_limit_watts": round_metric(_max(power_limit_values)),
                "power_stddev_watts": round_metric(power_stddev),
                "power_coefficient_of_variation_percent": round_metric(power_cv),
                "power_stability": classify_power_stability(power_cv),
                "temperature_avg_celsius": round_metric(_mean(temp_values)),
                "temperature_peak_celsius": round_metric(_max(temp_values)),
            }
        )

    return {
        "sample_count": len(samples),
        "duration_seconds": round_metric(duration_seconds),
        "process_peak_rss_mib": round_metric(_max(process_rss_values)) or current_process_max_rss_mib(),
        "process_cpu_avg_percent": round_metric(process_cpu_avg_percent),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpus": gpu_summaries,
    }


class RuntimeResourceSampler:
    def __init__(
        self,
        interval_seconds: float = 0.5,
        timer: Callable[[], float] = time.perf_counter,
        gpu_reader: Callable[[], list[dict[str, Any]]] = query_gpu_samples,
        rss_reader: Callable[[], float] = current_process_max_rss_mib,
        cpu_reader: Callable[[], float] = current_process_cpu_seconds,
    ) -> None:
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.timer = timer
        self.gpu_reader = gpu_reader
        self.rss_reader = rss_reader
        self.cpu_reader = cpu_reader
        self.samples: list[dict[str, Any]] = []
        self._started_at = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._started_at = self.timer()
        self._stop_event.clear()
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2.0))
        self._sample_once()
        return summarize_resource_samples(self.samples, duration_seconds=self.timer() - self._started_at)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        elapsed_seconds = self.timer() - self._started_at if self._started_at else 0.0
        self.samples.append(
            {
                "elapsed_seconds": round(float(elapsed_seconds), 3),
                "gpus": self.gpu_reader(),
                "process_max_rss_mib": self.rss_reader(),
                "process_cpu_seconds": self.cpu_reader(),
            }
        )


def _expected_step_id(window: dict[str, Any]) -> str | None:
    for field in GROUND_TRUTH_FIELDS:
        value = str(window.get(field, "")).strip()
        if value:
            return value
    return None


def summarize_accuracy_metrics(rows: list[dict[str, Any]], windows: list[dict[str, Any]]) -> dict[str, Any]:
    windows_by_id = {str(window.get("window_id", "")): window for window in windows}
    scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    margins = [float(row["margin"]) for row in rows if row.get("margin") is not None]
    evaluated = 0
    correct = 0
    for row in rows:
        window = windows_by_id.get(str(row.get("window_id", "")), {})
        expected = _expected_step_id(window)
        if expected is None:
            continue
        evaluated += 1
        if str(row.get("predicted_step_id", "")).strip() == expected:
            correct += 1

    return {
        "has_ground_truth": evaluated > 0,
        "evaluated_windows": evaluated,
        "correct_predictions": correct,
        "accuracy": round_metric(correct / evaluated) if evaluated else None,
        "mean_score": round_metric(_mean(scores)),
        "min_score": round_metric(_min(scores)),
        "mean_margin": round_metric(_mean(margins)),
        "min_margin": round_metric(_min(margins)),
    }


def _format_optional(value: Any, unit: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}{unit}"
    return f"{value}{unit}"


def append_accuracy_report_lines(lines: list[str], accuracy_summary: dict[str, Any]) -> None:
    if accuracy_summary["has_ground_truth"]:
        accuracy = float(accuracy_summary["accuracy"] or 0.0) * 100.0
        lines.append(
            f"- accuracy: {accuracy:.2f}% "
            f"({accuracy_summary['correct_predictions']}/{accuracy_summary['evaluated_windows']})"
        )
    else:
        lines.append("- accuracy: N/A (窗口清单未提供 expected_step_id/ground_truth_step_id 等真值字段)")
    lines.append(
        "- 置信度: mean_score={mean_score}, min_score={min_score}, mean_margin={mean_margin}, min_margin={min_margin}".format(
            mean_score=_format_optional(accuracy_summary["mean_score"]),
            min_score=_format_optional(accuracy_summary["min_score"]),
            mean_margin=_format_optional(accuracy_summary["mean_margin"]),
            min_margin=_format_optional(accuracy_summary["min_margin"]),
        )
    )


def append_resource_report_lines(lines: list[str], resource_summary: dict[str, Any]) -> None:
    lines.extend(
        [
            f"- 采样时长: {_format_optional(resource_summary['duration_seconds'], 's')}, "
            f"样本数: {resource_summary['sample_count']}, "
            f"进程峰值 RSS: {_format_optional(resource_summary['process_peak_rss_mib'], ' MiB')}, "
            f"进程CPU平均: {_format_optional(resource_summary['process_cpu_avg_percent'], '%')}",
            f"- CPU 核数: {resource_summary.get('cpu_count')}, CUDA_VISIBLE_DEVICES={resource_summary.get('cuda_visible_devices') or '未设置'}",
        ]
    )
    if not resource_summary["gpus"]:
        lines.append("- GPU: N/A (nvidia-smi 不可用或未采到 GPU 指标)")
    for gpu in resource_summary["gpus"]:
        lines.append(
            f"- GPU {gpu['gpu_index']} {gpu['gpu_name']}: "
            f"显存峰值 {_format_optional(gpu['memory_peak_mib'], ' MiB')}/"
            f"{_format_optional(gpu['memory_total_mib'], ' MiB')} "
            f"({_format_optional(gpu['memory_peak_percent'], '%')}), "
            f"平均显存 {_format_optional(gpu['memory_avg_mib'], ' MiB')}, "
            f"GPU利用率 avg/peak {_format_optional(gpu['gpu_utilization_avg_percent'], '%')}/"
            f"{_format_optional(gpu['gpu_utilization_peak_percent'], '%')}, "
            f"温度 avg/peak {_format_optional(gpu['temperature_avg_celsius'], ' C')}/"
            f"{_format_optional(gpu['temperature_peak_celsius'], ' C')}"
        )


def append_power_report_lines(lines: list[str], resource_summary: dict[str, Any]) -> None:
    if not resource_summary["gpus"]:
        lines.append("- N/A")
    for gpu in resource_summary["gpus"]:
        lines.append(
            f"- GPU {gpu['gpu_index']}: "
            f"avg {_format_optional(gpu['power_avg_watts'], ' W')}, "
            f"min/peak {_format_optional(gpu['power_min_watts'], ' W')}/"
            f"{_format_optional(gpu['power_peak_watts'], ' W')}, "
            f"stddev {_format_optional(gpu['power_stddev_watts'], ' W')}, "
            f"CV {_format_optional(gpu['power_coefficient_of_variation_percent'], '%')}, "
            f"stability={gpu['power_stability']}"
        )


def build_terminal_report(
    summary: dict[str, Any],
    resource_summary: dict[str, Any],
    accuracy_summary: dict[str, Any],
) -> str:
    lines = [
        "",
        "=== 推理运行指标 ===",
        "速度指标:",
        (
            f"- 最佳吞吐: batch={summary['best_throughput_batch_size']}, "
            f"{summary['best_windows_per_second']:.3f} windows/s"
        ),
        (
            f"- 最低平均 batch 延迟: batch={summary['lowest_mean_batch_latency_size']}, "
            f"{summary['lowest_mean_batch_latency_seconds']:.3f}s"
        ),
    ]
    for run in summary["runs"]:
        lines.append(
            "- batch={batch_size}: {windows_per_second:.3f} windows/s, "
            "实时倍率 {video_seconds_per_wall_second:.3f}x, "
            "mean/window {mean_seconds_per_window:.3f}s, p95/window {p95_seconds_per_window:.3f}s".format(**run)
        )

    if all("accuracy" in run for run in summary["runs"]):
        for run in summary["runs"]:
            lines.append(f"精度指标 batch={run['batch_size']}:")
            append_accuracy_report_lines(lines, run["accuracy"])
    else:
        lines.append("精度指标:")
        append_accuracy_report_lines(lines, accuracy_summary)

    if all("resources" in run for run in summary["runs"]):
        for run in summary["runs"]:
            lines.append(f"资源占用 batch={run['batch_size']}:")
            append_resource_report_lines(lines, run["resources"])
    else:
        lines.append("资源占用:")
        append_resource_report_lines(lines, resource_summary)

    if all("resources" in run for run in summary["runs"]):
        for run in summary["runs"]:
            lines.append(f"功耗稳定性 batch={run['batch_size']}:")
            append_power_report_lines(lines, run["resources"])
    else:
        lines.append("功耗稳定性:")
        append_power_report_lines(lines, resource_summary)
    return "\n".join(lines)


def encode_video_batch(
    model: Any,
    windows: list[dict[str, Any]],
    instruction: str,
    fps: float,
    max_frames: int,
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[np.ndarray, float]:
    inputs = [
        {
            "video": window["path"],
            "instruction": instruction,
            "fps": fps,
            "max_frames": max_frames,
        }
        for window in windows
    ]
    started = timer()
    embeddings = tensor_to_numpy(model.process(inputs))
    elapsed = timer() - started
    return embeddings.astype(np.float32), elapsed


class NativeEmbeddingBackend:
    name = "native"

    def __init__(self, model_config: dict[str, Any], project_root: Path) -> None:
        self.model = load_embedder(model_config, project_root)

    def encode_video_batch(
        self,
        windows: list[dict[str, Any]],
        instruction: str,
        fps: float,
        max_frames: int,
    ) -> tuple[np.ndarray, float]:
        return encode_video_batch(self.model, windows, instruction, fps, max_frames)


class VllmEmbeddingBackend:
    name = "vllm"

    def __init__(
        self,
        model_config: dict[str, Any],
        project_root: Path,
        fps: float,
        max_frames: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        enforce_eager: bool,
    ) -> None:
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor
            from vllm import LLM
        except ModuleNotFoundError as exc:
            raise RuntimeError("当前环境未安装 vLLM 或其依赖；请安装 vLLM>=0.14.0 后再运行 --backend vllm") from exc

        self.process_vision_info = process_vision_info
        self.model_path = resolve_model_name_or_path(str(model_config["path"]), project_root)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        engine_kwargs: dict[str, Any] = {
            "model": self.model_path,
            "runner": "pooling",
            "max_model_len": max_model_len,
            "limit_mm_per_prompt": {"video": 1},
            "mm_processor_kwargs": {"fps": fps},
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "trust_remote_code": True,
        }
        dtype_name = str(model_config.get("torch_dtype", "")).strip()
        if dtype_name:
            engine_kwargs["dtype"] = dtype_name
        if enforce_eager:
            engine_kwargs["enforce_eager"] = True
        self.llm = LLM(**engine_kwargs)
        self.fps = fps
        self.max_frames = max_frames

    def apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def encode_texts(self, texts: list[str], instruction: str) -> np.ndarray:
        prompts = [
            {
                "prompt": self.apply_chat_template(build_qwen_vllm_text_messages(text, instruction)),
            }
            for text in texts
        ]
        outputs = self.llm.embed(prompts)
        return np.vstack([extract_vllm_embedding(output) for output in outputs]).astype(np.float32)

    def build_video_prompt(self, window: dict[str, Any], instruction: str) -> dict[str, Any]:
        messages = build_qwen_vllm_video_messages(
            window["path"],
            instruction,
            fps=self.fps,
            max_frames=self.max_frames,
        )
        prompt = self.apply_chat_template(messages)
        _, video_inputs = self.process_vision_info(messages, return_video_metadata=True)
        if video_inputs is None:
            raise ValueError(f"无法从窗口读取视频输入: {window['path']}")
        return {
            "prompt": prompt,
            "multi_modal_data": {"video": video_inputs},
        }

    def encode_video_batch(
        self,
        windows: list[dict[str, Any]],
        instruction: str,
        fps: float,
        max_frames: int,
    ) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        prompts = [self.build_video_prompt(window, instruction) for window in windows]
        outputs = self.llm.embed(prompts)
        elapsed = time.perf_counter() - started
        return np.vstack([extract_vllm_embedding(output) for output in outputs]).astype(np.float32), elapsed


def encode_video_window(
    model: Any,
    window: dict[str, Any],
    instruction: str,
    fps: float,
    max_frames: int,
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[np.ndarray, float]:
    embeddings, elapsed = encode_video_batch(model, [window], instruction, fps, max_frames, timer)
    return embeddings[0].astype(np.float32), elapsed


def run_batch_benchmark(
    backend: Any,
    windows: list[dict[str, Any]],
    step_embeddings: np.ndarray,
    step_metadata: list[dict[str, Any]],
    instruction: str,
    fps: float,
    max_frames: int,
    batch_size: int,
    window_seconds: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    sample_index = 0
    for batch_index, window_batch in enumerate(batch_windows(windows, batch_size)):
        embeddings, elapsed = backend.encode_video_batch(window_batch, instruction, fps, max_frames)
        batch_latency = round(float(elapsed), 6)
        actual_batch_size = len(window_batch)
        seconds_per_window = round(float(elapsed / actual_batch_size), 6)
        batch_records.append(
            {
                "batch_size": batch_size,
                "batch_index": batch_index,
                "actual_batch_size": actual_batch_size,
                "window_ids": ",".join(window["window_id"] for window in window_batch),
                "time_ranges": ",".join(window["time_range"] for window in window_batch),
                "batch_latency_seconds": batch_latency,
                "seconds_per_window": seconds_per_window,
            }
        )
        for window, embedding in zip(window_batch, embeddings, strict=True):
            result = classify_embedding(embedding, step_embeddings, step_metadata)
            rows.append(
                {
                    "batch_size": batch_size,
                    "batch_index": batch_index,
                    "actual_batch_size": actual_batch_size,
                    "sample_index": sample_index,
                    "window_id": window["window_id"],
                    "time_range": window["time_range"],
                    "batch_latency_seconds": batch_latency,
                    "amortized_latency_seconds": seconds_per_window,
                    "predicted_step_id": result["predicted_step_id"],
                    "predicted_step_name": result["predicted_step_name"],
                    "score": result["score"],
                    "margin": result["margin"],
                    "second_score": result["second_score"],
                }
            )
            sample_index += 1
    return {
        "batch_size": batch_size,
        "summary": summarize_batch_records(batch_records, batch_size, window_seconds=window_seconds),
        "batches": batch_records,
        "samples": rows,
    }


def build_step_embeddings_with_vllm(
    backend: VllmEmbeddingBackend,
    steps: list[dict[str, Any]],
    instruction: str,
    batch_size: int,
) -> np.ndarray:
    prototypes: list[np.ndarray] = []
    for step in steps:
        descriptions = step.get("descriptions") or []
        if not descriptions:
            raise ValueError(f"步骤缺少 descriptions: {step.get('id')}")
        all_embeddings: list[np.ndarray] = []
        for batch in chunked(descriptions, batch_size):
            all_embeddings.append(backend.encode_texts(batch, instruction))
        prototypes.append(mean_normalized(np.vstack(all_embeddings)))
    return np.vstack(prototypes).astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="单卡模拟 1 秒视频并发输入的实时工步识别推理测速。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    parser.add_argument("--steps", default="configs/steps.operation_case.yaml", help="vLLM 模式下用于构建步骤 embedding 的步骤 YAML")
    parser.add_argument("--backend", default="native", choices=["native", "vllm"], help="推理后端: native 或 vllm")
    parser.add_argument("--limit", type=int, default=0, help="最多测试多少个窗口，0 表示全部")
    parser.add_argument("--warmup", type=int, default=1, help="正式统计前预热多少个窗口")
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="逗号分隔的并发 batch size，例如 1,2,4,8")
    parser.add_argument("--vllm-max-model-len", type=int, default=8192, help="vLLM max_model_len")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU 显存利用率")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=0, help="vLLM max_num_seqs，0 表示使用最大 batch size")
    parser.add_argument("--vllm-enforce-eager", action="store_true", help="vLLM 使用 enforce_eager")
    parser.add_argument("--resource-sample-interval", type=float, default=0.5, help="推理期间资源采样间隔秒数")
    parser.add_argument("--output-csv", default=None, help="逐窗口结果 CSV")
    parser.add_argument("--output-json", default=None, help="汇总结果 JSON")
    args = parser.parse_args()

    project_root = Path.cwd()
    config = load_yaml(resolve_project_path(args.config, project_root))
    outputs = config["outputs"]
    embedding_config = config["embedding"]
    model_config = config["model"]

    windows = read_windows_manifest(resolve_project_path(outputs["windows_csv"], project_root))
    if args.limit > 0:
        windows = windows[: args.limit]
    if not windows:
        raise ValueError("没有可测试的视频窗口")
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    backend_name = parse_backend(args.backend)

    model_load_started = time.perf_counter()
    if backend_name == "native":
        backend = NativeEmbeddingBackend(model_config, project_root)
    else:
        backend = VllmEmbeddingBackend(
            model_config=model_config,
            project_root=project_root,
            fps=float(embedding_config.get("fps", 1.0)),
            max_frames=int(embedding_config.get("max_frames", 16)),
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_num_seqs=args.vllm_max_num_seqs or max(batch_sizes),
            enforce_eager=args.vllm_enforce_eager,
        )
    model_load_seconds = time.perf_counter() - model_load_started

    instruction = str(model_config.get("instruction", "Represent the user's input"))
    fps = float(embedding_config.get("fps", 1.0))
    max_frames = int(embedding_config.get("max_frames", 16))

    step_embedding_started = time.perf_counter()
    if backend_name == "native":
        step_embeddings = np.load(resolve_project_path(outputs["step_embeddings"], project_root))
        step_metadata = read_json(resolve_project_path(outputs["step_metadata"], project_root))["steps"]
        step_embedding_source = str(resolve_project_path(outputs["step_embeddings"], project_root))
    else:
        steps_config = load_yaml(resolve_project_path(args.steps, project_root))
        steps = steps_config["steps"]
        step_embeddings = build_step_embeddings_with_vllm(
            backend=backend,
            steps=steps,
            instruction=instruction,
            batch_size=int(embedding_config.get("batch_size", 1)),
        )
        step_metadata = [
            {
                "id": step["id"],
                "name": step["name"],
                "descriptions": step.get("descriptions", []),
            }
            for step in steps
        ]
        step_embedding_source = args.steps
    step_embedding_seconds = time.perf_counter() - step_embedding_started

    warmup_count = max(0, min(args.warmup, len(windows)))
    if warmup_count > 0:
        backend.encode_video_batch(windows[:warmup_count], instruction, fps, max_frames)

    window_seconds = float(config["video"].get("window_seconds", 1.0))
    runs: list[dict[str, Any]] = []
    terminal_run_summaries: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        resource_sampler = RuntimeResourceSampler(interval_seconds=args.resource_sample_interval)
        resource_sampler.start()
        try:
            run = run_batch_benchmark(
                backend=backend,
                windows=windows,
                step_embeddings=step_embeddings,
                step_metadata=step_metadata,
                instruction=instruction,
                fps=fps,
                max_frames=max_frames,
                batch_size=batch_size,
                window_seconds=window_seconds,
            )
        finally:
            resource_summary = resource_sampler.stop()
        runs.append(run)
        terminal_run_summaries.append(
            {
                **run["summary"],
                "accuracy": summarize_accuracy_metrics(run["samples"], windows),
                "resources": resource_summary,
            }
        )

    run_summaries = [run["summary"] for run in runs]
    best_throughput = max(run_summaries, key=lambda summary: float(summary["windows_per_second"]))
    lowest_batch_latency = min(run_summaries, key=lambda summary: float(summary["mean_batch_latency_seconds"]))
    summary = {
        "backend": backend_name,
        "num_windows": len(windows),
        "batch_sizes": batch_sizes,
        "model_load_seconds": round(float(model_load_seconds), 6),
        "step_embedding_seconds": round(float(step_embedding_seconds), 6),
        "step_embedding_source": step_embedding_source,
        "fps": fps,
        "max_frames": max_frames,
        "warmup_windows": warmup_count,
        "best_throughput_batch_size": best_throughput["batch_size"],
        "best_windows_per_second": best_throughput["windows_per_second"],
        "lowest_mean_batch_latency_size": lowest_batch_latency["batch_size"],
        "lowest_mean_batch_latency_seconds": lowest_batch_latency["mean_batch_latency_seconds"],
        "runs": run_summaries,
    }

    default_dir = resolve_project_path(outputs["scores_csv"], project_root).parent
    default_prefix = "vllm_" if backend_name == "vllm" else ""
    csv_path = (
        resolve_project_path(args.output_csv, project_root)
        if args.output_csv
        else default_dir / f"{default_prefix}realtime_1s_concurrency_benchmark.csv"
    )
    json_path = (
        resolve_project_path(args.output_json, project_root)
        if args.output_json
        else default_dir / f"{default_prefix}realtime_1s_concurrency_benchmark.json"
    )
    rows = [row for run in runs for row in run["samples"]]
    accuracy_summary = summarize_accuracy_metrics(rows, windows)
    terminal_summary = {**summary, "runs": terminal_run_summaries}
    write_csv(csv_path, rows)
    write_json(json_path, {"summary": summary, "runs": runs})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(build_terminal_report(terminal_summary, {}, accuracy_summary))
    print(f"saved_csv: {csv_path}")
    print(f"saved_json: {json_path}")


if __name__ == "__main__":
    main()
