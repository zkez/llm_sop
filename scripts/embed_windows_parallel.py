from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_dir, load_yaml, read_json, resolve_project_path, write_json
from scripts.embed_windows import add_shard_suffix


def command_path(path: Path) -> str:
    return path.as_posix()


def build_shard_command(
    python_executable: str,
    config_path: str,
    shard_index: int,
    num_shards: int,
    profile_dir: Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        "scripts/embed_windows.py",
        "--config",
        config_path,
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
    ]
    if profile_dir is not None:
        command.extend(
            [
                "--profile-csv",
                command_path(profile_dir / f"shard_{shard_index:02d}.csv"),
                "--profile-json",
                command_path(profile_dir / f"shard_{shard_index:02d}.json"),
            ]
        )
    return command


def build_parallel_profile_summary(
    profile_dir: Path,
    gpus: list[str],
    elapsed_seconds: float,
    log_dir: Path,
) -> dict[str, Any]:
    shard_summaries: list[dict[str, Any]] = []
    per_window_samples: list[float] = []

    for shard_index, gpu in enumerate(gpus):
        profile_path = profile_dir / f"shard_{shard_index:02d}.json"
        if not profile_path.exists():
            continue
        profile = read_json(profile_path)
        summary = dict(profile["summary"])
        summary["shard_index"] = shard_index
        summary["gpu"] = gpu
        shard_summaries.append(summary)
        for record in profile.get("batches", []):
            per_window_samples.extend([float(record["seconds_per_window"])] * int(record["batch_size"]))

    if per_window_samples:
        percentiles = np.percentile(per_window_samples, [50, 90, 95, 99])
        mean_seconds_per_window = sum(per_window_samples) / len(per_window_samples)
    else:
        percentiles = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mean_seconds_per_window = 0.0

    return {
        "gpus": gpus,
        "num_shards": len(gpus),
        "num_windows": sum(int(summary["num_windows"]) for summary in shard_summaries),
        "parallel_elapsed_seconds": round(float(elapsed_seconds), 6),
        "mean_seconds_per_window": round(float(mean_seconds_per_window), 6),
        "p50_seconds_per_window": round(float(percentiles[0]), 6),
        "p90_seconds_per_window": round(float(percentiles[1]), 6),
        "p95_seconds_per_window": round(float(percentiles[2]), 6),
        "p99_seconds_per_window": round(float(percentiles[3]), 6),
        "log_dir": command_path(log_dir),
        "shards": shard_summaries,
    }


def run_shards(config_path: str, gpus: list[str], project_root: Path, log_dir: Path, profile_dir: Path | None = None) -> None:
    ensure_dir(log_dir)
    processes: list[tuple[int, str, subprocess.Popen[str], Any]] = []
    num_shards = len(gpus)
    for shard_index, gpu in enumerate(gpus):
        log_path = log_dir / f"shard_{shard_index:02d}_gpu_{gpu}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = build_shard_command(sys.executable, config_path, shard_index, num_shards, profile_dir)
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard_index, gpu, process, log_handle))

    failures: list[str] = []
    for shard_index, gpu, process, log_handle in processes:
        exit_code = process.wait()
        log_handle.close()
        if exit_code != 0:
            failures.append(f"shard={shard_index} gpu={gpu} exit={exit_code}")
    if failures:
        raise RuntimeError("视频窗口并行 embedding 失败: " + "; ".join(failures))


def merge_shards(config: dict[str, Any], project_root: Path, num_shards: int) -> None:
    final_embedding_path = resolve_project_path(config["outputs"]["window_embeddings"], project_root)
    final_metadata_path = resolve_project_path(config["outputs"]["window_metadata"], project_root)
    rows: list[tuple[int, dict[str, Any], np.ndarray]] = []

    for shard_index in range(num_shards):
        shard_embedding_path = add_shard_suffix(final_embedding_path, shard_index, num_shards)
        shard_metadata_path = add_shard_suffix(final_metadata_path, shard_index, num_shards)
        embeddings = np.load(shard_embedding_path)
        metadata = read_json(shard_metadata_path)
        windows = metadata["windows"]
        if len(windows) != len(embeddings):
            raise ValueError(f"分片元数据和 embedding 数量不一致: {shard_index}")
        for window, embedding in zip(windows, embeddings):
            rows.append((int(window["index"]), window, embedding))

    rows.sort(key=lambda item: item[0])
    ensure_dir(final_embedding_path.parent)
    np.save(final_embedding_path, np.vstack([row[2] for row in rows]).astype(np.float32))
    write_json(final_metadata_path, {"windows": [row[1] for row in rows]})


def main() -> None:
    parser = argparse.ArgumentParser(description="使用多张 GPU 并行生成视频窗口 embedding。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    parser.add_argument("--gpus", default="0,1,2,3", help="逗号分隔的 GPU 编号")
    parser.add_argument("--profile-dir", default=None, help="可选，保存每个分片的 profiling 结果目录")
    args = parser.parse_args()

    project_root = Path.cwd()
    config = load_yaml(resolve_project_path(args.config, project_root))
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus 至少需要一个 GPU 编号")

    log_dir = resolve_project_path(config["outputs"]["window_embeddings"], project_root).parent / "parallel_logs" / time.strftime("%Y%m%d-%H%M%S")
    profile_dir = resolve_project_path(args.profile_dir, project_root) if args.profile_dir else None
    if profile_dir is not None:
        ensure_dir(profile_dir)
    started = time.perf_counter()
    run_shards(args.config, gpus, project_root, log_dir, profile_dir)
    merge_shards(config, project_root, len(gpus))
    elapsed = time.perf_counter() - started
    if profile_dir is not None:
        write_json(profile_dir / "parallel_summary.json", build_parallel_profile_summary(profile_dir, gpus, elapsed, log_dir))
    print(f"并行视频窗口 embedding 完成: gpus={gpus}, elapsed_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
