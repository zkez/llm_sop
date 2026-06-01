from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import chunked, ensure_dir, load_yaml, resolve_model_name_or_path, resolve_project_path, write_json


PROFILE_FIELDNAMES = [
    "batch_index",
    "batch_size",
    "window_indices",
    "window_ids",
    "time_ranges",
    "elapsed_seconds",
    "seconds_per_window",
]


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def load_embedder(model_config: dict[str, Any], project_root: Path) -> Any:
    repo_path = resolve_project_path(model_config["qwen_repo_path"], project_root)
    if not repo_path.exists():
        raise FileNotFoundError(f"找不到 Qwen3-VL-Embedding 仓库目录: {repo_path}")
    sys.path.insert(0, str(repo_path))

    import torch
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    dtype_name = str(model_config.get("torch_dtype", "bfloat16"))
    torch_dtype = getattr(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "model_name_or_path": resolve_model_name_or_path(str(model_config["path"]), project_root),
        "torch_dtype": torch_dtype,
    }
    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return Qwen3VLEmbedder(**kwargs)


def read_windows_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_windows_for_shard(windows: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards 必须大于 0")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index 必须在 [0, num_shards) 范围内")
    return [window for index, window in enumerate(windows) if index % num_shards == shard_index]


def add_shard_suffix(path: Path, shard_index: int, num_shards: int) -> Path:
    return path.with_name(f"{path.stem}.shard{shard_index:02d}-of-{num_shards:02d}{path.suffix}")


def round_seconds(value: float) -> float:
    return round(float(value), 6)


def build_profile_summary(
    records: list[dict[str, Any]],
    model_load_seconds: float,
    encode_seconds: float,
    total_seconds: float,
    num_windows: int,
) -> dict[str, Any]:
    per_window_samples: list[float] = []
    for record in records:
        per_window_samples.extend([float(record["seconds_per_window"])] * int(record["batch_size"]))

    if per_window_samples:
        percentiles = np.percentile(per_window_samples, [50, 90, 95, 99])
        mean_seconds_per_window = sum(per_window_samples) / len(per_window_samples)
    else:
        percentiles = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mean_seconds_per_window = 0.0

    return {
        "num_windows": int(num_windows),
        "num_batches": len(records),
        "model_load_seconds": round_seconds(model_load_seconds),
        "encode_seconds": round_seconds(encode_seconds),
        "total_seconds": round_seconds(total_seconds),
        "mean_seconds_per_window": round_seconds(mean_seconds_per_window),
        "p50_seconds_per_window": round_seconds(float(percentiles[0])),
        "p90_seconds_per_window": round_seconds(float(percentiles[1])),
        "p95_seconds_per_window": round_seconds(float(percentiles[2])),
        "p99_seconds_per_window": round_seconds(float(percentiles[3])),
    }


def write_profile_outputs(
    csv_path: Path | None,
    json_path: Path | None,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if csv_path is not None:
        ensure_dir(csv_path.parent)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(records)

    if json_path is not None:
        write_json(json_path, {"summary": summary, "batches": records})


def encode_windows(
    model: Any,
    windows: list[dict[str, Any]],
    instruction: str,
    fps: float,
    max_frames: int,
    batch_size: int,
    profile_records: list[dict[str, Any]] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> np.ndarray:
    all_embeddings: list[np.ndarray] = []
    for batch_index, batch in enumerate(tqdm(chunked(windows, batch_size), desc="编码视频窗口")):
        inputs = [
            {
                "video": window["path"],
                "instruction": instruction,
                "fps": fps,
                "max_frames": max_frames,
            }
            for window in batch
        ]
        started = timer()
        embeddings = tensor_to_numpy(model.process(inputs))
        elapsed_seconds = round_seconds(timer() - started)
        if profile_records is not None:
            profile_records.append(
                {
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "window_indices": ",".join(str(window.get("index", "")) for window in batch),
                    "window_ids": ",".join(str(window.get("window_id", "")) for window in batch),
                    "time_ranges": "|".join(str(window.get("time_range", "")) for window in batch),
                    "elapsed_seconds": elapsed_seconds,
                    "seconds_per_window": round_seconds(elapsed_seconds / len(batch)),
                }
            )
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成视频窗口 embedding。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    parser.add_argument("--shard-index", type=int, default=0, help="当前分片编号，从 0 开始")
    parser.add_argument("--num-shards", type=int, default=1, help="总分片数")
    parser.add_argument("--profile-csv", default=None, help="可选，保存 batch 级 profiling CSV")
    parser.add_argument("--profile-json", default=None, help="可选，保存 profiling 汇总 JSON")
    args = parser.parse_args()

    total_started = time.perf_counter()
    project_root = Path.cwd()
    config = load_yaml(resolve_project_path(args.config, project_root))
    windows_path = resolve_project_path(config["outputs"]["windows_csv"], project_root)
    windows = read_windows_manifest(windows_path)
    if not windows:
        raise ValueError(f"窗口清单为空: {windows_path}")
    windows = filter_windows_for_shard(windows, args.shard_index, args.num_shards)
    if not windows:
        raise ValueError(f"分片没有可处理窗口: shard={args.shard_index}/{args.num_shards}")

    load_started = time.perf_counter()
    model = load_embedder(config["model"], project_root)
    model_load_seconds = time.perf_counter() - load_started
    profile_records: list[dict[str, Any]] = []
    encode_started = time.perf_counter()
    embeddings = encode_windows(
        model=model,
        windows=windows,
        instruction=str(config["model"].get("instruction", "Represent the user's input")),
        fps=float(config["embedding"].get("fps", 1.0)),
        max_frames=int(config["embedding"].get("max_frames", 16)),
        batch_size=int(config["embedding"].get("batch_size", 1)),
        profile_records=profile_records if args.profile_csv or args.profile_json else None,
    )
    encode_seconds = time.perf_counter() - encode_started

    embedding_path = resolve_project_path(config["outputs"]["window_embeddings"], project_root)
    metadata_path = resolve_project_path(config["outputs"]["window_metadata"], project_root)
    if args.num_shards > 1:
        embedding_path = add_shard_suffix(embedding_path, args.shard_index, args.num_shards)
        metadata_path = add_shard_suffix(metadata_path, args.shard_index, args.num_shards)
    ensure_dir(embedding_path.parent)
    np.save(embedding_path, embeddings)
    write_json(metadata_path, {"windows": windows})
    total_seconds = time.perf_counter() - total_started
    if args.profile_csv or args.profile_json:
        summary = build_profile_summary(
            records=profile_records,
            model_load_seconds=model_load_seconds,
            encode_seconds=encode_seconds,
            total_seconds=total_seconds,
            num_windows=len(windows),
        )
        write_profile_outputs(
            resolve_project_path(args.profile_csv, project_root) if args.profile_csv else None,
            resolve_project_path(args.profile_json, project_root) if args.profile_json else None,
            profile_records,
            summary,
        )
    print(f"已保存视频窗口 embedding: {embedding_path}")
    print(f"已保存视频窗口元数据: {metadata_path}")


if __name__ == "__main__":
    main()
