from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_dir, l2_normalize, load_yaml, resolve_project_path, write_json
from scripts.embed_steps import build_step_prototypes, load_embedder
from scripts.embed_windows import encode_windows


MATRIX_PREFIX_COLUMNS = ["video_id", "video_step_id", "video_step_name", "path"]
SCORE_FIELDNAMES = [
    "video_id",
    "video_step_id",
    "video_step_name",
    "path",
    "text_step_id",
    "text_step_name",
    "similarity",
    "rank_for_video",
    "is_target_step",
]
SUMMARY_FIELDNAMES = [
    "video_id",
    "video_step_id",
    "video_step_name",
    "path",
    "best_step_id",
    "best_step_name",
    "best_score",
    "second_step_id",
    "second_step_name",
    "second_score",
    "margin",
    "target_score",
    "target_rank",
    "target_margin",
]


def _score(value: float) -> float:
    return round(float(value), 6)


def read_video_manifest(path: Path, project_root: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"视频清单为空: {path}")

    videos: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        raw_path = row.get("path") or row.get("video") or row.get("video_path")
        if not raw_path:
            raise ValueError(f"视频清单第 {index} 行缺少 path/video/video_path")
        video_path = resolve_project_path(raw_path, project_root)
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")
        step_id = str(row.get("step_id") or row.get("target_step_id") or "").strip()
        step_name = str(row.get("step_name") or row.get("target_step_name") or "").strip()
        video_id = str(row.get("video_id") or step_id or video_path.stem).strip()
        videos.append(
            {
                "video_id": video_id,
                "step_id": step_id,
                "step_name": step_name,
                "path": str(video_path.resolve()),
            }
        )
    return videos


def build_single_video_record(
    video_path: Path,
    project_root: Path,
    video_id: str | None = None,
    step_id: str | None = None,
    step_name: str | None = None,
) -> dict[str, str]:
    resolved_path = resolve_project_path(video_path, project_root)
    if not resolved_path.exists():
        raise FileNotFoundError(f"视频不存在: {resolved_path}")
    resolved_step_id = step_id or ""
    return {
        "video_id": video_id or resolved_step_id or resolved_path.stem,
        "step_id": resolved_step_id,
        "step_name": step_name or "",
        "path": str(resolved_path.resolve()),
    }


def compute_video_text_similarity(video_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    video_vectors = l2_normalize(video_embeddings)
    text_vectors = l2_normalize(text_embeddings)
    return video_vectors @ text_vectors.T


def _ranked_step_indices(scores: np.ndarray) -> list[int]:
    return [int(index) for index in np.argsort(-np.asarray(scores, dtype=np.float32))]


def build_score_rows(
    videos: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    matrix: np.ndarray,
) -> list[dict[str, Any]]:
    if matrix.shape != (len(videos), len(steps)):
        raise ValueError("相似度矩阵形状必须等于 [视频数, 步骤数]")

    rows: list[dict[str, Any]] = []
    for video_index, video in enumerate(videos):
        ranked_indices = _ranked_step_indices(matrix[video_index])
        rank_by_step_index = {step_index: rank + 1 for rank, step_index in enumerate(ranked_indices)}
        for step_index in ranked_indices:
            step = steps[step_index]
            rows.append(
                {
                    "video_id": video.get("video_id", ""),
                    "video_step_id": video.get("step_id", ""),
                    "video_step_name": video.get("step_name", ""),
                    "path": video.get("path", ""),
                    "text_step_id": step["id"],
                    "text_step_name": step["name"],
                    "similarity": _score(matrix[video_index, step_index]),
                    "rank_for_video": rank_by_step_index[step_index],
                    "is_target_step": str(step["id"]) == str(video.get("step_id", "")),
                }
            )
    return rows


def build_summary_rows(
    videos: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    matrix: np.ndarray,
) -> list[dict[str, Any]]:
    if matrix.shape != (len(videos), len(steps)):
        raise ValueError("相似度矩阵形状必须等于 [视频数, 步骤数]")

    step_index_by_id = {str(step["id"]): index for index, step in enumerate(steps)}
    rows: list[dict[str, Any]] = []
    for video_index, video in enumerate(videos):
        ranked_indices = _ranked_step_indices(matrix[video_index])
        best_index = ranked_indices[0]
        second_index = ranked_indices[1] if len(ranked_indices) > 1 else best_index
        best_score = float(matrix[video_index, best_index])
        second_score = float(matrix[video_index, second_index]) if len(ranked_indices) > 1 else 0.0
        target_step_id = str(video.get("step_id", ""))
        target_index = step_index_by_id.get(target_step_id)
        if target_index is None:
            target_score: float | str = ""
            target_rank: int | str = ""
            target_margin: float | str = ""
        else:
            target_score_value = float(matrix[video_index, target_index])
            target_rank = ranked_indices.index(target_index) + 1
            best_non_target_score = max(
                (
                    float(matrix[video_index, step_index])
                    for step_index in range(len(steps))
                    if step_index != target_index
                ),
                default=0.0,
            )
            target_score = _score(target_score_value)
            target_margin = _score(target_score_value - best_non_target_score)

        rows.append(
            {
                "video_id": video.get("video_id", ""),
                "video_step_id": video.get("step_id", ""),
                "video_step_name": video.get("step_name", ""),
                "path": video.get("path", ""),
                "best_step_id": steps[best_index]["id"],
                "best_step_name": steps[best_index]["name"],
                "best_score": _score(best_score),
                "second_step_id": steps[second_index]["id"] if len(ranked_indices) > 1 else "",
                "second_step_name": steps[second_index]["name"] if len(ranked_indices) > 1 else "",
                "second_score": _score(second_score) if len(ranked_indices) > 1 else "",
                "margin": _score(best_score - second_score),
                "target_score": target_score,
                "target_rank": target_rank,
                "target_margin": target_margin,
            }
        )
    return rows


def write_similarity_matrix_csv(
    path: Path,
    videos: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    matrix: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    step_columns = [f"{step['id']}:{step['name']}" for step in steps]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_PREFIX_COLUMNS + step_columns)
        writer.writeheader()
        for video_index, video in enumerate(videos):
            row = {
                "video_id": video.get("video_id", ""),
                "video_step_id": video.get("step_id", ""),
                "video_step_name": video.get("step_name", ""),
                "path": video.get("path", ""),
            }
            for step_index, column in enumerate(step_columns):
                row[column] = _score(matrix[video_index, step_index])
            writer.writerow(row)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def encode_step_videos(
    model: Any,
    videos: list[dict[str, Any]],
    instruction: str,
    fps: float,
    max_frames: int,
    batch_size: int,
) -> np.ndarray:
    windows = [
        {
            "window_id": video["video_id"],
            "index": str(index),
            "time_range": "",
            "path": video["path"],
        }
        for index, video in enumerate(videos)
    ]
    return encode_windows(
        model=model,
        windows=windows,
        instruction=instruction,
        fps=fps,
        max_frames=max_frames,
        batch_size=batch_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验：不同长度工步视频与 SOP 文本的相似度。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    parser.add_argument("--steps", default="configs/steps.yaml", help="步骤配置 YAML 路径")
    parser.add_argument("--video", default=None, help="单个工步视频路径")
    parser.add_argument("--video-id", default=None, help="单个视频 ID，默认使用 step-id 或文件名")
    parser.add_argument("--video-step-id", default=None, help="单个视频对应的目标工步 ID")
    parser.add_argument("--video-step-name", default=None, help="单个视频对应的目标工步名称")
    parser.add_argument("--video-manifest", default=None, help="多个工步视频 CSV，列包含 path，可选 video_id/step_id/step_name")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--fps", type=float, default=None, help="覆盖视频采样 FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="覆盖每个视频最大采样帧数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    if not args.video and not args.video_manifest:
        raise ValueError("必须指定 --video 或 --video-manifest")
    if args.video and args.video_manifest:
        raise ValueError("--video 和 --video-manifest 只能指定一个")

    config = load_yaml(resolve_project_path(args.config, project_root))
    steps_config = load_yaml(resolve_project_path(args.steps, project_root))
    steps = steps_config["steps"]
    if args.video_manifest:
        videos = read_video_manifest(resolve_project_path(args.video_manifest, project_root), project_root)
    else:
        videos = [
            build_single_video_record(
                Path(args.video),
                project_root=project_root,
                video_id=args.video_id,
                step_id=args.video_step_id,
                step_name=args.video_step_name,
            )
        ]

    instruction = str(config["model"].get("instruction", "Represent the user's input"))
    batch_size = int(args.batch_size or config["embedding"].get("batch_size", 1))
    fps = float(args.fps if args.fps is not None else config["embedding"].get("fps", 1.0))
    max_frames = int(args.max_frames if args.max_frames is not None else config["embedding"].get("max_frames", 16))
    output_dir = ensure_dir(resolve_project_path(args.output_dir, project_root))

    model = load_embedder(config["model"], project_root)
    text_embeddings = build_step_prototypes(model=model, steps=steps, instruction=instruction, batch_size=batch_size)
    video_embeddings = encode_step_videos(
        model=model,
        videos=videos,
        instruction=instruction,
        fps=fps,
        max_frames=max_frames,
        batch_size=batch_size,
    )
    matrix = compute_video_text_similarity(video_embeddings, text_embeddings)

    np.save(output_dir / "video_embeddings.npy", video_embeddings)
    np.save(output_dir / "text_embeddings.npy", text_embeddings)
    np.save(output_dir / "video_text_similarity.npy", matrix)
    write_json(output_dir / "metadata.json", {"videos": videos, "steps": steps, "fps": fps, "max_frames": max_frames})
    write_similarity_matrix_csv(output_dir / "video_text_similarity.csv", videos, steps, matrix)
    write_csv(output_dir / "video_text_scores.csv", SCORE_FIELDNAMES, build_score_rows(videos, steps, matrix))
    write_csv(output_dir / "video_summary.csv", SUMMARY_FIELDNAMES, build_summary_rows(videos, steps, matrix))
    print(f"已保存视频-文本相似度矩阵: {output_dir / 'video_text_similarity.csv'}")
    print(f"已保存视频匹配汇总: {output_dir / 'video_summary.csv'}")


if __name__ == "__main__":
    main()
