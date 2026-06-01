from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import (
    ensure_dir,
    l2_normalize,
    load_yaml,
    read_json,
    resolve_project_path,
)


def compute_similarity_matrix(step_embeddings: np.ndarray, window_embeddings: np.ndarray) -> np.ndarray:
    step_vectors = l2_normalize(step_embeddings)
    window_vectors = l2_normalize(window_embeddings)
    return step_vectors @ window_vectors.T


def _window_margin(matrix: np.ndarray, step_index: int, window_index: int) -> float:
    score = float(matrix[step_index, window_index])
    if matrix.shape[0] == 1:
        return score
    other_scores = np.delete(matrix[:, window_index], step_index)
    return score - float(other_scores.max())


def _window_center(window: dict[str, Any]) -> float:
    return (float(window["start"]) + float(window["end"])) / 2.0


def _format_seconds(seconds: float) -> str:
    if float(seconds).is_integer():
        return str(int(seconds))
    return f"{seconds:g}"


def _context_window_indices(windows: list[dict[str, Any]], window_index: int, context_seconds: float) -> list[int]:
    if context_seconds <= 0:
        return [window_index]
    center = _window_center(windows[window_index])
    start = center - context_seconds
    end = center + context_seconds
    indices = [
        index
        for index, window in enumerate(windows)
        if start <= _window_center(window) <= end
    ]
    return indices or [window_index]


def _classification_score(
    matrix: np.ndarray,
    step_index: int,
    window_index: int,
    windows: list[dict[str, Any]],
    score_median_context_seconds: float,
) -> tuple[float, list[str]]:
    if score_median_context_seconds <= 0:
        return float(matrix[step_index, window_index]), []
    indices = _context_window_indices(windows, window_index, score_median_context_seconds)
    score = float(np.median(matrix[step_index, indices]))
    reason = f"使用候选窗口前后 {_format_seconds(score_median_context_seconds)} 秒中位数"
    return score, [reason]


def _classify_score(
    best_score: float,
    margin: float,
    completion_threshold: float,
    missing_threshold: float,
    margin_threshold: float,
    ignore_margin_for_completion: bool = False,
) -> tuple[str, list[str]]:
    reason_parts: list[str] = []
    if best_score < missing_threshold:
        status = "missing"
        reason_parts.append("最高分低于 missing_threshold")
    elif best_score < completion_threshold:
        status = "uncertain"
        reason_parts.append("最高分未达到 completion_threshold")
    elif ignore_margin_for_completion:
        status = "completed"
        reason_parts.append("分数达到阈值，已按配置忽略 margin")
    elif margin < margin_threshold:
        status = "uncertain"
        reason_parts.append("与同一窗口第二名步骤的 margin 偏小")
    else:
        status = "completed"
        reason_parts.append("分数和 margin 均达到阈值")
    return status, reason_parts


def _build_summary_row(
    step: dict[str, Any],
    window: dict[str, Any],
    best_score: float,
    margin: float,
    status: str,
    reason_parts: list[str],
) -> dict[str, Any]:
    return {
        "step_id": step["id"],
        "step_name": step["name"],
        "best_window_id": window["window_id"] if status != "missing" else "--",
        "best_time_range": window["time_range"] if status != "missing" else "--",
        "candidate_window_id": window["window_id"],
        "candidate_time_range": window["time_range"],
        "best_score": round(best_score, 6),
        "margin": round(margin, 6),
        "status": status,
        "reason": "；".join(reason_parts),
    }


def summarize_steps(
    matrix: np.ndarray,
    steps: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    completion_threshold: float,
    missing_threshold: float,
    margin_threshold: float,
    enforce_order: bool,
    ignore_margin_for_completion: bool = False,
    score_median_context_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    if matrix.shape != (len(steps), len(windows)):
        raise ValueError("相似度矩阵形状必须等于 [步骤数, 窗口数]")

    rows: list[dict[str, Any]] = []
    last_completed_start = -float("inf")

    for step_index, step in enumerate(steps):
        best_window_index = int(np.argmax(matrix[step_index]))
        best_window = windows[best_window_index]
        best_score, score_reason_parts = _classification_score(
            matrix,
            step_index,
            best_window_index,
            windows,
            score_median_context_seconds,
        )
        margin = _window_margin(matrix, step_index, best_window_index)

        status, reason_parts = _classify_score(
            best_score,
            margin,
            completion_threshold,
            missing_threshold,
            margin_threshold,
            ignore_margin_for_completion=ignore_margin_for_completion,
        )
        reason_parts = score_reason_parts + reason_parts

        if (
            enforce_order
            and status == "completed"
            and float(best_window["start"]) < last_completed_start
        ):
            status = "uncertain"
            reason_parts.append("顺序冲突：该步骤匹配到更早的时间窗口")

        if status == "completed":
            last_completed_start = float(best_window["start"])

        rows.append(_build_summary_row(step, best_window, best_score, margin, status, reason_parts))

    return rows


def decode_monotonic_path(matrix: np.ndarray, min_gap_windows: int = 1) -> list[int]:
    if matrix.ndim != 2:
        raise ValueError("相似度矩阵必须是二维数组")
    num_steps, num_windows = matrix.shape
    if num_steps == 0 or num_windows == 0:
        raise ValueError("相似度矩阵不能为空")
    if min_gap_windows < 0:
        raise ValueError("min_gap_windows 不能小于 0")
    required_windows = 1 + (num_steps - 1) * max(1, min_gap_windows)
    if required_windows > num_windows:
        raise ValueError("窗口数量不足，无法为所有步骤生成单调递增路径")

    scores = np.asarray(matrix, dtype=np.float32)
    dp = np.full((num_steps, num_windows), -np.inf, dtype=np.float32)
    back = np.full((num_steps, num_windows), -1, dtype=np.int32)
    dp[0] = scores[0]

    gap = max(1, min_gap_windows)
    for step_index in range(1, num_steps):
        best_previous_score = -np.inf
        best_previous_index = -1
        for window_index in range(num_windows):
            previous_index = window_index - gap
            if previous_index >= 0 and dp[step_index - 1, previous_index] > best_previous_score:
                best_previous_score = dp[step_index - 1, previous_index]
                best_previous_index = previous_index
            if best_previous_index >= 0:
                dp[step_index, window_index] = best_previous_score + scores[step_index, window_index]
                back[step_index, window_index] = best_previous_index

    final_index = int(np.argmax(dp[-1]))
    if not np.isfinite(dp[-1, final_index]):
        raise ValueError("无法生成有效的单调递增路径")

    path = [final_index]
    for step_index in range(num_steps - 1, 0, -1):
        final_index = int(back[step_index, final_index])
        if final_index < 0:
            raise ValueError("单调递增路径回溯失败")
        path.append(final_index)
    return list(reversed(path))


def summarize_steps_sequence(
    matrix: np.ndarray,
    steps: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    completion_threshold: float,
    missing_threshold: float,
    margin_threshold: float,
    min_gap_windows: int = 1,
    ignore_margin_for_completion: bool = False,
    score_median_context_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    if matrix.shape != (len(steps), len(windows)):
        raise ValueError("相似度矩阵形状必须等于 [步骤数, 窗口数]")

    path = decode_monotonic_path(matrix, min_gap_windows=min_gap_windows)
    rows: list[dict[str, Any]] = []
    for step_index, window_index in enumerate(path):
        step = steps[step_index]
        window = windows[window_index]
        best_score, score_reason_parts = _classification_score(
            matrix,
            step_index,
            window_index,
            windows,
            score_median_context_seconds,
        )
        margin = _window_margin(matrix, step_index, window_index)
        status, reason_parts = _classify_score(
            best_score,
            margin,
            completion_threshold,
            missing_threshold,
            margin_threshold,
            ignore_margin_for_completion=ignore_margin_for_completion,
        )
        reason_parts = score_reason_parts + reason_parts
        reason_parts.append("时序解码选择窗口")
        rows.append(_build_summary_row(step, window, best_score, margin, status, reason_parts))
    return rows


def write_similarity_matrix_csv(
    path: str | Path,
    matrix: np.ndarray,
    steps: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step_id", "step_name", *[window["time_range"] for window in windows]])
        for step, scores in zip(steps, matrix):
            writer.writerow([step["id"], step["name"], *[round(float(score), 6) for score in scores]])


def write_scores_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fieldnames = [
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
    ]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report_md(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    lines = [
        "# 视频步骤核验报告",
        "",
        "| 步骤 | 状态 | 最佳时间段 | 分数 | Margin | 说明 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {status} | {time_range} | {score:.6f} | {margin:.6f} | {reason} |".format(
                name=row["step_name"],
                status=row["status"],
                time_range=row["best_time_range"],
                score=float(row["best_score"]),
                margin=float(row["margin"]),
                reason=row["reason"],
            )
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `completed` 表示最高分和区分度都达到当前阈值。",
            "- `missing` 表示最高分低于缺失阈值。",
            "- `uncertain` 表示有候选片段，但置信度、区分度或顺序约束不足。",
            "- 这是一版 embedding-only 基线结果，需要结合热力图和原视频人工复核。",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")


def write_heatmap_png(
    path: str | Path,
    matrix: np.ndarray,
    steps: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    target = Path(path)
    ensure_dir(target.parent)
    fig_width = max(8, min(24, len(windows) * 0.45))
    fig_height = max(4, min(18, len(steps) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([step["id"] for step in steps])
    tick_step = max(1, len(windows) // 12)
    tick_indices = list(range(0, len(windows), tick_step))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([windows[index]["time_range"] for index in tick_indices], rotation=45, ha="right")
    ax.set_xlabel("video window")
    ax.set_ylabel("step_id")
    ax.set_title("Step-window similarity heatmap")
    fig.colorbar(image, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="计算步骤文本 embedding 与视频窗口 embedding 的相似度。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    args = parser.parse_args()

    project_root = Path.cwd()
    config = load_yaml(resolve_project_path(args.config, project_root))
    outputs = config["outputs"]
    scoring = config["scoring"]

    step_embeddings = np.load(resolve_project_path(outputs["step_embeddings"], project_root))
    window_embeddings = np.load(resolve_project_path(outputs["window_embeddings"], project_root))
    step_metadata = read_json(resolve_project_path(outputs["step_metadata"], project_root))
    window_metadata = read_json(resolve_project_path(outputs["window_metadata"], project_root))

    steps = step_metadata["steps"]
    windows = window_metadata["windows"]
    matrix = compute_similarity_matrix(step_embeddings, window_embeddings)
    decode_strategy = str(scoring.get("decode_strategy", "sequence")).lower()
    ignore_margin_for_completion = bool(scoring.get("ignore_margin_for_completion", False))
    score_median_context_seconds = float(scoring.get("score_median_context_seconds", 0.0))
    if decode_strategy == "sequence":
        rows = summarize_steps_sequence(
            matrix,
            steps,
            windows,
            completion_threshold=float(scoring["completion_threshold"]),
            missing_threshold=float(scoring["missing_threshold"]),
            margin_threshold=float(scoring["margin_threshold"]),
            min_gap_windows=int(scoring.get("sequence_min_gap_windows", 1)),
            ignore_margin_for_completion=ignore_margin_for_completion,
            score_median_context_seconds=score_median_context_seconds,
        )
    elif decode_strategy == "independent":
        rows = summarize_steps(
            matrix,
            steps,
            windows,
            completion_threshold=float(scoring["completion_threshold"]),
            missing_threshold=float(scoring["missing_threshold"]),
            margin_threshold=float(scoring["margin_threshold"]),
            enforce_order=bool(scoring.get("enforce_order", True)),
            ignore_margin_for_completion=ignore_margin_for_completion,
            score_median_context_seconds=score_median_context_seconds,
        )
    else:
        raise ValueError(f"未知 decode_strategy: {decode_strategy}")

    write_similarity_matrix_csv(resolve_project_path(outputs["similarity_matrix_csv"], project_root), matrix, steps, windows)
    write_scores_csv(resolve_project_path(outputs["scores_csv"], project_root), rows)
    write_heatmap_png(resolve_project_path(outputs["heatmap_png"], project_root), matrix, steps, windows)
    write_report_md(resolve_project_path(outputs["report_md"], project_root), rows)

    print(f"已生成报告: {resolve_project_path(outputs['report_md'], project_root)}")


if __name__ == "__main__":
    main()
