from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_dir, format_time_range, get_nested, load_yaml, resolve_project_path


def find_ffmpeg(config: dict[str, Any]) -> str:
    configured = get_nested(config, "video.ffmpeg_path")
    if configured:
        return str(configured)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise RuntimeError("找不到 ffmpeg，请安装系统 ffmpeg，或安装 imageio-ffmpeg。") from error


def find_ffprobe(config: dict[str, Any]) -> str | None:
    configured = get_nested(config, "video.ffprobe_path")
    if configured:
        return str(configured)
    return shutil.which("ffprobe")


def get_video_duration(video_path: Path, ffprobe_path: str | None = None) -> float:
    if ffprobe_path is None:
        return get_video_duration_with_decord(video_path)
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def get_video_duration_with_decord(video_path: Path) -> float:
    try:
        from decord import VideoReader

        reader = VideoReader(str(video_path))
        fps = float(reader.get_avg_fps())
        if fps <= 0:
            raise ValueError("decord 返回的视频 FPS 非法")
        return len(reader) / fps
    except Exception as error:
        raise RuntimeError("无法读取视频时长：系统缺少 ffprobe，且 decord 读取失败。") from error


def generate_windows(duration: float, window_seconds: float, stride_seconds: float, min_window_seconds: float) -> list[dict[str, float]]:
    if duration <= 0:
        raise ValueError("视频时长必须大于 0")
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds 和 stride_seconds 必须大于 0")

    windows: list[dict[str, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + window_seconds, duration)
        if end - start >= min_window_seconds:
            windows.append({"start": start, "end": end, "duration": end - start})
        if end >= duration:
            break
        start += stride_seconds

    if not windows:
        windows.append({"start": 0.0, "end": duration, "duration": duration})
    return windows


def split_window(ffmpeg_path: str, input_path: Path, output_path: Path, start: float, duration: float) -> None:
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["window_id", "index", "start", "end", "duration", "time_range", "path"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="将长视频切成有重叠的短窗口。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    args = parser.parse_args()

    project_root = Path.cwd()
    config = load_yaml(resolve_project_path(args.config, project_root))
    video_config = config["video"]
    outputs = config["outputs"]

    input_path = resolve_project_path(video_config["path"], project_root)
    windows_dir = ensure_dir(resolve_project_path(video_config["windows_dir"], project_root))
    manifest_path = resolve_project_path(outputs["windows_csv"], project_root)
    ffmpeg_path = find_ffmpeg(config)
    ffprobe_path = find_ffprobe(config)
    duration = get_video_duration(input_path, ffprobe_path)
    windows = generate_windows(
        duration=duration,
        window_seconds=float(video_config["window_seconds"]),
        stride_seconds=float(video_config["stride_seconds"]),
        min_window_seconds=float(video_config.get("min_window_seconds", 1.0)),
    )
    skip_existing = bool(video_config.get("skip_existing", True))

    rows: list[dict[str, Any]] = []
    for index, window in enumerate(tqdm(windows, desc="切分视频窗口")):
        window_id = f"window_{index + 1:06d}"
        output_path = windows_dir / f"{window_id}.mp4"
        if not (skip_existing and output_path.exists()):
            split_window(ffmpeg_path, input_path, output_path, window["start"], window["duration"])
        rows.append(
            {
                "window_id": window_id,
                "index": index,
                "start": f"{window['start']:.3f}",
                "end": f"{window['end']:.3f}",
                "duration": f"{window['duration']:.3f}",
                "time_range": format_time_range(window["start"], window["end"]),
                "path": str(output_path.resolve()),
            }
        )

    write_manifest(manifest_path, rows)
    print(f"已生成窗口清单: {manifest_path}")


if __name__ == "__main__":
    main()
