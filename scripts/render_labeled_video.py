from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_dir, load_yaml, resolve_project_path
from scripts.split_video import find_ffmpeg, find_ffprobe, get_video_duration


def parse_time_value(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"无法解析时间: {value}")


def parse_time_range(value: str) -> tuple[float, float]:
    if not value or value == "--" or "-" not in value:
        raise ValueError(f"无法解析时间段: {value}")
    start_text, end_text = value.split("-", 1)
    start = parse_time_value(start_text)
    end = parse_time_value(end_text)
    if end <= start:
        raise ValueError(f"时间段结束时间必须晚于开始时间: {value}")
    return start, end


def format_ass_time(seconds: float) -> str:
    centiseconds = int(round(float(seconds) * 100))
    total_seconds, centis = divmod(centiseconds, 100)
    minutes_total, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def read_score_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _score_value(row: dict[str, Any]) -> float:
    try:
        return float(row.get("best_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _candidate_events(
    rows: list[dict[str, Any]],
    included_statuses: set[str],
    time_field: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status", ""))
        if status not in included_statuses:
            continue
        time_range = str(row.get(time_field) or row.get("candidate_time_range") or "")
        try:
            start, end = parse_time_range(time_range)
        except ValueError:
            continue
        events.append(
            {
                "start": start,
                "end": end,
                "label": str(row.get("step_name") or row.get("step_id") or "unknown"),
                "score": _score_value(row),
            }
        )
    return events


def _merge_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in segments:
        if segment["end"] <= segment["start"]:
            continue
        if merged and merged[-1]["label"] == segment["label"] and abs(float(merged[-1]["end"]) - float(segment["start"])) < 1e-6:
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(segment)
    return merged


def build_label_segments(
    rows: list[dict[str, Any]],
    video_duration: float,
    included_statuses: set[str] | None = None,
    background_label: str = "background",
    time_field: str = "best_time_range",
) -> list[dict[str, Any]]:
    included = included_statuses or {"completed"}
    events = _candidate_events(rows, included, time_field)
    boundaries = {0.0, float(video_duration)}
    for event in events:
        boundaries.add(max(0.0, min(float(video_duration), float(event["start"]))))
        boundaries.add(max(0.0, min(float(video_duration), float(event["end"]))))

    sorted_boundaries = sorted(boundaries)
    segments: list[dict[str, Any]] = []
    for start, end in zip(sorted_boundaries, sorted_boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        active = [
            event
            for event in events
            if float(event["start"]) <= midpoint < float(event["end"])
        ]
        if active:
            selected = max(active, key=lambda item: float(item["score"]))
            label = selected["label"]
        else:
            label = background_label
        segments.append({"start": round(start, 6), "end": round(end, 6), "label": label})
    return _merge_adjacent_segments(segments)


def escape_ass_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def write_ass_subtitles(
    path: str | Path,
    segments: list[dict[str, Any]],
    font_name: str = "Noto Sans CJK SC",
    font_size: int = 32,
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Label,{font_name},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,"
        "-1,0,0,0,100,100,0,0,3,2,0,7,24,24,24,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for segment in segments:
        lines.append(
            "Dialogue: 0,{start},{end},Label,,0,0,0,,{text}".format(
                start=format_ass_time(float(segment["start"])),
                end=format_ass_time(float(segment["end"])),
                text=escape_ass_text(str(segment["label"])),
            )
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_subtitles_filter_path(path: Path) -> str:
    text = path.as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return f"'{text}'"


def render_video(ffmpeg_path: str, input_path: Path, ass_path: Path, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"subtitles={escape_subtitles_filter_path(ass_path)}",
        "-c:a",
        "copy",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def parse_statuses(value: str) -> set[str]:
    statuses = {item.strip() for item in value.split(",") if item.strip()}
    if not statuses:
        raise ValueError("至少需要一个 status")
    return statuses


def default_output_path(scores_path: Path) -> Path:
    return scores_path.with_name("labeled_output.mp4")


def default_ass_path(output_path: Path) -> Path:
    return output_path.with_suffix(".ass")


def main() -> None:
    parser = argparse.ArgumentParser(description="根据步骤打分结果生成左上角工步标签视频。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    parser.add_argument("--scores", default=None, help="可选，覆盖 scores_csv 路径")
    parser.add_argument("--output", default=None, help="可选，输出带标签 MP4 路径")
    parser.add_argument("--ass-output", default=None, help="可选，保存中间 ASS 字幕路径")
    parser.add_argument("--include-statuses", default="completed", help="逗号分隔，参与显示的状态")
    parser.add_argument("--background-label", default="background", help="未识别时显示的标签")
    parser.add_argument("--time-field", default="best_time_range", help="从 scores CSV 读取的时间段字段")
    parser.add_argument("--font-name", default="Noto Sans CJK SC", help="ASS 字幕字体名")
    parser.add_argument("--font-size", type=int, default=32, help="ASS 字幕字号")
    args = parser.parse_args()

    project_root = Path.cwd()
    config = load_yaml(resolve_project_path(args.config, project_root))
    video_path = resolve_project_path(config["video"]["path"], project_root)
    scores_path = resolve_project_path(args.scores or config["outputs"]["scores_csv"], project_root)
    output_path = resolve_project_path(args.output, project_root) if args.output else default_output_path(scores_path)
    ass_path = resolve_project_path(args.ass_output, project_root) if args.ass_output else default_ass_path(output_path)

    ffmpeg_path = find_ffmpeg(config)
    duration = get_video_duration(video_path, find_ffprobe(config))
    rows = read_score_rows(scores_path)
    segments = build_label_segments(
        rows,
        video_duration=duration,
        included_statuses=parse_statuses(args.include_statuses),
        background_label=args.background_label,
        time_field=args.time_field,
    )
    write_ass_subtitles(ass_path, segments, font_name=args.font_name, font_size=args.font_size)
    render_video(ffmpeg_path, video_path, ass_path, output_path)

    print(f"已生成字幕: {ass_path}")
    print(f"已生成标注视频: {output_path}")


if __name__ == "__main__":
    main()
