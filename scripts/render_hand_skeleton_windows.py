from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_dir, load_yaml, resolve_project_path
from scripts.embed_windows import read_windows_manifest


MANIFEST_FIELDNAMES = [
    "window_id",
    "index",
    "start",
    "end",
    "duration",
    "time_range",
    "path",
    "source_path",
    "total_frames",
    "hand_frames",
    "hand_visibility_rate",
]


def resolve_hand_skeleton_config(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    section = dict(config.get("hand_skeleton", {}))
    outputs = config.get("outputs", {})
    source_windows_csv = section.get("source_windows_csv") or outputs.get("windows_csv")
    if not source_windows_csv:
        raise ValueError("缺少 hand_skeleton.source_windows_csv 或 outputs.windows_csv")
    output_windows_csv = section.get("output_windows_csv") or outputs.get("windows_csv")
    if not output_windows_csv:
        raise ValueError("缺少 hand_skeleton.output_windows_csv 或 outputs.windows_csv")
    output_dir = section.get("output_dir")
    if not output_dir:
        output_dir = resolve_project_path(output_windows_csv, project_root).parent / "windows"

    return {
        "source_windows_csv": resolve_project_path(source_windows_csv, project_root),
        "output_windows_csv": resolve_project_path(output_windows_csv, project_root),
        "output_dir": resolve_project_path(output_dir, project_root),
        "max_num_hands": int(section.get("max_num_hands", 2)),
        "min_detection_confidence": float(section.get("min_detection_confidence", 0.5)),
        "min_tracking_confidence": float(section.get("min_tracking_confidence", 0.5)),
        "skip_existing": bool(section.get("skip_existing", True)),
        "line_thickness": int(section.get("line_thickness", 2)),
        "point_radius": int(section.get("point_radius", 2)),
    }


def build_output_row(window: dict[str, Any], output_path: Path, stats: dict[str, Any]) -> dict[str, Any]:
    total_frames = int(stats.get("total_frames", 0))
    hand_frames = int(stats.get("hand_frames", 0))
    visibility_rate = float(stats.get("hand_visibility_rate", 0.0))
    return {
        "window_id": str(window.get("window_id", "")),
        "index": str(window.get("index", "")),
        "start": str(window.get("start", "")),
        "end": str(window.get("end", "")),
        "duration": str(window.get("duration", "")),
        "time_range": str(window.get("time_range", "")),
        "path": str(output_path.resolve()),
        "source_path": str(window.get("path", "")),
        "total_frames": total_frames,
        "hand_frames": hand_frames,
        "hand_visibility_rate": round(visibility_rate, 6),
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def render_one_window(input_path: Path, output_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "缺少 mediapipe 或 opencv-python。请先安装依赖，例如: "
            "uv pip install mediapipe opencv-python"
        ) from error

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频窗口: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"视频窗口尺寸非法: {input_path}")

    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    mp_hands = mp.solutions.hands
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    total_frames = 0
    hand_frames = 0
    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=int(settings.get("max_num_hands", 2)),
            min_detection_confidence=float(settings.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(settings.get("min_tracking_confidence", 0.5)),
        ) as hands:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                total_frames += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    hand_frames += 1
                    for hand_landmarks in result.multi_hand_landmarks:
                        drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            drawing_styles.get_default_hand_landmarks_style(),
                            drawing_styles.get_default_hand_connections_style(),
                        )
                writer.write(frame)
    finally:
        capture.release()
        writer.release()

    visibility_rate = hand_frames / total_frames if total_frames else 0.0
    return {
        "total_frames": total_frames,
        "hand_frames": hand_frames,
        "hand_visibility_rate": visibility_rate,
    }


def render_windows(
    windows: list[dict[str, Any]],
    output_dir: Path,
    output_windows_csv: Path,
    settings: dict[str, Any],
    render_one: Callable[[Path, Path, dict[str, Any]], dict[str, Any]] = render_one_window,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ensure_dir(output_dir)
    for window in tqdm(windows, desc="渲染手部骨架窗口"):
        source_path = Path(str(window["path"]))
        output_path = output_dir / f"{window['window_id']}.mp4"
        if bool(settings.get("skip_existing", True)) and output_path.exists():
            stats = {
                "total_frames": int(window.get("total_frames", 0) or 0),
                "hand_frames": int(window.get("hand_frames", 0) or 0),
                "hand_visibility_rate": float(window.get("hand_visibility_rate", 0.0) or 0.0),
            }
        else:
            stats = render_one(source_path, output_path, settings)
        rows.append(build_output_row(window, output_path, stats))
    write_manifest(output_windows_csv, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="将 MediaPipe 手部关键点骨架叠加到视频窗口上。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    parser.add_argument("--source-windows-csv", default=None, help="可选，覆盖输入窗口清单路径")
    parser.add_argument("--output-windows-csv", default=None, help="可选，覆盖输出窗口清单路径")
    parser.add_argument("--output-dir", default=None, help="可选，覆盖输出视频窗口目录")
    parser.add_argument("--limit", type=int, default=None, help="可选，只处理前 N 个窗口")
    parser.add_argument("--no-skip-existing", action="store_true", help="重新渲染已存在的输出窗口")
    args = parser.parse_args()

    project_root = Path.cwd()
    config = load_yaml(resolve_project_path(args.config, project_root))
    settings = resolve_hand_skeleton_config(config, project_root)
    if args.source_windows_csv:
        settings["source_windows_csv"] = resolve_project_path(args.source_windows_csv, project_root)
    if args.output_windows_csv:
        settings["output_windows_csv"] = resolve_project_path(args.output_windows_csv, project_root)
    if args.output_dir:
        settings["output_dir"] = resolve_project_path(args.output_dir, project_root)
    if args.no_skip_existing:
        settings["skip_existing"] = False

    windows = read_windows_manifest(settings["source_windows_csv"])
    if args.limit is not None:
        windows = windows[: args.limit]
    if not windows:
        raise ValueError(f"窗口清单为空: {settings['source_windows_csv']}")

    rows = render_windows(
        windows=windows,
        output_dir=settings["output_dir"],
        output_windows_csv=settings["output_windows_csv"],
        settings=settings,
    )
    print(f"已生成手部骨架窗口清单: {settings['output_windows_csv']}")
    print(f"已生成手部骨架窗口数量: {len(rows)}")


if __name__ == "__main__":
    main()
