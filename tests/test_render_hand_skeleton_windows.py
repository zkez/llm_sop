import csv
from pathlib import Path

from scripts.render_hand_skeleton_windows import (
    build_output_row,
    resolve_hand_skeleton_config,
    render_windows,
)


def test_resolve_hand_skeleton_config_uses_section_defaults(tmp_path):
    config = {
        "hand_skeleton": {
            "source_windows_csv": "./outputs/source/windows.csv",
            "output_dir": "./data/hands",
            "output_windows_csv": "./outputs/hands/windows.csv",
            "max_num_hands": 1,
        },
        "outputs": {"windows_csv": "./outputs/fallback/windows.csv"},
    }

    resolved = resolve_hand_skeleton_config(config, tmp_path)

    assert resolved["source_windows_csv"] == tmp_path / "outputs/source/windows.csv"
    assert resolved["output_dir"] == tmp_path / "data/hands"
    assert resolved["output_windows_csv"] == tmp_path / "outputs/hands/windows.csv"
    assert resolved["max_num_hands"] == 1


def test_build_output_row_replaces_path_and_keeps_window_metadata(tmp_path):
    window = {
        "window_id": "window_000001",
        "index": "0",
        "start": "0.000",
        "end": "2.000",
        "duration": "2.000",
        "time_range": "00:00.000-00:02.000",
        "path": "/old/window_000001.mp4",
    }
    output_path = tmp_path / "window_000001.mp4"

    row = build_output_row(
        window,
        output_path,
        {"total_frames": 10, "hand_frames": 4, "hand_visibility_rate": 0.4},
    )

    assert row["window_id"] == "window_000001"
    assert row["path"] == str(output_path.resolve())
    assert row["source_path"] == "/old/window_000001.mp4"
    assert row["hand_frames"] == 4
    assert row["hand_visibility_rate"] == 0.4


def test_render_windows_writes_manifest_with_render_callback(tmp_path):
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake-video")
    windows = [
        {
            "window_id": "window_000001",
            "index": "0",
            "start": "0.000",
            "end": "2.000",
            "duration": "2.000",
            "time_range": "00:00.000-00:02.000",
            "path": str(source_video),
        }
    ]
    output_dir = tmp_path / "hands"
    manifest_path = tmp_path / "outputs" / "windows.csv"

    def fake_render(input_path: Path, output_path: Path, settings: dict):
        output_path.write_bytes(input_path.read_bytes())
        return {"total_frames": 5, "hand_frames": 2, "hand_visibility_rate": 0.4}

    rows = render_windows(
        windows=windows,
        output_dir=output_dir,
        output_windows_csv=manifest_path,
        settings={"skip_existing": False},
        render_one=fake_render,
    )

    assert len(rows) == 1
    assert Path(rows[0]["path"]).exists()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert manifest_rows[0]["window_id"] == "window_000001"
    assert manifest_rows[0]["hand_frames"] == "2"
