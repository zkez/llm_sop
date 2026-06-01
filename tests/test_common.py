from pathlib import Path

import numpy as np

from scripts.common import (
    format_time,
    format_time_range,
    get_nested,
    l2_normalize,
    mean_normalized,
    resolve_model_name_or_path,
    resolve_project_path,
)


def test_format_time_uses_minutes_and_seconds():
    assert format_time(65.25) == "01:05.250"


def test_format_time_range_formats_start_and_end():
    assert format_time_range(8, 16.5) == "00:08.000-00:16.500"


def test_l2_normalize_handles_rows_and_zero_vectors():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]])

    normalized = l2_normalize(vectors)

    np.testing.assert_allclose(normalized[0], np.array([0.6, 0.8]))
    np.testing.assert_allclose(normalized[1], np.array([0.0, 0.0]))


def test_mean_normalized_averages_normalized_vectors_then_normalizes_again():
    vectors = np.array([[3.0, 0.0], [0.0, 4.0]])

    prototype = mean_normalized(vectors)

    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(prototype, expected)


def test_resolve_project_path_keeps_absolute_paths_and_resolves_relative_paths(tmp_path):
    absolute = tmp_path / "file.txt"
    assert resolve_project_path(str(absolute), tmp_path) == absolute
    assert resolve_project_path("outputs/report.md", tmp_path) == Path(tmp_path / "outputs/report.md")


def test_resolve_model_name_or_path_keeps_huggingface_ids_and_resolves_local_paths(tmp_path):
    assert resolve_model_name_or_path("Qwen/Qwen3-VL-Embedding-2B", tmp_path) == "Qwen/Qwen3-VL-Embedding-2B"
    assert resolve_model_name_or_path("./models/Qwen3-VL-Embedding-2B", tmp_path) == str(
        Path(tmp_path / "models/Qwen3-VL-Embedding-2B").resolve()
    )


def test_get_nested_returns_defaults_for_missing_keys():
    data = {"video": {"ffmpeg_path": "/opt/bin/ffmpeg"}}

    assert get_nested(data, "video.ffmpeg_path", "ffmpeg") == "/opt/bin/ffmpeg"
    assert get_nested(data, "video.ffprobe_path", "ffprobe") == "ffprobe"
