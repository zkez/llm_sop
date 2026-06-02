import csv
from pathlib import Path

import numpy as np

from experiments.step_video_text_similarity import (
    build_score_rows,
    build_summary_rows,
    compute_video_text_similarity,
    read_video_manifest,
)


def test_read_video_manifest_resolves_paths_and_defaults_video_id(tmp_path):
    video_path = tmp_path / "videos" / "step01.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"fake-video")
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step_id", "step_name", "path"])
        writer.writeheader()
        writer.writerow(
            {
                "step_id": "step_01",
                "step_name": "接电连接",
                "path": "./videos/step01.mp4",
            }
        )

    videos = read_video_manifest(manifest_path, tmp_path)

    assert videos == [
        {
            "video_id": "step_01",
            "step_id": "step_01",
            "step_name": "接电连接",
            "path": str(video_path.resolve()),
        }
    ]


def test_compute_video_text_similarity_returns_video_by_step_matrix():
    video_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text_embeddings = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    matrix = compute_video_text_similarity(video_embeddings, text_embeddings)

    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix[0], [1.0, 0.70710677], rtol=1e-6)
    np.testing.assert_allclose(matrix[1], [0.0, 0.70710677], rtol=1e-6)


def test_build_summary_rows_reports_best_target_rank_and_margin():
    videos = [
        {"video_id": "v1", "step_id": "step_01", "step_name": "接电连接", "path": "/tmp/v1.mp4"},
        {"video_id": "v2", "step_id": "step_02", "step_name": "航插连接", "path": "/tmp/v2.mp4"},
    ]
    steps = [
        {"id": "step_01", "name": "接电连接"},
        {"id": "step_02", "name": "航插连接"},
        {"id": "step_03", "name": "扫入系统"},
    ]
    matrix = np.asarray(
        [
            [0.9, 0.4, 0.2],
            [0.5, 0.7, 0.6],
        ],
        dtype=np.float32,
    )

    summary_rows = build_summary_rows(videos, steps, matrix)
    score_rows = build_score_rows(videos, steps, matrix)

    assert summary_rows[0]["best_step_id"] == "step_01"
    assert summary_rows[0]["target_score"] == 0.9
    assert summary_rows[0]["target_rank"] == 1
    assert summary_rows[0]["margin"] == 0.5
    assert summary_rows[1]["best_step_id"] == "step_02"
    assert summary_rows[1]["target_rank"] == 1
    assert len(score_rows) == 6
    assert score_rows[0]["video_id"] == "v1"
    assert score_rows[0]["text_step_id"] == "step_01"
    assert score_rows[0]["rank_for_video"] == 1
