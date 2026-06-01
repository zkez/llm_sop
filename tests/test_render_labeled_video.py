from pathlib import Path

from scripts.render_labeled_video import (
    build_label_segments,
    format_ass_time,
    parse_time_range,
    write_ass_subtitles,
)


def test_parse_time_range_reads_minute_second_ranges():
    assert parse_time_range("01:52.000-01:53.500") == (112.0, 113.5)


def test_build_label_segments_fills_background_and_completed_steps():
    rows = [
        {
            "step_name": "接电连接",
            "best_time_range": "00:01.000-00:02.000",
            "candidate_time_range": "00:01.000-00:02.000",
            "best_score": "0.66",
            "status": "completed",
        },
        {
            "step_name": "航插连接",
            "best_time_range": "00:03.000-00:04.000",
            "candidate_time_range": "00:03.000-00:04.000",
            "best_score": "0.62",
            "status": "uncertain",
        },
    ]

    segments = build_label_segments(
        rows,
        video_duration=5.0,
        included_statuses={"completed"},
        background_label="background",
    )

    assert segments == [
        {"start": 0.0, "end": 1.0, "label": "background"},
        {"start": 1.0, "end": 2.0, "label": "接电连接"},
        {"start": 2.0, "end": 5.0, "label": "background"},
    ]


def test_build_label_segments_uses_highest_score_for_overlaps():
    rows = [
        {
            "step_name": "低分步骤",
            "best_time_range": "00:01.000-00:03.000",
            "candidate_time_range": "00:01.000-00:03.000",
            "best_score": "0.61",
            "status": "completed",
        },
        {
            "step_name": "高分步骤",
            "best_time_range": "00:02.000-00:04.000",
            "candidate_time_range": "00:02.000-00:04.000",
            "best_score": "0.82",
            "status": "completed",
        },
    ]

    segments = build_label_segments(rows, video_duration=4.0)

    assert segments == [
        {"start": 0.0, "end": 1.0, "label": "background"},
        {"start": 1.0, "end": 2.0, "label": "低分步骤"},
        {"start": 2.0, "end": 4.0, "label": "高分步骤"},
    ]


def test_write_ass_subtitles_writes_top_left_labels(tmp_path: Path):
    ass_path = tmp_path / "labels.ass"
    write_ass_subtitles(
        ass_path,
        [
            {"start": 0.0, "end": 1.25, "label": "background"},
            {"start": 1.25, "end": 2.0, "label": "接电连接"},
        ],
        font_name="Noto Sans CJK SC",
        font_size=30,
    )

    content = ass_path.read_text(encoding="utf-8")

    assert "Alignment,MarginL,MarginR,MarginV" in content
    assert "Style: Label,Noto Sans CJK SC,30" in content
    assert "Dialogue: 0,0:00:00.00,0:00:01.25,Label" in content
    assert "background" in content
    assert "接电连接" in content


def test_format_ass_time_uses_centiseconds():
    assert format_ass_time(3723.456) == "1:02:03.46"
