from pathlib import Path

import numpy as np

from scripts.embed_windows import (
    add_shard_suffix,
    build_profile_summary,
    encode_windows,
    filter_windows_for_shard,
    write_profile_outputs,
)


def test_filter_windows_for_shard_distributes_by_original_order():
    windows = [{"index": str(index)} for index in range(7)]

    shard_0 = filter_windows_for_shard(windows, shard_index=0, num_shards=3)
    shard_1 = filter_windows_for_shard(windows, shard_index=1, num_shards=3)
    shard_2 = filter_windows_for_shard(windows, shard_index=2, num_shards=3)

    assert [item["index"] for item in shard_0] == ["0", "3", "6"]
    assert [item["index"] for item in shard_1] == ["1", "4"]
    assert [item["index"] for item in shard_2] == ["2", "5"]


def test_add_shard_suffix_preserves_directory_and_extension():
    path = Path("outputs/window_embeddings.npy")

    assert add_shard_suffix(path, shard_index=2, num_shards=4) == Path(
        "outputs/window_embeddings.shard02-of-04.npy"
    )


def test_encode_windows_records_batch_profile():
    class FakeModel:
        def process(self, inputs):
            return np.ones((len(inputs), 3), dtype=np.float32)

    windows = [
        {
            "window_id": "window_000001",
            "index": "0",
            "start": "0.000",
            "end": "1.000",
            "duration": "1.000",
            "time_range": "00:00.000-00:01.000",
            "path": "a.mp4",
        },
        {
            "window_id": "window_000002",
            "index": "1",
            "start": "1.000",
            "end": "2.000",
            "duration": "1.000",
            "time_range": "00:01.000-00:02.000",
            "path": "b.mp4",
        },
        {
            "window_id": "window_000003",
            "index": "2",
            "start": "2.000",
            "end": "3.000",
            "duration": "1.000",
            "time_range": "00:02.000-00:03.000",
            "path": "c.mp4",
        },
    ]
    ticks = iter([10.0, 10.8, 11.0, 11.5])
    profile_records = []

    embeddings = encode_windows(
        model=FakeModel(),
        windows=windows,
        instruction="Represent video",
        fps=1.0,
        max_frames=2,
        batch_size=2,
        profile_records=profile_records,
        timer=lambda: next(ticks),
    )

    assert embeddings.shape == (3, 3)
    assert profile_records == [
        {
            "batch_index": 0,
            "batch_size": 2,
            "window_indices": "0,1",
            "window_ids": "window_000001,window_000002",
            "time_ranges": "00:00.000-00:01.000|00:01.000-00:02.000",
            "elapsed_seconds": 0.8,
            "seconds_per_window": 0.4,
        },
        {
            "batch_index": 1,
            "batch_size": 1,
            "window_indices": "2",
            "window_ids": "window_000003",
            "time_ranges": "00:02.000-00:03.000",
            "elapsed_seconds": 0.5,
            "seconds_per_window": 0.5,
        },
    ]


def test_write_profile_outputs_saves_csv_and_json(tmp_path):
    records = [
        {
            "batch_index": 0,
            "batch_size": 2,
            "window_indices": "0,1",
            "window_ids": "window_000001,window_000002",
            "time_ranges": "00:00.000-00:01.000|00:01.000-00:02.000",
            "elapsed_seconds": 0.8,
            "seconds_per_window": 0.4,
        },
        {
            "batch_index": 1,
            "batch_size": 1,
            "window_indices": "2",
            "window_ids": "window_000003",
            "time_ranges": "00:02.000-00:03.000",
            "elapsed_seconds": 0.5,
            "seconds_per_window": 0.5,
        },
    ]

    summary = build_profile_summary(
        records=records,
        model_load_seconds=2.0,
        encode_seconds=1.3,
        total_seconds=3.5,
        num_windows=3,
    )
    csv_path = tmp_path / "profile.csv"
    json_path = tmp_path / "profile.json"
    write_profile_outputs(csv_path, json_path, records, summary)

    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("batch_index,batch_size")
    assert '"num_windows": 3' in json_path.read_text(encoding="utf-8")
    assert summary["mean_seconds_per_window"] == 0.433333
    assert summary["p50_seconds_per_window"] == 0.4
