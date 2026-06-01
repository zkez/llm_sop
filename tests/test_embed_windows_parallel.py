from pathlib import Path

from scripts.embed_windows_parallel import build_shard_command


def test_build_shard_command_adds_profile_outputs():
    command = build_shard_command(
        python_executable="python",
        config_path="configs/run.yaml",
        shard_index=2,
        num_shards=4,
        profile_dir=Path("outputs/profile"),
    )

    assert command == [
        "python",
        "scripts/embed_windows.py",
        "--config",
        "configs/run.yaml",
        "--shard-index",
        "2",
        "--num-shards",
        "4",
        "--profile-csv",
        "outputs/profile/shard_02.csv",
        "--profile-json",
        "outputs/profile/shard_02.json",
    ]
