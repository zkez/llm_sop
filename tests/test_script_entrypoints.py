import subprocess
import sys
from pathlib import Path


def test_scripts_can_run_help_when_called_by_path():
    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        "scripts/split_video.py",
        "scripts/embed_steps.py",
        "scripts/embed_windows.py",
        "scripts/embed_windows_parallel.py",
        "scripts/render_labeled_video.py",
        "scripts/score_steps.py",
    ]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, "-B", script, "--help"],
            cwd=project_root,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, f"{script} failed: {result.stderr}"
