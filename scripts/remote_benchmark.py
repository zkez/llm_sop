from __future__ import annotations

import csv
import os
import shlex
import subprocess
import time
from pathlib import Path


PROJECT_DIR = Path(os.environ.get("BENCH_PROJECT_DIR", "/mnt/project/zy/step-video-embedding"))
CONFIG_PATH = os.environ.get("BENCH_CONFIG", "configs/run.operation_case_1s.yaml")
STEPS_PATH = os.environ.get("BENCH_STEPS", "configs/steps.operation_case.yaml")
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
SAMPLE_INTERVAL = float(os.environ.get("BENCH_SAMPLE_INTERVAL", "0.2"))
COMMAND_MODE = os.environ.get("BENCH_COMMANDS", "all")


def query_gpu_memory() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        return 0
    gpu_index = int(CUDA_VISIBLE_DEVICES.split(",")[0])
    return values[gpu_index]


def run_with_sampling(name: str, command: list[str], log_dir: Path) -> dict[str, float | int | str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    samples_path = log_dir / f"{name}_gpu_samples.csv"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    baseline = query_gpu_memory()
    peak = baseline
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        with samples_path.open("w", encoding="utf-8", newline="") as sample_handle:
            writer = csv.writer(sample_handle)
            writer.writerow(["elapsed_seconds", "memory_used_mib"])
            while process.poll() is None:
                elapsed = time.perf_counter() - started
                try:
                    current = query_gpu_memory()
                    peak = max(peak, current)
                    writer.writerow([f"{elapsed:.3f}", current])
                except Exception as error:
                    writer.writerow([f"{elapsed:.3f}", f"ERR:{error}"])
                time.sleep(SAMPLE_INTERVAL)
        exit_code = process.wait()
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "command": " ".join(shlex.quote(item) for item in command),
        "exit_code": exit_code,
        "elapsed_seconds": round(elapsed, 3),
        "baseline_mib": baseline,
        "peak_mib": peak,
        "delta_mib": max(0, peak - baseline),
        "log_path": str(log_path),
        "samples_path": str(samples_path),
    }


def main() -> int:
    log_dir = PROJECT_DIR / "outputs" / "benchmarks" / time.strftime("%Y%m%d-%H%M%S")
    command_map = {
        "embed_steps": (
            "embed_steps",
            [
                "python",
                "scripts/embed_steps.py",
                "--config",
                CONFIG_PATH,
                "--steps",
                STEPS_PATH,
            ],
        ),
        "embed_windows": ("embed_windows", ["python", "scripts/embed_windows.py", "--config", CONFIG_PATH]),
        "score_steps": ("score_steps", ["python", "scripts/score_steps.py", "--config", CONFIG_PATH]),
    }
    if COMMAND_MODE == "all":
        command_names = ["embed_steps", "embed_windows", "score_steps"]
    else:
        command_names = [name.strip() for name in COMMAND_MODE.split(",") if name.strip()]
    commands = [command_map[name] for name in command_names]
    rows = []
    for name, command in commands:
        print(f"running {name} ...")
        row = run_with_sampling(name, command, log_dir)
        rows.append(row)
        print(row)
        if row["exit_code"] != 0:
            break

    summary_path = log_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary: {summary_path}")
    return 0 if all(row["exit_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
