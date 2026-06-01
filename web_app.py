from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
import warnings
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

from scripts.common import ensure_dir, load_yaml, write_json

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import cgi


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "run.operation_case_1s.yaml"
DEFAULT_STEPS_CONFIG = PROJECT_ROOT / "configs" / "steps.operation_case.yaml"
DEFAULT_JOB_ROOT = PROJECT_ROOT / "outputs" / "web_jobs"
DEFAULT_UPLOAD_ROOT = PROJECT_ROOT / "data" / "web_uploads"
DEFAULT_WEB_INDEX = PROJECT_ROOT / "web" / "index.html"
DEFAULT_PYTHON = PROJECT_ROOT / "deps" / "Qwen3-VL-Embedding" / ".venv" / "bin" / "python"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_COMPLETION_THRESHOLD = 0.44
DEFAULT_MISSING_THRESHOLD = 0.40


def load_steps(path: Path = DEFAULT_STEPS_CONFIG) -> list[dict[str, Any]]:
    config = load_yaml(path)
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"步骤配置不能为空: {path}")
    return [
        {
            "id": str(step["id"]),
            "name": str(step["name"]),
            "descriptions": list(step.get("descriptions", [])),
        }
        for step in steps
    ]


def steps_match_metadata(steps: list[dict[str, Any]], metadata_path: Path) -> bool:
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    metadata_steps = metadata.get("steps")
    if not isinstance(metadata_steps, list):
        return False
    normalized = [
        {
            "id": str(step.get("id", "")),
            "name": str(step.get("name", "")),
            "descriptions": list(step.get("descriptions", [])),
        }
        for step in metadata_steps
    ]
    return normalized == steps


def build_job_config(
    base_config: dict[str, Any],
    upload_path: Path,
    windows_dir: Path,
    output_dir: Path,
    completion_threshold: float = DEFAULT_COMPLETION_THRESHOLD,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    video_config = dict(config.get("video", {}))
    video_config["path"] = str(upload_path)
    video_config["windows_dir"] = str(windows_dir)
    video_config["skip_existing"] = False
    config["video"] = video_config
    scoring_config = dict(config.get("scoring", {}))
    scoring_config["completion_threshold"] = completion_threshold
    scoring_config["missing_threshold"] = missing_threshold
    scoring_config["ignore_margin_for_completion"] = True
    scoring_config["score_median_context_seconds"] = 1.0
    config["scoring"] = scoring_config
    config["outputs"] = {
        "windows_csv": str(output_dir / "windows.csv"),
        "step_embeddings": str(output_dir / "step_embeddings.npy"),
        "step_metadata": str(output_dir / "step_metadata.json"),
        "window_embeddings": str(output_dir / "window_embeddings.npy"),
        "window_metadata": str(output_dir / "window_metadata.json"),
        "similarity_matrix_csv": str(output_dir / "similarity_matrix.csv"),
        "scores_csv": str(output_dir / "scores.csv"),
        "heatmap_png": str(output_dir / "heatmap.png"),
        "report_md": str(output_dir / "report.md"),
    }
    return config


def normalize_scoring_thresholds(completion_value: Any = None, missing_value: Any = None) -> tuple[float, float]:
    completion_threshold = _parse_threshold(
        completion_value,
        default=DEFAULT_COMPLETION_THRESHOLD,
        label="亮灯阈值",
    )
    missing_threshold = _parse_threshold(
        missing_value,
        default=DEFAULT_MISSING_THRESHOLD,
        label="复核下限",
    )
    if missing_threshold >= completion_threshold:
        raise ValueError("复核下限必须小于亮灯阈值")
    return completion_threshold, missing_threshold


def _parse_threshold(value: Any, default: float, label: str) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}必须是数字") from error
    if parsed < 0 or parsed > 1:
        raise ValueError(f"{label}必须在 0 到 1 之间")
    return round(parsed, 6)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def read_scores(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    results: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status", "missing"))
        results.append(
            {
                "step_id": str(row.get("step_id", "")),
                "step_name": str(row.get("step_name", "")),
                "status": status,
                "is_active": status == "completed",
                "best_time_range": str(row.get("best_time_range", "--")),
                "best_score": _parse_float(row.get("best_score")),
                "margin": _parse_float(row.get("margin")),
                "reason": str(row.get("reason", "")),
            }
        )
    return results


def select_decode_strategy(config: dict[str, Any], window_count: int, step_count: int) -> dict[str, Any]:
    selected = copy.deepcopy(config)
    scoring = dict(selected.get("scoring", {}))
    decode_strategy = str(scoring.get("decode_strategy", "sequence")).lower()
    min_gap_windows = int(scoring.get("sequence_min_gap_windows", 1))
    required_windows = 1 + (step_count - 1) * max(1, min_gap_windows)
    if decode_strategy == "sequence" and window_count < required_windows:
        scoring["decode_strategy"] = "independent"
        scoring["enforce_order"] = True
    selected["scoring"] = scoring
    return selected


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError("仅支持 mp4、mov、avi、mkv、webm、m4v 视频文件")
    return suffix


class JobStore:
    def __init__(self, job_root: Path = DEFAULT_JOB_ROOT) -> None:
        self.job_root = job_root
        self._lock = threading.Lock()

    def job_dir(self, job_id: str) -> Path:
        return self.job_root / job_id

    def status_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "status.json"

    def write_status(self, job_id: str, data: dict[str, Any]) -> None:
        ensure_dir(self.job_dir(job_id))
        with self._lock:
            write_json(self.status_path(job_id), data)

    def update_status(self, job_id: str, **updates: Any) -> dict[str, Any]:
        data = self.read_status(job_id)
        data.update(updates)
        data["updated_at"] = _utc_now()
        self.write_status(job_id, data)
        return data

    def read_status(self, job_id: str) -> dict[str, Any]:
        path = self.status_path(job_id)
        if not path.exists():
            raise FileNotFoundError(job_id)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


class Analyzer:
    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        base_config_path: Path = DEFAULT_BASE_CONFIG,
        steps_config_path: Path = DEFAULT_STEPS_CONFIG,
        job_root: Path = DEFAULT_JOB_ROOT,
        upload_root: Path = DEFAULT_UPLOAD_ROOT,
        python_executable: Path | str = DEFAULT_PYTHON,
    ) -> None:
        self.project_root = project_root
        self.base_config_path = base_config_path
        self.steps_config_path = steps_config_path
        self.job_root = job_root
        self.upload_root = upload_root
        self.python_executable = str(python_executable if Path(python_executable).exists() else sys.executable)
        self.store = JobStore(job_root)
        self.steps = load_steps(steps_config_path)

    def current_steps(self) -> list[dict[str, Any]]:
        self.steps = load_steps(self.steps_config_path)
        return self.steps

    def create_job(
        self,
        filename: str,
        source_file: Any,
        completion_threshold: float = DEFAULT_COMPLETION_THRESHOLD,
        missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
    ) -> dict[str, Any]:
        suffix = _safe_extension(filename)
        job_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = ensure_dir(self.job_root / job_id)
        upload_dir = ensure_dir(self.upload_root / job_id)
        upload_path = upload_dir / f"input{suffix}"
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(source_file, handle)

        steps = self.current_steps()
        base_config = load_yaml(self.base_config_path)
        config = build_job_config(
            base_config=base_config,
            upload_path=upload_path,
            windows_dir=upload_dir / "windows",
            output_dir=job_dir,
            completion_threshold=completion_threshold,
            missing_threshold=missing_threshold,
        )
        config_path = job_dir / "run.yaml"
        write_yaml(config_path, config)

        status = {
            "job_id": job_id,
            "state": "queued",
            "stage": "等待开始",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "filename": filename,
            "thresholds": {
                "completion_threshold": completion_threshold,
                "missing_threshold": missing_threshold,
            },
            "steps": self._initial_step_states(steps),
            "scores": [],
            "logs": [],
            "artifacts": {},
        }
        self.store.write_status(job_id, status)
        thread = threading.Thread(target=self.run_job, args=(job_id, config_path), daemon=True)
        thread.start()
        return status

    def run_job(self, job_id: str, config_path: Path) -> None:
        try:
            self._set_running(job_id, "切分视频窗口")
            self._run_command(job_id, [self.python_executable, "scripts/split_video.py", "--config", str(config_path)])

            config = load_yaml(config_path)
            self._prepare_step_embeddings(job_id, config, config_path)

            self._set_running(job_id, "生成视频窗口 embedding")
            self._run_command(job_id, [self.python_executable, "scripts/embed_windows.py", "--config", str(config_path)])

            config = self._adjust_decode_strategy(job_id, config, config_path)
            self._set_running(job_id, "计算步骤分数")
            self._run_command(job_id, [self.python_executable, "scripts/score_steps.py", "--config", str(config_path)])

            scores_path = Path(config["outputs"]["scores_csv"])
            scores = read_scores(scores_path)
            self.store.update_status(
                job_id,
                state="done",
                stage="识别完成",
                scores=scores,
                steps=self._merge_step_states(scores),
                artifacts={
                    "scores_csv": str(scores_path),
                    "heatmap_png": str(config["outputs"]["heatmap_png"]),
                    "report_md": str(config["outputs"]["report_md"]),
                },
            )
        except Exception as error:  # pragma: no cover - depends on model runtime
            logs = self.store.read_status(job_id).get("logs", [])
            logs.append(traceback.format_exc())
            self.store.update_status(
                job_id,
                state="error",
                stage="识别失败",
                error=str(error),
                logs=logs[-20:],
            )

    def _set_running(self, job_id: str, stage: str) -> None:
        self.store.update_status(job_id, state="running", stage=stage)

    def _run_command(self, job_id: str, command: list[str]) -> None:
        result = subprocess.run(command, cwd=self.project_root, check=False, capture_output=True, text=True)
        logs = self.store.read_status(job_id).get("logs", [])
        logs.append("$ " + " ".join(command))
        if result.stdout.strip():
            logs.append(result.stdout.strip())
        if result.stderr.strip():
            logs.append(result.stderr.strip())
        self.store.update_status(job_id, logs=logs[-20:])
        if result.returncode != 0:
            raise RuntimeError(f"命令执行失败: {' '.join(command)}")

    def _adjust_decode_strategy(self, job_id: str, config: dict[str, Any], config_path: Path) -> dict[str, Any]:
        metadata_path = Path(config["outputs"]["window_metadata"])
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        window_count = len(metadata.get("windows", []))
        selected = select_decode_strategy(config, window_count=window_count, step_count=len(self.current_steps()))
        if selected.get("scoring", {}).get("decode_strategy") != config.get("scoring", {}).get("decode_strategy"):
            self._set_running(job_id, "窗口较少，使用独立打分")
            write_yaml(config_path, selected)
        return selected

    def _prepare_step_embeddings(self, job_id: str, config: dict[str, Any], config_path: Path) -> None:
        output_step_embeddings = Path(config["outputs"]["step_embeddings"])
        output_step_metadata = Path(config["outputs"]["step_metadata"])
        source_outputs = load_yaml(self.base_config_path).get("outputs", {})
        source_step_embeddings = self.project_root / str(source_outputs.get("step_embeddings", ""))
        source_step_metadata = self.project_root / str(source_outputs.get("step_metadata", ""))
        if (
            source_step_embeddings.exists()
            and source_step_metadata.exists()
            and steps_match_metadata(self.current_steps(), source_step_metadata)
        ):
            self._set_running(job_id, "复用步骤 embedding")
            ensure_dir(output_step_embeddings.parent)
            shutil.copy2(source_step_embeddings, output_step_embeddings)
            shutil.copy2(source_step_metadata, output_step_metadata)
            return

        self._set_running(job_id, "生成步骤 embedding")
        self._run_command(
            job_id,
            [
                self.python_executable,
                "scripts/embed_steps.py",
                "--config",
                str(config_path),
                "--steps",
                str(self.steps_config_path),
            ],
        )

    def _initial_step_states(self, steps: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        if steps is None:
            steps = self.current_steps()
        return [
            {
                "step_id": step["id"],
                "step_name": step["name"],
                "status": "pending",
                "is_active": False,
                "best_time_range": "--",
                "best_score": None,
                "margin": None,
                "reason": "",
            }
            for step in steps
        ]

    def _merge_step_states(self, scores: list[dict[str, Any]], steps: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        if steps is None:
            steps = self.current_steps()
        by_id = {row["step_id"]: row for row in scores}
        return [
            by_id.get(
                step["id"],
                {
                    "step_id": step["id"],
                    "step_name": step["name"],
                    "status": "missing",
                    "is_active": False,
                    "best_time_range": "--",
                    "best_score": None,
                    "margin": None,
                    "reason": "未生成该步骤结果",
                },
            )
            for step in steps
        ]


class AppHandler(BaseHTTPRequestHandler):
    analyzer: Analyzer

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_file(DEFAULT_WEB_INDEX, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/steps":
                self._send_json({"steps": self.analyzer.current_steps()})
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = unquote(parsed.path.rsplit("/", 1)[-1])
                self._send_json(self.analyzer.store.read_status(job_id))
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except FileNotFoundError:
            self._send_json({"error": "任务不存在"}, status=HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path != "/api/jobs":
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
                self._send_json({"error": "视频大小不合法"}, status=HTTPStatus.BAD_REQUEST)
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": str(content_length),
                },
            )
            upload = form["video"] if "video" in form else None
            if upload is None or not getattr(upload, "filename", ""):
                self._send_json({"error": "缺少视频文件"}, status=HTTPStatus.BAD_REQUEST)
                return
            completion_threshold, missing_threshold = normalize_scoring_thresholds(
                form.getfirst("completion_threshold"),
                form.getfirst("missing_threshold"),
            )
            status = self.analyzer.create_job(
                upload.filename,
                upload.file,
                completion_threshold=completion_threshold,
                missing_threshold=missing_threshold,
            )
            self._send_json(status, status=HTTPStatus.ACCEPTED)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def create_server(host: str, port: int, analyzer: Analyzer) -> ThreadingHTTPServer:
    handler = type("ConfiguredAppHandler", (AppHandler,), {"analyzer": analyzer})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 SOP 视频行为识别网页。")
    parser.add_argument("--host", default=os.environ.get("LLM_SOP_WEB_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LLM_SOP_WEB_PORT", "7860")))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON), help="运行模型流水线的 Python")
    args = parser.parse_args()

    analyzer = Analyzer(python_executable=args.python)
    server = create_server(args.host, args.port, analyzer)
    print(f"Serving SOP web app on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
