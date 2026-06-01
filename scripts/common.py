from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path, project_root: str | Path = PROJECT_ROOT) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return (Path(project_root) / resolved).resolve()


def resolve_model_name_or_path(value: str, project_root: str | Path = PROJECT_ROOT) -> str:
    if value.startswith((".", "~")):
        return str(resolve_project_path(value, project_root))
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)
    local_candidate = Path(project_root) / raw_path
    if local_candidate.exists():
        return str(local_candidate.resolve())
    return value


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是对象: {path}")
    return data


def get_nested(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = data
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def l2_normalize(vectors: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    safe_norms = np.where(norms > eps, norms, 1.0)
    return array / safe_norms


def mean_normalized(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("mean_normalized 需要形状为 [n, dim] 的非空数组")
    return l2_normalize(l2_normalize(array).mean(axis=0))


def format_time(seconds: float) -> str:
    milliseconds = int(round(float(seconds) * 1000))
    total_seconds, millis = divmod(milliseconds, 1000)
    minutes_total, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes_total, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def format_time_range(start: float, end: float) -> str:
    return f"{format_time(start)}-{format_time(end)}"


def chunked(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]
