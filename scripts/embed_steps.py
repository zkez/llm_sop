from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import (
    chunked,
    ensure_dir,
    load_yaml,
    mean_normalized,
    resolve_model_name_or_path,
    resolve_project_path,
    write_json,
)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def load_embedder(model_config: dict[str, Any], project_root: Path) -> Any:
    repo_path = resolve_project_path(model_config["qwen_repo_path"], project_root)
    if not repo_path.exists():
        raise FileNotFoundError(f"找不到 Qwen3-VL-Embedding 仓库目录: {repo_path}")
    sys.path.insert(0, str(repo_path))

    import torch
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    dtype_name = str(model_config.get("torch_dtype", "bfloat16"))
    torch_dtype = getattr(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "model_name_or_path": resolve_model_name_or_path(str(model_config["path"]), project_root),
        "torch_dtype": torch_dtype,
    }
    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return Qwen3VLEmbedder(**kwargs)


def encode_texts(model: Any, texts: list[str], instruction: str, batch_size: int) -> np.ndarray:
    all_embeddings: list[np.ndarray] = []
    for batch in tqdm(chunked(texts, batch_size), desc="编码步骤描述"):
        inputs = [{"text": text, "instruction": instruction} for text in batch]
        embeddings = tensor_to_numpy(model.process(inputs))
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings).astype(np.float32)


def build_step_prototypes(model: Any, steps: list[dict[str, Any]], instruction: str, batch_size: int) -> np.ndarray:
    prototypes: list[np.ndarray] = []
    for step in steps:
        descriptions = step.get("descriptions") or []
        if not descriptions:
            raise ValueError(f"步骤缺少 descriptions: {step.get('id')}")
        embeddings = encode_texts(model, descriptions, instruction, batch_size)
        prototypes.append(mean_normalized(embeddings))
    return np.vstack(prototypes).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成步骤文本原型 embedding。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    parser.add_argument("--steps", default="configs/steps.yaml", help="步骤配置 YAML 路径")
    args = parser.parse_args()

    project_root = Path.cwd()
    config = load_yaml(resolve_project_path(args.config, project_root))
    steps_config = load_yaml(resolve_project_path(args.steps, project_root))
    steps = steps_config["steps"]

    model = load_embedder(config["model"], project_root)
    prototypes = build_step_prototypes(
        model=model,
        steps=steps,
        instruction=str(config["model"].get("instruction", "Represent the user's input")),
        batch_size=int(config["embedding"].get("batch_size", 1)),
    )

    embedding_path = resolve_project_path(config["outputs"]["step_embeddings"], project_root)
    metadata_path = resolve_project_path(config["outputs"]["step_metadata"], project_root)
    ensure_dir(embedding_path.parent)
    np.save(embedding_path, prototypes)
    write_json(
        metadata_path,
        {
            "steps": [
                {
                    "id": step["id"],
                    "name": step["name"],
                    "descriptions": step.get("descriptions", []),
                    "negative_notes": step.get("negative_notes", []),
                }
                for step in steps
            ]
        },
    )
    print(f"已保存步骤 embedding: {embedding_path}")
    print(f"已保存步骤元数据: {metadata_path}")


if __name__ == "__main__":
    main()
