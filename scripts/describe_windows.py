from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_dir, load_yaml, resolve_model_name_or_path, resolve_project_path
from scripts.embed_windows import filter_windows_for_shard, read_windows_manifest


DEFAULT_DESCRIPTION_PROMPT = """请只根据视频画面描述可见事实，不要推测不可见内容，不要判断 SOP 是否完成。
重点描述：
1. 人的手部动作
2. 使用的工具
3. 接触的工件或部件
4. 动作是否已经完成
5. 画面中可读文字或标签

请输出 JSON，不要输出 Markdown：
{
  "description": "一句话描述该 2 秒窗口中最重要的可见动作",
  "visible_objects": ["画面中可见的关键物体"],
  "human_action": "人的主要动作",
  "tool_action": "工具或部件的动作，没有则写空字符串",
  "completion_evidence": "能证明动作进展或完成的可见证据，没有则写空字符串",
  "uncertainty": "low|medium|high"
}
"""

DESCRIPTION_FIELDNAMES = [
    "window_id",
    "index",
    "start",
    "end",
    "time_range",
    "path",
    "description",
    "visible_objects",
    "human_action",
    "tool_action",
    "completion_evidence",
    "uncertainty",
    "raw_response",
]


def parse_json_response(response: str) -> dict[str, Any]:
    text = response.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            text = object_match.group(0).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"description": response.strip()}
    if not isinstance(parsed, dict):
        return {"description": response.strip()}
    return parsed


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def build_description_record(window: dict[str, Any], response: str) -> dict[str, Any]:
    parsed = parse_json_response(response)
    return {
        "window_id": str(window.get("window_id", "")),
        "index": str(window.get("index", "")),
        "start": str(window.get("start", "")),
        "end": str(window.get("end", "")),
        "time_range": str(window.get("time_range", "")),
        "path": str(window.get("path", "")),
        "description": _stringify(parsed.get("description")),
        "visible_objects": _stringify(parsed.get("visible_objects")),
        "human_action": _stringify(parsed.get("human_action")),
        "tool_action": _stringify(parsed.get("tool_action")),
        "completion_evidence": _stringify(parsed.get("completion_evidence")),
        "uncertainty": _stringify(parsed.get("uncertainty")),
        "raw_response": response,
    }


def resolve_description_outputs(
    config: dict[str, Any],
    project_root: Path,
    output_jsonl: str | None,
    output_csv: str | None,
) -> tuple[Path, Path]:
    outputs = config["outputs"]
    windows_csv = resolve_project_path(outputs["windows_csv"], project_root)
    output_dir = windows_csv.parent
    jsonl_value = output_jsonl or outputs.get("window_descriptions_jsonl") or output_dir / "window_descriptions.jsonl"
    csv_value = output_csv or outputs.get("window_descriptions_csv") or output_dir / "window_descriptions.csv"
    return resolve_project_path(jsonl_value, project_root), resolve_project_path(csv_value, project_root)


def write_description_outputs(jsonl_path: Path, csv_path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(jsonl_path.parent)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    ensure_dir(csv_path.parent)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DESCRIPTION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def _model_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    configured = dict(config.get("description_model") or config.get("vlm") or {})
    if args.model:
        configured["path"] = args.model
    if "path" not in configured:
        raise ValueError("请通过 --model 或配置 description_model.path 指定生成式 Qwen-VL 模型路径")
    configured.setdefault("torch_dtype", args.torch_dtype)
    configured.setdefault("device_map", args.device_map)
    if args.attn_implementation:
        configured["attn_implementation"] = args.attn_implementation
    return configured


def load_generation_model(model_config: dict[str, Any], project_root: Path) -> tuple[Any, Any]:
    raw_model_path = str(model_config["path"])
    if _looks_like_local_path(raw_model_path):
        local_model_path = resolve_project_path(raw_model_path, project_root)
        if not local_model_path.exists():
            raise FileNotFoundError(
                f"生成式 Qwen-VL 模型目录不存在: {local_model_path}\n"
                "请先下载生成式模型，或把 --model 改成可访问的 Hugging Face/ModelScope 模型 ID。"
            )

    import torch
    from transformers import AutoProcessor

    model_path = resolve_model_name_or_path(raw_model_path, project_root)
    dtype_name = str(model_config.get("torch_dtype", "bfloat16"))
    torch_dtype = getattr(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": model_config.get("device_map", "auto"),
    }
    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model_class_names = [
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]
    import transformers

    last_error: Exception | None = None
    for class_name in model_class_names:
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            model = model_class.from_pretrained(model_path, **kwargs)
            processor = AutoProcessor.from_pretrained(model_path)
            return model, processor
        except Exception as error:
            last_error = error
    raise RuntimeError(f"无法加载生成式 Qwen-VL 模型: {model_path}；底层错误: {last_error}") from last_error


def _looks_like_local_path(value: str) -> bool:
    expanded = Path(value).expanduser()
    return (
        value.startswith((".", "~"))
        or expanded.is_absolute()
        or value.startswith(("models/", "deps/", "data/", "outputs/", "configs/"))
    )


def describe_window(
    model: Any,
    processor: Any,
    window: dict[str, Any],
    prompt: str,
    fps: float,
    max_new_tokens: int,
) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(window["path"]), "fps": fps},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def describe_windows(
    model: Any,
    processor: Any,
    windows: list[dict[str, Any]],
    prompt: str,
    fps: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for window in tqdm(windows, desc="描述视频窗口"):
        response = describe_window(
            model=model,
            processor=processor,
            window=window,
            prompt=prompt,
            fps=fps,
            max_new_tokens=max_new_tokens,
        )
        records.append(build_description_record(window, response))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="使用生成式 Qwen-VL 描述每个视频窗口。")
    parser.add_argument("--config", default="configs/run.yaml", help="运行配置 YAML 路径")
    parser.add_argument("--model", default=None, help="生成式 Qwen-VL 模型路径或 Hugging Face 名称")
    parser.add_argument("--output-jsonl", default=None, help="可选，输出 JSONL 路径")
    parser.add_argument("--output-csv", default=None, help="可选，输出 CSV 路径")
    parser.add_argument("--limit", type=int, default=None, help="可选，只处理前 N 个窗口，便于快速测试")
    parser.add_argument("--shard-index", type=int, default=0, help="当前分片编号，从 0 开始")
    parser.add_argument("--num-shards", type=int, default=1, help="总分片数")
    parser.add_argument("--fps", type=float, default=1.0, help="传给 Qwen-VL 的视频采样 FPS")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="每个窗口描述最多生成 token 数")
    parser.add_argument("--torch-dtype", default="bfloat16", help="模型加载 dtype")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--attn-implementation", default=None, help="可选，注意力实现")
    args = parser.parse_args()

    project_root = Path.cwd()
    config = load_yaml(resolve_project_path(args.config, project_root))
    windows_path = resolve_project_path(config["outputs"]["windows_csv"], project_root)
    windows = read_windows_manifest(windows_path)
    if not windows:
        raise ValueError(f"窗口清单为空: {windows_path}")
    windows = filter_windows_for_shard(windows, args.shard_index, args.num_shards)
    if args.limit is not None:
        windows = windows[: args.limit]
    if not windows:
        raise ValueError("没有可描述的视频窗口")

    description_config = config.get("description", {})
    prompt = str(description_config.get("prompt") or DEFAULT_DESCRIPTION_PROMPT)
    model, processor = load_generation_model(_model_config(config, args), project_root)
    records = describe_windows(
        model=model,
        processor=processor,
        windows=windows,
        prompt=prompt,
        fps=float(description_config.get("fps", args.fps)),
        max_new_tokens=int(description_config.get("max_new_tokens", args.max_new_tokens)),
    )
    jsonl_path, csv_path = resolve_description_outputs(config, project_root, args.output_jsonl, args.output_csv)
    if args.num_shards > 1:
        jsonl_path = jsonl_path.with_name(f"{jsonl_path.stem}.shard{args.shard_index:02d}-of-{args.num_shards:02d}{jsonl_path.suffix}")
        csv_path = csv_path.with_name(f"{csv_path.stem}.shard{args.shard_index:02d}-of-{args.num_shards:02d}{csv_path.suffix}")
    write_description_outputs(jsonl_path, csv_path, records)
    print(f"已保存窗口描述 JSONL: {jsonl_path}")
    print(f"已保存窗口描述 CSV: {csv_path}")


if __name__ == "__main__":
    main()
