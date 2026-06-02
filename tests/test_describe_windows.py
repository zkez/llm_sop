import csv
import json

from scripts.describe_windows import (
    build_description_record,
    load_generation_model,
    parse_json_response,
    resolve_description_outputs,
    write_description_outputs,
)


def test_parse_json_response_reads_fenced_json():
    response = """```json
{"description": "工人正在扫码", "visible_objects": ["扫码枪", "标签"], "uncertainty": "low"}
```"""

    parsed = parse_json_response(response)

    assert parsed["description"] == "工人正在扫码"
    assert parsed["visible_objects"] == ["扫码枪", "标签"]
    assert parsed["uncertainty"] == "low"


def test_parse_json_response_falls_back_to_raw_description():
    parsed = parse_json_response("画面中工人拿起工具。")

    assert parsed["description"] == "画面中工人拿起工具。"


def test_build_description_record_flattens_structured_fields():
    window = {
        "window_id": "window_000001",
        "index": "0",
        "start": "0.000",
        "end": "2.000",
        "time_range": "00:00.000-00:02.000",
        "path": "/tmp/window_000001.mp4",
    }
    response = json.dumps(
        {
            "description": "工人手持扫码枪扫描标签。",
            "visible_objects": ["扫码枪", "标签"],
            "human_action": "扫码",
            "tool_action": "扫码枪靠近标签",
            "completion_evidence": "扫码枪已对准标签",
            "uncertainty": "low",
        },
        ensure_ascii=False,
    )

    record = build_description_record(window, response)

    assert record["window_id"] == "window_000001"
    assert record["description"] == "工人手持扫码枪扫描标签。"
    assert record["visible_objects"] == "扫码枪; 标签"
    assert record["human_action"] == "扫码"
    assert record["raw_response"] == response


def test_resolve_description_outputs_uses_config_or_defaults(tmp_path):
    config = {
        "outputs": {
            "windows_csv": str(tmp_path / "outputs" / "windows.csv"),
            "window_descriptions_jsonl": str(tmp_path / "custom" / "descriptions.jsonl"),
        }
    }

    jsonl_path, csv_path = resolve_description_outputs(config, tmp_path, None, None)

    assert jsonl_path == tmp_path / "custom" / "descriptions.jsonl"
    assert csv_path == tmp_path / "outputs" / "window_descriptions.csv"


def test_write_description_outputs_writes_jsonl_and_csv(tmp_path):
    records = [
        {
            "window_id": "window_000001",
            "index": "0",
            "time_range": "00:00.000-00:02.000",
            "path": "/tmp/window_000001.mp4",
            "description": "工人正在扫码。",
            "visible_objects": "扫码枪; 标签",
            "human_action": "扫码",
            "tool_action": "扫码枪对准标签",
            "completion_evidence": "扫码动作可见",
            "uncertainty": "low",
            "raw_response": "{\"description\":\"工人正在扫码。\"}",
        }
    ]
    jsonl_path = tmp_path / "window_descriptions.jsonl"
    csv_path = tmp_path / "window_descriptions.csv"

    write_description_outputs(jsonl_path, csv_path, records)

    assert json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])["description"] == "工人正在扫码。"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["window_id"] == "window_000001"
    assert rows[0]["visible_objects"] == "扫码枪; 标签"


def test_load_generation_model_reports_missing_local_model_path(tmp_path):
    missing_model = tmp_path / "models" / "Qwen2.5-VL-7B-Instruct"

    try:
        load_generation_model({"path": str(missing_model)}, tmp_path)
    except FileNotFoundError as error:
        assert "模型目录不存在" in str(error)
        assert str(missing_model) in str(error)
    else:
        raise AssertionError("expected missing local model path to fail before loading transformers")
