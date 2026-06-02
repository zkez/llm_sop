# LLM SOP 视频工步识别

LLM SOP 是一个基于多模态 embedding 的 SOP 视频工步识别项目。它将标准操作步骤描述和视频时间窗口映射到同一向量空间，通过相似度、顺序约束和阈值判断每个步骤是否完成。

当前实现使用 `Qwen3-VL-Embedding` 作为文本和视频 embedding 模型，适合做轻量零样本验证、工步漏检分析和视频片段定位。

## 工作流程

```text
步骤描述 -> 文本 embedding -> 步骤向量
视频 -> 重叠时间窗口 -> 视频窗口向量
步骤向量 x 视频窗口向量 -> 相似度矩阵
相似度矩阵 -> 工步状态、热力图、报告
```

项目不会把整段长视频直接压成一个向量，而是先切成短窗口，再在时间轴上查找每个 SOP 步骤的相似度峰值。

## 功能

- 按固定窗口和步长切分视频
- 为 SOP 步骤描述生成文本 embedding
- 为视频窗口生成多模态 embedding
- 支持单 GPU 和多 GPU 视频窗口编码
- 输出步骤状态、相似度矩阵、热力图和 Markdown 报告
- 可根据识别结果生成带工步标签的视频
- 提供简单 Web 页面用于上传视频并查看识别结果
- 提供不加载大模型的单元测试

## 目录结构

```text
configs/
  run.example.yaml              # 通用运行配置模板
  run.operation_case_1s.yaml    # 示例工步案例配置
  steps.example.yaml            # 通用步骤模板
  steps.operation_case.yaml     # 示例工步案例步骤
scripts/
  split_video.py                # 切分视频窗口
  describe_windows.py           # 可选，用生成式 Qwen-VL 描述视频窗口
  embed_steps.py                # 生成步骤文本 embedding
  embed_windows.py              # 生成视频窗口 embedding
  embed_windows_parallel.py     # 多 GPU 并行生成视频窗口 embedding
  score_steps.py                # 计算相似度、分数和报告
  render_hand_skeleton_windows.py # 可选，叠加 MediaPipe 手部骨架窗口
  render_labeled_video.py       # 生成带工步标签的视频
web/
  index.html                    # Web 页面
web_app.py                      # 内置 HTTP 服务
tests/                          # 单元测试
docs/                           # 设计文档
```

运行时会生成或使用以下目录，默认不进入 Git：

```text
deps/       # 外部依赖仓库和虚拟环境
models/     # 模型权重
data/       # 输入视频、上传视频和切片窗口
outputs/    # embedding、分数、热力图、报告和 Web 任务结果
```

## 环境准备

### 1. 克隆项目

```bash
git clone git@github.com:zkez/llm_sop.git
cd llm_sop
```

### 2. 安装 Qwen3-VL-Embedding

```bash
mkdir -p deps
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git deps/Qwen3-VL-Embedding

cd deps/Qwen3-VL-Embedding
bash scripts/setup_environment.sh
cd ../..
```

激活官方环境后安装本项目轻量依赖：

```bash
source deps/Qwen3-VL-Embedding/.venv/bin/activate
pip install -r requirements.txt
```

默认配置使用 `attn_implementation: sdpa`。如果环境已正确安装 `flash-attn`，可以在配置中改为 `flash_attention_2`。

### 3. 下载模型

Hugging Face：

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir ./models/Qwen3-VL-Embedding-2B
```

ModelScope：

```bash
mkdir -p models
modelscope download --model qwen/Qwen3-VL-Embedding-2B --local_dir ./models/Qwen3-VL-Embedding-2B
```

建议先用 2B 模型跑通流程，再根据显存和精度需求切换更大的模型。

## 配置

复制配置模板：

```bash
cp configs/run.example.yaml configs/run.yaml
cp configs/steps.example.yaml configs/steps.yaml
```

修改 `configs/run.yaml` 中的模型、视频和输出路径：

```yaml
model:
  path: ./models/Qwen3-VL-Embedding-2B
  qwen_repo_path: ./deps/Qwen3-VL-Embedding

video:
  path: ./data/input.mp4
  windows_dir: ./data/windows
```

修改 `configs/steps.yaml`，将示例步骤替换为真实 SOP 步骤。每个步骤建议写 1-5 条描述，重点描述“该步骤完成时画面中应该出现什么”。

## 命令行运行

### 1. 切分视频

```bash
python scripts/split_video.py --config configs/run.yaml
```

### 2. 生成步骤文本 embedding

```bash
python scripts/embed_steps.py --config configs/run.yaml --steps configs/steps.yaml
```

### 3. 生成视频窗口 embedding

```bash
python scripts/embed_windows.py --config configs/run.yaml
```

多 GPU 并行：

```bash
python scripts/embed_windows_parallel.py --config configs/run.yaml --gpus 0,1,2,3
```

### 4. 打分并生成报告

```bash
python scripts/score_steps.py --config configs/run.yaml
```

### 5. 生成标注视频

```bash
python scripts/render_labeled_video.py --config configs/run.yaml
```

### 可选：手部骨架叠加窗口对比

如果希望测试“手部关键点骨架可视化后再做 embedding”的效果，可以先安装 MediaPipe：

```bash
uv pip install mediapipe
```

然后生成带手部骨架的视频窗口：

```bash
python scripts/render_hand_skeleton_windows.py --config configs/run.operation_case_1s_hands.yaml
```

再按普通 embedding 流程跑对比配置：

```bash
python scripts/embed_windows.py --config configs/run.operation_case_1s_hands.yaml
python scripts/score_steps.py --config configs/run.operation_case_1s_hands.yaml
```

对比原始结果和手部骨架结果：

```text
outputs/operation_case_1s/report.md
outputs/operation_case_1s_hands/report.md
```

### 可选：生成窗口画面描述

如果需要检查每个视频窗口里模型能看到什么，可以单独准备生成式 Qwen-VL 模型，并运行：

```bash
mkdir -p models
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir ./models/Qwen2.5-VL-7B-Instruct
```

```bash
python scripts/describe_windows.py \
  --config configs/run.yaml \
  --model ./models/Qwen2.5-VL-7B-Instruct \
  --limit 10
```

该脚本只生成调试用描述文件，不参与现有 embedding 打分流程。输出默认写到：

```text
outputs/.../window_descriptions.jsonl
outputs/.../window_descriptions.csv
```

## Web 页面

启动内置服务：

```bash
source deps/Qwen3-VL-Embedding/.venv/bin/activate
python web_app.py --host 0.0.0.0 --port 7860 --python deps/Qwen3-VL-Embedding/.venv/bin/python
```

访问：

```text
http://127.0.0.1:7860
```

Web 服务默认使用示例配置：

```text
configs/run.operation_case_1s.yaml
configs/steps.operation_case.yaml
```

如需使用自己的工步和视频路径，可以修改配置文件或在代码中调整 `Analyzer` 初始化参数。

## 输出

运行完成后会在配置指定的 `outputs` 路径下生成：

```text
windows.csv                 # 视频窗口清单
step_embeddings.npy         # 步骤文本 embedding
step_metadata.json          # 步骤元数据
window_embeddings.npy       # 视频窗口 embedding
window_metadata.json        # 视频窗口元数据
similarity_matrix.csv       # 步骤 x 窗口相似度矩阵
scores.csv                  # 步骤识别结果
heatmap.png                 # 相似度热力图
report.md                   # Markdown 报告
labeled_output.mp4          # 可选，带工步标签的视频
```

`scores.csv` 是主要结果文件：

```text
step_id, step_name, best_time_range, best_score, margin, status, reason
```

状态说明：

- `completed`：该步骤在视频中有较明确的匹配窗口。
- `missing`：所有候选窗口分数较低，可能未完成。
- `uncertain`：存在候选窗口，但分数、区分度或顺序约束不足，需要人工复核。

## 阈值调参

可在运行配置中调整：

```yaml
scoring:
  completion_threshold: 0.60
  missing_threshold: 0.45
  margin_threshold: 0.05
  enforce_order: true
```

漏检较多时可以降低 `completion_threshold`；误检较多时可以提高 `completion_threshold` 或 `margin_threshold`。建议结合 `heatmap.png`、`scores.csv` 和 `report.md` 一起判断。

## 测试

单元测试不加载大模型，主要验证配置、向量处理、打分逻辑和 Web 服务辅助逻辑：

```bash
python -m pytest tests -v
python -m compileall scripts tests web_app.py
```

## 数据和模型

以下内容默认不提交到 Git：

- `data/`
- `models/`
- `outputs/`
- `deps/`
- 本地私有配置，如 `configs/run.yaml` 和 `configs/steps.yaml`

这样可以保持仓库只包含可复用代码、示例配置、测试和文档。
