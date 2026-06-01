# LLM SOP 视频工步识别

本项目用于识别 SOP 操作视频中的工步完成情况。当前核心方案是用 `Qwen3-VL-Embedding` 对标准工步文本和视频时间窗口分别生成 embedding，再用相似度、顺序约束和阈值生成 `completed`、`missing`、`uncertain` 结果。

仓库只保存代码、测试、Web 页面、示例配置和可复用文档。模型权重、输入视频、切片、输出报告、官方依赖仓库和本地私有配置不进入 Git。

## 核心流程

```text
步骤描述 -> 文本 embedding -> 步骤原型向量
视频 -> 重叠时间窗口 -> 视频窗口 embedding
步骤原型向量 x 视频窗口向量 -> cosine 相似度矩阵
相似度矩阵 + 阈值/顺序策略 -> 工步状态、热力图、报告、可选标注视频
```

几分钟长视频不要直接压成一个向量。默认配置会把视频切成短窗口，再查找每个工步在时间轴上的相似度峰值。

## 目录结构

```text
configs/
  run.example.yaml              # 通用运行配置模板
  run.operation_case_1s.yaml    # 工步案例默认配置，Web 服务也读取它
  steps.example.yaml            # 通用工步模板
  steps.operation_case.yaml     # 工步案例步骤定义
scripts/
  split_video.py                # 切分视频窗口
  embed_steps.py                # 生成步骤文本 embedding
  embed_windows.py              # 生成视频窗口 embedding
  embed_windows_parallel.py     # 多 GPU 并行生成视频窗口 embedding
  score_steps.py                # 打分、热力图、CSV 和 Markdown 报告
  render_labeled_video.py       # 根据识别结果生成带工步标签的视频
  remote_*.py                   # 远端环境探测、上传和基准测试辅助脚本
web/
  index.html                    # Web 上传和识别页面
web_app.py                      # 内置 HTTP 服务
tests/                          # 不加载大模型的单元测试
docs/                           # 设计和实现记录
```

运行时在服务器或本地自行创建以下目录，默认都被 `.gitignore` 忽略：

```text
deps/       # 官方 Qwen3-VL-Embedding 仓库和虚拟环境
models/     # 模型权重
data/       # 输入视频、上传视频和切片窗口
outputs/    # embedding、分数、热力图、报告和 Web 任务结果
```

## 快速开始

### 1. 准备项目

```bash
git clone git@github.com:zkez/llm_sop.git
cd llm_sop
```

如果是在当前开发机器上维护核心代码，推荐路径是：

```bash
cd /Users/junk/zk/junk/llm_sop
```

远端 4090 服务器默认部署路径：

```bash
ssh 4090
cd /mnt/project/zk/llm_sop
```

### 2. 安装官方依赖环境

官方模型代码单独放在 `deps/`，不要提交到本仓库。

```bash
mkdir -p deps
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git deps/Qwen3-VL-Embedding

cd deps/Qwen3-VL-Embedding
bash scripts/setup_environment.sh
cd ../..

source deps/Qwen3-VL-Embedding/.venv/bin/activate
pip install -r requirements.txt
```

默认配置使用 `attn_implementation: sdpa`，优先保证可跑通。如果服务器已正确安装 `flash-attn`，可以在配置里改为 `flash_attention_2`。

### 3. 下载模型权重

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

第一轮建议先用 2B 模型跑通流程，确认方案有效后再按显存切换到 8B。

### 4. 放置数据和配置

通用实验：

```bash
cp configs/run.example.yaml configs/run.yaml
cp configs/steps.example.yaml configs/steps.yaml
mkdir -p data
# 把输入视频放到 data/input.mp4，或修改 configs/run.yaml 里的 video.path
```

当前工步案例：

```bash
mkdir -p data/operation_case
# 把案例视频放到 data/operation_case/input.mp4
# 使用 configs/run.operation_case_1s.yaml 和 configs/steps.operation_case.yaml
```

每个步骤建议写 1-5 条能证明“已完成状态”的描述，避免只写物体名称。负例和边界条件可以写进 `negative_notes`，供后续规则或人工复核使用。

## 命令行运行

通用配置：

```bash
python scripts/split_video.py --config configs/run.yaml
python scripts/embed_steps.py --config configs/run.yaml --steps configs/steps.yaml
python scripts/embed_windows.py --config configs/run.yaml
python scripts/score_steps.py --config configs/run.yaml
```

当前工步案例：

```bash
python scripts/split_video.py --config configs/run.operation_case_1s.yaml
python scripts/embed_steps.py --config configs/run.operation_case_1s.yaml --steps configs/steps.operation_case.yaml
python scripts/embed_windows.py --config configs/run.operation_case_1s.yaml
python scripts/score_steps.py --config configs/run.operation_case_1s.yaml
```

多 GPU 并行生成视频窗口 embedding：

```bash
python scripts/embed_windows_parallel.py --config configs/run.operation_case_1s.yaml --gpus 0,1,2,3
```

生成带工步标签的视频：

```bash
python scripts/render_labeled_video.py --config configs/run.operation_case_1s.yaml
```

主要输出：

```text
outputs/.../windows.csv
outputs/.../step_embeddings.npy
outputs/.../window_embeddings.npy
outputs/.../similarity_matrix.csv
outputs/.../scores.csv
outputs/.../heatmap.png
outputs/.../report.md
outputs/.../labeled_output.mp4
```

`scores.csv` 是最重要的结果表，包含：

```text
step_id, step_name, best_time_range, best_score, margin, status, reason
```

状态含义：

- `completed`：最高分达到完成阈值，且通过当前排序或 margin 规则。
- `missing`：所有候选窗口分数都低，可能漏做该步骤。
- `uncertain`：有候选窗口，但分数、margin 或顺序约束不足，需要复核。

## Web 页面

服务器上启动：

```bash
source deps/Qwen3-VL-Embedding/.venv/bin/activate
python web_app.py --host 0.0.0.0 --port 7860 --python deps/Qwen3-VL-Embedding/.venv/bin/python
```

本地通过 SSH 端口转发访问：

```bash
ssh -L 7860:127.0.0.1:7860 4090
```

然后打开：

```text
http://127.0.0.1:7860
```

Web 服务默认读取：

```text
configs/run.operation_case_1s.yaml
configs/steps.operation_case.yaml
```

上传视频会写入 `data/web_uploads/`，任务输出会写入 `outputs/web_jobs/`，这些目录不会进入 Git。

## 本地和远端开发流程

当前约定：

- 本地核心仓库：`/Users/junk/zk/junk/llm_sop`
- 远端运行目录：`4090:/mnt/project/zk/llm_sop`
- GitHub 仓库：`git@github.com:zkez/llm_sop.git`
- 数据、模型和运行输出保留在远端，不从远端拉回 Git 仓库。

从远端同步核心代码到本地：

```bash
rsync -av \
  --exclude data/ \
  --exclude models/ \
  --exclude outputs/ \
  --exclude deps/ \
  --exclude __pycache__/ \
  --exclude .pytest_cache/ \
  --exclude '*.pyc' \
  4090:/mnt/project/zk/llm_sop/ /Users/junk/zk/junk/llm_sop/
```

本地修改后同步到远端：

```bash
rsync -av \
  --exclude .git/ \
  --exclude data/ \
  --exclude models/ \
  --exclude outputs/ \
  --exclude deps/ \
  --exclude __pycache__/ \
  --exclude .pytest_cache/ \
  --exclude '*.pyc' \
  /Users/junk/zk/junk/llm_sop/ 4090:/mnt/project/zk/llm_sop/
```

远端测试：

```bash
ssh 4090
cd /mnt/project/zk/llm_sop
source deps/Qwen3-VL-Embedding/.venv/bin/activate
python -m pytest tests -v
python -m compileall scripts tests web_app.py
```

需要加载模型的端到端验证在远端跑：

```bash
python scripts/split_video.py --config configs/run.operation_case_1s.yaml
python scripts/embed_steps.py --config configs/run.operation_case_1s.yaml --steps configs/steps.operation_case.yaml
python scripts/embed_windows.py --config configs/run.operation_case_1s.yaml
python scripts/score_steps.py --config configs/run.operation_case_1s.yaml
```

## Git 提交流程

首次初始化本地仓库：

```bash
cd /Users/junk/zk/junk/llm_sop
git init
git branch -M main
git remote add origin git@github.com:zkez/llm_sop.git
```

常规提交：

```bash
git status --short
git add README.md .gitignore configs scripts tests web web_app.py docs
git commit -m "整理项目说明"
git push -u origin main
```

提交信息使用简单中文概括即可，例如：

```text
整理项目说明
更新识别流程
修复窗口打分
补充远端测试
```

## 本地轻量验证

这些测试不加载模型，适合在本地或远端快速检查代码结构：

```bash
python -m pytest tests -v
python -m compileall scripts tests web_app.py
```

如果本地没有完整模型环境，只保证这些轻量测试通过即可；模型推理、真实视频和 Web 任务以远端 4090 服务器结果为准。

## 阈值调参

第一轮不要完全依赖固定阈值。建议先看 `outputs/.../heatmap.png`、`scores.csv` 和 `report.md`，再调整：

```yaml
scoring:
  completion_threshold: 0.50
  missing_threshold: 0.471
  margin_threshold: 0.05
  enforce_order: true
```

漏检多时降低 `completion_threshold`；误检多时提高 `completion_threshold` 或 `margin_threshold`。窗口太少导致顺序解码不可用时，Web 服务会自动回退到独立打分。

## 注意事项

- `data/`、`models/`、`outputs/`、`deps/` 不入库。
- `configs/run.yaml` 和 `configs/steps.yaml` 是个人本地配置，不入库。
- `configs/run.operation_case_1s.yaml` 是可复用案例配置，可以提交。
- 远端已有数据和模型时，同步代码不要覆盖这些目录。
- 如果热力图整体没有明显峰值，下一步再考虑加入 VLM 二次判定或更细的步骤描述。
