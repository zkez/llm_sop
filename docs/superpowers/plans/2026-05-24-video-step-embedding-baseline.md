# 视频步骤 Embedding 基线实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可在服务器上运行的零样本视频步骤核验实验工程。

**Architecture:** 项目分成配置层、视频切片层、embedding 层和打分报告层。embedding 层依赖官方 `Qwen3-VL-Embedding` 仓库，其余逻辑保持轻量，便于本地测试和服务器迁移。

**Tech Stack:** Python 3.10+、Transformers、Qwen3-VL-Embedding、PyTorch、ffmpeg/ffprobe、NumPy、pandas、matplotlib、PyYAML。

---

## 文件结构

- Create: `README.md`：中文使用说明、服务器部署步骤、运行命令。
- Create: `.gitignore`：忽略模型、视频、切片、embedding 和输出文件。
- Create: `requirements.txt`：轻量依赖，重型 PyTorch 按服务器 CUDA 单独安装。
- Create: `configs/run.example.yaml`：运行配置样例。
- Create: `configs/steps.example.yaml`：步骤描述配置样例。
- Create: `scripts/common.py`：配置读取、路径处理、向量归一化、时间格式化等通用函数。
- Create: `scripts/split_video.py`：用 ffprobe/ffmpeg 将长视频切成重叠窗口，并输出窗口清单。
- Create: `scripts/embed_steps.py`：加载 Qwen3-VL-Embedding，生成步骤文本原型向量。
- Create: `scripts/embed_windows.py`：加载 Qwen3-VL-Embedding，生成视频窗口向量。
- Create: `scripts/score_steps.py`：计算相似度矩阵、步骤状态、热力图和 Markdown 报告。
- Create: `tests/test_common.py`：验证通用函数。
- Create: `tests/test_scoring.py`：用小型假 embedding 验证打分逻辑。

## Task 1: 创建配置、说明和基础文件

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `configs/run.example.yaml`
- Create: `configs/steps.example.yaml`

- [ ] **Step 1: 写入项目说明**

创建中文 `README.md`，包含目标、目录结构、服务器准备、模型下载、运行顺序和输出说明。

- [ ] **Step 2: 写入忽略规则**

创建 `.gitignore`，忽略 `data/`、`models/`、`outputs/`、`deps/`、Python 缓存和虚拟环境。

- [ ] **Step 3: 写入依赖文件**

创建 `requirements.txt`，包含 `numpy`、`pandas`、`matplotlib`、`PyYAML`、`tqdm`、`Pillow`。PyTorch 和 Transformers 在 README 中按服务器 CUDA 版本安装。

- [ ] **Step 4: 写入运行配置样例**

创建 `configs/run.example.yaml`，包含模型路径、官方仓库路径、视频路径、切片参数、embedding 批大小、打分阈值和输出路径。

- [ ] **Step 5: 写入步骤配置样例**

创建 `configs/steps.example.yaml`，包含 3 个中文示例步骤，每个步骤 3 条正向描述和若干负例说明。

## Task 2: 实现通用工具和测试

**Files:**
- Create: `scripts/common.py`
- Create: `tests/test_common.py`

- [ ] **Step 1: 写通用函数测试**

测试 `format_time_range`、`l2_normalize`、`mean_normalized`、`resolve_project_path`。

- [ ] **Step 2: 实现通用函数**

实现 YAML/JSON 读写、目录创建、向量归一化、时间格式化、项目相对路径解析。

- [ ] **Step 3: 运行测试**

Run: `python -m pytest tests/test_common.py -v`

Expected: 所有测试通过。

## Task 3: 实现视频切片脚本

**Files:**
- Create: `scripts/split_video.py`

- [ ] **Step 1: 实现 ffprobe 时长读取**

用 `ffprobe -show_entries format=duration` 获取视频秒数。

- [ ] **Step 2: 实现窗口生成**

根据 `window_seconds` 和 `stride_seconds` 生成 `[start, end)` 窗口，最后一个窗口不超过视频时长。

- [ ] **Step 3: 实现 ffmpeg 切片**

对每个窗口运行 ffmpeg，输出 `window_000001.mp4` 这类文件，并生成 `outputs/windows.csv`。

- [ ] **Step 4: 保留可重复运行能力**

如果配置启用 `skip_existing` 且窗口文件已存在，则跳过该窗口。

## Task 4: 实现步骤文本 embedding

**Files:**
- Create: `scripts/embed_steps.py`

- [ ] **Step 1: 动态加载官方 Qwen3-VL-Embedding 仓库**

从配置中的 `model.qwen_repo_path` 加入 `sys.path`，导入 `src.models.qwen3_vl_embedding.Qwen3VLEmbedder`。

- [ ] **Step 2: 编码步骤描述**

对每条步骤正向描述分别调用 `model.process([{"text": ..., "instruction": ...}])`。

- [ ] **Step 3: 聚合为步骤原型**

对同一步骤的描述 embedding 先归一化，再平均，再归一化，保存为 `outputs/step_embeddings.npy` 和 `outputs/step_metadata.json`。

## Task 5: 实现视频窗口 embedding

**Files:**
- Create: `scripts/embed_windows.py`

- [ ] **Step 1: 读取窗口清单**

读取 `outputs/windows.csv` 中的窗口 id、时间范围和视频路径。

- [ ] **Step 2: 批量编码视频窗口**

对窗口构造 `{"video": path, "instruction": ..., "fps": ..., "max_frames": ...}` 输入，按配置批量调用 `model.process`。

- [ ] **Step 3: 保存视频窗口 embedding**

保存 `outputs/window_embeddings.npy` 和 `outputs/window_metadata.json`。

## Task 6: 实现打分、热力图和报告

**Files:**
- Create: `scripts/score_steps.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: 写打分测试**

用小型假向量验证 cosine 矩阵、最佳窗口、margin、`completed/missing/uncertain` 判定和顺序冲突。

- [ ] **Step 2: 实现打分函数**

实现 `compute_similarity_matrix`、`summarize_steps`、`write_similarity_matrix_csv`、`write_scores_csv`、`write_report_md`。

- [ ] **Step 3: 实现热力图**

使用 matplotlib 输出 `outputs/heatmap.png`，横轴为时间窗口，纵轴为步骤。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/test_scoring.py -v`

Expected: 所有测试通过。

## Task 7: 全项目验证

**Files:**
- Modify: none

- [ ] **Step 1: 运行轻量测试**

Run: `python -m pytest tests -v`

Expected: 所有测试通过。

- [ ] **Step 2: 做语法检查**

Run: `python -m compileall scripts tests`

Expected: 所有 Python 文件编译通过。

- [ ] **Step 3: 检查文档语言**

确认 README、配置样例和报告模板说明都使用中文。

