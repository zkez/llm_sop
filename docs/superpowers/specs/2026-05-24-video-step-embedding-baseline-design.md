# 视频步骤 Embedding 基线实验设计

## 目标

构建一个小型零样本实验，用来判断一段较长的操作流程视频中，是否包含了预期的所有步骤。第一版只使用多模态 embedding，不做微调、不训练分类器，也不加入语言模型的二次判定。

## 范围

实验在远程服务器上运行，通过 Transformers 直接加载本地模型权重。本地工作区只保存轻量内容：代码、配置、文档和同步/运行说明。模型权重、输入视频、切片视频、embedding 结果和报告等较大文件都放在服务器上。

## 模型

第一版默认使用 `Qwen/Qwen3-VL-Embedding-2B`。项目需要允许通过配置切换模型路径，例如切到服务器本地模型目录，或后续换成 `Qwen/Qwen3-VL-Embedding-8B`。

## 数据流程

1. 在 `configs/steps.yaml` 中定义预期流程步骤。
2. 为每个步骤写多条详细文本描述，并计算文本 embedding。
3. 将同一步骤的多条文本 embedding 归一化后取平均，得到该步骤的原型向量。
4. 将长视频切成有重叠的时间窗口。
5. 对每个视频窗口计算视频 embedding。
6. 计算每个步骤原型向量与每个视频窗口向量之间的 cosine 相似度。
7. 输出步骤-时间窗口相似度矩阵、每个步骤的最佳匹配结果，以及人可读的核验报告。

## 步骤描述

每个步骤包含一个 id、一个显示名称，以及三到五条详细的正向描述。描述应该写清楚画面中可见的动作和完成状态，而不是只写步骤名称。

可以在配置中写负例说明，帮助人工整理步骤边界；但第一版实现不会把负例用于打分。

## 视频切片

第一版默认使用 8 秒窗口、4 秒步长，每个窗口最多采样 16 帧。配置中需要暴露 `window_seconds`、`stride_seconds` 和 `max_frames`。

后续可以继续测试更短或更长的窗口。第一版的核心目标是观察相似度热力图中，预期步骤是否能在合理时间段形成明显峰值。

## 打分逻辑

对每个步骤：

- 找到相似度最高的视频窗口。
- 计算该窗口中最佳步骤分数和第二名步骤分数之间的 margin。
- 根据可配置阈值，将步骤标记为 `completed`、`missing` 或 `uncertain`。
- 可以选择启用顺序约束，让后面的步骤尽量匹配到更靠后的视频窗口。

阈值在第一版中只是实验参数。报告应该暴露原始分数和 margin，不要只输出最终结论。

## 输出结果

第一版实现需要生成：

- `outputs/scores.csv`：每个步骤一行，包含最佳时间窗口、分数、margin 和状态。
- `outputs/similarity_matrix.csv`：完整的步骤-视频窗口相似度矩阵。
- `outputs/heatmap.png`：用于人工检查的相似度热力图。
- `outputs/report.md`：简洁的步骤核验报告。

## 服务器运行流程

完整运行环境在服务器上：

1. 创建 Python 环境。
2. 安装 PyTorch、Transformers、Qwen3-VL-Embedding 所需依赖、ffmpeg 工具、pandas、numpy、matplotlib 和 PyYAML。
3. 下载或挂载模型权重。
4. 将输入视频放到 `data/` 目录。
5. 从项目根目录运行脚本。

本地机器可以通过 `rsync` 或 `scp` 将轻量项目同步到服务器。

## 第一版项目文件

计划创建这些文件：

- `README.md`
- `configs/run.example.yaml`
- `configs/steps.example.yaml`
- `scripts/split_video.py`
- `scripts/embed_steps.py`
- `scripts/embed_windows.py`
- `scripts/score_steps.py`

## 验收标准

基线实验成功的标准是：给定一段长视频和一份步骤列表，系统可以生成一份可读报告，展示每个步骤可能是已完成、缺失还是不确定，并给出对应时间戳和原始相似度证据。

