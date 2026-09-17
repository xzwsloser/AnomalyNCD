# Task4 & Task5 执行计划：TAC 文本引导融合

> 说明：任务4 与任务5 均基于论文 *Image Clustering with External Guidance*（TAC, ICML 2024, Oral），
> 属于同一条「文本引导（外部知识）」流水线，故**合并为一份计划**执行。
> - Task4：应用 `Text Counterpart Construction`，两模态特征直接拼接用于后续训练。
> - Task5：融入 `Cross-modal Mutual Distillation`（含端到端/非端到端方案的选择与说明）。
>
> 分支：`feat_task_4`（已对齐 `feat_task_3` 最新 `a9ba641`）。

- 创建：2026-09-16
- 状态：待用户 review

---

## 1. 目标

1. **Task4**：参考 TAC 的 `Text Counterpart Construction`，为 AnomalyNCD 训练/推理所用的子图（sub-image）离线构造「文本对应特征」（text counterpart）。训练时将该文本特征与图像特征**直接拼接**后送入后续投影分类头训练，跑通 MVTec AD，对比 Baseline 分析 NMI/ARI/F1 指标提升/下降/几乎不变的原因。
2. **Task5**：将 TAC 的 `Cross-modal Mutual Distillation`（跨模态互蒸馏）融入 AnomalyNCD，选择融合时机（训练前/中/后），可选端到端或非端到端，跑通并分析指标升降原因。
3. **治理**：补充 `AGENTS.md`（papers 对应表 + reference 说明）；登记 `plans/index.md`、`sessions/index.md`（任务5并入任务4）。

## 2. 背景与论文解读（TAC, 2310.11989）

TAC 的核心：在没有「类别名先验」的情况下，利用 CLIP 预训练模型提供的**跨模态语义**来引导图像聚类。

### 2.1 Text Counterpart Construction（Task4 对应模块）

离线流水线（官方即非端到端，参考 `reference/2024-ICML-TAC/`）：
1. `image_embedding.py`：CLIP(ViT-B/32) 编码所有图像 → 图像特征（512 维）。
2. `text_embedding.py`：CLIP 文本编码器编码 WordNet 名词表（多 prompt 模板取平均）→ 名词嵌入（512 维）。
3. `filter_nouns.py`：
   - 对图像特征做 k-means（faiss spherical k-means，`cluster_num` 个中心）得到图像语义中心。
   - 用零样本反向分类：把名词划分到最近的图像中心，每个中心取 topK 个置信度最高的名词 → 形成**判别性名词子集** `filtered_nouns`。
4. `retrieve_text.py`：对每个图像，计算图像特征与名词子集的 softmax 相似度（`tau=0.005`），加权求和名词嵌入得到文本对应向量，并 L2 归一化：
   `text_i = normalize( softmax( feats_i @ nouns.T / tau ) @ nouns )`

### 2.2 Cross-modal Mutual Distillation（Task5 对应模块）

`train_head.py` + `loss_utils.py` + `models.py::ClusterHead`：
- `ClusterHead`：文本分支 + 图像分支，各自对（文本/图像）特征输出聚类分配 logits。
- `DistillLoss`：InfoNCE 式对比损失，拉近「图像分支对某样本的分配」与「文本分支对同一样本（或邻居）的分配」（互为教师-学生）。
- `consistency_loss`：对齐两分支的分配分布（binary CE）。
- `entropy`：负熵项防止退化（全部样本分到同一类）。
- 配合 `mine_nearest_neighbors`（faiss）用邻居做跨模态信息互蒸馏。

## 3. 端到端 vs 非端到端训练（概念与本次选择）

### 3.1 概念解释

- **端到端训练（end-to-end）**：把**整条流水线**放入同一个可微计算图中一次训练。即「原始图像 → 图像编码 → 文本编码 → 文本检索/文本对应生成 → 拼接/跨模态蒸馏损失 → 分类头」全程可导，训练时损失从最后一级**反向传播到最前面的编码器**，所有模块参数（含 CLIP 文本/图像编码器或可学习的检索部分）随每次迭代一起更新。
  - 优点：各模块可协同优化，理论上最贴合任务。
  - 代价：计算图大、训练慢、显存占用高、更易不稳定；需要把 CLIP 引入并被训练，偏离「利用冻结外部知识」的设定。

- **非端到端训练（two-stage / offline）**：把部分阶段**离线/冻结**，训练只更新下游主模型。
  - 本场景的语义：**文本对应生成**（CLIP 图像编码、名词检索、text counterpart）在训练前离线完成、参数**冻结**，生成的文本特征作为**固定输入**喂给 AnomalyNCD；训练只反向传播更新 AnomalyNCD 自身（MGViT + 投影头 + 可选的小文本投影/文本头）。
  - 优点：与 TAC 官方一致（本就离线）、训练便宜、稳定、易复现、显存/磁盘开销小（共享服务器、磁盘仅剩约 30GB）。

### 3.2 本次选择（推荐）

- **Task4**：**非端到端**。文本对应特征离线生成、冻结；训练时与图像特征拼接后喂入投影头。这与任务字面「两模态特征直接拼接进行后续训练」一致。
- **Task5**：融合时机选**训练中（during training）**，方式**非端到端**。即：文本特征冻结，AnomalyNCD 内新增一个可训练的**文本聚类头**与跨模态蒸馏损失，训练阶段加入该损失，与主模型联合优化（此处的「训练」更新的是 MGViT + 图像投影头 + 文本头，不更新 CLIP）。
- 端到端作为**可选项**在设计上预留开关，但因成本与共享资源约束，**首版不启用**。
- 依据：TAC 官方实现即为非端到端且效果最好；AnomalyNCD 图像主干是 DINO ViT 而非 CLIP，文本引导的「外部知识」正应保持冻结；便于复现与消融对比（与 baseline/task2 对齐）。

## 4. 总体技术方案

在 AnomalyNCD 现有「MGViT → MultiHead 投影/分类」主链上，增加一条**冻结的文本侧输入**：

```text
图像分支（主干，可训练）:
  crop image --MGViT--> cls_token(768) ──┐
                                         ├─ 直接拼接(concatenate) ─> MultiHead projector( in_dim 增大 )
文本侧（离线冻结，训练时常量输入）:        │
  预计算 text counterpart (512, CLIP,离线) ─┴─> [可选] 小线性投影(可训练) / 文本聚类头(Task5)
```

- **Task4**：`cls_token` 与 `text_feat` 直接拼接 → 送入 `MultiHead`（`in_dim` 从 768 增加到 768+512=1280）。
- **Task5**：新增文本聚类头 `TextClusterHead`（借鉴 TAC `ClusterHead` 文本分支），对 `text_feat` 输出跨类分配 logits；在训练循环中加入跨模态互蒸馏损失（`L_distill + L_consist - λ·H`）。

## 5. 数据集与依赖

- 数据集仍是 **MVTec AD**（`dataset_path=/mnt/data/xrli/mvtec_anomaly_detection`，服务器共享路径），逐 category 训练（bottle/cable/.../zipper 共 15 类）。
- 文本对应的对象是 MEBin 产出的**子图**（crop sub-image，`data/mvtec_musc_crop/{category}/images/...`，训练+测试都要），因为 AnomalyNCD 训练/推理均在该子图上进行。
- 需新增依赖：
  - `clip`（openai/CLIP，用于离线提取图像/名词文本特征）。
  - `faiss-gpu`（filter_nouns 的 k-means 与 retrieve 相似度检索）。
  - WordNet 名词表：复用 TAC `reference/2024-ICML-TAC/data/WordNetNouns.csv`（拷贝到服务器共享数据根，体积小）。
- 磁盘控制：文本特征为 `.npy`（N×512 float32；全部 MVTec 子图约几十万 × 512 ≈ 数百 MB），放到共享 untracked 路径（如 `data_store/...`），不提交进 git。

## 6. 改动方案（涉及文件与具体修改）

### 6.1 离线文本对应生成（Task4 前置）

新增模块 `models/modules/_tac_text.py`（带中文注释，保留 TAC 出处）：
- `encode_image_clip(paths, clip_model)`：批处理编码 crop 子图 → N×512。
- `encode_nouns(clip_model)`：编码 `WordNetNouns.csv`（7 个 SIMPLE_IMAGENET_TEMPLATES 平均）。
- `filter_nouns(...)`：faiss spherical k-means（k=`cluster_num`）+ 每中心 topK 名词筛选。
- `retrieve_text(...)`：softmax(feat·noun^T/tau) @ noun → 归一化，输出 text counterpart。
- `build_or_load(...)`：对 train/test 需要的所有子图生成并**落盘**（`image_path→text vector`，如 `{crop_root}/text_counterpart/{category}.npy` + `image_paths.json`，或目录对齐）。

新增脚本 `scripts/build_text_counterpart.sh`：按 category 依次调用上面模块生成文本特征并保存到共享路径。

> 作用对象：默认对**训练+测试**所有会进入 loaders 的子图（含 base 与 novel）生成文本对应。
> 若 base 类（AeBAD）无法便捷复用，则 base 样本文本特征可用「类别名模板」的 CLIP 文本嵌入兜底（config 可开关）。具体在实现期定稿，计划先锁定 novel（待聚类）子图为主。

### 6.2 数据加载注入文本特征

- `datasets/dataset.py`：`Dataset_AnomalyNCD` 增加可选 `text_feat_root` / `use_text_feat`，在 `__getitem__` 中按 `image_path` 查找并返回 `text_feat`（新增返回位）。
- `datasets/data_utils.py`：`get_datasets` 透传文本特征根路径；`MergedDataset.__getitem__` 拼接/透传文本特征，使 `MGRL` 与两个 predict 函数都能取到与每个样本对齐的文本特征。

### 6.3 模型改动（Task4：拼接）

- `models/modules/_classifier.py` 或新文件：可选新增一个小的可训练文本投影 `TextProjector`（默认仅一个 Linear，将文本特征投影到与图像特征可拼接的维度；也可直接原样拼接，config 控制 `project_text: True/False`）。
- `models/AnomalyNCD.py::load_model`：
  - 当 `text_counterpart.enabled=True` 时，`MultiHead(in_dim = feat_dim + text_dim)`（默认 768+512=1280）。
  - 若启用文本投影，则文本特征先过 `TextProjector` 再拼接。
  - 模型包装改为 `nn.Sequential(MGViT, projector)` 之外另行保存文本分支（或用 ModuleDict），保证 `model.state_dict()` 兼容保存/加载。

### 6.4 模型改动（Task5：Cross-modal Mutual Distillation）

- 新增 `models/modules/_tac_text.py::TextClusterHead`（借鉴 TAC `ClusterHead` 文本分支，输出跨类分配 logits；带中文注释与出处）。
- 新增 `models/loss/_tac_loss.py`（带中文注释与出处）：
  - `TACDistillLoss`（InfoNCE 式跨模态蒸馏，等价 TAC `DistillLoss`）。
  - `consistency_loss`（对齐两分支分配分布）。
  - `entropy`（负熵正则，避免退化）。
- `models/AnomalyNCD.py::MGRL`（熔入**训练中**）：
  - 从 batch 取 `text_feat`。
  - 前向：图像分支（原有）得到 `student_out`；文本分支 `text_feat → TextClusterHead → text_logits`。
  - 计算 CMD 损失：`L_cmd = w_cmd * ( L_distill(image_logits, text_logits) + L_distill(text_logits, image_logits) + L_consist - λ_cmd * entropy )`。
  - 加入总损失：`loss += L_cmd`，`backward/step` 更新 MGViT、投影头、文本头。
- `sub_image_predict` / `region_merge_predict`：推理时需同时取到 `text_feat` 并走相应分支；若 Task5 文本头也参与推理可纳入 logits 融合（第一版：推理只走图像分支，文本分支仅用于训练蒸馏，减少对评测逻辑的改动；具体实现期定稿）。

### 6.5 配置与入口

- `configs/AnomalyNCD.yaml`（及新 `configs/AnomalyNCD_task4.yaml`）新增段：
  ```yaml
  text_counterpart:
    enabled: false          # 是否启用文本对应
    root: data_store/text_counterpart   # 离线特征根（共享 untracked 路径）
    top_k: 5                # filter_nouns 每中心保留名词数
    tau: 0.005              # retrieve softmax 温度
    cluster_num: null       # null => 自动取各 category 的 novel 类别数
    concat_in_features: true  # Task4: 文本特征与图像特征直接拼接
    project_text: false       # 是否先经可训练线性投影再拼接
  cmd:                     # Task5: Cross-modal Mutual Distillation
    enabled: false
    weight: 0.1            # 跨模态蒸馏损失权重
    entropy_weight: 1.0    # 负熵项权重
  ```
- `examples/anomalyncd_main.py::load_args`：解析上述新配置到 `args`。
- run：`scripts/anomalyncd_task4.sh`（逐 category，`--runner_name mvtec_musc_crop_task4`）。

### 6.6 治理文档

- `AGENTS.md`：
  - §3 papers 表新增 `papers/2310.11989v3.pdf`（TAC, ICML 2024）与 `reference/2024-ICML-TAC/` 对应关系及模块说明。
  - §4/§2 说明 reference 目录内 TAC 参考实现的结构（`filter_nouns/retrieve_text/train_head/loss_utils/models` 等）。
- `plans/index.md`：登记任务4（含「任务5已并入任务4」）；`sessions/index.md` 待执行后登记。

## 7. 验证方式

1. **本地离线冒烟**：对单个 category，先用一小组子图运行 `_tac_text.build_or_load`，检查文本特征 shape/对齐（`image_path→text`），勿跑完整 CLIP（如需下载 CLIP 权重需网络/权限，服务器上准备）。
2. **本地 smoke**：`examples/anomalyncd_main.py --only_test` 载入 checkpoint 确认拼接/文本分支推理路径不报错（或先构造最小 batch 单测）。
3. **服务器训练冒烟**：单 category、`text_counterpart.enabled=True`（Task4）与 `cmd.enabled=True`（Task5）各跑 1–2 epoch，确认 loss 正常且含 CMD 分量。
4. **正式实验**：15 个 category 各跑（复用现有 `mvtec_musc_crop*` 流程），记录 region-merged NMI/ARI/F1。
5. **对比分析**：与 baseline（task1）及 task2 结果对比，说明指标提升/下降/几乎不变的原因（外部文本引导对哪些缺陷类型有效、哪些因文本空间与工业视觉差异失效等）。

## 8. 风险与依赖

- **依赖安装**：服务器需装 `clip`、`faiss-gpu`（与 CUDA 12.0 匹配）；需在训练前确认。
- **CLIP 权重下载**：离线文本特征阶段需要 CLIP 预训练权重，服务器需能访问或预置。
- **文本空间与工业缺陷语义偏差**：WordNet 名词来自自然图像语料，与 MVTec 工业缺陷（如 bottle 的 broken_large）可能语义不对齐，导致 task4/5 增益有限甚至下降——这正是要求分析的重点。
- **维度/拼接改动**：调整 `MultiHead.in_dim` 会连带 checkpoint、模型参数 `state_dict` 变更，需保证保存/加载路径一致（新增文本分支参数需纳入）。
- **磁盘/显存**：文本特征与 run 日志需放共享 untracked 路径；继续控制体积。
- **分支纪律**：本轮所有改动（含 AGENTS.md、plans/sessions）均在 `feat_task_4` 分支维护。

## 9. 修改记录

- 2026-09-16：创建本计划（合并任务4+任务5）。待用户 review。
- 2026-09-17：补记「服务器 CLIP 离线加载」修复计划。根因：服务器无外网，
  `open_clip` 在加载 ViT-B-32 权重时回退到 `huggingface.co` 在线下载，出现
  `Network unreachable` 并持续重试直至 `KeyboardInterrupt`。方案：在
  `_tac_text.build_or_load` 中，当 `CLIP_CHECKPOINT` 未设置时自动探测共享
  `data_store/clip/ViT-B-32.pt`（已预置的 openai JIT 权重，open_clip 2.32 可直接
  加载），根除联网依赖；同时在服务器 repo 建立 `data_store` → 共享数据根符号链接。
