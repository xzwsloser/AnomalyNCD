# 任务1（Task1）执行计划

> 状态：已 review；多轮修改均在文件内维护（见 §13 修改记录）
> 创建时间：2026-09-12（依据飞书考核文档与用户确认更新）
> 需求来源：飞书《AnomalyNCD考核任务》· Task1
> 对应记录：`sessions/task1_session.md`

## 1. 任务需求（原文）
> **Task1**：初始化 `classifier` 的时候，改为采用论文《Novel Class Discovery for Long-tailed Recognition》中提到的 `ETF` 模块，并在训练之前将图像特征以及 `classifier` 中的 `prototype` 可视化到一张图中（`t-SNE`）。

### 全局说明（来自考核文档）
- 基于 `AnomalyNCD` 原版代码，数据集使用 **MVTec AD**（15 类，资源不足时可选取若干类）。

## 2. 已确认范围（用户决定）
- **不需要训练**：Task1 只做「训练前」的两个动作 —— 将 classifier 换为 ETF 模块、训练前输出 t-SNE 可视化图。
- 训练、指标评测、baseline 对比 **不属于 Task1 范围**（可留作后续任务或额外验证，不在本次交付内）。

## 3. 目标拆解
1. 将 classification 头改为 **ETF（equiangular tight frame，等角原型）模块**，替换 `MultiHead` 的可学习 `last_layer`。
2. 训练开始前，利用**预训练编码器（MGViT）输出特征**与 **ETF 的 prototype（`ori_M` 行向量）**，用 **t-SNE** 降维并绘制到**同一张图**。

## 4. 关键概念澄清
- **“训练前的图像特征”**：指预训练 MGViT（直接加载 DINO 预训练权重）对图像的前向输出特征（768 维）。**不经过 ETF 模块**（经过 ETF 得到的是 logits，不是特征）。
- **“classifier 中的 prototype”**：指 ETF 分类器的等角矩阵 `ori_M` 的每个类别向量（每个类一根原型）。与图像特征同处 768 维空间，因此可一起做 t-SNE。
- 可视化 = 图像特征点 + 原型点，投影到同一个 2D 平面画在一张图。

## 5. 数据集与实验范围
- MVTec AD：15 类（bottle、cable、capsule、carpet、grid、hazelnut、leather、metal_nut、pill、screw、tile、toothbrush、transistor、wood、zipper）。
- 由于不训练，可视化只需少量 MVTec 图像（每类若干张）即可，资源占用很小。
- 目前服务器上未见 `dachuang` 自己的 MVTec 数据集，需确认/放置一次（见 §8 执行步骤与 AGENTS.md 数据集说明）。

## 6. 现状分析（相关代码）
- `models/modules/_classifier.py::MultiHead`：`last_layer` 为可学习 weight-norm 线性头（`n_head=4`）；`mlp` 产出 `x_proj` 仅用于对比损失。
  - `logits = last_layer[i](normalize(x))`；`x_proj = mlp(x)`。
- ETF 参考实现：`reference/NCDLR/nets/vit.py::ETF_Classifier`，`logits = normalize(x) @ ori_M`。
- 特征抽取：可直接加载 pre-trained backbone 前向（无需求助训练逻辑）。

## 7. 改动方案（涉及文件）
- **`models/modules/_classifier.py`**
  - 新增 `ETF_Classifier`（含 `generate_random_orthogonal_matrix`），参考 `reference/NCDLR/nets/vit.py`。
  - 将 `MultiHead` 的 `last_layer` 替换为固定 ETF 原型矩阵 `ori_M`；`ori_M` 用 `.to(device)` 放置（避免 `.cuda()` 硬编码）。
  - 保留 `mlp` / `x_proj`；关键位置加中文注释并注明论文/参考实现出处。
- **实现可视化（放 `examples/` 或新增脚本，如 `examples/visualize_tsne.py`）**
  - 用预训练 MGViT 抽取若干 MVTec 图像特征；
  - 取 ETF `ori_M` 作为 prototype；
  - 合并后用 `sklearn.manifold.TSNE` 降维至 2D，`matplotlib` 绘制一张图（特征散点 + 原型星形），保存到输出路径。
- **`utils/`（可选）**：可视化辅助函数；**`configs/AnomalyNCD.yaml`（可选）**：可视化规模/输出路径配置。

## 8. 分工与执行步骤

### 8.1 分工（重要）
- **本侧（coding agent）**：只负责修改代码与仓库配置（`.gitignore` 等），以及在 `plans/`、`sessions/` 记录文档。
- **用户侧**：负责把改动提交到 github、在服务器拉取，并完成服务器上的数据放置/下载、环境与 GPU 等运维操作。
- coding agent **不直接**代做服务器数据放置、提交、GPU 调度等操作。

### 8.2 执行步骤（代码侧）
1. 实现 `ETF_Classifier` 并替换分类头（`models/modules/_classifier.py`）。
2. 实现训练前 t-SNE 可视化脚本（预训练特征 + prototype 单图，默认 `examples/`）。
3. 静态验证：前向 logits 形状正确、`ori_M` 为等角矩阵、多 head 匹配 `n_head`。
4. （提交后由用户在服务器运行）运行可视化脚本，确认单图生成成功。
5. 填写 `sessions/task1_session.md` 记录执行详情。

> 服务器侧待办（用户执行）：确认/放置 MVTec AD 数据（优先复用已有数据，避免重复下载）、安装依赖、在 `feat_task_1` 分支拉取并运行。

## 9. 验证方式（范围：不训练）
- 单步前向：`ETF_Classifier` 输出 `[B, num_classes]` logits，与 softmax 用法兼容。
- 验证 `ori_M` 等角结构（`feat_dim >= num_classes`、正交等模长）与 device 放置。
- 多 head：确认为共享/复制的 `n_head` 组 logits 与现有 `n_head` 循环匹配。
- 可视化：训练前 t-SNE 图生成成功，图中同时包含特征点与 prototype 点。

> 训练收敛、MVTec 指标等不在 Task1 范围内（已确认无需训练）。

## 10. 风险与依赖
- ETF 要求 `feat_dim >= num_classes`：当前 `feat_dim=768`，MVTec 类别数远小 768，满足；保留参考实现的维度扩展逻辑以防万一。
- 依赖 `sklearn`、`matplotlib`（已含于 `requirements.txt`）；若服务器环境缺 `torch`/`timm` 需先装。
- t-SNE 需控制样本量，避免耗时/内存过大。
- 数据集磁盘问题见 AGENTS.md：先复用服务器已有数据，避免重复下载（磁盘仅剩约 30GB）。

## 11. 待定事项
- ETF 多 head 处理：4 个 head 共享一个 `ori_M` / 各自复制 / 改为单 head。
- t-SNE 样本规模与每类样本数。
- MVTec 用于可视化的类选取（全 15 类 or 若干类）。
- 可视化输出路径约定。

## 12. 讨论总结
### 12.1 原分类器是「MLP → softmax」吗？
- `mlp` 输出 `x_proj` 仅用于对比损失；真正生产 logits 的是 `last_layer`（weight-norm 线性头）+ softmax。

### 12.2 只是换分类器为等角矩阵吗？
- 仅替换 `last_layer`；`mlp`/`x_proj` 不变。

### 12.3 损失函数需要改吗？
- 不需要：现有损失消费 `[B, num_classes]` logits 并走 softmax，ETF 输出形状相同。多 head 需匹配 `n_head`。
- 因本任务不训练，损失改动无关紧要，仅保持形状兼容即可。

### 12.4 one-hot 标签 vs 等角原型
- one-hot 决定“贴哪个原型”，等角矩阵决定“原型之间如何摆”，不冲突。本任务只做训练前可视化，不涉训练收敛验证。

## 13. 修改记录（多轮执行，单文件维护）
> 同一任务的多次改动统一记录在本文件，不另建版本文件（规则见 AGENTS.md §1.1 第 5 条）。

### v2（2026-09-12）修复可视化脚本模块导入路径
- 问题：服务器直接运行脚本报 `ModuleNotFoundError: No module named 'models'`（脚本所在目录 `examples/` 已在 `sys.path`，但仓库根目录不在）。
- 改动：`examples/visualize_tsne.py` 顶部新增 `import sys`，并加入 `sys.path.append(os.getcwd())`（含中文注释），与 `examples/anomalyncd_main.py` 保持一致。
- 验证：本机 `python3 -m py_compile` 通过。

### v3（2026-09-12）修复可视化脚本 mask 维度
- 问题：特征抽取时报 `ValueError: not enough values to unpack (expected 4, got 3)`，定位 `_MGViT.py::prepare_mask`。
- 根因：`prepare_mask` 内 `mask_downscaling` 为 `AvgPool2d`，要求 4D `[1,1,H,W]`；脚本却传 3D `[1,H,W]`。
- 改动：`examples/visualize_tsne.py::extract_features` 中 mask 由 `torch.ones(1, H, W)` 改为 `torch.ones(1, 1, H, W)`（补通道维），加中文注释。
- 验证：本机 `python3 -m py_compile` 通过。
