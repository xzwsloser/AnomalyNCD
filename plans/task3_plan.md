# 任务3 执行计划

## 修改记录

| 时间 | 修改说明 |
| --- | --- |
| 2026-09-13 | 根据飞书任务三要求重写计划：明确 AMEND 代码适配、MEBin 输入输出兼容、损失实现、实验设置与验证方式。 |
| 2026-09-14 | 复核飞书文档 revision 716、AMEND 论文与当前代码：修正直接邻居默认值与损失实现口径，明确与 AnomalyNCD 兼容时保持的超参数偏差，并补充规划阶段 session 记录。 |
| 2026-09-15 | 执行代码实现与本地验证；服务器 bottle smoke test 暴露 projector 权重设备迁移问题，已改为动态获取 `last_layer` 分类器并记录待复跑。 |

## 目标

- 以 `MEBin` 产出的 crop 图像与 mask 作为输入，按论文公式最小复现并运行 AMEND 的两个核心模块：Expanded Neighborhood 对比学习和 Adaptive Margin 原型正则。
- AMEND 的输出格式与 AnomalyNCD 保持一致：返回子图像分类 logits，并复用 AnomalyNCD 的子图像预测与 region merging 评估流程，计算 NMI、ARI、F1。
- 在 MVTec AD 15 类上与 AnomalyNCD baseline 使用相同种子、epoch、数据路径和评估协议对比；若某个类别前存在可复用 baseline 结果，优先复用，否则补跑 baseline。

## 背景

### 飞书任务三要求

> 修改论文《AMEND: Adaptive Margin and Expanded Neighborhood for Efficient Generalized Category Discovery》的代码，使其兼容 AnomalyNCD 的输入输出（以 MEBin 处理后的结果作为输入），并运行代码计算指标。

AMEND 论文全文没有给出官方代码仓库、项目页或 supplementary 链接；本仓库 `reference/` 目录也没有对应参考实现。因此这里的执行口径不是克隆/修改官方仓库，而是**论文公式复现**：保留 AnomalyNCD 的 MEBin 数据流、mask-guided backbone 与评估协议，把 AMEND 的邻域对比损失和 adaptive margin 正则加入训练。

飞书文档版本为 `revision_id=716`。整体约束为使用 MVTec AD，每个任务基于 AnomalyNCD 原版代码；若资源不足，可只选部分类别。

### 论文核心方法

论文 PDF：`papers/Banerjee_AMEND_Adaptive_Margin_and_Expanded_Neighborhood_for_Efficient_Generalized_Category_WACV_2024_paper.pdf`。

1. **Neighborhood loss**：对 l2 normalized projection 特征 `z` 维护 feature bank；对每个 anchor 从 bank 取 top-N 直接邻居，按论文公式 (1) 计算直接邻域对比损失。
2. **Expanded neighborhood loss**：对每个直接邻居再从 bank 检索 top-M 二级邻居；扩展邻居列表保留重复项，使重复出现的近邻对损失产生更大贡献。按论文公式 (2)(3) 以 `lambda_en=0.1` 加权。
3. **Adaptive Margin loss**：分类原型 l2 normalized 后，按论文公式 (6)(7) 惩罚过于接近的原型对；`lambda_am=1.0`。
4. **分类损失**：labelled 样本用监督分类损失，unlabelled 样本用 teacher soft pseudo-label 与 mean-entropy regularizer；AMEND 使用 ViT-DINO 主干、三层 MLP projection（输出 256 维）、feature bank 容量 2048、扩展邻居数 M=5。论文中 fine-grained benchmark 的直接邻居数为 N=4，coarse-grained benchmark 才使用 N=5；MVTec 子图像更接近细粒度场景，因此本任务默认采用 N=4。

AMEND 论文默认训练 200 epochs。本任务为了与 AnomalyNCD baseline 可比，采用仓库现有 `epochs=50`，不照搬论文 200 epochs；AMEND 特有超参数按论文设置。

**兼容性超参口径**：为了把实验差异限制在 AMEND 的两个核心模块，不改变 AnomalyNCD 的训练协议，本任务继续使用 `sup_weight=0.3` 与 `memax_weight=4`，不照搬论文中的 `lambda=0.35` 与 `epsilon=2`。这一偏差必须在实验分析中明确记录，避免被误解为严格复现 AMEND。

## 改动方案

### 文件与具体修改

| 文件 | 修改内容 |
| --- | --- |
| `models/amend.py` | 新增 `AMEND(AnomalyNCD)` runner。复用 `train_init()` 的数据/模型/日志初始化和 `main()` 的训练、保存、评估流程；重写 `load_model()` 与 `MGRL()`，保持 AnomalyNCD 的输入输出接口。该文件是论文方法在本仓库中的复现实现。 |
| `models/loss/_amend_loss.py` | 新增 `AMENDNeighborhoodLoss` 与 `AdaptiveMarginLoss`。前者维护 2048 容量 FIFO feature bank，从 bank 检索 top-N/top-M 邻居并用 mini-batch 作 negatives；后者从 projector 原型计算 adaptive margin。损失公式来源均为 AMEND 论文，不引用不存在的官方代码。 |
| `configs/AnomalyNCD_amend.yaml` | 在原配置基础上新增 `amend:` 配置块：`neighbors=4`、`expanded_neighbors=5`、`bank_size=2048`、`expanded_affinity=0.1`、`margin_weight=1.0`、`projection_dim=256`；`n_head` 设为 `1`，其余训练/MEBin 参数保持原配置。 |
| `examples/amend_main.py` | 新增命令入口，参数与 `examples/anomalyncd_main.py` 保持一致，仅加载 `models.amend.AMEND`。 |
| `scripts/anomalyncd_task3.sh` | 新增 MVTec 15 类批量实验脚本，数据路径与 Task2 脚本一致，输出目录命名为 `mvtec_musc_crop_task3_amend`。 |
| `plans/index.md` | 更新任务3状态为“待 review”。 |

### 数据接口约定

1. **输入**：不重新生成 MEBin 输出，直接使用共享 untracked 路径：
   - `binary_data_path=data/mvtec_musc`
   - `crop_data_path=data/mvtec_musc_crop`
   - `anomaly_map_path=data/mvtec_musc_anomaly_map`
   - `dataset_path` 指向服务器上的 MVTec AD 原始数据。
2. **Dataset 返回**：沿用 `Dataset_AnomalyNCD` 与 `MergedDataset` 的 `(views, labels, image_paths, masks, mask_paths)` 接口；masks 一起送入 `MGViT`，避免丢弃 AnomalyNCD 的 mask guidance。
3. **标签与伪标签**：AeBAD_crop 中的 labelled 类沿用现有 `mask_lab`；novel 类沿用 crop anomaly score 的 pseudo-label correction 与 MEMax，保证评估对象与 AnomalyNCD 一致。
4. **模型输出**：继续输出 `(x_proj, [logit])`。`x_proj` 用于 AMEND neighborhood loss，`logit` 用于分类、蒸馏、region merging 和指标计算。

### 训练损失与实现细节

保持 AnomalyNCD 主训练目标结构不变，做最小 AMEND 替换：

```text
loss
  = (1 - sup_weight) * cluster_loss
  + sup_weight * cls_loss
  + (1 - sup_weight) * AMEND_neighborhood_loss
  + sup_weight * supervised_contrastive_loss
  + amend.margin_weight * adaptive_margin_loss
```

即把原 InfoNCE 项替换为 AMEND 的直接/扩展邻域损失，其余蒸馏、伪标签修正、MEMax、supervised contrastive loss 保持不变。

Feature bank 细节：

1. 每个训练 step 将 batch 中两个 view 的 `x_proj` 做 L2 normalization 后写入 2048 容量 FIFO bank。projection 输出维度 256 通过 `amend.projection_dim` 显式传入，便于检查论文设置。
2. 直接邻居与扩展邻居只从历史 feature bank 检索；denominator 的 negatives 只取当前 mini-batch 的两份 view 特征，并排除同一原图的另一个增强 view，避免把潜在正样本当 negatives。实现时需按 crop 文件名/原图 ID 判断同一原图。
3. 对缺少足够邻居的 warmup 阶段：如果 bank 中邻居数少于 N/M，可用现有特征数执行，损失分母使用实际邻居数；bank 为空时该邻域项跳过。
4. AMEND 的分类前向与 Adaptive Margin 均对 prototype 做 L2 normalization，符合论文公式 (8)；baseline 的 `MultiHead` 行为保持不变。
5. 日志中分别记录 direct neighborhood、expanded neighborhood、总和 neighborhood 和 adaptive margin 四项损失，便于后续解释指标变化，而不只记录总和。

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | MVTec AD，15 类：`bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper` |
| MEBin 输入 | 复用 `data/mvtec_musc` 与 `data/mvtec_musc_crop`，不重复二值化和裁剪 |
| 训练超参数 | `configs/AnomalyNCD.yaml` 中 `batch_size=32`、`epochs=50`、`lr=0.003`、`seed=3407` 保持不变 |
| AMEND 超参数 | `neighbors=4`、`expanded_neighbors=5`、`bank_size=2048`、`expanded_affinity=0.1`、`margin_weight=1.0`；projection 维度沿用 `MultiHead` 内置的 256 |
| 对比 baseline | 优先复用服务器已有 baseline 结果；若无，则用原 `scripts/anomalyncd.sh` 补跑相同类别、seed 与 epoch |
| 指标 | 子图像 NMI / ARI / F1，以及 region merged NMI / ARI / F1 |

训练前必须在服务器重新查询：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
df -h /
```

只使用空闲且内存占用极低的 GPU；优先选择编号最小的空闲卡。输出、checkpoint 与日志仍写入 `outputs/`，不提交到 git。

## 验证方式

### 本地验证

1. `python -m py_compile examples/amend_main.py models/amend.py models/loss/_amend_loss.py`。
2. 用随机张量做 loss 单元级检查：
   - neighbor loss 在同 batch 中不把同一原图的另一个 view 当作 negative；
   - expanded neighbors 数量、权重与去重策略符合实现约定；
   - adaptive margin loss 可反向传播且对 normalized prototypes 数值有限；
   - `AMENDNeighborhoodLoss` 的 FIFO bank 容量不超过 2048。
3. 构造小 batch 调用 `AMEND` 模型和 loss，确认 `x_proj`、logits、梯度与 `n_head=1` 输出形状正确。

### 服务器实验验证

1. 确认 MVTec AD、MuSc anomaly map、MEBin binary/crop 数据与 `scores_json` 均存在且路径有效。
2. 先跑 `bottle` 一个类别作为 smoke test，确认训练、checkpoint、日志和 NMI/ARI/F1 输出正常。
3. smoke test 通过后批量运行 15 类；若共享 GPU 排队严重或磁盘余量不足，与用户确认后再缩减类别。
4. 汇总每个类别的 baseline 与 AMEND 指标，计算 NMI/ARI/F1 的 per-category 与 mean 差异，并分析上升、下降或几乎不变的原因。
5. 执行完成后更新 `sessions/task3_session.md` 与 `sessions/index.md`。

## 分析口径

1. 对比实验以“同一份 MEBin crop 数据 + 同一种子/epoch/评估协议 + 仅替换邻域对比与加入 adaptive margin”为主结论；明确记录 `sup_weight` 与 `memax_weight` 沿用 AnomalyNCD 而非 AMEND 论文设置。
2. 分析上升/下降/几乎不变时，结合分项损失曲线判断：邻域损失可能提高实例一致性与局部聚类性，但错误邻居也可能强化错误正样本；adaptive margin 理论上应增大原型间隔，但可能改变分类边界。
3. 不在主实验中网格搜索 `neighbors`、`expanded_neighbors` 或 `margin_weight`；如果 smoke test 暴露实现问题，先修复实现，不做选择性调参。

## 风险与依赖

- 当前 `reference/` 目录没有 AMEND 官方源码，论文正文也没有代码链接；本计划按论文公式在仓库内最小复现。若后续确认存在官方实现或用户提供实现，先放入 `reference/AMEND/`，再以官方实现为准核对细节并调整本计划。
- AMEND feature bank 会增加少量显存/内存；`batch_size=32`、两个 view、bank 容量 2048、projection 维度 256，预期可运行在 RTX 3090 24GB，但启动前仍需实际检查显存。
- `MultiHead` 原本支持多 head；AMEND 论文使用一个 prototype 分类头。本任务设 `n_head=1`，避免引入与论文无关的 head 集成逻辑。
- AnomalyNCD 的 mask guidance 不是 AMEND 论文原生输入；为满足“兼容 AnomalyNCD 输入”的约束，保留 `MGViT(images, masks)` 而不替换为普通 DINO ViT。
- 若 crop 数据的 pseudo-label score 与 baseline 不一致，会导致比较失真；实验前必须使用同一份共享数据，不重新生成。
