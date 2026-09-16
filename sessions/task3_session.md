# 任务3（Task3）执行记录

> 状态：代码实现与本地验证完成；服务器 smoke test 待用户复跑  
> 创建时间：2026-09-14  
> 对应计划：`plans/task3_plan.md`

## 1. 实际改动内容

### 本轮仅包含规划文档

1. 复核飞书文档《AnomalyNCD考核任务》revision 716，确认任务三要求：使 AMEND 兼容 AnomalyNCD 输入输出（以 MEBin 处理结果作为输入），运行并计算指标。
2. 阅读 AMEND 论文中的 neighborhood loss、expanded neighborhood loss、adaptive margin regularizer 与 final loss 定义。
3. 核对当前 `models/AnomalyNCD.py`、`models/loss/_contrastive_loss.py`、`models/modules/_classifier.py` 与配置/脚本的数据流。
4. 更新 `plans/task3_plan.md`：
   - 将 MVTec 子图像场景的直接邻居数从 5 修正为论文 fine-grained 设置的 4；
   - 明确 expanded neighbors 保留重复项的加权口径；
   - 明确 negatives 只来自当前 mini-batch，并排除同一原图的两个 view；
   - 明确为保持与 AnomalyNCD 可比，继续使用 `sup_weight=0.3` 与 `memax_weight=4`，不照搬论文 0.35 与 2；
   - 删除重复修改 `AGENTS.md` 的计划项，因为该论文索引已经存在；
   - 补充分项损失日志与指标分析口径。
5. 新增本文件，用于同一任务后续多轮执行时持续回填，不另建版本文件。

## 2. 运行 / 验证结果

- 本轮只修改治理文档，未运行训练、推理或代码生成任务。
- 尚未执行计划中列出的 `py_compile`、随机张量 loss 检查、smoke test 与 MVTec 15 类实验。

## 3. 遇到的问题与解决方式

1. 初版计划写为 AMEND 直接邻居数 `N=5`。论文补充材料实验设置显示 fine-grained benchmark 使用 `N=4`，coarse-grained benchmark 才使用 `N=5`；已按 MVTec 子图像的细粒度特性修正为 `N=4`。
2. 初版计划将修改 `AGENTS.md` 列为待办。当前 `AGENTS.md` 已包含 AMEND 论文与 Task3 预期实现路径的索引；本轮删除重复改动项。
3. 若直接照搬 AMEND 论文的 `lambda=0.35` 与 `epsilon=2`，会同时改变 AnomalyNCD 的训练协议，难以判断指标变化来自 AMEND 模块还是训练权重。因此计划改为沿用 AnomalyNCD 设置，并在分析中记录该兼容性偏差。

## 4. 遗留事项

1. 等待用户 review `plans/task3_plan.md`；review 通过前不创建模型、loss、入口或实验脚本，不改训练代码。
2. review 通过后按计划执行本地单元验证、`bottle` smoke test、MVTec 15 类实验，并回填实际命令、指标、日志与问题。

## 7. 修改记录（多轮执行，单文件维护）

### v1（2026-09-14）规划核对与计划修订

- 完成：飞书任务要求核对、AMEND 论文方法核对、AnomalyNCD 代码接口核对、`plans/task3_plan.md` 修订与本文档创建。
- 结果：Task3 方案仍处于待 review 状态，未开始代码实现。

### v2（2026-09-15）代码实现与 smoke test 记录

#### 实际改动内容

1. 新增 `models/loss/_amend_loss.py`：
   - `AMENDNeighborhoodLoss` 维护 FIFO feature bank，检索 top-N 直接邻居；每个直接邻居再检索 top-M 二级邻居，且保留重复项。
   - 当前 mini-batch 特征作为 negatives，并排除同一原图两个 view；bank 为空时跳过邻域损失但仍写入 bank。
   - `AdaptiveMarginLoss` 对分类原型做 L2 normalization，按论文公式 (6)(7) 计算 adaptive margin。
2. 新增 `models/amend.py`：
   - `AMENDProjector` 使用三层 MLP 输出 256 维 projection；分类 logits 对 hidden feature 和 prototype 同时做 L2 normalization。
   - `AMEND(AnomalyNCD)` 复用 AnomalyNCD 数据流与评估主流程，重写 `load_model()` 与 `MGRL()`。
   - 训练目标保持 baseline 权重结构，把 InfoNCE 替换为 direct/expanded neighborhood loss，并加入 adaptive margin loss；蒸馏、伪标签修正、MEMax 和 supervised contrastive loss 保持不变。
3. 新增 `configs/AnomalyNCD_amend.yaml`，AMEND 设置为 `neighbors=4`、`expanded_neighbors=5`、`bank_size=2048`、`expanded_affinity=0.1`、`margin_weight=1.0`、`projection_dim=256`。
4. 新增 `examples/amend_main.py` 与 `scripts/anomalyncd_task3.sh`；后者覆盖 MVTec 15 类并输出到 `mvtec_musc_crop_task3_amend`。
5. 更新 `plans/task3_plan.md`、`plans/index.md` 与本文档；未提交 git。

#### 运行 / 验证结果

1. 本地环境 `/home/xzw/app/anaconda3/envs/AnomalyNCD/bin/python` 验证通过：
   - `python -m py_compile examples/amend_main.py models/amend.py models/loss/_amend_loss.py`
   - `bash -n scripts/anomalyncd_task3.sh`
   - `git diff --check`
   - `python examples/amend_main.py --help`
   - 配置解析结果符合计划：N=4、M=5、bank=2048、lambda_en=0.1、lambda_am=1.0、projection=256。
2. 随机张量单元级验证通过：
   - FIFO bank 按 batch 写入并保持容量上限；
   - bank 为空时跳过邻域损失；
   - direct positives 数量为 `B*N`，expanded positives 数量为 `B*N*M`；
   - 同一原图的两个 view 不进入 negatives；
   - margin loss 和 projector 输出均可反向传播且数值有限。
3. 服务器首次 bottle smoke test：
   - 使用 `DCproject` 环境，启动前 GPU0 空闲；模型构建成功，显存约 1.9GB。
   - 训练启动后在 `AMENDProjector.forward` 出现 `cuda:0` 与 `cpu` 张量设备不一致错误。
   - 根据用户要求停止服务器操作；该错误已在本地修复：不再额外注册分类器引用，改为动态从 `projector.last_layer[0]` 获取分类器，避免旧版 PyTorch 中引用迁移歧义。修复后尚未在服务器复跑。

#### 遇到的问题与解决方式

1. 本地随机张量测试暴露 negatives mask 与 repeat 后 anchors 维度不匹配，已修正为按 anchor 维度 mask 当前 batch negatives。
2. `_topk_neighbors` 初版误用 bank 自相似矩阵处理二级邻居查询，已改为用实际邻居特征查询 bank。
3. direct positives 初版未展平成 `(anchors, feature_dim)`，已修正。
4. 服务器 smoke test 暴露分类器权重设备迁移错误，已改为动态获取 `last_layer[0]` 引用；等待用户自行复跑确认。
5. 为避免服务器数据风险，未执行任何数据删除、覆盖或清理操作；共享 MEBin 数据仅由原有流程做幂等校验/生成。

#### 遗留事项

1. 用户在服务器 `feat_task_3` 分支使用 `DCproject` 环境复跑 `bottle` smoke test。
2. smoke test 通过后批量运行 MVTec 15 类，汇总子图像和 region merged NMI/ARI/F1。
3. 与 baseline 对比时记录 `sup_weight=0.3`、`memax_weight=4` 沿用 AnomalyNCD，而非 AMEND 论文的 `0.35` 与 `2`。

### v3（2026-09-16）修复服务器 device 不一致报错

#### 实际改动内容

1. 新增 `models/amend.py::AMENDProjector.classifier_weight()`：由 `weight_g` 与 `weight_v` 手动重建 weight-norm 分类头权重，返回 shape `(num_classes, feat_dim)` 且位于正确设备。
2. `AMENDProjector.forward` 中使用 `classifier_weight()` 构造归一化原型（替代 bug 的 `self.classifier.weight`）。
3. `AMEND.MGRL` 的 `AdaptiveMarginLoss` 入参改为 `projector.classifier_weight()`，保证 margin loss 在 cuda 上计算。
4. 已同步到服务器 `feat_task_3` 分支 `~/anomaly_ncd/AnomalyNCD`。

#### 运行 / 验证结果

1. 服务器 `DCproject` 环境复现确认根因：torch 2.0.1 中 `weight_norm` 模块 `weight_g`/`weight_v` 均位于 `cuda:0`，但 `module.weight` 属性仍返回 CPU 张量。
2. 服务器 GPU smoke test：`AMENDProjector(in_dim=768, out_dim=20)` 迁到 cuda 后输入 cuda 特征，`x_proj` 与 logits 均为 `cuda:0`，无 device 报错。
3. 服务器 GPU 验证 `AdaptiveMarginLoss(P.classifier_weight())` 返回 `cuda:0` 张量，loss 数值有限。
4. 服务器 `py_compile models/amend.py` 通过（DCproject 环境）。

#### 遇到的问题与解决方式

1. 初版 plan 记录"改为动态获取 `last_layer[0]`"并不能根治问题：device 迁移后 `last_layer[0].weight` 属性本身在 torch 2.0.1 下返回 CPU 张量。解决方式是绕过 `.weight` 属性，手动由 `weight_g * normalize(weight_v)` 重建，与 weight_norm 的 reparametrization 定义一致。

#### 遗留事项

1. 用户在服务器 `feat_task_3` 分支用 `DCproject` 环境复跑 `bottle` smoke test，确认训练、checkpoint、日志与 NMI/ARI/F1 输出正常。
2. smoke test 通过后批量运行 MVTec 15 类并汇总指标。
