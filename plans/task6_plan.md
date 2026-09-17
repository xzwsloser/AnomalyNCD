# Task6 执行计划：训练断点续训（epoch 级 resume）

> 背景：本次任务起因于服务器上 Task4&Task5 实验（`task45`，逐 category 训练）因其他任务
> 挤占显存导致 OOM 中断。当前仓库**没有训练断点恢复机制**，中断类别需要整轮从零重训。
> 本任务为通用训练流程新增「epoch 级断点续训」，让中断类别从最近一次已保存 epoch 继续训练。

- 分支：`feat_task_4`（本功能直接在其中实施，不另建分支）
- 创建：2026-09-17
- 状态：已 review；实施中（在 `feat_task_4` 分支）

---

## 1. 目标

1. 为 AnomalyNCD 训练主循环增加 `--resume` 支持：从指定 `model.pt` 恢复「模型权重、文本分支、优化器、学习率调度、epoch 计数」，从中断点继续训练。
2. 改为**每个 epoch 结束时都保存**到同一路径 `checkpoints/model.pt`（覆盖写、不新增文件），保证任意时刻中断最多丢失 <1 个 epoch，且能稳定续训。
3. 功能对 Baseline（Task1-3）与 Task4/5（text counterpart + CMD）统一生效，不改动推理/评测逻辑。

## 2. 背景与现状

- **当前无续训**：训练只在最后一个 epoch 才 `torch.save`（`models/AnomalyNCD.py:803-810`），中断则该类别无 checkpoint。
- 唯一的加载逻辑在 `--only_test` 推理分支（`models/AnomalyNCD.py:746-760`），训练分支不加载权重。
- 每个类别为独立进程，`load_model()` 每次都从预训练 DINO 主干重新初始化（`models/AnomalyNCD.py:95`），**不跨类别复用权重**；因此续训只针对「同一类别本轮训练中断」的场景。
- `save_dict` 已包含续训所需的全部状态：`model`、`text_modules`、`optimizer`、`epoch`、`loss_list`、`category`（`models/AnomalyNCD.py:792-802`），为续训打下基础。
- `cluster_loss_head`（`loss_list`）是 `[0]*n_head` 的逐头均值损失列表，仅在最终 predict 用于头部选择（`models/AnomalyNCD.py:209`），续训时重新从 0 累积即可。

## 3. 改动方案

### 3.1 `examples/anomalyncd_main.py`：新增 `--resume`
- 新增参数 `--resume`（默认 `None`，type=str），语义：训练启动时从该 checkpoint 恢复。
- 路径兼容两种输入：给目录时自动补 `checkpoints/model.pt`（与 `--checkpoint_path` 的用法一致）；给文件时直接用。

### 3.2 `models/AnomalyNCD.py`：`main()` 训练分支加续训逻辑 + 每 epoch 保存
位置：`main()` 的 else（训练）分支，在构建 `optimizer` / `exp_lr_scheduler` / `DistillLoss` 之后、epoch 循环之前（约 `models/AnomalyNCD.py:762`）。

具体修改（保持最小、仅新增分支，不影响不传 `--resume` 的原路径）：
1. 初始化 `start_epoch = 0`；若 `args.resume` 有值：`torch.load` 后依次
   - `self.model.load_state_dict(checkpoint['model'])`；
   - 若 `checkpoint.get('text_modules')` 且 `len(self.text_modules)`，逐个恢复文本分支（复用 `--only_test` 分支同样的逻辑）；
   - `optimizer.load_state_dict(checkpoint['optimizer'])`；
   - `start_epoch = int(checkpoint['epoch'])`（即已完成的 epoch 数）；
   - 一致性校验：若 `checkpoint.get('category')` 存在且 `!= args.category`，打印 warning（类别不匹配可能导致无效续训）。
   - 恢复 `exp_lr_scheduler`：`CosineAnnealingLR` 需要对调度器 `step()` `start_epoch` 次，使 `last_epoch`/LR 与断点一致。
2. epoch 循环改为 `for epoch in range(start_epoch, self.args.epochs)`。
3. `exp_lr_scheduler.step()` 仍在每个 epoch 训练后调用（与原逻辑一致）。
4. 保存策略调整：
   - 原「仅最后 epoch 保存」改为「**每个 epoch 结束都 `torch.save(save_dict, self.args.model_path)`**（覆盖同一文件）」，仍然只有在 `epoch + 1 == self.args.epochs` 时才执行 `sub_image_predict` / `region_merge_predict` 评测。
   - `save_dict` 内容不变（含 `epoch=epoch+1`、`optimizer`、`text_modules`、`category` 等），保证可被下一轮 `--resume` 读取。

> 说明：`--only_test` 推理分支保持不变；仅在需要时把其中的权重/文本分支加载逻辑抽成小函数 `_load_state(checkpoint)` 复用，避免重复代码。

### 3.3 脚本 / 运行方式（可选，随 review 确认）
- 不强制改 `scripts/anomalyncd.sh` / `anomalyncd_task5.sh` 主流程。
- 续训时按「同一 category 单条命令」方式运行，追加 `--resume <该类别 checkpoints 目录>` 即可。
- 增强（已实现）：新增 `scripts/anomalyncd_task45_resume.sh`，逐 category 判断「已完成(跳过) /
  中断待续(--resume) / 未开始(重跑)」，幂等可反复执行。判断依据：checkpoint 中本次新增写入的
  `epochs` 与已保存 `epoch` 比较，缺 `epochs` key（旧代码生成，只在最后 epoch 保存）视为已完成跳过。

### 3.4 文档
- `README.md` 的「运行方式」补一句 `--resume` 用法说明。
- 续训会重新执行 `binarization()`（`models/AnomalyNCD.py:425`，会 rmtree 并重建该类别 crop 数据）与幂等的 `build_text_counterpart()`；必要时后续可再加 `--skip_preprocess`，本期不纳入。

## 4. 验证方式

1. **本地语法检查**：`python -m py_compile` 编译 `models/AnomalyNCD.py`、`examples/anomalyncd_main.py`。
2. **服务器 smoke（占 1 张空闲 GPU）**：
   - 单 category、`epochs` 临时调小（如 5）；
   - 跑 2-3 个 epoch 后手动中断；
   - 用同一命令追加 `--resume`（指向该类别 `checkpoints` 目录）重启，确认日志中 epoch 从断点继续、loss 正常、最终 `model.pt` 被刷新覆盖；
   - 分别在 `use_text_feat`/`cmd` 关闭与开启各验证一次，确认文本分支权重与 `text_modules` 恢复正确。
3. **回归**：不传 `--resume` 的默认训练路径行为与改动前一致（第 3 点 epoch 循环从 0 开始、保存路径不变）。

## 5. 风险与依赖

- **非位级可复现**：续训因 sampler/seed 重排不会与「未中断」完全一致，但指标差异应可忽略，不影响结论。
- **`--resume` 与 `--only_test` 互斥**：二者都非 None 时以 warning 提示并忽略 `--resume`（续训属训练分支）。
- **类别一致性**：误用其他类别的 `model.pt` 会导致无效续训，通过 `checkpoint['category']` 校验并告警。
- **磁盘**：每 epoch 覆盖写同一个 `model.pt`，不新增文件；需确认单个文件体积不大（预计几十 MB），符合磁盘受限约束。
- **依赖**：无新增第三方库；复用 torch 原生 `load_state_dict` / optimizer 恢复。

## 6. 修改记录

- 2026-09-17：创建本计划，待用户 review。
- 2026-09-17：review 通过；按用户要求直接在 `feat_task_4` 分支实施（不另建 `feat_task_6`），实施中。
