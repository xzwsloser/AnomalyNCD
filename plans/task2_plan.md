# 任务2（Task2）执行计划

> 状态：已 review；实验二数据补充与分析  
> 创建时间：2026-09-12  
> 最近更新：2026-09-13

## 1. 目标

1. 实现 Task2 的 novel 训练子集筛选：缺陷样本按 MuSc 最大原始异常分数取高分，`good` 取低分。
2. 保持测试集与训练损失不变，仅过滤 `train_unlabelled`。
3. 在服务器完成 baseline 与 Task2 的 15 类对比实验。
4. 回填实验二设置、逐类结果、整体差异与指标上升/下降原因分析。

## 2. 背景与实验设置

- baseline：`configs/AnomalyNCD.yaml` + `scripts/anomalyncd.sh`，`runner_name=mvtec_musc_crop`。
- Task2：`configs/AnomalyNCD_task2.yaml` + `scripts/anomalyncd_task2.sh`，`runner_name=mvtec_musc_crop_task2`。
- Task2 相比 baseline 只启用 `train_subset.enabled=True`，并将实验名改为 `AnomalyNCD_Task2`。
- 筛选依据为 `score_definition=max_raw_anomaly_map`，即每张原图的 MuSc anomaly map 最大值。
- `ratio=0.5`，缺陷类按分数降序选择，`normal_class=good` 按分数升序选择。
- 种子为 3407，主干为 `dino_vitb8`，`use_etf=False`，训练 50 epochs。

## 3. 改动方案

1. `datasets/data_utils.py`：增加可复现的 image-level 子集筛选、JSON 保存和 `train_unlabelled` 过滤。
2. `models/AnomalyNCD.py`：在 MEBin/cropping 阶段记录每张原图最大异常分数并生成筛选清单。
3. `configs/AnomalyNCD_task2.yaml`：启用 Task2 筛选。
4. `scripts/anomalyncd_task2.sh`：隔离 Task2 输出目录并复用共享数据路径。
5. `sessions/task2_session.md`：回填实验二设置、指标表和原因分析。

## 4. 验证方式

1. 本地 `py_compile` 与 shell `bash -n`。
2. 用 fixture 验证高分缺陷、低分 good、tie-break 与 crop 文件名还原。
3. 服务器完成后读取 `metrics.csv`，对齐 15 类 NMI/ARI/F1。
4. 计算 baseline 与 Task2 的逐类差异和均值差异。
5. 检查代码路径，确认过滤只作用于 `train_unlabelled`，测试集不变。

## 5. 风险与依赖

- 结果基于 `seed=3407` 的单次运行，不做统计显著性结论。
- MuSc 分数作为外部先验，若某类 map 分数和缺陷语义不一致，筛选会引入偏差。
- 小类别样本量较少，减半后可能降低训练多样性。

## 6. 修改记录

- v5（2026-09-13）：补充实验二设置、逐类指标、整体均值差异与代码关联的原因分析。

## 7. 飞书文档同步计划

### 目标

将本文件与 `sessions/task2_session.md` 中已经完成的实验配置、实验结果和指标分析同步到用户提供的飞书 wiki 文档，并插入到文档中的 `Task2` 标题下。

### 执行方式

1. 检查 `lark-cli` 是否可用，查看其 wiki / docx 相关命令帮助。
2. 定位飞书 wiki 页面对应的 `obj_token` 和 `obj_type`。
3. 在文档块树中查找 `Task2` 标题块。
4. 在该标题块之后插入实验配置、实验结果、实验分析三部分内容。
5. 写入后再次读取文档块树，确认内容位于 `Task2` 标题下且没有重复插入。
6. 在 `sessions/task2_session.md` 回填实际写入结果。

### 验证与风险

- 验证方式：`lark-cli` 返回成功，且文档块树中 `Task2` 标题下包含三部分内容；`git status` 不出现实验代码改动。
- 前置依赖：本地已安装并登录 `lark-cli`，当前账号具有目标文档编辑权限，文档中存在明确的 `Task2` 标题。
- 主要风险：飞书块 ID 会随编辑变化，写入前必须重新定位；若存在多个 `Task2` 标题且语境无法唯一判断，需先向用户确认；内容较长时可分块插入。
