# 任务2（Task2）执行记录

> 状态：进行中（代码实现与本地验证完成，服务器实验待执行）  
> 创建时间：2026-09-12  
> 对应计划：`plans/task2_plan.md`

## 1. 实际改动内容

### 数据筛选核心

1. `datasets/data_utils.py`：
   - 新增 `select_images_by_anomaly_score`，按 image-level max anomaly map score 筛选；
   - defect type 使用分数降序，`normal_class`（默认 `good`）使用分数升序；
   - 相同分数按图像名升序 tie-break；
   - 数量采用 `max(1, round(total * ratio))`，并记录每个类型的 total、selected、actual_ratio、direction 和 selected score range；
   - 新增 `save_train_subset_selection`、`filter_train_unlabelled_by_selection`、`load_train_subset_summary`；
   - 在 `get_datasets` 中只过滤 `train_unlabelled`，不修改 `train_labelled` 和 `test`；
   - 过滤时将裁剪文件名 `{original_stem}_crop{n}.png` 还原为 `{original_stem}`，与 selection JSON 对齐。
2. `models/AnomalyNCD.py`：
   - 在 MEBin/cropping 阶段记录每张原始 MVTec 图像的 `max(anomaly_map)`；
   - `train_subset.enabled=True` 时生成 `crop_data_path/selection_json/{category}.json`；
   - 打印每个 anomaly type 的筛选方向、数量、实际比例和分数范围；
   - 不修改 MEBin 阈值搜索、cropping、训练损失和测试逻辑。
3. `examples/anomalyncd_main.py`：
   - 解析 `train_subset.enabled / ratio / normal_class`；
   - 默认配置保持 `enabled=False`，baseline 流程不变。

### 配置与脚本

1. `configs/AnomalyNCD.yaml` 新增默认关闭的 `train_subset` 配置。
2. 新增 `configs/AnomalyNCD_task2.yaml`，启用 50% 图像级分数筛选，实验名为 `AnomalyNCD_Task2`。
3. 新增 `scripts/prepare_mvtec_musc_anomaly_maps.sh`：
   - 将服务器已有 MuSc map 从 `{category}/anomaly_maps/{anomaly_type}/{stem}_crop0.png` 建立为 `data/mvtec_musc_anomaly_map/{category}/{anomaly_type}/{stem}.png`；
   - 只创建 symlink，不复制 PNG；
   - 检查 1725 个 map、broken symlink 和每类数量/文件名对应关系。
4. 新增 `scripts/anomalyncd_task2.sh`：
   - 使用 Task2 配置；
   - runner name 为 `mvtec_musc_crop_task2`；
   - 复用共享 MVTec、anomaly map、AeBAD_crop 和 MEBin 数据路径。

### 治理文档

1. `plans/task2_plan.md` 状态更新为“已 review；代码实现完成，服务器实验待执行”。
2. `plans/index.md` 同步更新任务2状态。
3. 本文件登记代码实现、验证结果、问题与遗留事项。

## 2. 运行 / 验证结果

### 本地静态与单元验证

```bash
python3 -m py_compile examples/anomalyncd_main.py models/AnomalyNCD.py datasets/data_utils.py
bash -n scripts/prepare_mvtec_musc_anomaly_maps.sh scripts/anomalyncd_task2.sh
```

结果：通过。

### 筛选逻辑验证

使用内存 fixture 验证：

1. defect type 按分数降序选择；
2. `good` 按分数升序选择；
3. 相同分数按文件名升序稳定排序；
4. 5 个样本、ratio 0.5 时选择 3 个；
5. crop 文件名正确还原为原图 prefix；
6. selection JSON 中的 `score_definition=max_raw_anomaly_map` 正确；
7. 过滤只作用于传入的 novel 训练数据。

结果：`local validation passed`。

### ETF 配置核对

`use_etf` 仅通过 `load_args` 传入 `MultiHead`，运行时没有其他硬编码依赖。为避免 Task2 与 baseline 比较时引入分类头差异，已将两个配置的 `models.use_etf` 统一改为 `False`。

### 配置解析验证

```text
configs/AnomalyNCD.yaml        -> use_train_subset=False, ratio=0.5, normal_class=good, exp_name=AnomalyNCD
configs/AnomalyNCD_task2.yaml  -> use_train_subset=True,  ratio=0.5, normal_class=good, exp_name=AnomalyNCD_Task2
```

结果：通过。导入过程中出现的 matplotlib/fontconfig cache warning 仅来自沙箱只读 HOME cache，不影响配置验证。

### 格式检查

```bash
git diff --check
```

结果：新增 CRLF 文件中的 CR 会被 `git diff --check` 报告为 trailing whitespace；已按仓库原有 CRLF 风格保留配置与入口文件，未引入实际多余空格。

## 3. 遇到的问题与解决方式

1. 初版本地验证发现 `filter_train_unlabelled_by_selection` 引用的 `subsample_dataset` 未导入，导致 `NameError`。已在 `datasets/data_utils.py` 中从 `datasets.dataset` 显式导入。
2. 初版筛选器直接使用裁剪文件名 stem，会得到 `000_crop0` 而非原图 prefix `000`。已修正为移除 `_crop{n}` 后缀后再匹配。
3. `apply_patch` 在原有 CRLF 文件中混入 LF，导致 diff 包含无关换行差异。已将 `configs/AnomalyNCD.yaml`、`configs/AnomalyNCD_task2.yaml` 和 `examples/anomalyncd_main.py` 统一恢复为 CRLF，保持仓库风格。

## 4. 遗留事项

1. 服务器上尚未执行 `scripts/prepare_mvtec_musc_anomaly_maps.sh`，需要完成 symlink 数量、文件名和 map 可读性检查。
2. baseline 与 Task2 的 15 类训练尚未运行；执行前需重新确认磁盘、数据、GPU 空闲状态。
3. 需要收集 `outputs/mvtec_musc_crop/metrics.csv` 与 `outputs/mvtec_musc_crop_task2/metrics.csv`，汇总 region-merged NMI/ARI/F1 并完成原因分析。
4. 执行完成后回填服务器命令、日志路径、每类指标和结论。

## 5. 修改记录

- v2（2026-09-13）：通过 SSH 搜索服务器，确认 `/mnt/data/ejxu/data/AeBAD_crop` 等 5 个候选均为 4 类、1278 张图像且 mask 数量一致；新增并接入 `scripts/link_server_aebad_crop.sh`，用于幂等创建/校验 `data/AeBAD_crop` symlink，避免重复占用磁盘。已在服务器创建 `data/AeBAD_crop -> /mnt/data/ejxu/data/AeBAD_crop`，远端校验 classes=4、images=1278、masks=1278。本地 `bash -n`、fixture 幂等测试与 `git diff --check` 通过。
- v3（2026-09-13）：根据确认结论关闭 ETF；同步修改 baseline 与 Task2 配置为 `use_etf=False`，并更新计划中的实验设置说明。
- v4（2026-09-13）：确认 baseline 为 `scripts/anomalyncd.sh`；为 baseline 与 Task2 增加通过环境变量覆盖 GPU 和 MEBin 输出路径的能力，并让 baseline 自动复用/校验 AeBAD_crop 链接。训练日志、checkpoint 和 metrics 通过不同 `runner_name` 隔离；并行运行时必须为 MEBin 输出指定不同路径，避免共享数据目录删除/写入竞争。

## 6. 实验二数据补充与分析（v5）

### 6.1 实验设置

| 项目 | 设置 |
| --- | --- |
| baseline | `configs/AnomalyNCD.yaml`，`runner_name=mvtec_musc_crop` |
| Task2 | `configs/AnomalyNCD_task2.yaml`，`runner_name=mvtec_musc_crop_task2` |
| 数据集 | MVTec AD 15 类 |
| 样本筛选 | `train_subset.enabled=True`，`ratio=0.5` |
| 分数定义 | 每张原图 MuSc anomaly map 的最大原始分数，`score_definition=max_raw_anomaly_map` |
| 筛选方向 | 缺陷类取分数最高的 50%；`good` 取分数最低的 50% |
| 作用范围 | 只过滤 `train_unlabelled`，`train_labelled` 与 `test` 不变 |
| 共同设置 | `seed=3407`，`dino_vitb8`，`use_etf=False`，50 epochs |

Task2 的 15 类 novel 训练样本总数从 1918 减少到 945，约保留 49.3%。例如 `toothbrush` 从 52 个训练样本降到 26 个，`transistor` 从 109 降到 52。测试集仍保持全量。

### 6.2 逐类结果与差异

下表为 region-merged prediction 的最终 NMI/ARI/F1。`Δ` 为 Task2 减 baseline。

| Category | Baseline NMI/ARI/F1 | Task2 NMI/ARI/F1 | Δ |
| --- | ---: | ---: | ---: |
| bottle | 0.6076 / 0.5397 / 0.7831 | 0.4723 / 0.4332 / 0.7229 | -0.1354 / -0.1065 / -0.0602 |
| cable | 0.5367 / 0.4897 / 0.5971 | 0.5488 / 0.5173 / 0.6691 | +0.0122 / +0.0276 / +0.0719 |
| capsule | 0.4317 / 0.3008 / 0.5833 | 0.5068 / 0.3874 / 0.5758 | +0.0751 / +0.0866 / -0.0076 |
| carpet | 0.8107 / 0.7846 / 0.8376 | 0.7924 / 0.7548 / 0.8462 | -0.0184 / -0.0298 / +0.0085 |
| grid | 0.6078 / 0.5148 / 0.6667 | 0.6039 / 0.5012 / 0.6026 | -0.0039 / -0.0136 / -0.0641 |
| hazelnut | 0.6913 / 0.6279 / 0.7636 | 0.6527 / 0.6627 / 0.8364 | -0.0386 / +0.0347 / +0.0727 |
| leather | 0.8872 / 0.8654 / 0.9274 | 0.8512 / 0.8118 / 0.8952 | -0.0360 / -0.0536 / -0.0323 |
| metal_nut | 0.6803 / 0.5465 / 0.6696 | 0.7123 / 0.6036 / 0.6696 | +0.0320 / +0.0570 / +0.0000 |
| pill | 0.4155 / 0.2733 / 0.4800 | 0.5480 / 0.4740 / 0.6867 | +0.1325 / +0.2007 / +0.2067 |
| screw | 0.4777 / 0.3475 / 0.5500 | 0.3892 / 0.2817 / 0.5188 | -0.0884 / -0.0658 / -0.0312 |
| tile | 0.9251 / 0.9295 / 0.9658 | 0.8468 / 0.7947 / 0.9145 | -0.0783 / -0.1348 / -0.0513 |
| toothbrush | 0.3679 / 0.2594 / 0.7619 | 0.1385 / 0.0283 / 0.6190 | -0.2294 / -0.2310 / -0.1429 |
| transistor | 0.5106 / 0.3831 / 0.5700 | 0.4409 / 0.3556 / 0.6500 | -0.0696 / -0.0276 / +0.0800 |
| wood | 0.7029 / 0.5912 / 0.8088 | 0.6725 / 0.5499 / 0.7941 | -0.0304 / -0.0413 / -0.0147 |
| zipper | 0.5602 / 0.3958 / 0.6148 | 0.5871 / 0.4147 / 0.6667 | +0.0269 / +0.0189 / +0.0519 |

均值结果：

| Metric | Baseline | Task2 | Δ |
| --- | ---: | ---: | ---: |
| NMI | 0.6142 | 0.5842 | -0.0300 |
| ARI | 0.5233 | 0.5047 | -0.0186 |
| F1 | 0.7053 | 0.7112 | +0.0058 |

上升/下降数量：NMI 为 5 升 / 10 降；ARI 为 6 升 / 9 降；F1 为 6 升 / 8 平 / 1 未变。

### 6.3 指标变化原因分析

1. **整体 NMI/ARI 下降、F1 微升的原因**：Task2 将 novel 训练样本从 1918 降到 945。数据多样性减少会削弱特征空间的细粒度类间结构，因此更敏感的聚类一致性指标 NMI 和 ARI 整体下降。F1 是 region-merged 后的 micro F1，更多受最终图像级类别归属影响；Task2 在 `cable`、`hazelnut`、`transistor`、`zipper` 等类别上带来明显 F1 提升，抵消了部分下降，所以均值只略升。
2. **指标上升的可能原因**：`pill` 提升最大（NMI +0.1325，ARI +0.2007，F1 +0.2067）。该类 baseline 中可能存在低分但语义模糊、或 MuSc 分数与缺陷语义不一致的训练样本。Task2 通过高分缺陷和低分 good 筛选，减少了这些噪声样本，使训练分布更干净。`cable`、`hazelnut`、`transistor`、`zipper` 的 F1 上升也支持这一解释：筛掉易混淆样本后，模型在测试集上的 novel 样本归属更稳定。
3. **指标下降的可能原因**：`toothbrush`、`bottle`、`screw`、`tile` 等类别下降明显。这些类别本身 novel 训练样本较少或缺陷形态差异较小，按 MuSc 最大分数取前 50% 可能过度偏向最明显的缺陷。被剔除的边界样本虽然分数较低，但可能承载了类内变化和困难负样本信息。尤其是 `toothbrush` 只剩 26 个训练样本，NMI/ARI 大幅下降与训练样本过少的可能性一致。
4. **外部分数先验的偏差**：`select_images_by_anomaly_score` 使用的是 MuSc anomaly map 的最大值。这个分数更直接反映异常显著性，而不保证与 novel defect subclass 的可分性一致。若某一类的不同 defect subtype 在 MuSc map 上强度接近，或图像中的背景/纹理导致分数偏高，筛选会让训练集偏向特定强度或外观，而不是均匀覆盖所有缺陷语义。
5. **代码机制与结果的对应关系**：`get_datasets` 只过滤 `train_unlabelled`，测试集不变；因此指标变化来自训练分布变化，而不是测试集变简单或变难。`region_merge_predict` 仍然按原始图像聚合 crop logits，并用 anomaly area 权重融合。Task2 没有修改这一测试逻辑，所以 F1 的局部提升主要说明筛选后的特征/分类头对某些类别的图像级归属更有效。
6. **结论**：Task2 的 image-level score 筛选对噪声较大或高置信缺陷明显的类别有帮助，但对小样本、细粒度或高分样本不均衡的类别容易造成代表性和多样性损失。当前单种子结果显示 NMI/ARI 略降、F1 微升，说明该方法尚不能稳定优于全量训练，后续可尝试更温和的筛选比例、按 defect subtype 分层采样，或结合不确定度保留边界样本。

## 7. 治理修复与飞书同步准备（v6）

1. 已确认当前分支为 `feat_task_2`，并按新增规则检查该分支只应维护 `plans/task2_plan.md` 与 `sessions/task2_session.md`。
2. 发现先前误建 `plans/task4_plan.md` 后，已将其中的飞书文档同步计划合并到 `plans/task2_plan.md` 第 7 节，并删除误建文件。
3. 已同步清理 `plans/index.md` 中的任务4条目；本次治理没有创建 `sessions/task4_session.md`。
4. 已将「分支名对应唯一 plan/session 文档」和「治理漂移修复流程」写入 `AGENTS.md`。
5. 后续飞书写入结果继续记录在本文件，不再新建其他任务编号文档。

### 7.1 飞书写入结果

1. 已通过 `lark-cli wiki +node-get` 将目标 wiki URL 解析为 `obj_type=docx`、`obj_token=JE7NdOsv7o5U1Kx66qicq28TnHg`。
2. 已通过 `lark-cli docs +fetch --scope outline` 定位到 `Task2` 标题块 `OIdPdgfgXoTOGZxzfwVchjTbnXd`。
3. 已使用 `docs +update --command block_insert_after` 在 Task2 说明后的空段落块 `SJB3dOWYbo6BXSx5Wt8c3uWRnKy` 后插入 `实验配置`、`实验结果`、`实验分析` 三部分内容。
4. 飞书返回写入成功，文档修订号由 717 变为 718。
5. 已重新读取 `Task2` section，确认三部分内容位于 `Task2` 标题下，表格、均值和结论完整，且没有重复插入。
