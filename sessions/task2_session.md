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
