# 任务2（Task2）执行计划

> 状态：已 review；代码实现完成，服务器实验待执行  
> 创建时间：2026-09-12  
> 需求来源：飞书《AnomalyNCD考核任务》· Task2  
> 对应记录：执行通过后创建并登记 `sessions/task2_session.md`

## 1. 任务需求（原文）

> 训练集修改，每种类型缺陷仅使用图像级异常分数最高的 50% 样本训练，正常类别使用图像级异常分数最低的 50% 样本训练，运行代码分析指标提升/下降/几乎不变的原因；

## 2. 目标与范围

1. 在不改变测试集的前提下，只对 MVTec AD novel/unlabeled 训练样本做筛选：
   - 异常缺陷类型：按图像级异常分数从高到低取约 50%；
   - 正常类型（MVTec AD 中的 `good`）：按图像级异常分数从低到高取约 50%；
   - 图像级异常分数定义为该原始图像对应异常概率图的全局最大值，与论文中的 anomaly score 定义一致。
2. 保留现有模型结构、损失函数、增强、优化器、学习率调度、随机种子和 50 epoch 训练配置，避免引入无关变量。
3. 在相同数据与配置下对比 baseline 和 Task2 筛选集，报告 `NMI / ARI / F1`，并分析变化原因。
4. 复用服务器上已生成的 MVTec MuSc anomaly map；MuSc 属于 zero-shot AD 推理，不需要重新训练，也不重复下载或复制大体积数据。

### 范围说明

- 当前实现的 labeled base 数据来自 `AeBAD_crop`，novel/unlabeled 数据由 MVTec AD 原图经 MuSc anomaly map 与 MEBin 处理得到。本任务只筛选 MVTec novel 训练样本，不改动 `AeBAD_crop`。
- 当前训练与测试使用同一份 MEBin 裁剪数据目录，因此实现上会生成筛选清单并在加载训练集时过滤，而不是物理删除裁剪结果；测试集必须继续读取全量裁剪样本。
- 若用户对“训练集”的理解包含 labeled base 数据，需要另行确认；本计划默认任务需求针对 MVTec AD 的 unlabeled novel 训练数据。

## 3. 当前实现分析

### 3.1 数据生成

- `models/AnomalyNCD.py::binarization` 按产品类读取 MVTec 原图和对应 anomaly map，调用 `MEBin` 生成二值 mask，并进行 anomaly-centered cropping。
- `bin.crop_sub_image_mask` 会返回每个 crop 的异常分数，随后保存到 `crop_data_path/scores_json/{category}.json`。
- 当前 `scores_json` 主要用于训练中 `get_pseudo_label_weights` 的伪标签修正，不是用于按原始图像筛选训练集。

### 3.2 数据加载

- `datasets/dataset.py::get_anomalyncd_datasets` 从同一份裁剪数据构造 `train_labelled`、`train_unlabelled` 和 `test`。
- `datasets/data_utils.py::get_datasets` 将 labelled 与 unlabelled 合成训练集，测试集独立返回。
- `subsample_dataset` 已支持按索引过滤 `data_to_iterate` 和 `uq_idxs`，可复用于训练集筛选。

### 3.3 训练与评估

- `models/AnomalyNCD.py::load_datasets` 在 `get_datasets` 后构造 `WeightedRandomSampler`， labelled/unlabeled 的相对权重基于过滤后的数据长度计算。
- `MGRL` 训练时继续读取 `scores_json/{category}.json`，对 novel 样本做伪标签权重修正。
- 最终指标来自 `region_merge_predict` 的 image-level region merging 结果；`sub_image_predict` 结果可作为诊断，但主表以 region merged prediction 为准。
- `split_cluster_acc` 中的 `NMI / ARI / F1` 在 novel 类别集合上计算，`good` 与各 defect type 都属于当前 novel/unlabeled 类别。

## 4. 改动方案

### 4.1 配置与参数

1. 在 `configs/AnomalyNCD.yaml` 中新增训练筛选开关，默认关闭，保证 baseline 配置可复现：
   ```yaml
   train_subset:
     enabled: false
     ratio: 0.5
     normal_class: good
   ```
2. 新增 `configs/AnomalyNCD_task2.yaml`：
   - 继承默认训练超参数；
   - `train_subset.enabled: true`；
   - `experiment.exp_name: AnomalyNCD_Task2`，避免与 baseline 日志混淆。
3. `examples/anomalyncd_main.py::load_args` 读取并解析上述字段：
   - `args.use_train_subset`
   - `args.train_subset_ratio`
   - `args.train_subset_normal_class`

### 4.2 图像级分数筛选工具

1. 在 `datasets/data_utils.py` 新增纯函数式筛选工具，便于静态测试：
   - 输入每个 anomaly type 的 `{image_prefix: image_score}`；
   - defect type 按 `score` 降序排序；
   - `good` 按 `score` 升序排序；
   - 对相同分数使用图像文件名升序作为确定性 tie-breaker；
   - 数量使用四舍五入：`max(1, floor(total * ratio + 0.5))`，保证非空类别至少保留 1 张；
   - 返回选中图像 prefix、每类总数/选中数/实际比例、分数范围和排序方向。
2. 输出结构中保存：
   ```json
   {
     "ratio": 0.5,
     "normal_class": "good",
     "score_definition": "max_raw_anomaly_map",
     "selected": {"anomaly_type": ["image_prefix"]},
     "stats": {"anomaly_type": {"total": 0, "selected": 0, "actual_ratio": 0.0, "direction": "high|low", "score_min": 0.0, "score_max": 0.0}}
   }
   ```
3. 筛选文件保存到 `crop_data_path/selection_json/{category}.json`，与现有 `scores_json` 分离，避免混淆 crop 级分数和 image 级筛选结果。

### 4.3 MEBin 阶段生成筛选清单

修改 `models/AnomalyNCD.py::binarization`：

1. 在逐图读取 anomaly map 时记录 `image_prefix = basename(image_path).split('.')[0]` 与 `score = float(np.max(anomaly_map))`。
2. 在保存现有 `scores_json` 后，若 `use_train_subset` 为真，则调用筛选工具生成并保存 `selection_json/{category}.json`。
3. 打印每个 anomaly type 的总数、选中数、实际比例与分数范围，便于服务器日志检查。
4. 若 `use_train_subset` 为假，不生成筛选文件，行为与 baseline 完全一致。
5. 不改变现有 MEBin 阈值搜索、二值化、crop 逻辑和 `scores_json` 格式。

### 4.4 只过滤训练 unlabelled 数据

修改 `datasets/data_utils.py::get_datasets`：

1. 在构造 `MergedDataset` 之前，只对 `datasets['train_unlabelled']` 应用筛选。
2. 读取 `crop_data_path/selection_json/{category}.json`。
3. 将 dataset 中 `image_path` 的文件名前缀与 `selected[anomaly_type]` 匹配后，调用现有 `subsample_dataset`。
4. 不修改 `datasets['test']`，确保测试集仍包含全部 anomaly type 样本。
5. 不修改 `datasets['train_labelled']`，保持 AeBAD labeled base 数据不变。
6. 训练 sampler 使用过滤后的 `labelled_dataset` 和 `unlabelled_dataset` 长度，因此无需额外修改 `models/AnomalyNCD.py::load_datasets`。

### 4.5 Task2 运行脚本

新增 `scripts/anomalyncd_task2.sh`：

1. 与现有 `scripts/anomalyncd.sh` 保持相同的 15 个 MVTec AD 类别和数据路径。
2. 使用 `--config configs/AnomalyNCD_task2.yaml`。
3. 使用独立 `run_exp="mvtec_musc_crop_task2"`，使日志和 `outputs/{runner_name}/metrics.csv` 与 baseline 分开。
4. 不复制或重复存储 MVTec / anomaly map / MEBin 裁剪数据，继续复用共享 untracked 数据路径。

### 4.6 服务器异常图准备

服务器上已存在与 MVTec AD 测试集一一对应的 MuSc anomaly map，不需要重新运行 MuSc，更不需要训练 MuSc。已确认的路径如下：

| 内容 | 服务器路径 |
| --- | --- |
| MVTec AD 原始数据 | `/mnt/data/xrli/mvtec_anomaly_detection` |
| MuSc + MEBin 处理结果 | `/mnt/data/xrli/mvtec_musc_mebin_nocrop` |
| MuSc anomaly map | `/mnt/data/xrli/mvtec_musc_mebin_nocrop/{category}/anomaly_maps/{anomaly_type}/{image_stem}_crop0.png` |

官方处理结果已经核对：

1. 覆盖 MVTec AD 全部 15 个产品类别。
2. anomaly map 总数为 1725，与 MVTec AD 测试集总数一致。
3. map 为 8-bit grayscale PNG，尺寸约为 `900×900`。

新增 `scripts/prepare_mvtec_musc_anomaly_maps.sh`，只建立 symlink，不复制 PNG：

1. 输入源目录：
   ```bash
   SRC=/mnt/data/xrli/mvtec_musc_mebin_nocrop
   DST=data/mvtec_musc_anomaly_map
   ```
2. 对每个类别、每个 anomaly type，将：
   ```text
   $SRC/{category}/anomaly_maps/{anomaly_type}/{image_stem}_crop0.png
   ```
   映射为当前代码期望的结构：
   ```text
   $DST/{category}/{anomaly_type}/{image_stem}.png
   ```
3. 目标使用 `ln -s` 指向源文件；若目标 symlink 已存在且指向正确源文件，则跳过。
4. 若目标路径存在但不是正确 symlink，报错退出，避免覆盖已有数据。
5. 脚本最后检查：
   - symlink 数量是否为 1725；
   - 是否存在 broken symlink；
   - 每个 MVTec 类别的原始测试图数量与 symlink 数量是否一致；
   - 去掉 `_crop0` 后的 map 文件名是否能与 `test/{category}/{anomaly_type}` 下的原图一一匹配。
6. 该脚本只生成 `data/mvtec_musc_anomaly_map` 的轻量 symlink 结构，后续仍按当前 `binarization()` 流程重新生成本仓库需要的 MEBin binary mask 与 crop 数据。

## 5. 实验设置

### 5.1 数据与模型

| 项目 | 设置 |
| --- | --- |
| 数据集 | MVTec AD，15 类全部参与 |
| anomaly map | `data/mvtec_musc_anomaly_map`，由 `scripts/prepare_mvtec_musc_anomaly_maps.sh` 软链接到 `/mnt/data/xrli/mvtec_musc_mebin_nocrop/{category}/anomaly_maps/{anomaly_type}` |
| MVTec 原始数据 | `/mnt/data/xrli/mvtec_anomaly_detection` |
| MEBin 数据 | `data/mvtec_musc` 与 `data/mvtec_musc_crop`，共享 untracked 数据路径 |
| labeled base | `data/AeBAD_crop`，不筛选 |
| backbone | DINO `dino_vitb8` |
| 分类头 | MultiHead + ETF，`n_head=4` |
| evaluation set | 全量 MVTec novel 裁剪样本，不因训练筛选而缩减 |

### 5.2 训练超参数

| 项目 | 设置 |
| --- | --- |
| seed | 3407 |
| epochs | 50 |
| batch size | 32 |
| learning rate | 0.003 |
| optimizer | SGD |
| momentum | 0.9 |
| weight decay | 0.00005 |
| scheduler | CosineAnnealingLR |
| n_views | 2 |
| subset ratio | defect 类取最高分数 50%，`good` 取最低分数 50% |

### 5.3 论文实验设置对齐

以下设置来自 `papers/2410.14379v2.pdf` §4.1、Appendix A/B/C，并与当前仓库配置对照。Task2 不修改这些设置，除非本计划显式说明。

| 模块 | 论文 / 官方实现 | 当前仓库配置 | 说明 |
| --- | --- | --- | --- |
| backbone | DINO ViT-B/8，最后 9 层 self-attention 替换为 mask-guided attention；只微调最后一层 block | `pretrained_backbone=dino_vitb8`，`mask_layers=9`，`grad_from_block=11` | 语义一致：冻结 block 0–10，训练 block 11 |
| 输入 | 224×224 | `image_size=224`，`crop_pct=0.875` | 代码中通过 resize + center/random crop 实现 |
| batch size / epochs | 32 / 50 | 32 / 50 | 与论文一致 |
| optimizer | SGD，lr 0.003 | SGD，lr 0.003，momentum 0.9，weight decay 0.00005，CosineAnnealingLR | 仓库补充了 optimizer 细节和调度器 |
| views / augmentation | 2 views；random crop、flip、color jitter、Gaussian blur、rotation、posterize、sharpness | `n_views=2`，`MVTecTransformWithMaskTrain` 应用同类增强且 mask 同步增强 | 与论文描述一致 |
| classifier | multi-head，推理选择最小 loss 的 head | `n_head=4`；Task2 沿用 Task1 的 `use_etf=True` | ETF 是 Task1 改动；论文原表结果未使用 ETF，因此论文数值只作为参考，不做逐位复现要求 |
| loss weights | supervised loss 权重 λ=0.3；mean-entropy maximization 权重 µ=4 | `sup_weight=0.3`，`memax_weight=4` | 一致 |
| teacher / student temperature | τt 从 0.07 线性降到 0.04，前 40 epochs 每 4 epochs 更新；τs=0.1 | `warmup_teacher_temp=0.07`，`teacher_temp=0.04`，`warmup_teacher_temp_epochs=40`，`repeat_times=4`，student temp 0.1 | 一致 |
| MEBin | 采样 64 个阈值，最小稳定区间 τ=4 | `sample_rate=4`（等效 64 个采样点），`min_interval_len=4`，`erode=True` | 代码还使用 6×6 erosion kernel、1 次迭代 |
| cropping | 方形最小外接区域，padding 10%，最小 crop 为图像边长 1% | `crop_sub_image_mask` 中 `padding=0.1`、`min_crop_size=0.1` | `min_crop_size` 在代码中为 ratio 参数 |
| region merging | MVTec AD 的区域温度 τα=100 | MVTec 分支 `temps=[100]` | 一致 |

论文与当前代码存在两处温度记录差异，Task2 不做修改以避免引入无关变量：论文 Appendix B 记录 self-supervised contrastive τu=0.07、supervised contrastive τc=1.0；当前代码 `info_nce_logits` 默认 temperature 为 1.0，`SupConLoss` 默认 temperature/base temperature 为 0.07。Task2 baseline 和 Task2 都沿用当前代码默认值。

### 5.4 论文实验数据基准

论文 MVTec AD 实验使用 10 个 object 类和 5 个 texture 类，移除 combined anomaly class；unlabeled novel 数据为 MVTec AD 测试图像，labeled base 为去掉 normal class 的 AeBAD-S。MuSc + AnomalyNCD 的测试图像总数为 1725。

各产品测试图像数量已在服务器上核对；论文 Table 17 给出的 MuSc + AnomalyNCD 类别级结果如下。数值仅作为参考基准，Task2 主比较仍是同一代码/数据/seed 下的 baseline vs Task2。

| product | MVTec test images | paper NMI | paper ARI | paper F1 |
| --- | ---: | ---: | ---: | ---: |
| bottle | 83 | 0.613 | 0.583 | 0.819 |
| cable | 150 | 0.597 | 0.492 | 0.626 |
| capsule | 132 | 0.445 | 0.335 | 0.591 |
| carpet | 117 | 0.852 | 0.837 | 0.906 |
| grid | 78 | 0.622 | 0.578 | 0.731 |
| hazelnut | 110 | 0.662 | 0.582 | 0.718 |
| leather | 124 | 0.863 | 0.838 | 0.911 |
| metal_nut | 115 | 0.643 | 0.467 | 0.565 |
| pill | 167 | 0.439 | 0.291 | 0.513 |
| screw | 160 | 0.399 | 0.265 | 0.488 |
| tile | 117 | 0.885 | 0.850 | 0.940 |
| toothbrush | 42 | 0.368 | 0.259 | 0.762 |
| transistor | 100 | 0.531 | 0.421 | 0.620 |
| wood | 79 | 0.743 | 0.672 | 0.868 |
| zipper | 151 | 0.526 | 0.417 | 0.615 |
| **mean / total** | **1725** | **0.613** | **0.526** | **0.712** |

论文还报告 MEBin 在 MVTec AD 上的 FPR/FNR 分别为 0.153 / 0.035；使用 labeled abnormal data 相比不使用时，MVTec AD 的 NMI/ARI/F1 从 0.583 / 0.506 / 0.689 提升到 0.613 / 0.526 / 0.712。这些可作为诊断参考。

### 5.5 实验指标观测

当前训练循环默认只在最后一个 epoch 执行 sub-image prediction 和 region merging prediction，不在每个 epoch 做 validation。Task2 不修改该评估节奏，避免改变训练/评估语义；指标通过训练日志和 `metrics.csv` 记录。

| 观测层级 | 指标 / 字段 | 来源 | 观测方式 |
| --- | --- | --- | --- |
| 训练过程 | epoch、total loss、`cls_loss`、`cluster_loss`、`sup_con_loss`、`contrastive_loss` | `MGRL` 训练日志 | 检查 loss 是否发散、不同 head 的 cluster loss 是否稳定 |
| 数据选择 | 每个 defect type / `good` 的原始图数、选中数、实际比例、score range | `selection_json/{category}.json` | 启动后逐类检查筛选方向和数量 |
| 数据规模 | filtered/unfiltered labeled、unlabeled、test sub-image 数量 | 启动日志或手动脚本 | 确认只缩小训练集，不缩小测试集 |
| 子图分类 | sub-image `NMI / ARI / F1` | `sub_image_predict` 日志 | 作为诊断，不作为主结论 |
| 图像分类 | region-merged `NMI / ARI / F1` | `region_merge_predict` 日志与 `outputs/{runner_name}/metrics.csv` | 主结论指标 |
| 评估 head | 最小 cluster loss 的 head 编号 | `loss_list` / evaluation 日志 | 记录是否在类别间不稳定 |
| 效率 | GPU 编号、显存峰值、单类训练时长 | `nvidia-smi`、`tmux` / `nohup` 日志 | 控制共享服务器资源 |

每个类别运行完成后至少记录一行汇总：

```text
runner, setting, category, seed, train_unlabeled_before, train_unlabeled_after, test_images, test_subimages, head_id, subimage_NMI, subimage_ARI, subimage_F1, image_NMI, image_ARI, image_F1, train_seconds
```

其中 `image_NMI / image_ARI / image_F1` 是主指标。汇总时对 15 类取算术平均，并计算 Task2 − baseline 的差值；同时保留每类差值，避免平均掩盖小类退化。

### 5.6 对比实验

1. **Baseline**：
   - 使用 `configs/AnomalyNCD.yaml`，`train_subset.enabled=false`；
   - runner name 为 `mvtec_musc_crop`；
   - 如服务器上已存在同一代码版本、同一 seed 和同一数据路径的可信 baseline 结果，可复用；否则重跑 15 类。
2. **Task2**：
   - 使用 `configs/AnomalyNCD_task2.yaml`；
   - runner name 为 `mvtec_musc_crop_task2`；
   - 除训练样本筛选外，其余设置与 baseline 相同。
3. 主指标对比 `outputs/{runner_name}/metrics.csv` 中的 region merged image-level `NMI / ARI / F1`。
4. 辅助查看训练日志中的 sub-image prediction，但原因分析以主指标为准。
5. 可选诊断：若 Task2 显著下降，可在小类上额外分析样本数量、MEBin crop 数和分数分布；不默认加入主对比。

## 6. 验证方式

### 6.1 本地静态验证

1. 运行 `python3 -m py_compile examples/anomalyncd_main.py models/AnomalyNCD.py datasets/data_utils.py`。
2. 用小型内存 fixture 验证筛选函数：
   - defect type 按分数降序；
   - `good` 按分数升序；
   - 相同分数按文件名稳定排序；
   - 单样本类别至少保留 1 张；
   - 奇数样本时输出数量与记录的 `actual_ratio` 正确。
3. 检查默认配置仍为 `train_subset.enabled=false`，保证原流程不受影响。

### 6.2 服务器数据检查

1. 启动前重新确认目标数据集、anomaly map 和磁盘可用空间。
2. 对某一类别检查 `selection_json/{category}.json`：
   - defect type 的选中样本分数应不低于未选样本；
   - `good` 的选中样本分数应不高于未选样本；
   - selected prefix 必须能在裁剪图像目录中找到；
   - 测试 dataset 长度保持不变。
3. 打印过滤前后：
   - `len(train_dataset.labelled_dataset)`；
   - `len(train_dataset.unlabelled_dataset)`；
   - `len(test_dataset)`；
   - 每个 novel type 的过滤前后样本数。
4. 启动训练前检查异常图：
   - `data/mvtec_musc_anomaly_map` 下的 symlink 无 broken link；
   - 15 个类别合计 1725 个 map；
   - 每个 map 与 MVTec 测试原图一一对应；
   - 抽样读取 map 的 shape 为 `(900, 900)`、dtype 为 `uint8`、取值范围为 `[0, 255]`。

### 6.3 GPU 服务器纪律

1. 每次选卡前重新执行：
   ```bash
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
   ```
2. 只使用无进程且显存占用极低的 GPU，优先选择编号最小的空闲卡。
3. 多类实验可按实际空闲 GPU 分配，但必须通过 `tmux` 或 `nohup` 后台运行，避免终端断开。
4. 根分区剩余空间通常紧张；先确认磁盘余量，禁止重复下载数据集或重复保存大体积数据。
5. 每个实验完成后只保留日志、metrics、必要 checkpoint，不把任何数据、PNG 或模型文件提交 git。

### 6.4 结果验收标准

1. baseline 与 Task2 的 15 类 `metrics.csv` 均完整。
2. 汇总表包含每类的 `NMI / ARI / F1` 和 15 类平均值，以及 Task2 相对 baseline 的差值。
3. 结果结论明确分为提升、下降或几乎不变，并结合筛选机制给出可验证原因。
4. 计划通过 review 并执行后，将实际命令、运行结果、问题与结论记录到 `sessions/task2_session.md`，同时更新 `sessions/index.md`。

## 7. 指标变化的分析框架

结果完成后按以下机制归因，而不是只报告数值：

1. **样本纯度**：
   - 高分 defect 训练样本更可能包含真实异常区域，可能减少正常伪标签噪声；
   - 低分 `good` 训练样本更可能没有明显异常，可能降低正常类与缺陷类的混淆。
2. **样本多样性与数量**：
   - 筛选会减少约一半 novel 训练样本，可能导致小样本缺陷类别或外观变化大的类别欠拟合；
   - `WeightedRandomSampler` 的总采样数会随过滤后的训练集长度变化，需要记录每类筛选数量。
3. **分数质量**：
   - MuSc anomaly map 的分数跨 defect type 不一定可比；本方案按每个 type 内部排序，但 AD 方法漏检或误检仍会影响筛选；
   - 若高分 defect 中混入误检正常图，会强化错误伪标签；若低分 `good` 中包含弱异常，会引入难样本。
4. **与 MEBin / 伪标签机制的交互**：
   - 训练筛选使用 raw anomaly map 的全局最大值；
   - 训练中的 normal correction 使用 MEBin crop 后的分数，二者尺度与粒度不同，需分开解释；
   - 筛选可能改变每个 batch 中 labeled / unlabeled 与 normal / defect 的分布。
5. **类别语义**：
   - 若某些 defect 的分数与类型语义弱相关，只用高分样本可能丢失低但异常类型明确的样本；
   - `good` 保留低分样本有助于正常簇，但可能让正常类样本更同质，影响决策边界。
6. **评估口径**：
   - 测试集必须保持全量；若测试集被意外过滤，指标提升属于评估偏差而非真实提升；
   - 记录每类过滤前后的 test 长度作为否证检查。

## 8. 风险与依赖

1. 本地仓库当前没有数据集，无法直接运行完整训练；只能做静态验证，完整实验依赖服务器数据与空闲 GPU。
2. MVTec `good` 目录与 anomaly map 的 `good` 目录必须存在且文件名一一对应；服务器现有数据已满足数量对账要求，但执行前仍需做 broken symlink 与文件名映射检查。
3. 服务器异常图源目录位于 `/mnt/data/xrli`；若权限或挂载状态变化，需要重新确认可读性。
4. baseline 结果必须与当前代码、seed、数据路径、MEBin 输出一致；否则不能直接比较。
5. 服务器是共享环境，GPU 可能随时被占用；必须重新查询后选卡，禁止沿用文档中的历史 GPU 状态。
6. 训练产物较大，需遵守磁盘约束；异常图使用 symlink，不复制 PNG，不重复下载，不提交任何大文件。
7. `only_test` 路径仍会执行 `binarization()`；如复用已有裁剪数据，需确认不会误覆盖共享数据或与其他任务冲突。

## 9. 执行顺序

1. 用户 review 本计划。
2. 在 `feat_task_2` 分支实施配置、筛选工具、MEBin 清单生成和训练集过滤。
3. 新增异常图 symlink 准备脚本。
4. 本地静态验证与小型 fixture 测试。
5. 同步代码到服务器，确认分支、数据、磁盘和 GPU 状态。
6. 生成并检查 `data/mvtec_musc_anomaly_map` symlink 结构。
7. 先运行或复用 baseline，再运行 Task2 筛选集。
8. 检查 selection JSON、数据长度和日志。
9. 汇总 15 类指标并完成原因分析。
10. 创建 `sessions/task2_session.md`，更新 `sessions/index.md`。

## 10. 修改记录

- v1（2026-09-12）：创建 Task2 计划，明确只过滤 MVTec novel 训练集、保留全量测试集，并定义 baseline / Task2 对比与原因分析框架。
- v2（2026-09-12）：补充服务器已有 MVTec MuSc anomaly map 的来源、路径、数量对账、symlink 映射方案和验证步骤；明确 MuSc 为 zero-shot 推理，不需要重新训练。
- v3（2026-09-12）：根据论文 §4.1 与 Appendix A/B/C 补充论文实验设置、代码配置对照、MVTec 数据量和 MuSc 基准指标；新增训练过程与最终指标的观测方案。
- v4（2026-09-12）：按 review 完成代码实现：新增 train_subset 配置、selection 生成/过滤逻辑、异常图 symlink 准备脚本和 Task2 训练脚本；本地静态验证与筛选单元验证通过，服务器实验待执行。
