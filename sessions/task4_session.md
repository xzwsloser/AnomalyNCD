# Task4 & Task5 执行记录：TAC 文本引导融合

> 对应计划：`plans/task4_plan.md`（任务5 已并入任务4）。
> 分支：`feat_task_4`。

- 创建：2026-09-16
- 状态：**代码实现与本机语法验证完成；服务器训练/实验待执行**

---

## 1. 本次执行内容

依据计划 §6 改动方案，完成 Task4（Text Counterpart Construction + 特征拼接）与
Task5（Cross-modal Mutual Distillation）的代码实现，全部为非端到端（文本特征离线冻结）。

### 1.1 新增文件
- `models/modules/_tac_text.py`
  - 离线流水线：`encode_nouns` / `encode_images` / `filter_nouns`（faiss spherical k-means +
    反向分类挑选 topK 判别名词）/ `retrieve_text`（softmax(feat·noun^T/tau)@noun + L2 归一化）/
    `build_or_load`（幂等落盘 `{category}.npy` + `{category}_image_paths.json`）。
  - 可训练文本分支：`TextProjector`（可选线性投影）、`TextClusterHead`（TAC `ClusterHead`
    文本分支：Linear→BN→ReLU→Linear→Softmax，用于 Task5）。
  - CLIP / faiss 仅在离线构建路径内按需导入，文本引导关闭时不引入依赖。
- `models/loss/_tac_loss.py`
  - `TACDistillLoss`（InfoNCE 式跨模态蒸馏，`mask` 按实际设备惰性构建）、
    `consistency_loss`（分配一致性 binary CE）、`entropy`（负熵防退化）。
- `configs/AnomalyNCD_task4.yaml`（`text_counterpart.enabled=True`、`concat_in_features=True`、
  `cmd.enabled=False`）。
- `configs/AnomalyNCD_task5.yaml`（`text_counterpart.enabled=True`、`concat_in_features=False`、
  `cmd.enabled=True`）。
- `scripts/build_text_counterpart.sh`（按 category 离线构建文本对应，输出到共享 untracked 根）。
- `scripts/anomalyncd_task4.sh` / `scripts/anomalyncd_task5.sh`（逐 category 运行 Task4 / Task5）。

### 1.2 修改文件
- `models/AnomalyNCD.py`
  - `load_model`：启用 `concat_text` 时 `MultiHead.in_dim = feat_dim + text_dim`（768+512=1280）；
    新增 `text_projector` / `text_cluster_head` / `text_modules` 文本分支（独立于
    `nn.Sequential(MGViT, projector)`，保证 state_dict 兼容）。
  - 新增 `trainable_params`（图像分支 + 文本分支参数分组）与 `build_text_counterpart`
    （主流程 `main()` 中裁剪后、训练前调用，幂等）。
  - `MGRL`：batch 注入 `text_feat`、多视图时按 2 份复制、Task4 拼接、Task5 计算 CMD 损失
    `w_cmd*(L_distill+L_consist - λ·entropy)`。
  - `sub_image_predict` / `region_merge_predict`：启用拼接时同样取文本特征并拼接（推理只走
    图像分支，文本分支仅训练阶段使用，符合计划第一版约定）。
  - `main()`：保存/加载文本分支状态；`only_test` 时一并载入 `text_modules`。
- `datasets/dataset.py`
  - `Dataset_AnomalyNCD` 支持 `use_text_feat` / `text_feat_root`，懒加载
    `{root}/{novel_class}.npy + _image_paths.json` 构建 path→向量 字典，`__getitem__` 追加
    `text_feat`（文本关闭时不追加，保持 baseline 与 Task3 兼容）。
  - `get_anomalyncd_datasets` 透传文本开关与根路径。
- `datasets/data_utils.py`
  - `get_datasets` 透传；`MergedDataset` 支持文本特征透传（关闭时保持 5 元返回，不破坏 Task1-3）。
- `examples/anomalyncd_main.py`
  - `load_args` 解析 `text_counterpart` / `cmd` 新配置段。
- `configs/AnomalyNCD.yaml`
  - 新增默认关闭的 `text_counterpart` 与 `cmd` 段，保持 baseline 可复现。

## 2. 涉及文件清单
- 新增：`models/modules/_tac_text.py`、`models/loss/_tac_loss.py`、
  `configs/AnomalyNCD_task4.yaml`、`configs/AnomalyNCD_task5.yaml`、
  `configs/AnomalyNCD_task45.yaml`（Task4+5 叠加）、
  `scripts/build_text_counterpart.sh`、`scripts/anomalyncd_task4.sh`、`scripts/anomalyncd_task5.sh`、
  `scripts/anomalyncd_task45.sh`（Task4+5 叠加）
- 修改：`models/AnomalyNCD.py`、`datasets/dataset.py`、`datasets/data_utils.py`、
  `examples/anomalyncd_main.py`、`configs/AnomalyNCD.yaml`

## 3. 运行 / 验证结果
- 所有改动的 Python 模块通过 `python -m py_compile` 语法校验；Bash 脚本通过 `bash -n` 校验。
- 本机环境不含 `torch` / `numpy` / `yaml` / `clip` / `faiss`，且无 MVTec/裁剪数据，
  故无法在本机执行完整离线构建与训练/推理 smoke test。
- 按计划 §7，服务器侧验证待执行：单 category、`text_counterpart.enabled=True`（Task4）与
  `cmd.enabled=True`（Task5）各跑 1–2 epoch，确认 loss 正常且含 CMD 分量，再做 15 category 正式实验。

## 4. 遇到的问题与解决方式
- 设备初始化顺序：`build_text_counterpart_` 在 `main()` 中需于 `train_init` 之前调用，而
  `self.device` 原本只在 `train_init` 中设置 → 将设备初始化提前到 `AnomalyNCD.__init__`。
- `MergedDataset` / `Dataset_AnomalyNCD` 返回元素数量：为不破坏 Task1-3（尤其 Task3 的
  `amend.py` 仍按 5 元解包），文本特征仅在 `use_text_feat=True` 时追加到返回元组末尾。
- `TACDistillLoss` 的 mask 设备：改为在 `forward` 中按输入设备惰性构建，避免旧 PyTorch
  迁移歧义。

## 5. 遗留事项 / 下一步
- 服务器：安装/确认 `clip`、`faiss-gpu`（匹配 CUDA 12.0）、CLIP 权重可访问。
- 服务器先跑 Task4 / Task5 smoke test，再跑 15 category 正式实验并记录 region-merged
  NMI/ARI/F1，与 baseline（task1）及 task2 对比分析文本引导对各类缺陷的增益/失效原因。
- （可选）端到端开关与邻居挖掘（`mine_nearest_neighbors`）按计划作为后续可选增强保留。

## 6. 修改记录
- 2026-09-16：创建本记录；代码实现与语法验证完成，服务器实验待执行。
- 2026-09-16：应需求补充 Task4+5 叠加运行的组合配置
  `configs/AnomalyNCD_task45.yaml`（`concat_in_features=True` 且 `cmd.enabled=True`）
  与运行脚本 `scripts/anomalyncd_task45.sh`；`concat_text` 与 `cmd_enabled` 两个开关相互独立，
  可在同一次训练中同时启用（拼接仅用于图像分支特征，CMD 仅在训练阶段、推理不走文本分支）。
- 2026-09-17：解决服务器 CLIP 下载挂起问题。
  - 根因：服务器无外网，`open_clip` 默认从 `huggingface.co` 下载 ViT-B-32 权重，
    出现 `Network unreachable` 并重试挂起直至 `KeyboardInterrupt`。
  - 修复：`models/modules/_tac_text.py::build_or_load` 增加本地 checkpoint 自动探测
    （`data_store/clip/ViT-B-32.pt`，`CLIP_CHECKPOINT` 仍可覆盖）；服务器 repo 建立
    `data_store` → `/home/dachuang/data_store/AnomalyNCD_data` 符号链接。
  - 验证：open_clip 2.32.0 离线加载该 openai JIT 权重并 `encode_text` 正常；
    `scripts/build_text_counterpart.sh` 在 DCproject + GPU2 后台跑 bottle，CLIP 本地
    加载成功、GPU2 利用率正常、不再联网挂起（余下 category 后台继续）。
  - 说明：修复文件已同步到服务器 `feat_task_4` 工作区，未提交。
