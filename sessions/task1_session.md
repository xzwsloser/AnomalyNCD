# 任务1（Task1）执行记录

> 状态：已完成（含服务器运行结果回填与图片分析，见 §7 修改记录 v5）
> 创建时间：2026-09-12
> 对应计划：`plans/task1_plan.md`

## 1. 任务概述
- 将 classifier 初始化改为论文《Novel Class Discovery for Long-tailed Recognition》中的 **ETF（等角原型）模块**。
- 训练之前将图像特征与 classifier 中 prototype 用 t-SNE 可视化到一张图。
- 已确认范围：**不需要训练**，仅做「训练前 ETF + t-SNE」。

## 2. 实际改动
| 文件 | 改动说明 |
| --- | --- |
| `models/modules/_classifier.py` | 新增 `ETF_Classifier`（等角紧框架金字塔，含 `_generate_orthogonal_matrix`）；`MultiHead` 增加 `use_etf` 开关，ETF 模式下 `last_layer` 替换为固定等角原型矩阵 `ori_M`，并暴露 `prototypes()` |
| `models/AnomalyNCD.py` | `load_model` 创建 projector 时传入 `use_etf=self.args.use_etf` |
| `examples/anomalyncd_main.py` | 从配置读取 `use_etf` 到 args |
| `configs/AnomalyNCD.yaml` | `models` 下新增 `use_etf: True` |
| `examples/visualize_tsne.py` | 新增训练前 t-SNE 可视化脚本（预训练 MGViT 特征 + ETF 原型单图，输出到 `outputs/`） |
| `.gitignore` | 更新：忽略 `data/`、`outputs/`、模型/缓存等；`AGENTS.md`/`plans/`/`sessions/` 改为纳入 git；服务器敏感信息拆到私有文件 |

## 3. 静态验证结果（AnomalyNCD conda 环境，CPU）
- `ETF_Classifier`：`ori_M` 形状 `(768, 15)`，行归一化后 Gram 矩阵非对角最大幅值 `0.0714 = 1/(C-1)`，符合等角紧框架性质。
- `MultiHead(use_etf=True)` 保持接口：返回 `(x_proj shape (B,256), n_head=4 组 logits shape (B,15))`；多 head 共享同一等角矩阵（各 head logits 一致）。
- `prototypes()` 返回 `(num_classes, feat_dim)`，ETF 与非 ETF 两种模式一致。
- 4 个改动文件 `py_compile` 通过。
- ETF 权重为 buffer（固定不训练），梯度会完全流向特征主干，符合设计预期。

## 4. 服务器运行待办（用户侧）
- 在 `feat_task_1` 分支拉取代码，放置 MVTec AD 数据集（优先复用服务器已有数据）。
- 运行训练前可视化：`python examples/visualize_tsne.py --dataset_path data/mvtec_anomaly_detection --output outputs/task1_tsne.png`
- 确认输出单图中同时包含图像特征点与 ETF 原型点。

## 5. 问题与说明
- 本地无 CUDA，仅做 CPU 静态验证；完整可视化需服务器 GPU（MGViT 的 `prepare_mask` 含 `.cuda()` 硬编码）。
- `models/AnomalyNCD.py` 在改动前已存在未提交的空格清理（历史改动），与本次 `use_etf` 改动同在工作区，提交时请留意。
- 原始仓库部分文件为 CRLF，已对 `configs/AnomalyNCD.yaml`、`examples/anomalyncd_main.py` 保持 CRLF，diff 仅含意图改动。

## 6. 遗留事项
- 训练前可视化的实际效果图需在服务器跑出后回填确认。
- 多 head 处理采用「共享同一个等角矩阵」方案，如需改动可后续调整。

## 7. 修改记录（多轮执行，单文件维护）
> 同一任务的多次改动统一记录在本文件，不另建版本文件（规则见 AGENTS.md §1.1 第 5 条）。

### v2（2026-09-12）修复可视化脚本模块导入路径
- 问题：服务器运行 `python examples/visualize_tsne.py --dataset_path ...` 报 `ModuleNotFoundError: No module named 'models'`。
- 改动：`examples/visualize_tsne.py` 顶部新增 `import sys` + `sys.path.append(os.getcwd())`，与 `anomalyncd_main.py` 一致。
- 验证：本机 `python3 -m py_compile` 通过；服务器待重跑确认。

### v3（2026-09-12）修复可视化脚本 mask 维度
- 问题：特征抽取报 `ValueError: not enough values to unpack (expected 4, got 3)`，定位 `_MGViT.py::prepare_mask`。
- 根因：脚本传 3D mask `[1,H,W]`，而 `prepare_mask` 的 `AvgPool2d` 需 4D `[1,1,H,W]`。
- 改动：`examples/visualize_tsne.py::extract_features` 中 mask 改为 `torch.ones(1, 1, H, W)`。
- 验证：本机 `python3 -m py_compile` 通过；服务器待重跑确认。

### v4（2026-09-12）添加进度可视化（进度条 + 分阶段耗时）
- 问题：特征抽取为静默逐张循环，运行中看不到进度与已消耗时长。
- 改动：`examples/visualize_tsne.py`
  - 新增 `import time`、`from tqdm import tqdm`。
  - `extract_features` 用 `tqdm` 显示进度条（进度、s/it、ETA），并打印总抽取耗时。
  - `main()` 打印并计时各阶段：主干加载（含 DINO 下载）、特征抽取、t-SNE 降维、绘图，以及总耗时。
- 验证：本机 `python3 -m py_compile` 通过；服务器重跑应能看到进度条与分阶段耗时。
- 遗留：服务器端实际输出效果待回填确认。

### v5（2026-09-12）服务器运行结果回填与 t-SNE 图分析
- 运行命令（服务器，DCproject 环境，feat_task_1 分支）：
  `python examples/visualize_tsne.py --dataset_path /home/xrli/data/mvtec_anomaly_detection --output outputs/task1_tsne.png`
- 运行结果：成功生成 `outputs/task1_tsne.png`（1483×1182 PNG），任务一「训练前 ETF + t-SNE」交付物完成。
- 图片内容：150 个彩色散点 = MVTec AD 15 类各 10 张测试图像的预训练 DINO CLS 特征（768 维 → t-SNE 2D）；15 个红色星形（编号 1–15）= ETF 分类头的等角原型，与特征同平面投影。
- 分析结论：
  1. 15 类预训练特征在 2D t-SNE 中各自形成清晰、紧凑、互不重叠的簇，类间区分度很高——这是无标签类别发现的良好起点。
  2. 15 个 ETF 原型聚集在图中心一小团。这是等角紧框架的几何性质（768 维中范数相同、两两夹角相等，即"高维等距"），t-SNE 无法在 2D 保持 15 点等距而压缩成团；此前静态验证已确认 Gram 矩阵非对角幅值 = 1/(C-1)，结构正确，非 bug。
  3. 原型与任何一类特征簇均未对齐（位于中央、离各簇较远），符合"训练前"的预期起点：ETF 原型初始化为随机等角矩阵，与特征分布无对齐关系；训练目标即让各类特征簇向其对应原型收敛。
- 可选改进（非本次必改）：将 15 个原型按所属类别着色（与特征点同色），训练后再次绘制可直观对比特征向原型收敛的效果。
- 遗留：训练后的对比可视化与训练流程不在 Task1 范围；`models/AnomalyNCD.py` 仍存在改动前遗留的未提交空格清理，提交时留意。
