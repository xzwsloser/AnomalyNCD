# 任务1（Task1）执行记录

> 状态：**代码已实现 + 静态验证通过；训练前可视化待服务器运行**
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
