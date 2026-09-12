# 任务1（Task1）执行计划 v2 —— 修复 visualize_tsne.py 模块导入路径

> 状态：已 review（用户明确确认：“可以，请帮我进行修改”）
> 创建时间：2026-09-12
> 对应记录：`sessions/task1_session_v2.md`
> 基于：`plans/task1_plan.md`（Task1 主体：ETF 模块 + 训练前 t-SNE 可视化，已完成）

## 1. 问题描述（服务器实测）
- 服务器运行 `python examples/visualize_tsne.py --dataset_path /home/xrli/data/mvtec_anomaly_detection` 时报错：
  `ModuleNotFoundError: No module named 'models'`
- 根因：直接运行脚本时 Python 仅把脚本所在目录 `examples/` 加入 `sys.path`，仓库根目录不在其中，导致
  `from models.modules.load_backbone import load_backbone` 失败。
- 对比：官方入口 `examples/anomalyncd_main.py` 第 3 行显式执行 `sys.path.append(os.getcwd())`，故能正常运行。
  本次新增的 `visualize_tsne.py` 遗漏了这一步。

## 2. 改动方案
| 文件 | 改动 |
| --- | --- |
| `examples/visualize_tsne.py` | 在文件顶部 import 区加入 `import sys`，并在 import 后添加 `sys.path.append(os.getcwd())`，与 `examples/anomalyncd_main.py` 保持一致，保证从仓库根目录以 `python examples/visualize_tsne.py` 方式启动时能导入 `models` 等顶层包 |

- 需确认目标脚本当前是否 `import os`（若未导入需一并补上）。
- 保持与官方入口相同的写法，加中文注释说明意图。

## 3. 验证方式
- 本机 `py_compile examples/visualize_tsne.py` 通过。
- 本机以 `PYTHONPATH` 模拟/直接 import 检查：`cd 仓库根 && python -c "import sys; sys.argv=['x']; exec(open('examples/visualize_tsne.py').read().split('if __name__')[0])"`（可选，仅验证 import 段不报错）。
- 服务器侧最终验证：用户从仓库根直接 `python examples/visualize_tsne.py --dataset_path ...` 不再报 `No module named 'models'`。

## 4. 风险与依赖
- 单行路径修复，风险极低。
- 依赖服务器端有 MVTec 数据（用户已提供 `/home/xrli/data/mvtec_anomaly_detection`）与 DINO 权重联网下载。
