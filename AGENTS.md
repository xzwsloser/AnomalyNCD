# AGENTS.md

本文件是 `AnomalyNCD` 仓库内所有 coding agent 的统一行为约束。任何在本仓库执行任务的 coding agent 都必须遵守下面的约定。若本文件与其他 `AGENTS.md` 或指令冲突，以本文件为准。

## 1. 任务执行流程（plans / sessions）

仓库通过 `plans/`（计划）与 `sessions/`（执行记录）两个目录管理任务的执行节奏，要求「先计划、后执行、再记录」。

### 1.1 核心规则

1. 每次用户要求**新建任务 / 执行任务**时，coding agent 必须**先编写执行计划（plan）**文档，等待用户 review 通过后，方可修改代码 / 执行命令。
2. 计划通过 review 并完成执行后，必须填写对应的**执行记录（session）**文档，记录实际改动、运行结果与问题。
3. 每一个任务在 `plans/` 与 `sessions/` 下各维护一个 `index.md`，用于建立「任务 → 文档」的索引关系。
4. 任务只在用户通过 prompt 明确「创建 / 执行某个任务」时才新建，不要在每次会话中自动新增任务。
5. 每个任务在 `plans/` 与 `sessions/` 下各维护 **一份** `taskN_plan.md` / `taskN_session.md`。同一任务的多轮修改（不同会话、多次执行）**不新建独立版本文件**，而是在同一份文档内通过「修改记录 / 更新记录」小节区分每一次改动，并同步更新对应 `index.md`。

### 1.2 目录与命名约定

```text
plans/
├── index.md            # 所有任务计划索引：任务名 / 文档 / 状态 / 创建时间
└── task1_plan.md       # 任务1 的执行计划（执行前编写，需用户 review）

sessions/
├── index.md            # 所有任务执行记录索引：任务名 / 文档 / 状态 / 时间
└── task1_session.md    # 任务1 的执行记录（执行后填写）
```

- 任务编号使用「任务N」（如 任务1、任务2、任务3）；文档文件名统一使用英文小写 `taskN_*` 前缀。同一任务的多轮改动在文件内用「修改记录」小节区分即可，**不新建版本文件**（例如不出现 `taskN_plan_v2.md`）。
- **plan 文档**必须包含：目标、背景、改动方案（涉及文件与具体修改）、验证方式、风险与依赖。
- **session 文档**必须包含：实际改动内容、涉及文件、运行 / 验证结果、遇到的问题与解决方式、遗留事项。

## 2. 仓库结构

本项目为论文 *AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios*（arXiv:2410.14379）的官方 PyTorch 实现。

```text
AnomalyNCD/
├── AGENTS.md                        # 本约束文件
├── README.md                        # 项目说明（方法、数据集、运行方式）
├── LICENSE
├── requirements.txt                 # Python 依赖（torch/timm 由外部安装，见 README）
├── configs/
│   └── AnomalyNCD.yaml              # 训练 / 模型 / 损失 / 实验超参数
├── datasets/
│   ├── dataset.py                   # 数据集定义与加载
│   ├── data_utils.py                # 类别划分、数据加载、伪标签权重等工具
│   ├── transform.py                 # 数据增强与多视角生成
│   ├── aebad_preprocess.py          # AeBAD 数据集预处理
│   └── mtd_preprocess.py            # MTD 数据集预处理
├── models/
│   ├── AnomalyNCD.py                # 主模型：训练 / 推理 / 评估主逻辑
│   ├── loss/
│   │   ├── _contrastive_loss.py     # 对比损失（info-nce、SupCon）
│   │   └── _distill_loss.py         # 蒸馏损失（teacher-student / pseudo-label）
│   └── modules/
│       ├── _MEBin.py                # 主元素二值化 MEBin
│       ├── _MGViT.py                # Mask-Guided Vision Transformer（掩码引导主干）
│       ├── _classifier.py           # 分类头 MultiHead（当前为 weight-norm, DINO/SwAV 风格）
│       └── load_backbone.py         # 预训练主干加载（DINO ViT）
├── examples/
│   └── anomalyncd_main.py           # 命令行入口（解析参数并启动 AnomalyNCD）
├── scripts/
│   ├── anomalyncd.sh                # 训练脚本（MVTec AD / MTD）
│   └── anomalyncd_test.sh           # 推理脚本（加载 checkpoint 测试）
├── utils/
│   ├── general_utils.py             # 配置加载、实验初始化、日志等通用工具
│   └── cluster_and_log_utils.py     # 聚类评估（NMI/ARI/F1）与日志工具
├── assets/                          # 项目相关静态资源
├── papers/                          # 论文 PDF（见 §3 对应关系）
├── reference/                       # 参考实现代码（见 §3 对应关系）
├── plans/                           # 任务执行计划
└── sessions/                        # 任务执行记录
```

## 3. papers 目录与论文对应关系

`papers/` 目录存放论文 PDF，`reference/` 存放对应论文的官方参考实现。后续 coding agent 若需参考相关论文方法，请按下表定位：

| papers 文件 | 论文名称 | 说明 | 对应参考实现 |
| --- | --- | --- | --- |
| `papers/2308.02989v3.pdf` | *Novel Class Discovery for Long-tailed Recognition*（NCDLR，TMLR 2023，arXiv:2308.02989） | 提出 equiangular（等角）prototype 的 ETF 分类头，用于长尾场景的类别发现 | `reference/NCDLR/` |

- **ETF 模块**：`reference/NCDLR/nets/vit.py` 中的 `ETF_Classifier`（含 `generate_random_orthogonal_matrix`），构造等角分类 prototype 矩阵 `ori_M`，前向时对特征归一化后与 `ori_M` 做矩阵乘法得到 logits。当前主仓库的 `models/modules/_classifier.py` 仍是 weight-norm 风格，尚未采用 ETF，是后续改造目标。
- 主仓库自身对应论文 *AnomalyNCD*（arXiv:2410.14379），其内容在 `README.md`，不在 `papers/` 目录内。

## 4. 代码风格约束

1. 代码风格保持干净、简洁，遵循仓库现有组织方式与命名习惯。
2. **关键位置必须添加中文注释**，说明逻辑意图与实现依据（如对应论文的哪个模块）。
3. 改动保持最小化、聚焦当前任务，不做无关的大范围重构。
4. 新增代码按功能放入对应目录：模型 → `models/`，工具 → `utils/`，数据集 → `datasets/`，可视化可新增到 `utils/`。
5. 涉及引用外部实现（如 ETF）时，保留来源出处注释。

## 5. 运行与验证参考

- 配置文件：`configs/AnomalyNCD.yaml`。
- 入口脚本：`python examples/anomalyncd_main.py --config configs/AnomalyNCD.yaml`。
- 训练：`scripts/anomalyncd.sh`；推理：`scripts/anomalyncd_test.sh`。
- 依赖：见 `requirements.txt`（`torch`、`timm` 需自行按 README 安装）。


## 6. 训练服务器（GPU）资源与约束

> 训练/推理均在本仓库本地 PC 之外的目标服务器上进行。以下信息为该服务器资源与使用纪律，供 coding agent 连接与调度 GPU 时遵守。

### 6.1 机器与连接
- 训练/推理在目标服务器上进行（连接方式见被 `.gitignore` 忽略的本地私有文件 `server_private.private.md`，勿提交到 git）。
- 服务器是**共享机器**：有 **17 个用户**登录、系统负载较高（load average 约为 16），且**存在其他用户/任务正在运行**。

### 6.2 GPU 资源（2026-09-12 实测）
- 共 **4 张 NVIDIA RTX 3090**（各 24GB 显存），Driver 525.105.17、CUDA 12.0：
  - `GPU 0`：空闲
  - `GPU 1`：空闲
  - `GPU 2`：空闲
  - `GPU 3`：空闲
- 本轮检查时刻 `nvidia-smi` 显示 4 张 GPU 均 **0% 利用、无运行进程**，均可选用。

### 6.3 磁盘约束
- 根分区 `/` 使用率约 **97%**，仅剩约 **30GB** 可用。训练输出、checkpoint、日志需控制体积，避免写满磁盘。

### 6.4 使用纪律（强制）
- **选卡前必须重新查询** `nvidia-smi`，确认实际空闲的 GPU，而不是直接沿用本文档记录（资源随时可能变化）。
- **务必只使用空闲 GPU，避免占用其他用户正在使用的 GPU**；优先选择编号最小的空闲卡（建议默认 `CUDA_VISIBLE_DEVICES=0`，若 0 被占用则顺延到下一个空闲卡）。
- 启动训练/推理任务前，须在服务器上运行 `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` 确认可用性与内存（无运行进程且内存占用极低）。连接命令见私有文件 `server_private.private.md`。
- 长任务建议通过 `tmux` / `nohup` 在服务器后台运行，避免终端断开导致中断。
- 注意规避共享冲突：加载负载本身很高，训练启动前确认目标 GPU 仍为空闲。

### 6.5 安全与版本控制说明
- `AGENTS.md`、`plans/`、`sessions/` 属于治理文档，**纳入 git** 以便提交 github 后在服务器拉取。
- 服务器**连接信息（IP、SSH 命令、用户名、私钥路径）属于敏感信息**，一律放在被 `.gitignore` 忽略的 `server_private.private.md`，**不得写入任何会被 git 跟踪的文件**。coding agent 请严格遵守。
- 大体积数据与产物（`data/`、`outputs/`、`*.pt`、`*.png` 等）已被 `.gitignore` 忽略，勿提交。

## 7. 数据集与多任务分支协作

### 7.1 项目里没有自动下载脚本
- 本仓库**不含数据集自动下载逻辑**。数据集需手动放置到 `./data` 目录（README §Datasets 提供下载链接），脚本通过 `--dataset_path` / `data/...` 路径引用（见 `scripts/anomalyncd.sh`）。
- MVTec AD 官方链接、anomaly map（MuSc 等）与 `AeBAD_crop` 的 Google Drive 链接见 `README.md` 的 Datasets 与 anomaly maps 小节。

### 7.2 每任务一个 git 分支
- 服务器仓库位于 `~/anomaly_ncd/AnomalyNCD`，每个任务在独立分支上完成（Task1 已有 `feat_task_1` 分支）。
- 建议分支命名：`feat_taskN`（如 `feat_task_1`、`feat_task_2`）。
- 所有任务共享**同一个工作区**，通过在同一个 repo 内 `git checkout` 切换分支。

### 7.3 避免重复下载 / 磁盘浪费（磁盘仅剩约 30GB）
- 大体积数据（MVTec、AeBAD、anomaly maps、MEBin 中间产物）放**共享 untracked 路径**，不要放进某个分支提交。
- 未跟踪目录在一个工作区里跨分支切换时**不会变化**，因此只在共享工作区放置一次即可，切换分支不会产生重复。
- 推荐做法：
  1. 在服务器**下载并解压一次**，放到 `data_store` 或共享数据根（如 `~/data_store/AnomalyNCD_data/`）。
  2. 用 `data` 符号链接指向共享数据根（`ln -s`），或通过脚本参数 `--dataset_path` 指向共享数据根，避免每个分支各自放一份。
  3. 把 `data/`、`outputs/`、`*.pt`、`*.png` 等大文件加入 `.gitignore`，严禁提交进分支。
  4. 优先复用服务器上**已存在的数据集**（其他用户目录下的 MVTec 副本等），经确认后通过 symlink 复用，而不是重新下载。
- coding agent 每次启动前需重新确认目标数据集是否存在及磁盘可用空间。

### 7.4 git 提交纪律
- coding agent 写文档/代码后不应擅自提交；确需提交时先与用户确认分支与 commit 信息。
- 不得将敏感信息（服务器 SSH、私钥路径等）写入会被 git 跟踪的文件。
