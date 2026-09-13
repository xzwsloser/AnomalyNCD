# 任务3执行记录：下载服务器日志与 CSV 输出

## 执行时间

- 开始时间：2026-09-13
- 结束时间：2026-09-13

## 实际改动内容

1. 按计划先创建 `plans/task3_plan.md`；发现任务2已被仓库既有工作占用后，删除误建的重复计划文件并改用任务3。
2. 根据用户更新后的指令，取消两个约 10.95 GB 的 `.zip` 输出文件下载，任务范围改为只下载日志文件与 CSV 文件。
3. 连接服务器确认 `outputs/` 下共有 30 个 `log.txt` 日志文件和 2 个 `metrics.csv` 文件。
4. 使用 SSH 流式 tar 传输并本地解包，仅包含 `*.log`、`*.txt`、`*.csv` 匹配文件；解包时逐文件显示名称，未传输 checkpoint 或权重文件。
5. 更新 `plans/index.md` 与 `sessions/index.md` 中任务3的状态。

## 涉及文件

### 新增

- `plans/task3_plan.md`
- `sessions/task3_session.md`
- `outputs/mvtec_musc_crop/metrics.csv`
- `outputs/mvtec_musc_crop_task2/metrics.csv`
- `outputs/mvtec_musc_crop/log/*/log.txt`
- `outputs/mvtec_musc_crop_task2/log/*/log.txt`

### 更新

- `plans/index.md`
- `sessions/index.md`

## 运行 / 验证结果

1. 服务器端筛选结果显示目标文件为 30 个日志文件和 2 个 CSV 文件。
2. 日志目录内同时存在 `checkpoints/model.pt`，已在下载清单中显式排除。
3. 下载完成后本地 SHA-256 校验列表与服务器端一致。
4. 本地 `outputs/` 下目标文件数量为 32，总有效数据量约 4.06 MB。
5. 本地未发现 `.zip`、`.pt`、`.pth`、`.ckpt` 文件。
6. `git status` 未显示 `outputs/` 下文件进入跟踪范围；未将服务器 IP、用户名、私钥路径写入 git 跟踪文档。

## 遇到的问题与解决方式

1. 本机 SSH 首次连接时受 `/etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf` 权限问题影响；改用 `-F /dev/null` 跳过系统级 SSH 配置后连接成功。
2. 初始 zip 文件传输速度约 0.4 MB/s，预计耗时约 4 小时；按用户指令中止传输并改为只下载日志/CSV。
3. 日志目录内存在大量 `model.pt` checkpoint；改用服务器端按文件名筛选后打包，避免递归复制目录时误下载权重。

## 遗留事项

- 用户最初要求下载的两个 `.zip` 压缩包仍未下载；如后续需要，可另行确认后再执行。
- 服务器上的 checkpoint 与权重文件均未下载，仍保留在服务器端。
