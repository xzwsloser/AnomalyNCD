#!/usr/bin/env bash
set -euo pipefail

# Task4+5 断点续训脚本：逐 category 检查已保存的 model.pt（每 epoch 覆盖保存），
# 做三选一动作：
#   - 已完成（epoch >= 配置 epochs） => 跳过；
#   - 中断待续（epoch < epochs，新代码已有的中间 checkpoint） => 追加 --resume 续训；
#   - 未开始 / 无 checkpoint（旧代码跑过且未完成） => 从零重跑。
# 中途再次中断只需重新执行本脚本即可，幂等。
gpu="${gpu:-0}"
categories=(
    bottle cable capsule carpet grid hazelnut leather metal_nut
    pill screw tile toothbrush transistor wood zipper
)
dataset_path="${dataset_path:-/mnt/data/xrli/mvtec_anomaly_detection}"
anomaly_map_path="data/mvtec_musc_anomaly_map"
base_data_path="data/AeBAD_crop"
binary_data_path="${binary_data_path:-data/mvtec_musc}"
crop_data_path="${crop_data_path:-data/mvtec_musc_crop}"
run_exp="mvtec_musc_crop_task45"
config="configs/AnomalyNCD_task45.yaml"

# 从配置文件读取 epochs（用于判断类别是否训练完成）
epochs="$(sed -n 's/^[[:space:]]*epochs:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$config" | head -1)"
if [[ -z "${epochs:-}" ]]; then
    echo "ERROR: cannot read 'epochs' from $config" >&2
    exit 1
fi

./scripts/link_server_aebad_crop.sh

if [[ ! -d "$anomaly_map_path/bottle" ]]; then
    echo "Anomaly map symlinks are missing. Run scripts/prepare_mvtec_musc_anomaly_maps.sh first." >&2
    exit 1
fi

# 判断一个类别该跳过(skip)/续训(resume)/重跑(fresh)：输出对应动作名。
# 用法：action="$(classify '<latest model.pt or empty>' '<epochs>')"
classify() {
    local pt="$1" max_epochs="$2" res
    if [[ -z "$pt" ]]; then
        echo "fresh"
        return
    fi
    res="$(python - "$pt" "$max_epochs" <<'PY'
import sys
import torch
pt, max_epochs = sys.argv[1], int(sys.argv[2])
ck = torch.load(pt, map_location='cpu')
epoch = int(ck.get('epoch', 0))
ck_epochs = ck.get('epochs')          # 新代码保存；旧代码无此 key
if ck_epochs is not None:
    ck_epochs = int(ck_epochs)
# 缺 epochs key 说明是旧代码生成的 checkpoint：旧代码只在最后 epoch 保存 => 已完成，跳过。
if ck_epochs is None or epoch >= max(ck_epochs, max_epochs):
    print('skip')
else:
    print('resume')
PY
    )"
    echo "$res"
}

for category in "${categories[@]}"
do
    # 取该类别最新一份 checkpoint（按 mtime 排序）
    latest_pt="$(ls -td outputs/${run_exp}/log/*${category}_*/checkpoints/model.pt 2>/dev/null | head -1 || true)"
    action="$(classify "$latest_pt" "$epochs")"

    if [[ "$action" == "skip" ]]; then
        echo "[$(date '+%F %T')] skip $category (already finished: $latest_pt)"
        continue
    fi

    args_extra=()
    if [[ "$action" == "resume" ]]; then
        latest_dir="$(dirname "$latest_pt")"
        echo "[$(date '+%F %T')] resume $category from $latest_pt"
        args_extra=(--resume "$latest_dir")
    else
        echo "[$(date '+%F %T')] run $category (fresh)"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" python examples/anomalyncd_main.py \
        --config "$config" \
        --runner_name "$run_exp" \
        --dataset "mvtec" \
        --category "$category" \
        --dataset_path "$dataset_path" \
        --anomaly_map_path "$anomaly_map_path" \
        --binary_data_path "$binary_data_path" \
        --crop_data_path "$crop_data_path" \
        --base_data_path "$base_data_path" \
        "${args_extra[@]}"
done

echo "[$(date '+%F %T')] All categories processed."
