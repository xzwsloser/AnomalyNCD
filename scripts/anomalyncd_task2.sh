#!/usr/bin/env bash
set -euo pipefail

gpu="${gpu:-0}"
categories=(
    bottle cable capsule carpet grid hazelnut leather metal_nut
    pill screw tile toothbrush transistor wood zipper
)

# MVTec AD 数据
dataset_path="${dataset_path:-/mnt/data/xrli/mvtec_anomaly_detection}"
anomaly_map_path="data/mvtec_musc_anomaly_map"
base_data_path="data/AeBAD_crop"

# MEBin 输出保持共享 untracked 数据路径，避免重复占用磁盘。
binary_data_path="${binary_data_path:-data/mvtec_musc}"
crop_data_path="${crop_data_path:-data/mvtec_musc_crop}"

# 实验名与 baseline 分开，便于对比 metrics。
run_exp="mvtec_musc_crop_task2"
config="configs/AnomalyNCD_task2.yaml"

# 启动前自动复用服务器上的 AeBAD_crop；已存在且合法时该脚本是幂等的。
./scripts/link_server_aebad_crop.sh

if [[ ! -d "$anomaly_map_path/bottle" ]]; then
    echo "Anomaly map symlinks are missing. Run scripts/prepare_mvtec_musc_anomaly_maps.sh first." >&2
    exit 1
fi

for category in "${categories[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" python examples/anomalyncd_main.py \
        --config "$config" \
        --runner_name "$run_exp" \
        --dataset "mvtec" \
        --category "$category" \
        --dataset_path "$dataset_path" \
        --anomaly_map_path "$anomaly_map_path" \
        --binary_data_path "$binary_data_path" \
        --crop_data_path "$crop_data_path" \
        --base_data_path "$base_data_path"
done
