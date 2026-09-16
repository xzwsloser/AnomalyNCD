#!/usr/bin/env bash
set -euo pipefail

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
run_exp="mvtec_musc_crop_task3_amend"
config="configs/AnomalyNCD_amend.yaml"

./scripts/link_server_aebad_crop.sh

if [[ ! -d "$anomaly_map_path/bottle" ]]; then
    echo "Anomaly map symlinks are missing. Run scripts/prepare_mvtec_musc_anomaly_maps.sh first." >&2
    exit 1
fi

for category in "${categories[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" python examples/amend_main.py \
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
