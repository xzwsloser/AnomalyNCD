#!/usr/bin/env bash
set -euo pipefail

# MVTec AD dataset
gpu="${gpu:-0}"
categories=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")
# input data path
dataset_path="${dataset_path:-/mnt/data/xrli/mvtec_anomaly_detection}"
anomaly_map_path="data/mvtec_musc_anomaly_map"
base_data_path="data/AeBAD_crop"
# output data path
binary_data_path="${binary_data_path:-data/mvtec_musc}" # the MEBin's output path of the binary data
crop_data_path="${crop_data_path:-data/mvtec_musc_crop}" # the MEBin's output path of the crop data
# experiment name
run_exp="mvtec_musc_crop"

# Baseline 复用服务器已有的 AeBAD_crop；已链接时该脚本是幂等的。
./scripts/link_server_aebad_crop.sh

if [[ ! -d "$anomaly_map_path/bottle" ]]; then
    echo "Anomaly map symlinks are missing. Run scripts/prepare_mvtec_musc_anomaly_maps.sh first." >&2
    exit 1
fi

for category in "${categories[@]}"
do
    CUDA_VISIBLE_DEVICES=$gpu python examples/anomalyncd_main.py \
        --runner_name "$run_exp" \
        --dataset "mvtec" \
        --category "$category" \
        --dataset_path "$dataset_path" \
        --anomaly_map_path "$anomaly_map_path" \
        --binary_data_path "$binary_data_path" \
        --crop_data_path "$crop_data_path" \
        --base_data_path "$base_data_path"
done


# # MTD dataset 
# gpu=0
# categories=("MTD")
# # input data path
# dataset_path="data/mtd_anomaly_detection/test"
# anomaly_map_path="data/mtd_musc_anomaly_map"
# base_data_path="data/AeBAD_crop"
# # output data path
# binary_data_path="data/mtd_musc"
# crop_data_path="data/mtd_musc_crop"
# run_exp="mtd_musc_crop"
# for category in "${categories[@]}"
# do
#     CUDA_VISIBLE_DEVICES=$gpu python examples/anomalyncd_main.py \
#         --runner_name "$run_exp" \
#         --dataset "mtd" \
#         --category "$category" \
#         --dataset_path "$dataset_path" \
#         --anomaly_map_path "$anomaly_map_path" \
#         --binary_data_path "$binary_data_path" \
#         --crop_data_path "$crop_data_path" \
#         --base_data_path "$base_data_path"
# done
