#!/usr/bin/env bash
set -euo pipefail

# 离线构建 MVTec AD 各 category 子图的文本对应特征（Task4/5 前置）。
# 依赖：服务器需安装 clip 与 faiss；子图需先经 anomalyncd 流程裁剪生成。
# 输出写在共享 untracked 路径，幂等（已存在则跳过）。

categories=(
    bottle cable capsule carpet grid hazelnut leather metal_nut
    pill screw tile toothbrush transistor wood zipper
)
crop_data_path="${crop_data_path:-data/mvtec_musc_crop}"
base_data_path="${base_data_path:-data/AeBAD_crop}"
out_root="${out_root:-data_store/text_counterpart}"
noun_csv="reference/2024-ICML-TAC/data/WordNetNouns.csv"

if [[ ! -d "$crop_data_path/bottle" ]]; then
    echo "子图目录不存在：$crop_data_path/bottle。请先运行 anomalyncd 完成裁剪。" >&2
    exit 1
fi

for category in "${categories[@]}"; do
    echo "== build text counterpart: $category =="
    python - "$category" <<PYEOF
import os, sys
from models.modules._tac_text import build_or_load

category = sys.argv[1]
build_or_load(
    novel_image_root=os.path.join("$crop_data_path", category),
    base_image_root="$base_data_path",
    category=category,
    out_root="$out_root",
    noun_csv="$noun_csv",
)
PYEOF
done
echo "All text counterparts built."
