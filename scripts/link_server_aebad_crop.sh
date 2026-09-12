#!/usr/bin/env bash
set -euo pipefail

# 任务2 使用服务器上已有的 AeBAD_crop 作为 labeled base 数据；
# 通过 symlink 复用共享数据，避免在紧张的根分区重复复制数据集。
AEBAD_CROP_SRC="${AEBAD_CROP_SRC:-/mnt/data/ejxu/data/AeBAD_crop}"
AEBAD_CROP_DST="${AEBAD_CROP_DST:-data/AeBAD_crop}"

if [[ ! -d "$AEBAD_CROP_SRC/images" || ! -d "$AEBAD_CROP_SRC/masks" ]]; then
    echo "Invalid AeBAD_crop source: $AEBAD_CROP_SRC (images/ and masks/ required)" >&2
    exit 1
fi

if [[ -L "$AEBAD_CROP_DST" ]]; then
    current_target="$(readlink "$AEBAD_CROP_DST")"
    if [[ "$current_target" == "$AEBAD_CROP_SRC" ]]; then
        echo "AeBAD_crop link already exists: $AEBAD_CROP_DST -> $current_target"
    else
        rm "$AEBAD_CROP_DST"
        ln -s "$AEBAD_CROP_SRC" "$AEBAD_CROP_DST"
        echo "Updated AeBAD_crop link: $AEBAD_CROP_DST -> $AEBAD_CROP_SRC"
    fi
elif [[ -e "$AEBAD_CROP_DST" ]]; then
    echo "Refusing to overwrite non-symlink path: $AEBAD_CROP_DST" >&2
    exit 1
else
    mkdir -p "$(dirname "$AEBAD_CROP_DST")"
    ln -s "$AEBAD_CROP_SRC" "$AEBAD_CROP_DST"
    echo "Created AeBAD_crop link: $AEBAD_CROP_DST -> $AEBAD_CROP_SRC"
fi

image_classes="$(find "$AEBAD_CROP_DST/images" -mindepth 1 -maxdepth 1 -type d | wc -l)"
mask_classes="$(find "$AEBAD_CROP_DST/masks" -mindepth 1 -maxdepth 1 -type d | wc -l)"
image_count="$(find "$AEBAD_CROP_DST/images" -type f | wc -l)"
mask_count="$(find "$AEBAD_CROP_DST/masks" -type f | wc -l)"

if [[ "$image_classes" -eq 0 || "$image_classes" -ne "$mask_classes" || "$image_count" -eq 0 || "$image_count" -ne "$mask_count" ]]; then
    echo "Incomplete AeBAD_crop data: image_classes=$image_classes mask_classes=$mask_classes image_count=$image_count mask_count=$mask_count" >&2
    exit 1
fi

echo "AeBAD_crop ready: classes=$image_classes images=$image_count masks=$mask_count"
