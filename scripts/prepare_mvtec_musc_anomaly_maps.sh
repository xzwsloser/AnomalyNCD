#!/usr/bin/env bash
set -euo pipefail

# 将服务器上已有的 MuSc anomaly map 建立为 AnomalyNCD 期望的 symlink 结构，避免复制 PNG。
SRC="${1:-/mnt/data/xrli/mvtec_musc_mebin_nocrop}"
DST="${2:-data/mvtec_musc_anomaly_map}"
MVTec_ROOT="${3:-/mnt/data/xrli/mvtec_anomaly_detection}"

categories=(
    bottle cable capsule carpet grid hazelnut leather metal_nut
    pill screw tile toothbrush transistor wood zipper
)

if [[ ! -d "$SRC" ]]; then
    echo "MuSc map source does not exist: $SRC" >&2
    exit 1
fi
if [[ ! -d "$MVTec_ROOT" ]]; then
    echo "MVTec AD root does not exist: $MVTec_ROOT" >&2
    exit 1
fi

mkdir -p "$DST"
for category in "${categories[@]}"; do
    src_category="$SRC/$category/anomaly_maps"
    if [[ ! -d "$src_category" ]]; then
        echo "Missing source map directory: $src_category" >&2
        exit 1
    fi

    while IFS= read -r anomaly_type; do
        target_dir="$DST/$category/$anomaly_type"
        mkdir -p "$target_dir"
        for source_map in "$src_category/$anomaly_type"/*_crop0.png; do
            [[ -e "$source_map" ]] || continue
            source_map="$(readlink -f "$source_map")"
            filename="$(basename "$source_map")"
            original_name="${filename%_crop0.png}.png"
            target="$target_dir/$original_name"

            if [[ -L "$target" ]]; then
                current_source="$(readlink -f "$target")"
                if [[ "$current_source" != "$source_map" ]]; then
                    echo "Unexpected symlink target: $target -> $current_source (expected $source_map)" >&2
                    exit 1
                fi
            elif [[ -e "$target" ]]; then
                echo "Refusing to overwrite existing file: $target" >&2
                exit 1
            else
                ln -s "$source_map" "$target"
            fi
        done
    done < <(find "$src_category" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
done

total_original=0
total_links=0
for category in "${categories[@]}"; do
    original_count="$(find "$MVTec_ROOT/$category/test" -type f -name '*.png' | wc -l)"
    link_count="$(find "$DST/$category" -type l | wc -l)"
    missing=0

    while IFS= read -r anomaly_type; do
        for original_image in "$MVTec_ROOT/$category/test/$anomaly_type"/*.png; do
            [[ -e "$original_image" ]] || continue
            filename="$(basename "$original_image")"
            if [[ ! -L "$DST/$category/$anomaly_type/$filename" ]]; then
                echo "Missing map symlink: $DST/$category/$anomaly_type/$filename" >&2
                missing=$((missing + 1))
            fi
        done
    done < <(find "$MVTec_ROOT/$category/test" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

    if [[ "$original_count" -ne "$link_count" || "$missing" -ne 0 ]]; then
        echo "Count mismatch for $category: original=$original_count links=$link_count missing=$missing" >&2
        exit 1
    fi

    printf '%-12s original=%3d links=%3d\n' "$category" "$original_count" "$link_count"
    total_original=$((total_original + original_count))
    total_links=$((total_links + link_count))
done

broken_links="$(find "$DST" -xtype l | wc -l)"
if [[ "$broken_links" -ne 0 ]]; then
    echo "Found $broken_links broken symlinks under $DST" >&2
    exit 1
fi
if [[ "$total_original" -ne 1725 || "$total_links" -ne 1725 ]]; then
    echo "Expected 1725 maps, got original=$total_original links=$total_links" >&2
    exit 1
fi

echo "Prepared and verified $total_links anomaly map symlinks under $DST"
