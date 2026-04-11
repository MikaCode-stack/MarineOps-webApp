#Applying preprocessing techniques and saving the new cleaned dataset(still with COCO format annotations)
# This script addresses key dataset challenges identified during EDA:
# - Extremely small objects (noise)
# - Dense scenes (object overlap)
# - Class imbalance (dominant "fragment" class)
# - Inconsistent image resolutions
#
# Core strategy:
# 1. Filtering noisy annotations
# 2. Applying tiling to improve small object visibility
# 3. Merging semantically similar classes
# 4. Downsampling dominant classes to reduce bias
import os
import json
import math
import random
import shutil
import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# Computing area of COCO bounding boc [x, y ,w, h] - this will be useful for filtering small object and 
# - evaluating retained aread after clipping during yiling
def coco_bbox_area(bbox):
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


#Cliping a bounding box to the current tile region to prevent objects from spanning multiple tiles(allows to keep only the visible portion inside the tile)
def clip_box_to_tile(bbox, tile_x, tile_y, tile_w, tile_h):
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    tx1 = tile_x
    ty1 = tile_y
    tx2 = tile_x + tile_w
    ty2 = tile_y + tile_h

    ix1 = max(x1, tx1)
    iy1 = max(y1, ty1)
    ix2 = min(x2, tx2)
    iy2 = min(y2, ty2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    if iw <= 0 or ih <= 0:
        return None

    return [ix1 - tile_x, iy1 - tile_y, iw, ih]

#Adjusting segmentation polygons to tile coordinates 
def segmentation_to_tile(segmentation, tile_x, tile_y, tile_w, tile_h):
    if not isinstance(segmentation, list):
        return []

    clipped_segments = []
    for poly in segmentation:
        if not poly or len(poly) < 6:
            continue

        pts = []
        for i in range(0, len(poly), 2):
            px = poly[i]
            py = poly[i + 1]

            if tile_x <= px <= tile_x + tile_w and tile_y <= py <= tile_y + tile_h:
                pts.extend([px - tile_x, py - tile_y])

        if len(pts) >= 6:
            clipped_segments.append(pts)

    return clipped_segments

#mapping from original categories to merged categories 
#rarest classes are ambiguous, therefore are merged for better representation
#e.g.g "other_bottle", "other_container" are merged -> "others"
def build_category_mapping(categories, merge_map):
    old_id_to_name = {c["id"]: c["name"] for c in categories}

    final_names = []
    for c in categories:
        src = c["name"]
        dst = merge_map.get(src, src)
        if dst not in final_names:
            final_names.append(dst)

    final_categories = []
    name_to_new_id = {}
    for idx, name in enumerate(final_names, start=1):
        final_categories.append({"id": idx, "name": name, "supercategory": "plastic_litter"})
        name_to_new_id[name] = idx

    old_id_to_new_id = {}
    for old_id, old_name in old_id_to_name.items():
        merged_name = merge_map.get(old_name, old_name)
        old_id_to_new_id[old_id] = name_to_new_id[merged_name]

    return final_categories, old_id_to_new_id, old_id_to_name

#Core preprocessing pipeline.
def preprocess_and_tile_coco(
    coco_json_path,
    images_dir,
    output_root,
    split_name,
    tile_size=640,
    overlap=0.2,
    min_norm_area=1e-5,
    min_pixel_area=16,
    min_retained_ratio=0.30,
    merge_map=None,
    fragment_keep_ratio=1.0,
    keep_empty_tiles=False,
    seed=42,
):
    random.seed(seed)

    merge_map = merge_map or {}

    coco = load_json(coco_json_path)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    final_categories, old_id_to_new_id, old_id_to_name = build_category_mapping(categories, merge_map)

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    output_images_dir = Path(output_root) / split_name / "images"
    output_json_path = Path(output_root) / split_name / "annotations.json"

    ensure_dir(output_images_dir)

    new_images = []
    new_annotations = []

    new_img_id = 1
    new_ann_id = 1

    #computing strides based on overlap, this ensures:
    # - objects near tile borders are not lost
    # - better coverage of dense scenes
    stride = max(1, int(tile_size * (1.0 - overlap)))

    stats = {
        "original_images": len(images),
        "original_annotations": len(annotations),
        "dropped_tiny": 0,
        "dropped_low_retained_ratio": 0,
        "dropped_small_after_clip": 0,
        "dropped_fragment_downsample": 0,
        "written_tiles": 0,
        "written_annotations": 0,
    }

    for img in images:
        image_id = img["id"]
        file_name = img["file_name"]
        width = img["width"]
        height = img["height"]

        image_path = Path(images_dir) / file_name
        if not image_path.exists():
            print(f"[WARN] Missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        image_anns = anns_by_image.get(image_id, [])

        filtered_anns = []
        image_area = width * height

        for ann in image_anns:
            bbox = ann["bbox"]
            area = coco_bbox_area(bbox)
            norm_area = area / image_area if image_area > 0 else 0.0

            #filtering noisy annotations(very small objects)
            if norm_area < min_norm_area:
                stats["dropped_tiny"] += 1
                continue

            class_name = old_id_to_name[ann["category_id"]]

            #Downsampling dominant class(fragment classes cause bias)
            if class_name == "fragment" and fragment_keep_ratio < 1.0:
                if random.random() > fragment_keep_ratio:
                    stats["dropped_fragment_downsample"] += 1
                    continue

            ann_copy = ann.copy()
            ann_copy["category_id"] = old_id_to_new_id[ann["category_id"]]
            filtered_anns.append(ann_copy)

        x_starts = list(range(0, max(1, width - tile_size + 1), stride))
        y_starts = list(range(0, max(1, height - tile_size + 1), stride))

        if not x_starts or x_starts[-1] != max(0, width - tile_size):
            x_starts.append(max(0, width - tile_size))
        if not y_starts or y_starts[-1] != max(0, height - tile_size):
            y_starts.append(max(0, height - tile_size))

        for ty in sorted(set(y_starts)):
            for tx in sorted(set(x_starts)):
                tile_w = min(tile_size, width - tx)
                tile_h = min(tile_size, height - ty)

                tile_anns = []

                for ann in filtered_anns:
                    orig_bbox = ann["bbox"]
                    orig_area = coco_bbox_area(orig_bbox)

                    clipped_bbox = clip_box_to_tile(orig_bbox, tx, ty, tile_w, tile_h)
                    if clipped_bbox is None:
                        continue

                    clipped_area = coco_bbox_area(clipped_bbox)
                    retained_ratio = clipped_area / orig_area if orig_area > 0 else 0.0

                    #ensuring sufficient portion of object is kept in tile
                    if retained_ratio < min_retained_ratio:
                        stats["dropped_low_retained_ratio"] += 1
                        continue
                    
                    #removing objects that are too msall after clippping
                    if clipped_area < min_pixel_area:
                        stats["dropped_small_after_clip"] += 1
                        continue

                    clipped_seg = segmentation_to_tile(
                        ann.get("segmentation", []), tx, ty, tile_w, tile_h
                    )

                    new_ann = {
                        "id": new_ann_id,
                        "image_id": new_img_id,
                        "category_id": ann["category_id"],
                        "bbox": [round(v, 2) for v in clipped_bbox],
                        "area": round(clipped_area, 2),
                        "iscrowd": ann.get("iscrowd", 0),
                        "segmentation": clipped_seg,
                    }
                    tile_anns.append(new_ann)
                    new_ann_id += 1

                #skipping tiles without annotation
                if not tile_anns and not keep_empty_tiles:
                    continue

                tile = image.crop((tx, ty, tx + tile_w, ty + tile_h))
                tile_file_name = f"{Path(file_name).stem}_x{tx}_y{ty}.png"
                tile_out_path = output_images_dir / tile_file_name
                tile.save(tile_out_path)

                new_images.append({
                    "id": new_img_id,
                    "file_name": tile_file_name,
                    "width": tile_w,
                    "height": tile_h,
                    "license": img.get("license", 1),
                })

                new_annotations.extend(tile_anns)

                new_img_id += 1
                stats["written_tiles"] += 1
                stats["written_annotations"] += len(tile_anns)

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": final_categories,
    }

    save_json(new_coco, output_json_path)

    print("\n=== PREPROCESSING COMPLETE ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"output_json: {output_json_path}")
    print(f"output_images_dir: {output_images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and tile COCO dataset for small-object detection.")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to input COCO json.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images folder for the split.")
    parser.add_argument("--output_root", type=str, required=True, help="Output root folder.")
    parser.add_argument("--split_name", type=str, required=True, help="Split name, e.g. train / val / test.")
    parser.add_argument("--tile_size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min_norm_area", type=float, default=1e-5)
    parser.add_argument("--min_pixel_area", type=float, default=16)
    parser.add_argument("--min_retained_ratio", type=float, default=0.30)
    parser.add_argument("--fragment_keep_ratio", type=float, default=0.7)
    parser.add_argument("--keep_empty_tiles", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merge_map = {
        "other_bottle": "others",
        "other_container": "others",
        "other_fishing_gear": "others",
        "other_string": "others",
    }

    preprocess_and_tile_coco(
        coco_json_path=args.coco_json,
        images_dir=args.images_dir,
        output_root=args.output_root,
        split_name=args.split_name,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_norm_area=args.min_norm_area,
        min_pixel_area=args.min_pixel_area,
        min_retained_ratio=args.min_retained_ratio,
        merge_map=merge_map,
        fragment_keep_ratio=args.fragment_keep_ratio,
        keep_empty_tiles=args.keep_empty_tiles,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()