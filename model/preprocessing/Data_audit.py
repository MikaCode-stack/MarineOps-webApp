from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Expected dataset structure:
#
# bepli/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── annotations/
#     ├── train.json
#     ├── val.json
#     └── test.json
#
# Change only DATASET_ROOT if folder is elsewhere.
DATASET_ROOT = Path("bepli")
IMAGES_ROOT = DATASET_ROOT / "images"
ANNOTATIONS_ROOT = DATASET_ROOT / "annotations"

SPLITS = {
    "train": ANNOTATIONS_ROOT / "train.json",
    "val": ANNOTATIONS_ROOT / "val.json",
    "test": ANNOTATIONS_ROOT / "test.json",
}

# Folder where all audit outputs will be saved
OUTPUT_DIR = Path("audit_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# HELPER FUNCTIONS
def load_coco_json(json_path: Path):
    """
    Robust JSON loader.

    Why this version?
    - On some Windows / OneDrive setups, json.load(file_object)
      can behave oddly on large files.
    - Reading bytes first, then decoding, is often more stable.
    """
    with open(json_path, "rb") as f:
        raw = f.read()

    text = raw.decode("utf-8-sig", errors="replace")
    return json.loads(text)


def summarize_numeric(series):
    """
    Return a dictionary of summary statistics for a numeric pandas Series.
    """
    series = pd.Series(series).dropna()

    if len(series) == 0:
        return {
            "count": 0,
            "min": np.nan,
            "p1": np.nan,
            "p5": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "mean": np.nan
        }

    return {
        "count": int(series.count()),
        "min": float(series.min()),
        "p1": float(series.quantile(0.01)),
        "p5": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
        "mean": float(series.mean())
    }


def save_bar_plot(series, title, xlabel, ylabel, out_path, rotation=45):
    """
    Save a bar plot from a pandas Series.
    """
    plt.figure(figsize=(12, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_hist(series, title, xlabel, out_path, bins=50, log_x=False, log_y=False):
    """
    Save a histogram from a numeric pandas Series.
    """
    data = pd.Series(series).dropna()

    if log_x:
        data = data[data > 0]

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins)

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# MAIN AUDIT FUNCTION
def audit_split(split_name, json_path):
    """
    Audit a single dataset split (train / val / test).

    This function:
    - loads COCO JSON
    - checks image consistency between JSON and disk
    - flattens annotations into a DataFrame
    - computes summary statistics
    - flags suspicious boxes
    - prints key results
    - saves CSV and plots
    """

    print(f"\n{'=' * 70}")
    print(f"AUDITING SPLIT: {split_name.upper()}")
    print(f"JSON: {json_path}")
    print(f"{'=' * 70}")

    coco = load_coco_json(json_path)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    print(f"Images in JSON: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")

    # Map category IDs to category names
    # Important: COCO category IDs may start at 1, not 0
    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # Map image IDs to image metadata
    image_id_to_info = {img["id"]: img for img in images}

    # Check image files on disk
    image_dir = IMAGES_ROOT / split_name
    actual_image_files = set()

    if image_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            actual_image_files.update([p.name for p in image_dir.glob(ext)])
    else:
        print(f"WARNING: image directory not found: {image_dir}")

    json_image_names = set(img.get("file_name", "") for img in images)

    # Images listed in JSON but not present in folder
    missing_on_disk = sorted(json_image_names - actual_image_files)

    # Images present in folder but not listed in JSON
    extra_on_disk = sorted(actual_image_files - json_image_names)

    # Flatten annotations into a table
    ann_rows = []
    invalid_bbox_count = 0
    missing_image_ref_count = 0

    for ann in annotations:
        image_id = ann.get("image_id")
        img = image_id_to_info.get(image_id)

        # Annotation references an image_id that does not exist in images list
        if img is None:
            missing_image_ref_count += 1
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            invalid_bbox_count += 1
            continue

        # Convert bbox safely to floats
        try:
            x, y, w, h = map(float, bbox)
        except Exception:
            invalid_bbox_count += 1
            continue

        img_w = float(img.get("width", np.nan))
        img_h = float(img.get("height", np.nan))

        # Compute bbox area
        bbox_area = np.nan
        norm_bbox_area = np.nan

        if pd.notna(w) and pd.notna(h):
            bbox_area = w * h

            # Normalized area = bbox area / image area
            if pd.notna(img_w) and pd.notna(img_h) and img_w > 0 and img_h > 0:
                norm_bbox_area = bbox_area / (img_w * img_h)

        # Count invalid sizes
        if w <= 0 or h <= 0:
            invalid_bbox_count += 1

        ann_rows.append({
            "annotation_id": ann.get("id"),
            "image_id": image_id,
            "file_name": img.get("file_name"),
            "img_width": img_w,
            "img_height": img_h,
            "category_id": ann.get("category_id"),
            "category_name": category_id_to_name.get(
                ann.get("category_id"),
                f"cat_{ann.get('category_id')}"
            ),
            "x": x,
            "y": y,
            "bbox_width": w,
            "bbox_height": h,
            "bbox_area": bbox_area,
            "norm_bbox_area": norm_bbox_area,
            "iscrowd": ann.get("iscrowd", 0),
            "ann_area_field": ann.get("area", np.nan),
        })

    ann_df = pd.DataFrame(ann_rows)

    if ann_df.empty:
        print("No valid annotations found.")
        return None, None, None

    # Compare annotation area field to bbox area
    # This is informative only; segmentation area can naturally be smaller than bbox area.
    ann_df["area_ratio_ann_to_bbox"] = ann_df["ann_area_field"] / ann_df["bbox_area"]

    # Build image-level table
    # Count annotations per image
    image_counts = ann_df.groupby("image_id").size().rename("num_annotations").reset_index()

    # All images, including those with zero annotations
    all_images_df = pd.DataFrame(images)[["id", "file_name", "width", "height"]].rename(
        columns={"id": "image_id"}
    )

    all_images_df = all_images_df.merge(image_counts, on="image_id", how="left")
    all_images_df["num_annotations"] = all_images_df["num_annotations"].fillna(0).astype(int)
    all_images_df["exists_on_disk"] = all_images_df["file_name"].isin(actual_image_files)

    # Category summaries
    ann_per_category = ann_df["category_name"].value_counts().sort_values(ascending=False)
    img_per_category = ann_df.groupby("category_name")["image_id"].nunique().sort_values(ascending=False)

    category_summary = pd.DataFrame({
        "annotations": ann_per_category,
        "images_with_class": img_per_category
    }).fillna(0).astype(int)

    # Suspicious annotation flags
    # These do not automatically mean "bad labels".
    # They only highlight cases to inspect statistically.
    suspicious = pd.DataFrame({
        "tiny_abs_4px_area": ann_df["bbox_area"] < 16,
        "tiny_abs_8px_area": ann_df["bbox_area"] < 64,
        "tiny_norm_1e_5": ann_df["norm_bbox_area"] < 1e-5,
        "tiny_norm_1e_4": ann_df["norm_bbox_area"] < 1e-4,
        "very_thin_width_lt_2": ann_df["bbox_width"] < 2,
        "very_thin_height_lt_2": ann_df["bbox_height"] < 2,
        "bbox_w_le_0": ann_df["bbox_width"] <= 0,
        "bbox_h_le_0": ann_df["bbox_height"] <= 0,
        "bbox_x_lt_0": ann_df["x"] < 0,
        "bbox_y_lt_0": ann_df["y"] < 0,
        "bbox_exceeds_img_w": (ann_df["x"] + ann_df["bbox_width"]) > ann_df["img_width"],
        "bbox_exceeds_img_h": (ann_df["y"] + ann_df["bbox_height"]) > ann_df["img_height"],
        "iscrowd_eq_1": ann_df["iscrowd"] == 1,
        "ann_area_gt_bbox_area": ann_df["ann_area_field"] > ann_df["bbox_area"],
    })

    suspicious_summary = suspicious.sum().astype(int).sort_values(ascending=False)

    # Global numeric summaries
    global_box_stats = pd.DataFrame([
        {"metric": "bbox_width", **summarize_numeric(ann_df["bbox_width"])},
        {"metric": "bbox_height", **summarize_numeric(ann_df["bbox_height"])},
        {"metric": "bbox_area", **summarize_numeric(ann_df["bbox_area"])},
        {"metric": "norm_bbox_area", **summarize_numeric(ann_df["norm_bbox_area"])},
        {"metric": "annotations_per_image", **summarize_numeric(all_images_df["num_annotations"])},
        {"metric": "area_ratio_ann_to_bbox", **summarize_numeric(ann_df["area_ratio_ann_to_bbox"])},
    ])

    # Per-class size statistics
    per_class_rows = []

    for cat_name, group in ann_df.groupby("category_name"):
        per_class_rows.append({
            "category_name": cat_name,
            "annotations": len(group),
            "images_with_class": group["image_id"].nunique(),
            "median_bbox_area": group["bbox_area"].median(),
            "median_norm_bbox_area": group["norm_bbox_area"].median(),
            "p5_norm_bbox_area": group["norm_bbox_area"].quantile(0.05),
            "p25_norm_bbox_area": group["norm_bbox_area"].quantile(0.25),
            "p75_norm_bbox_area": group["norm_bbox_area"].quantile(0.75),
            "tiny_norm_lt_1e_4_count": int((group["norm_bbox_area"] < 1e-4).sum()),
            "tiny_norm_lt_1e_5_count": int((group["norm_bbox_area"] < 1e-5).sum())
        })

    per_class_df = pd.DataFrame(per_class_rows).sort_values(by="annotations", ascending=False)

    # Densest images
    densest_images = all_images_df.sort_values(by="num_annotations", ascending=False).head(20)

    # Images listed in JSON but missing on disk
    missing_image_files_df = all_images_df[~all_images_df["exists_on_disk"]].copy()

    # PRINT RESULTS
    print("\n--- File consistency ---")
    print(f"Images listed in JSON but missing on disk: {len(missing_on_disk)}")
    print(f"Images on disk but not listed in JSON: {len(extra_on_disk)}")
    print(f"Missing image references in annotations: {missing_image_ref_count}")
    print(f"Invalid bbox entries: {invalid_bbox_count}")
    print(f"Images with zero annotations: {(all_images_df['num_annotations'] == 0).sum()}")

    print("\n--- Top categories by annotations ---")
    print(category_summary.head(15).to_string())

    print("\n--- Annotation density per image ---")
    print(pd.Series(summarize_numeric(all_images_df["num_annotations"])).to_string())

    print("\n--- Bounding box size summary ---")
    print(global_box_stats.to_string(index=False))

    print("\n--- Suspicious annotation summary ---")
    print(suspicious_summary.to_string())

    print("\n--- Top 20 densest images ---")
    print(densest_images[["file_name", "num_annotations", "width", "height"]].to_string(index=False))

    print("\n--- Smallest classes by annotation count ---")
    print(category_summary.sort_values(by="annotations", ascending=True).head(10).to_string())

    # SAVE OUTPUTS
    split_out = OUTPUT_DIR / split_name
    split_out.mkdir(parents=True, exist_ok=True)

    ann_df.to_csv(split_out / "annotations_flat.csv", index=False)
    all_images_df.to_csv(split_out / "image_annotation_counts.csv", index=False)
    category_summary.to_csv(split_out / "category_summary.csv")
    global_box_stats.to_csv(split_out / "global_box_stats.csv", index=False)
    suspicious_summary.rename("count").to_csv(split_out / "suspicious_summary.csv")
    per_class_df.to_csv(split_out / "per_class_size_stats.csv", index=False)
    densest_images.to_csv(split_out / "densest_images.csv", index=False)
    missing_image_files_df.to_csv(split_out / "missing_images_from_disk.csv", index=False)

    # Save lists of image name mismatches
    with open(split_out / "missing_on_disk.txt", "w", encoding="utf-8") as f:
        for name in missing_on_disk:
            f.write(name + "\n")

    with open(split_out / "extra_on_disk.txt", "w", encoding="utf-8") as f:
        for name in extra_on_disk:
            f.write(name + "\n")

    # SAVE PLOTS
    save_bar_plot(
        ann_per_category,
        f"{split_name.upper()} - Annotations per category",
        "Category",
        "Annotations",
        split_out / "annotations_per_category.png"
    )

    save_bar_plot(
        img_per_category,
        f"{split_name.upper()} - Images with category",
        "Category",
        "Images",
        split_out / "images_per_category.png"
    )

    save_hist(
        all_images_df["num_annotations"],
        f"{split_name.upper()} - Annotations per image",
        "Annotations per image",
        split_out / "annotations_per_image_hist.png"
    )

    save_hist(
        ann_df["bbox_area"],
        f"{split_name.upper()} - BBox area",
        "BBox area (pixels^2)",
        split_out / "bbox_area_hist_logx_logy.png",
        bins=60,
        log_x=True,
        log_y=True
    )

    save_hist(
        ann_df["norm_bbox_area"],
        f"{split_name.upper()} - Normalized bbox area",
        "BBox area / image area",
        split_out / "norm_bbox_area_hist_logx_logy.png",
        bins=60,
        log_x=True,
        log_y=True
    )

    save_hist(
        ann_df["area_ratio_ann_to_bbox"],
        f"{split_name.upper()} - Annotation area / BBox area",
        "Annotation area ratio",
        split_out / "ann_area_ratio_hist.png",
        bins=60
    )

    return ann_df, all_images_df, per_class_df


# =========================================================
# RUN AUDIT FOR ALL SPLITS
# =========================================================
all_ann = []
all_img = []
all_cls = []

for split_name, json_path in SPLITS.items():
    if not json_path.exists():
        print(f"Skipping {split_name}: JSON not found -> {json_path}")
        continue

    ann_df, img_df, cls_df = audit_split(split_name, json_path)

    if ann_df is not None:
        ann_df["split"] = split_name
        img_df["split"] = split_name
        cls_df["split"] = split_name

        all_ann.append(ann_df)
        all_img.append(img_df)
        all_cls.append(cls_df)

# =========================================================
# COMBINED DATASET SUMMARY
# =========================================================
if all_ann:
    combined_ann = pd.concat(all_ann, ignore_index=True)
    combined_img = pd.concat(all_img, ignore_index=True)
    combined_cls = pd.concat(all_cls, ignore_index=True)

    combined_ann.to_csv(OUTPUT_DIR / "all_annotations_flat.csv", index=False)
    combined_img.to_csv(OUTPUT_DIR / "all_image_annotation_counts.csv", index=False)
    combined_cls.to_csv(OUTPUT_DIR / "all_per_class_size_stats.csv", index=False)

    print("\n" + "=" * 70)
    print("COMBINED DATASET SUMMARY")
    print("=" * 70)

    print("\n--- Total annotations by class ---")
    print(combined_ann["category_name"].value_counts().to_string())

    print("\n--- Overall normalized bbox area summary ---")
    print(pd.Series(summarize_numeric(combined_ann["norm_bbox_area"])).to_string())

    print("\n--- Overall annotations per image summary ---")
    print(pd.Series(summarize_numeric(combined_img["num_annotations"])).to_string())