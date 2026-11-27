#!/usr/bin/env python3
"""
Annotate annotation-free images (the retractor crops) with a must-not-fire
region, so the box head - not just the presence token - gets supervised.

Each such image gets ONE full-image box written as:

    {"category_id": 0, "is_negative_region": true, "bbox": [0, 0, W, H], ...}

`category_id: 0` / `is_negative_region` annotations are deliberately NOT
Objects in COCOSegmentDataset: they never reach the matcher, never count in
`num_boxes`, and only feed `IABCEMdetr.loss_neg_region` (gamma=0, per region).

Dry-run by default; pass --write to modify the annotation files. Idempotent:
re-running rewrites the same annotation instead of duplicating it. A
`_annotations.coco.json.bak` is created once, on the first modification.

Usage:
    python3 add_retractor_negatives.py --data-dir /data/dataset/sam3_train_coco
    python3 add_retractor_negatives.py --data-dir /data/dataset/sam3_train_coco --write
    python3 add_retractor_negatives.py --data-dir /data/dataset/sam3_train_coco --revert
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ANN_NAME = "_annotations.coco.json"


def is_negative(ann):
    return ann.get("category_id", 0) == 0 or ann.get("is_negative_region", False)


def process_split(split_dir, write=False):
    ann_file = Path(split_dir) / ANN_NAME
    if not ann_file.exists():
        print(f"  skip (no {ANN_NAME})")
        return

    data = json.loads(ann_file.read_text())
    counts = Counter(a["image_id"] for a in data["annotations"] if not is_negative(a))
    empty = [img for img in data["images"] if counts.get(img["id"], 0) == 0]
    if not empty:
        print("  nothing to do (no annotation-free images)")
        return

    empty_ids = {img["id"] for img in empty}
    kept = [a for a in data["annotations"]
            if not (is_negative(a) and a["image_id"] in empty_ids)]

    next_id = max((a.get("id", 0) for a in data["annotations"]), default=0) + 1
    added = 0
    for img in empty:
        w, h = img.get("width"), img.get("height")
        if not w or not h:
            print(f"  warn: image {img['id']} has no width/height, skipped")
            continue
        kept.append({
            "id": next_id,
            "image_id": img["id"],
            "category_id": 0,
            "iscrowd": 0,
            "bbox": [0, 0, w, h],
            "area": float(w * h),
            "segmentation": [],
            "is_negative_region": True,
        })
        next_id += 1
        added += 1

    print(f"  {added} must-not-fire regions "
          f"(annotations {len(data['annotations'])} -> {len(kept)})")
    if not write:
        print("  dry-run: pass --write to modify the file")
        return

    bak = ann_file.with_suffix(ann_file.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(ann_file, bak)
        print(f"  backup -> {bak.name}")
    data["annotations"] = kept
    ann_file.write_text(json.dumps(data))
    print(f"  wrote {ann_file}")


def revert_split(split_dir):
    ann_file = Path(split_dir) / ANN_NAME
    bak = ann_file.with_suffix(ann_file.suffix + ".bak")
    if bak.exists():
        shutil.copy2(bak, ann_file)
        print(f"  restored {ann_file.name} from {bak.name}")
    else:
        print("  no .bak, nothing to restore")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data/dataset/sam3_train_coco")
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    p.add_argument("--write", action="store_true",
                   help="actually write the annotation files (default: dry-run)")
    p.add_argument("--revert", action="store_true", help="restore from .bak")
    args = p.parse_args()

    for split in args.splits:
        split_dir = Path(args.data_dir) / split
        print(f"[{split}] {split_dir}")
        if not split_dir.exists():
            print("  skip (missing)")
            continue
        if args.revert:
            revert_split(split_dir)
        else:
            process_split(split_dir, write=args.write)


if __name__ == "__main__":
    main()
