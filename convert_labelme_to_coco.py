#!/usr/bin/env python3
"""
Convert LabelMe annotations to the COCO layout expected by the SAM3 LoRA trainers.

Input: a directory (flat or nested) containing LabelMe `<image>.json` files next
to their images. Each JSON may hold any number of polygon / rectangle / circle
shapes; every shape becomes one COCO annotation under the category named by its
`label`.

Output matches the layout documented in README.md "1. Prepare Your Data":

    <out>/train/<image files>
    <out>/train/_annotations.coco.json
    <out>/valid/<image files>
    <out>/valid/_annotations.coco.json
    <out>/test/...

Usage:
    # 90/10 train/valid split, images hard-linked (no disk duplication)
    python convert_labelme_to_coco.py --src /data/dataset/sam3_train \
        --out /data/dataset/sam3_train_coco

    # explicit split ratios + a held-out test split
    python convert_labelme_to_coco.py --src /data/dataset/sam3_train \
        --out /data/dataset/sam3_train_coco --valid-frac 0.1 --test-frac 0.05

    # keep images in place instead of linking/copying
    python convert_labelme_to_coco.py --src ... --out ... --link-mode symlink

Notes
- Image and annotation ids are **0-based and contiguous** (0..N-1), so the
  COCO `image["id"]` equals the dataset position the loaders index with. A
  1-based export would make `validate_sam3_lora.py` read the wrong image and
  overrun the last index.
- Polygon masks are stored as COCO polygon segmentation, so no mask rasterization
  is needed; the SAM3 COCO loader decodes polygons through pycocotools anyway.
- `is_instance_exhaustive` is set on every image: `validate_sam3_lora.py`
  needs it for the cgF1 metric.
- Optional per-class text prompts: pass `--descriptions classes.json` mapping
  `{"scissors": "surgical scissors ..."}` and a `_descriptions.json` sidecar is
  written into each split, which the trainer picks up automatically
  (`prompt_source: auto`).
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def polygon_area(points):
    """Shoelace area, absolute."""
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def bbox_from_points(points):
    pts = np.asarray(points, dtype=np.float64)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def circle_to_polygon(points, num_segments=64):
    """LabelMe circle = [center, edge_point] -> polygon approximation."""
    (cx, cy), (ex, ey) = points[0], points[1]
    r = float(np.hypot(ex - cx, ey - cy))
    angles = np.linspace(0.0, 2.0 * np.pi, num_segments, endpoint=False)
    return [[float(cx + r * np.cos(a)), float(cy + r * np.sin(a))] for a in angles]


def rectangle_to_polygon(points):
    (x0, y0), (x1, y1) = points[0], points[1]
    return [[float(x0), float(y0)], [float(x1), float(y0)],
            [float(x1), float(y1)], [float(x0), float(y1)]]


def ring_to_polygon(points):
    (cx, cy), (px, py) = points[0], points[1]
    r = float(np.hypot(px - cx, py - cy))
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    outer = [[float(cx + r * np.cos(a)), float(cy + r * np.sin(a))] for a in angles]
    inner_r = r * 0.5
    inner = [[float(cx + inner_r * np.cos(a)), float(cy + inner_r * np.sin(a))]
             for a in reversed(angles)]
    return outer + inner + [outer[0]]


def line_to_polygon(points, half_width=2.0):
    """LabelMe line -> thin rectangle so it can carry a mask."""
    (x0, y0), (x1, y1) = points[0], points[1]
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy))
    if length == 0:
        return None
    nx, ny = -dy / length * half_width, dx / length * half_width
    return [[float(x0 + nx), float(y0 + ny)], [float(x1 + nx), float(y1 + ny)],
            [float(x1 - nx), float(y1 - ny)], [float(x0 - nx), float(y0 - ny)]]


def shape_to_polygons(shape):
    """Return a list of COCO polygon rings (each a flat [x1, y1, x2, y2, ...])."""
    stype = shape.get("shape_type", "polygon")
    pts = shape.get("points", [])

    if stype == "polygon":
        polygons = [pts]
    elif stype == "rectangle":
        polygons = [rectangle_to_polygon(pts)] if len(pts) >= 2 else []
    elif stype == "circle":
        polygons = [circle_to_polygon(pts)] if len(pts) >= 2 else []
    elif stype == "ring":
        polygons = [ring_to_polygon(pts)] if len(pts) >= 2 else []
    elif stype == "line" or stype == "linestrip":
        poly = line_to_polygon(pts) if len(pts) >= 2 else None
        polygons = [poly] if poly else []
    elif stype == "point":
        polygons = []
    else:
        polygons = [pts] if len(pts) >= 3 else []

    out = []
    for ring in polygons:
        if len(ring) < 3:
            continue
        flat = []
        for p in ring:
            flat.extend([float(p[0]), float(p[1])])
        if polygon_area(ring) > 0.0:
            out.append(flat)
    return out


# ----------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------

IGNORED_DIRS = {"wandb", "checkpoints", "logs", "dumps", "tensorboard",
               "__pycache__", ".git", "outputs"}


def find_annotation_files(src, recursive=False):
    """LabelMe JSONs in `src` (flat by default); skips unrelated sub-projects."""
    root = Path(src)
    if not recursive:
        candidates = sorted(root.glob("*.json"))
    else:
        candidates = sorted(root.rglob("*.json"))

    out = []
    for p in candidates:
        if p.name.startswith("_") or p.name == "_descriptions.json":
            continue
        if any(part.lower() in IGNORED_DIRS or part.lower().startswith("coco_format")
               for part in p.relative_to(root).parts[:-1]):
            continue
        out.append(p)
    return out


def resolve_image(json_path, data):
    """Locate the image referenced by a LabelMe JSON."""
    candidates = []
    if data.get("imagePath"):
        rel = Path(data["imagePath"])
        candidates.append(json_path.parent / rel.name)
        candidates.append(json_path.parent / rel)
        candidates.append(Path(rel))
    stem = json_path.with_suffix("")
    candidates.extend(Path(str(stem) + ext) for ext in IMAGE_EXTS)
    candidates.extend(Path(str(stem) + ext.upper()) for ext in IMAGE_EXTS)
    for c in candidates:
        if c.is_file():
            return c
    return None


def read_labelme(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def build_split(records, out_dir, link_mode, descriptions):
    """Write one split directory: images + _annotations.coco.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted({label for rec in records for label in rec["labels"]})
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(labels)]
    cat_id = {c["name"]: c["id"] for c in categories}

    coco = {
        "info": {"description": f"SAM3 LoRA dataset ({out_dir.name})",
                 "version": "1.0", "year": 2026},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    ann_id = 0
    image_id_of = {}
    for rec in records:
        seg = rec["segmentation"]
        bbox = rec["bbox"]
        area = float(rec["area"])

        file_name = rec["file_name"]
        image_id = image_id_of.get(file_name)
        if image_id is None:
            # 0-based contiguous ids. The loaders index the dataset positionally
            # (`self.image_ids[idx]`) while SAM3's query metadata carries
            # `original_image_id`, so ids must equal positions or validation
            # reads the wrong image (and overruns by one at the end).
            image_id = len(coco["images"])
            image_id_of[file_name] = image_id
            coco["images"].append({
                "id": image_id,
                "file_name": file_name,
                "width": int(rec["width"]),
                "height": int(rec["height"]),
                "is_instance_exhaustive": True,
            })

        if seg:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id[rec["category"]],
                "segmentation": seg,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "ignore": 0,
            })
            ann_id += 1

    with open(out_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)

    _check_zero_based_ids(coco, out_dir)

    if descriptions:
        sidecar = out_dir / "_descriptions.json"
        with open(sidecar, "w") as f:
            json.dump({"categories": descriptions}, f, indent=2)

    # Stage image files (after JSON, so a failure keeps annotations discoverable)
    linked, copied = 0, 0
    for rec in records:
        dst = out_dir / rec["file_name"]
        if dst.exists():
            continue
        src = rec["src"]
        try:
            if link_mode == "hardlink":
                os.link(src, dst)
                linked += 1
            elif link_mode == "symlink":
                os.symlink(src.resolve(), dst)
                linked += 1
            else:
                shutil.copy2(src, dst)
                copied += 1
        except OSError:
            shutil.copy2(src, dst)
            copied += 1

    n_empty = sum(1 for r in records if not r["segmentation"])
    print(f"  {out_dir}: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations, {len(categories)} categories"
          f" ({n_empty} empty images), staged {linked} linked / {copied} copied")
    return coco


def _check_zero_based_ids(coco, out_dir):
    """Fail loudly if image/annotation ids are not contiguous and 0-based."""
    img_ids = sorted(img["id"] for img in coco["images"])
    expected = list(range(len(img_ids)))
    if img_ids != expected:
        raise ValueError(
            f"{out_dir}: image ids must be 0-based and contiguous "
            f"(got {img_ids[:5]}...{img_ids[-3:]}, expected 0..{len(img_ids) - 1})"
        )
    ann_ids = sorted(a["id"] for a in coco["annotations"])
    if ann_ids != list(range(len(ann_ids))):
        raise ValueError(
            f"{out_dir}: annotation ids must be 0-based and contiguous"
        )
    known = set(img_ids)
    stray = {a["image_id"] for a in coco["annotations"]} - known
    if stray:
        raise ValueError(f"{out_dir}: annotations reference unknown image ids {sorted(stray)[:5]}")


def main():
    ap = argparse.ArgumentParser(description="LabelMe -> COCO for SAM3 LoRA")
    ap.add_argument("--src", required=True, help="LabelMe root directory")
    ap.add_argument("--out", required=True, help="Output root (gets train/valid/test)")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"],
                    default="hardlink")
    ap.add_argument("--descriptions", default=None,
                    help="JSON mapping {class_name: text_prompt}")
    ap.add_argument("--min-area", type=float, default=1.0,
                    help="Drop shapes smaller than this area in pixels")
    ap.add_argument("--drop-empty", action="store_true",
                    help="Exclude images that have no shapes at all")
    ap.add_argument("--dedupe-stems", action="store_true",
                    help="Deduplicate by file stem (first JSON wins)")
    ap.add_argument("--recursive", action="store_true",
                    help="Scan subdirectories too (existing coco_format*/wandb dirs are skipped)")
    args = ap.parse_args()

    descriptions = None
    if args.descriptions:
        with open(args.descriptions) as f:
            descriptions = json.load(f)

    json_files = find_annotation_files(args.src, recursive=args.recursive)
    if not json_files:
        raise SystemExit(f"No LabelMe JSON files found under {args.src}")

    print(f"Found {len(json_files)} LabelMe JSON files under {args.src}")

    per_image = {}          # stem -> list of annotation records
    seen_stems = set()
    missing_images = []
    skipped_shapes = Counter()

    for jf in json_files:
        data = read_labelme(jf)
        img_path = resolve_image(jf, data)
        if img_path is None:
            missing_images.append(str(jf))
            continue

        if args.dedupe_stems:
            if jf.stem in seen_stems:
                skipped_shapes["duplicate_stem"] += 1
                continue
            seen_stems.add(jf.stem)

        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as im:
                width, height = im.size

        records = []
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if not label:
                skipped_shapes["no_label"] += 1
                continue
            polygons = shape_to_polygons(shape)
            if not polygons:
                skipped_shapes[shape.get("shape_type", "polygon")] += 1
                continue
            # Multi-ring labels (polygon with holes) stay separate annotations:
            # the loaders group by text, so each ring is still one instance.
            for ring in polygons:
                pts = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
                # LabelMe lets points sit slightly outside the image; clip so the
                # COCO bbox never exceeds the declared width/height.
                pts[:, 0] = np.clip(pts[:, 0], 0.0, float(width) - 1.0)
                pts[:, 1] = np.clip(pts[:, 1], 0.0, float(height) - 1.0)
                ring = pts.reshape(-1).tolist()
                area = polygon_area(pts.tolist())
                if area < args.min_area:
                    skipped_shapes["tiny"] += 1
                    continue
                records.append({
                    "src": img_path,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                    "category": label,
                    "segmentation": [ring],
                    "bbox": bbox_from_points(pts.tolist()),
                    "area": area,
                    "labels": {label},
                })

        if not records:
            if args.drop_empty:
                skipped_shapes["empty_image"] += 1
                continue
            records = [{
                "src": img_path,
                "file_name": img_path.name,
                "width": width,
                "height": height,
                "category": "__none__",
                "segmentation": [],
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "area": 0.0,
                "labels": set(),
            }]

        key = str(img_path.resolve())
        per_image.setdefault(key, []).extend(records)

    if missing_images:
        print(f"[warn] {len(missing_images)} JSON files had no image, e.g. {missing_images[:3]}")
    if skipped_shapes:
        print(f"[warn] skipped shapes: {dict(skipped_shapes)}")

    # Group records by image so each image is emitted exactly once per split.
    grouped = []
    for key, records in per_image.items():
        # Merge the per-shape records of one image into a single image entry.
        base = records[0]
        grouped.append({
            "src": base["src"],
            "file_name": base["file_name"],
            "width": base["width"],
            "height": base["height"],
            "records": records,
            "labels": set().union(*[r["labels"] for r in records]),
        })

    if not grouped:
        raise SystemExit("No usable images found.")

    rng = random.Random(args.seed)
    rng.shuffle(grouped)

    n = len(grouped)
    n_test = int(round(n * args.test_frac))
    n_valid = int(round(n * args.valid_frac))
    n_train = n - n_valid - n_test
    if n_train <= 0:
        raise SystemExit("Split fractions leave no training images.")

    splits = {
        "train": grouped[:n_train],
        "valid": grouped[n_train:n_train + n_valid],
        "test": grouped[n_train + n_valid:],
    }

    print(f"Split sizes: train={n_train}, valid={n_valid}, test={n_test}")

    out_root = Path(args.out)
    stats = {}
    for name, images in splits.items():
        if not images:
            continue
        flat = []
        for img in images:
            for rec in img["records"]:
                flat.append(rec)
        coco = build_split(flat, out_root / name, args.link_mode, descriptions)
        stats[name] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "categories": [c["name"] for c in coco["categories"]],
        }

    labels = Counter()
    for img in grouped:
        for rec in img["records"]:
            if rec["category"] != "__none__":
                labels[rec["category"]] += 1
    print("\nClass distribution (whole dataset):")
    for name, count in labels.most_common():
        print(f"  {name:20s} {count}")

    with open(out_root / "conversion_stats.json", "w") as f:
        json.dump({"splits": stats, "class_counts": dict(labels),
                   "source": str(Path(args.src).resolve())}, f, indent=2)

    print(f"\nDone. COCO dataset at {out_root}")
    print(f"Point training.training.data_dir at: {out_root}")


if __name__ == "__main__":
    main()
