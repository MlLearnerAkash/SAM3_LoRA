#!/usr/bin/env python3
"""
Overlay the top needle query and the needle ground truth on the source image.

For every validation image containing a needle annotation:
  1. one forward with the `needle` prompt (LoRA weights),
  2. the highest-scoring needle query is selected,
  3. its predicted box (green) and the needle GT box (red) are drawn on the
     ORIGINAL full-resolution image,
  4. only that overlay is saved (no crops).

Usage:
  python needle_attn_crops.py --config ... --weights ... \
      --val_data_dir /data/dataset/sponge_needle/coco/valid \
      --out_dir outputs/.../needle_attention_needle_only --limit 500
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image as PILImage, ImageDraw, ImageFont

from needle_patch_ablation import build_model, run

NEEDLE = "needle"
CID = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val_data_dir", required=True)
    ap.add_argument("--out_dir", default="outputs/needle_attn_overlay")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--score-thr", type=float, default=0.0,
                    help="only draw the query box when its score exceeds this")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.config, args.weights, dev)

    vdir = Path(args.val_data_dir)
    coco = json.load(open(vdir / "_annotations.coco.json"))
    imgs = {i["id"]: i for i in coco["images"]}
    cid = [c["id"] for c in coco["categories"] if c["name"].lower() == NEEDLE]
    cid = cid[0] if cid else CID
    gts = {}
    for a in coco["annotations"]:
        if a["category_id"] != cid:
            continue
        x, y, w, h = a["bbox"]
        gts.setdefault(a["image_id"], []).append([x, y, x + w, y + h])

    ids = sorted(gts)
    if args.files:
        ids = [i for i in ids if any(f in imgs[i]["file_name"] for f in args.files)]
    ids = ids[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"images: {len(ids)} -> {out_dir}")

    n_hit = 0
    lines = []
    for k, img_id in enumerate(ids):
        info = imgs[img_id]
        pil = PILImage.open(vdir / info["file_name"]).convert("RGB")
        r = run(model, pil, dev)

        qi = int(torch.argmax(r["scores"]))
        qscore = float(r["scores"][qi])
        qbox = r["det"][qi].tolist()

        dr = ImageDraw.Draw(pil)
        try:
            fnt = ImageFont.load_default()
        except Exception:
            fnt = None
        lw = max(2, round(min(pil.size) / 300))

        if qscore >= args.score_thr:
            dr.rectangle(qbox, outline=(0, 255, 90), width=lw)
            dr.text((max(2, qbox[0] + 3), max(2, qbox[1] + 2)),
                    f"q{qi} p={qscore:.2f}", fill=(0, 255, 90), font=fnt,
                    stroke_width=2, stroke_fill=(0, 0, 0))

        inside = False
        for gx0, gy0, gx1, gy1 in gts[img_id]:
            dr.rectangle([gx0, gy0, gx1, gy1], outline=(255, 30, 30), width=lw)
            dr.text((max(2, gx0 + 3), max(2, gy0 - 12)), NEEDLE,
                    fill=(255, 30, 30), font=fnt,
                    stroke_width=2, stroke_fill=(0, 0, 0))
            if qbox[0] <= gx0 and qbox[1] <= gy0 and qbox[2] >= gx1 and qbox[3] >= gy1:
                inside = True

        stem = Path(info["file_name"]).stem
        out_path = out_dir / f"{stem}_overlay.png"
        pil.save(out_path)

        n_hit += int(inside)
        lines.append({"stem": stem, "query": qi, "score": qscore,
                      "qbox": qbox, "gt": gts[img_id], "gt_in_query": inside})
        print(f"[{k+1}/{len(ids)}] {info['file_name']} q{qi} p={qscore:.2f} "
              f"gt_in_query={inside} -> {out_path}")

    print(f"\nGT inside top-query box: {n_hit}/{len(ids)}")
    (out_dir / "overlays.json").write_text(json.dumps(lines, indent=2))


if __name__ == "__main__":
    main()
