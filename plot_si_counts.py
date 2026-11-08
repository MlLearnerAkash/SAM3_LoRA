#!/usr/bin/env python3
"""Overlaid train/valid instance-count barplot per surgical instrument (SI)."""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/dataset/sam3_train_coco")
SPLITS = {"train": "#1f77b4", "valid": "#ff7f0e"}

counts, img_counts, cats = {}, {}, {}
for split in SPLITS:
    d = json.load(open(ROOT / split / "_annotations.coco.json"))
    names = {c["id"]: c["name"] for c in d["categories"]}
    cats[split] = names
    inst, per_img = Counter(), Counter()
    for a in d["annotations"]:
        inst[names[a["category_id"]]] += 1
    for img in d["images"]:
        pass
    # images containing each class
    img_cls = {}
    for a in d["annotations"]:
        img_cls.setdefault(a["image_id"], set()).add(names[a["category_id"]])
    for cls in img_cls.values():
        for c in cls:
            per_img[c] += 1
    counts[split], img_counts[split] = inst, per_img

labels = sorted(set(cats["train"].values()) | set(cats["valid"].values()))
x = np.arange(len(labels))
w = 0.6

fig, ax = plt.subplots(figsize=(11, 5.5))
for i, (split, color) in enumerate(SPLITS.items()):
    vals = [counts[split].get(l, 0) for l in labels]
    ax.bar(x, vals, width=w, color=color, alpha=0.55, label=f"{split} instances",
           edgecolor=color, linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels([l.replace("_", "\n") for l in labels])
ax.set_xlabel("Surgical instrument (SI)")
ax.set_ylabel("Instance count")
ax.set_title("SI instance counts - train vs valid (overlaid)")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.legend(frameon=False)

for xi, l in zip(x, labels):
    tr, va = counts["train"].get(l, 0), counts["valid"].get(l, 0)
    ax.text(xi, max(tr, va), f"{tr} / {va}", ha="center", va="bottom", fontsize=8)

out = ROOT / "si_counts_train_valid.png"
fig.tight_layout()
fig.savefig(out, dpi=160)
print("saved", out)
for l in labels:
    print(f"{l:15s} train={counts['train'].get(l,0):5d} ({img_counts['train'].get(l,0)} imgs)"
          f"  valid={counts['valid'].get(l,0):5d} ({img_counts['valid'].get(l,0)} imgs)")
