#!/usr/bin/env python3
"""
Ablation metric: are same-class instances merged into one big box?

For each validation image and each class prompt, compare the number of predicted
instances (score > threshold, after NMS) with the number of GT instances of that
prompt. ratio = preds / gt:

    ratio ~ 1.0  -> instances kept separate (good)
    ratio < 1.0  -> merging (e.g. 0.5 = two gauzes reported as one)
    ratio > 1.0  -> fragmentation / duplicates

Usage:
    python ablation_merge.py --config configs/full_lora_config_custom.yaml \
        --weights outputs/.../best_lora_weights.pt --num-samples 60
    python ablation_merge.py --base --num-samples 60
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

from validate_sam3_lora import COCOSegmentDataset, move_to_device


def box_iou(a, b):
    xa = (a[:, 2] - a[:, 0]).clamp(min=0)
    ya = (a[:, 3] - a[:, 1]).clamp(min=0)
    xb = (b[:, 2] - b[:, 0]).clamp(min=0)
    yb = (b[:, 3] - b[:, 1]).clamp(min=0)
    inter_x = (torch.min(a[:, 2], b[:, 2]) - torch.max(a[:, 0], b[:, 0])).clamp(min=0)
    inter_y = (torch.min(a[:, 3], b[:, 3]) - torch.max(a[:, 1], b[:, 1])).clamp(min=0)
    inter = inter_x * inter_y
    union = xa * ya + xb * yb - inter
    return inter / union.clamp(min=1e-9)


def greedy_nms(boxes_cxcywh, scores, iou_thr):
    """Greedy NMS on normalized cx,cy,w,h boxes; returns kept indices."""
    if boxes_cxcywh.numel() == 0:
        return []
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(xyxy[i].unsqueeze(0), xyxy[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_thr]
    return keep


def build_model(config_path, weights_path, base):
    model = build_sam3_image_model(
        device="cuda", compile=False, load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", eval_mode=False,
    )
    if base:
        return model.to("cuda").eval()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    lc = cfg["lora"]
    model = apply_lora_to_model(model, LoRAConfig(
        rank=lc["rank"], alpha=lc["alpha"], dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        apply_to_vision_encoder=lc["apply_to_vision_encoder"],
        apply_to_text_encoder=lc["apply_to_text_encoder"],
        apply_to_geometry_encoder=lc["apply_to_geometry_encoder"],
        apply_to_detr_encoder=lc["apply_to_detr_encoder"],
        apply_to_detr_decoder=lc["apply_to_detr_decoder"],
        apply_to_mask_decoder=lc["apply_to_mask_decoder"],
    ))
    if weights_path:
        load_lora_weights(model, weights_path)
    return model.to("cuda").eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full_lora_config_custom.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--base", action="store_true", help="stock SAM3, no LoRA")
    ap.add_argument("--data-dir", default="/data/dataset/sam3_train_coco")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--num-samples", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    args = ap.parse_args()

    model = build_model(args.config, args.weights, args.base)
    ds = COCOSegmentDataset(args.data_dir, split=args.split)
    n = min(args.num_samples, len(ds))
    loader = DataLoader(range(n), batch_size=1,
                        collate_fn=lambda ids: [ds[int(i)] for i in ids])

    gt = defaultdict(int)
    pred = defaultdict(int)
    images = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            dp = batch[0]
            collated = collate_fn_api([dp], dict_key="input", with_seg_masks=False)["input"]
            collated = move_to_device(collated, torch.device("cuda"))
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(collated)
            final = out[-1] if isinstance(out, (list, tuple)) else out

            logits = final["pred_logits"].float()
            boxes = final["pred_boxes"].float()
            masks = final.get("pred_masks")
            if masks is not None:
                masks = masks.float()

            texts = [q.query_text for q in dp.find_queries]
            n_gt_per_prompt = [len(q.object_ids_output) for q in dp.find_queries]

            for i, text in enumerate(texts):
                key = text.lower()
                images[key] += 1
                gt[key] += n_gt_per_prompt[i]
                if n_gt_per_prompt[i] == 0:
                    continue

                lg = logits[i]                      # [nq, 1]
                score = torch.sigmoid(lg).squeeze(-1)
                if "presence_logit_dec" in final:
                    pres = final["presence_logit_dec"][i]
                    score = score * torch.sigmoid(pres).reshape(-1)[:1]

                keep = score > args.threshold
                if keep.sum() == 0:
                    continue
                n_keep = len(greedy_nms(boxes[i][keep], score[keep], args.nms_iou))
                pred[key] += int(n_keep)

    print(f"\n{'prompt':16s} {'images':>7s} {'gt':>6s} {'pred':>6s} {'pred/gt':>8s}")
    tot_gt = tot_pred = 0
    for k in sorted(gt):
        r = pred[k] / gt[k] if gt[k] else float('nan')
        print(f"{k:16s} {images[k]:7d} {gt[k]:6d} {pred[k]:6d} {r:8.2f}")
        tot_gt += gt[k]
        tot_pred += pred[k]
    print(f"{'ALL':16s} {'':7s} {tot_gt:6d} {tot_pred:6d} {tot_pred / max(1, tot_gt):8.2f}")


if __name__ == "__main__":
    main()
